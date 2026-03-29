"""Cloudflare Access OAuth integration for OpenAugi MCP server.

Uses Cloudflare Access as an OAuth/OIDC authorization server.
Validates JWTs issued by Cloudflare Access and proxies OAuth endpoints
to handle a Claude.ai bug where it ignores metadata URLs.

Only imported when --auth cloudflare is passed to serve/up.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import httpx
import jwt

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# JWKS cache TTL in seconds
_JWKS_CACHE_TTL = 3600


class CloudflareTokenVerifier:
    """Verify JWTs issued by Cloudflare Access.

    Fetches public keys from Cloudflare's JWKS endpoint and validates
    token signature, audience, and expiry.
    """

    def __init__(self, team_name: str, audience: str):
        self.certs_url = f"https://{team_name}.cloudflareaccess.com/cdn-cgi/access/certs"
        self.audience = audience
        self._jwks_cache: dict | None = None
        self._jwks_fetched_at: float = 0

    def _get_jwks(self) -> dict:
        """Fetch and cache JWKS keys from Cloudflare Access."""
        now = time.time()
        if self._jwks_cache is not None and (now - self._jwks_fetched_at) < _JWKS_CACHE_TTL:
            return self._jwks_cache

        resp = httpx.get(self.certs_url, timeout=10)
        resp.raise_for_status()
        result: dict = resp.json()
        self._jwks_cache = result
        self._jwks_fetched_at = now
        return result

    def verify(self, token: str) -> dict | None:
        """Verify a JWT token. Returns decoded payload or None on failure."""
        try:
            jwks = self._get_jwks()
            # Get the key ID from the token header
            unverified = jwt.get_unverified_header(token)
            kid = unverified.get("kid")
            if not kid:
                logger.warning("JWT missing kid header")
                return None

            # Find the matching public key
            public_key = None
            for key_data in jwks.get("keys", []):
                if key_data.get("kid") == kid:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)  # type: ignore[attr-defined]
                    break

            if public_key is None:
                logger.warning("No matching key found for kid=%s", kid)
                return None

            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.audience,
                options={"verify_exp": True},
            )
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("JWT validation failed: %s", e)
            return None
        except httpx.HTTPError as e:
            logger.error("Failed to fetch JWKS: %s", e)
            return None


def configure_cloudflare_auth(
    mcp_server: FastMCP,
    *,
    team_name: str,
    client_id: str,
    audience: str,
    server_url: str,
) -> None:
    """Configure Cloudflare Access OAuth on a FastMCP server.

    Adds:
    - ASGI middleware that validates Bearer tokens on /mcp
    - /.well-known/oauth-authorization-server metadata
    - /.well-known/oauth-protected-resource metadata
    - /authorize proxy -> Cloudflare Access
    - /token proxy -> Cloudflare Access
    - /register fake DCR handler

    Must be called BEFORE mcp.run().
    """
    cf_base = f"https://{team_name}.cloudflareaccess.com"
    cf_auth_url = f"{cf_base}/cdn-cgi/access/sso/oidc/{client_id}/authorization"
    cf_token_url = f"{cf_base}/cdn-cgi/access/sso/oidc/{client_id}/token"

    verifier = CloudflareTokenVerifier(team_name, audience)

    # Store verifier on the mcp instance so middleware can access it
    mcp_server._cf_verifier = verifier  # type: ignore[attr-defined]

    _register_routes(mcp_server, cf_auth_url, cf_token_url, client_id, server_url)
    _register_auth_middleware(mcp_server, verifier)

    logger.info(
        "Cloudflare Access OAuth configured (team=%s, server=%s)",
        team_name,
        server_url,
    )


def _register_routes(
    mcp_server: FastMCP,
    cf_auth_url: str,
    cf_token_url: str,
    client_id: str,
    server_url: str,
) -> None:
    """Register OAuth metadata and proxy routes on the MCP server."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse, RedirectResponse, Response

    @mcp_server.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def oauth_as_metadata(request: Request) -> Response:
        return JSONResponse(
            {
                "issuer": server_url,
                "authorization_endpoint": f"{server_url}/authorize",
                "token_endpoint": f"{server_url}/token",
                "registration_endpoint": f"{server_url}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": [
                    "client_secret_post",
                    "client_secret_basic",
                ],
            }
        )

    @mcp_server.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def oauth_pr_metadata(request: Request) -> Response:
        return JSONResponse(
            {
                "resource": server_url,
                "authorization_servers": [server_url],
                "bearer_methods_supported": ["header"],
            }
        )

    @mcp_server.custom_route("/authorize", methods=["GET", "POST"])
    async def authorize_proxy(request: Request) -> Response:
        """Redirect to Cloudflare Access authorization, passing through params."""
        params = str(request.query_params)
        redirect_url = f"{cf_auth_url}?{params}" if params else cf_auth_url
        return RedirectResponse(redirect_url, status_code=302)

    @mcp_server.custom_route("/token", methods=["POST", "OPTIONS"])
    async def token_proxy(request: Request) -> Response:
        """Forward token exchange to Cloudflare Access."""
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                },
            )

        body = await request.body()
        content_type = request.headers.get("content-type", "application/x-www-form-urlencoded")

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                cf_token_url,
                content=body,
                headers={"Content-Type": content_type},
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={"Content-Type": resp.headers.get("content-type", "application/json")},
        )

    @mcp_server.custom_route("/register", methods=["POST", "OPTIONS"])
    async def register_handler(request: Request) -> Response:
        """Fake Dynamic Client Registration.

        Cloudflare Access doesn't support DCR. We echo back the registration
        request with the real client_id so Claude can proceed. Users can also
        skip this by entering client_id/secret manually in Claude's Advanced Settings.
        """
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                },
            )

        body = await request.body()
        try:
            reg_request = json.loads(body) if body else {}
        except json.JSONDecodeError:
            reg_request = {}

        # Echo back a valid registration response with the real client_id
        reg_response = {
            "client_id": client_id,
            "client_name": reg_request.get("client_name", "claude"),
            "redirect_uris": reg_request.get("redirect_uris", []),
            "grant_types": reg_request.get("grant_types", ["authorization_code"]),
            "response_types": reg_request.get("response_types", ["code"]),
            "token_endpoint_auth_method": reg_request.get(
                "token_endpoint_auth_method", "client_secret_post"
            ),
        }

        return JSONResponse(reg_response, status_code=201)


def _register_auth_middleware(mcp_server: FastMCP, verifier: CloudflareTokenVerifier) -> None:
    """Register ASGI middleware that validates Bearer tokens on /mcp requests.

    We hook into FastMCP's streamable_http_app() by monkey-patching it to
    wrap the returned Starlette app with our auth middleware. This avoids
    touching FastMCP internals beyond the one method override.
    """
    original_http_app = mcp_server.streamable_http_app

    def patched_http_app():
        app = original_http_app()
        from starlette.types import ASGIApp, Receive, Scope, Send

        class CloudflareAuthMiddleware:
            """ASGI middleware: require valid Bearer token on /mcp requests."""

            def __init__(self, app: ASGIApp):
                self.app = app

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] != "http":
                    await self.app(scope, receive, send)
                    return

                path = scope.get("path", "")

                # Only protect /mcp — let .well-known, /authorize, /token, /register through
                if not path.startswith("/mcp"):
                    await self.app(scope, receive, send)
                    return

                # Extract Bearer token from Authorization header
                headers = dict(scope.get("headers", []))
                auth_header = headers.get(b"authorization", b"").decode()

                if not auth_header.startswith("Bearer "):
                    from starlette.responses import JSONResponse

                    www_auth = 'Bearer resource_metadata="/.well-known/oauth-protected-resource"'
                    response = JSONResponse(
                        {"error": "unauthorized", "error_description": "Bearer token required"},
                        status_code=401,
                        headers={"WWW-Authenticate": www_auth},
                    )
                    await response(scope, receive, send)
                    return

                token = auth_header[7:]  # Strip "Bearer "
                payload = verifier.verify(token)

                if payload is None:
                    from starlette.responses import JSONResponse

                    response = JSONResponse(
                        {
                            "error": "invalid_token",
                            "error_description": "Token verification failed",
                        },
                        status_code=401,
                        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                    )
                    await response(scope, receive, send)
                    return

                # Token valid — proceed
                await self.app(scope, receive, send)

        # Wrap the app with our middleware
        wrapped = CloudflareAuthMiddleware(app)
        return wrapped

    mcp_server.streamable_http_app = patched_http_app  # type: ignore[method-assign]
