"""OpenAugi auth — optional OAuth layer for remote MCP access.

Only imported when `openaugi serve --auth cloudflare` is used.
Default stdio/HTTP modes have zero auth code involved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def configure_auth(mcp_server: FastMCP, provider: str, config: dict) -> None:
    """Configure OAuth authentication on the MCP server.

    Must be called BEFORE mcp_server.run(). Sets up token verification
    and registers OAuth proxy routes on the existing FastMCP instance.

    Args:
        mcp_server: The FastMCP instance to configure.
        provider: Auth provider name. Currently only "cloudflare".
        config: Full openaugi config dict (reads [remote] section).
    """
    if provider != "cloudflare":
        raise ValueError(f"Unknown auth provider: {provider!r}. Supported: cloudflare")

    remote = config.get("remote", {})
    required = ["team_name", "client_id", "audience", "server_url"]
    missing = [k for k in required if not remote.get(k)]
    if missing:
        raise ValueError(
            f"Missing [remote] config keys: {', '.join(missing)}. "
            "Add them to ~/.openaugi/config.toml:\n\n"
            "[remote]\n"
            'team_name = "your-cf-team"\n'
            'client_id = "your-cf-client-id"\n'
            'audience = "your-cf-audience-tag"\n'
            'server_url = "https://mcp.yourdomain.com"\n'
        )

    from openaugi.auth.cloudflare import configure_cloudflare_auth

    configure_cloudflare_auth(
        mcp_server,
        team_name=remote["team_name"],
        client_id=remote["client_id"],
        audience=remote["audience"],
        server_url=remote["server_url"],
    )
