---
name: releasing
description: How a release reaches PyPI — version bump, CHANGELOG, scripts/release.sh (checks, push, wait for CI, tag), the publish workflow and its approval, and verifying the published package.
---

# Releasing

## When to use

Cutting a new version of `openaugi` on PyPI, or debugging why one didn't publish.

## How it works

Two workflows. **CI** (`.github/workflows/ci.yml`) runs lint, pyright and the
tests on Python 3.12 and 3.13 for every push to `main` and every PR.
**Publish** (`.github/workflows/publish.yml`) runs on a `v*.*.*` tag: the same
tests, then a build, then an upload to PyPI that waits for a reviewer to
approve the `pypi` environment.

The rule: **a tag goes out only after CI is green on the commit it points at.**
Pushing `main` and a tag together starts both workflows at once, so a failure
shows up only after the release has started. `scripts/release.sh` enforces the
order.

1. Bump the version in `pyproject.toml` and `src/openaugi/_version.py`, run
   `uv lock` (it only rewrites openaugi's own version line), and turn
   `## Unreleased` in `CHANGELOG.md` into `## X.Y.Z — YYYY-MM-DD` with upgrade
   notes. Commit as `release: X.Y.Z`.
2. `./scripts/release.sh X.Y.Z` — checks the version is consistent, runs
   `scripts/check.sh`, pushes `main`, waits for that commit's CI run, and only
   if it passes, tags and pushes `vX.Y.Z`.
3. Approve the `pypi` environment on the publish run (the script prints its URL).
4. Verify from PyPI in a clean environment:
   `uvx --from openaugi==X.Y.Z openaugi --version`.

Versioning: 0.x — a new feature bumps the minor (0.3 → 0.4); fixes bump the patch.

## Tests behave the same on a laptop and in CI

`tests/conftest.py` makes every run hermetic before any test module is
imported: HOME is an empty temp folder (no `~/.openaugi/config.toml` or `.env`),
`OPENAUGI_*` and API-key variables are cleared, colour is off and CI detection
(`GITHUB_ACTIONS`, `CI`, `FORCE_COLOR`) is removed. A test that needs a config or
a variable sets it with monkeypatch. So `scripts/check.sh` on a configured
machine sees what CI sees, and a test can't pass only because of the
developer's own setup.
