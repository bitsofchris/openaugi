#!/usr/bin/env bash
# Cut a release: local checks → push main → wait for CI on that commit →
# tag → the tag's publish workflow uploads to PyPI (after you approve the
# `pypi` environment on GitHub). The tag goes out only once CI is green on
# the exact commit it points at.
#
# Usage: ./scripts/release.sh 0.4.1
# Before: bump the version in pyproject.toml and src/openaugi/_version.py,
# `uv lock`, and add a "## 0.4.1 — <date>" section to CHANGELOG.md; commit.
# See docs/reference/releasing.md.

set -euo pipefail

version="${1:?usage: scripts/release.sh X.Y.Z}"
tag="v${version}"

fail() { echo "✗ $*" >&2; exit 1; }

[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || fail "not on main"
[ -z "$(git status --porcelain)" ] || fail "working tree not clean"
grep -q "^version = \"${version}\"$" pyproject.toml || fail "pyproject.toml is not ${version}"
grep -q "__version__ = \"${version}\"" src/openaugi/_version.py || fail "_version.py is not ${version}"
grep -q "^## ${version} " CHANGELOG.md || fail "CHANGELOG.md has no '## ${version} — <date>' section"
git rev-parse -q --verify "refs/tags/${tag}" >/dev/null && fail "tag ${tag} already exists"

echo "=== Local checks ==="
./scripts/check.sh

sha="$(git rev-parse HEAD)"
echo "=== Push main (${sha:0:7}) ==="
git push origin main

echo "=== Wait for CI on ${sha:0:7} ==="
run_id=""
for _ in $(seq 1 30); do
  run_id="$(gh run list --workflow ci.yml --commit "$sha" --limit 1 --json databaseId -q '.[0].databaseId' || true)"
  [ -n "$run_id" ] && break
  sleep 5
done
[ -n "$run_id" ] || fail "no CI run appeared for ${sha:0:7}"
gh run watch "$run_id" --exit-status --interval 15 >/dev/null || fail "CI failed: $(gh run view "$run_id" --json url -q .url)"
echo "✓ CI green"

echo "=== Tag ${tag} ==="
git tag -a "$tag" -m "$version"
git push origin "$tag"

sleep 5
publish="$(gh run list --workflow publish.yml --limit 1 --json url -q '.[0].url' || true)"
echo ""
echo "✓ ${tag} pushed. Approve the pypi environment to publish:"
echo "  ${publish:-https://github.com/bitsofchris/openaugi/actions}"
