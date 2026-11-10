"""Every control the Augi pane draws has a name in its usage feature list.

The pane logs presses by control name (`augi-ui/hooks/register.tsx`, `usageName`)
and its report lists the names never pressed, from `USAGE_FEATURES`. A Button or
Input added without a name there would never show as unused, so this reads the
source and fails on the gap. The plugin's own tests cannot read files.
"""

import re
from pathlib import Path

SOURCE = Path(__file__).parent.parent / "augi-ui" / "hooks" / "register.tsx"

# Setup's own fields and the first-run screens are settings, not features to keep or cut.
NOT_FEATURES = re.compile(
    r"^(set-|vault$|found$|editor$|file-editor$|starter$|use-starter$|skip-step$)"
)
STILL_FEATURES = {"set-openOnStart", "set-trackUsage"}
# The fold-button helper builds its key from the card it folds: each is listed by name.
HELPER_KEYS = {"head-x"}


def _strings(source: str, name: str) -> list[str]:
    block = re.search(rf"(?:export )?const {name} = \[(.*?)\n\]", source, re.S)
    assert block, f"{name} not found in {SOURCE}"
    return re.findall(r"'([^']+)'", block.group(1))


def _usage_name(element: str, whole: list[str], prefixes: list[str]) -> str:
    """The same rule as `usageName` in the source."""
    if element in whole:
        return element
    for prefix in sorted(prefixes, key=len, reverse=True):
        if element.startswith(f"{prefix}-"):
            return prefix
    return "other" if re.search(r"[\s/\\:.]", element) or len(element) > 40 else element


def test_every_pane_control_is_a_usage_feature() -> None:
    source = SOURCE.read_text()
    whole = re.findall(r"'([^']+)'", re.search(r"const USAGE_WHOLE = \[(.*?)\]", source).group(1))
    prefixes = _strings(source, "USAGE_PREFIXES")
    features = set(_strings(source, "USAGE_FEATURES"))

    found = re.findall(r'<(?:Button|Input)\s+key=(?:"([^"]+)"|\{`([^`]+)`\})', source)
    keys = [plain or re.sub(r"\$\{.*$", "x", templated) for plain, templated in found]
    assert len(keys) > 80, "the key pattern no longer matches the source"

    names = {_usage_name(key, whole, prefixes) for key in keys}
    names = {
        name for name in names if not NOT_FEATURES.match(name) or name in STILL_FEATURES
    } - HELPER_KEYS
    assert sorted(names - features) == []
