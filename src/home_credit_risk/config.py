from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class ProjectConfig:
    """Typed access to the pipeline configuration."""

    raw: dict[str, Any]
    root: Path

    def get(self, *keys: str, default: Any | None = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


def load_config(config_path: str | Path) -> ProjectConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _parse_simple_yaml(text)
    return ProjectConfig(raw=data, root=path.parent)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, result)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        key = key.strip()
        value = value.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not value:
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent + 2, new_dict))
            continue

        current[key] = _coerce_scalar(value)

    return result


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value