from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_CANDIDATES = [
    REPO_ROOT / "configs" / "multi_agent_api.local.yaml",
    REPO_ROOT / "configs" / "multi_agent_api.yaml",
    REPO_ROOT / "configs" / "multi_agent_api.example.yaml",
]


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value.strip("\"'")


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, root)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]

        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if not isinstance(current, list):
                raise ValueError("Invalid YAML list structure")
            current.append(_parse_scalar(item))
            continue

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            next_container: Any = {}
            if key in {"planner_tool_descriptions", "executor_tools", "verifier_tools"}:
                next_container = []
            current[key] = next_container
            stack.append((indent + 2, next_container))
        else:
            current[key] = _parse_scalar(value)

    return root


def load_multi_agent_config(path: str = "") -> Dict[str, Any]:
    config_path = path or os.environ.get("NMR_MULTI_AGENT_CONFIG", "").strip()
    candidates = [Path(config_path)] if config_path else []
    candidates.extend(DEFAULT_CONFIG_CANDIDATES)

    chosen = None
    for candidate in candidates:
        if candidate and candidate.exists():
            chosen = candidate
            break

    if chosen is None:
        return {}

    payload = _parse_simple_yaml(chosen.read_text(encoding="utf-8"))
    payload["_config_path"] = str(chosen)
    return payload
