from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_CANDIDATES = [
    REPO_ROOT / "configs" / "runtime_assets.local.json",
    REPO_ROOT / "configs" / "runtime_assets.json",
    REPO_ROOT / "configs" / "runtime_assets.example.json",
]


_loaded = False


def _resolve_path(value: str, repo_root: Path) -> str:
    if not value:
        return value
    expanded = os.path.expanduser(value)
    if expanded.startswith(("http://", "https://")):
        return expanded
    p = Path(expanded)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def _load_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_runtime_assets(force: bool = False) -> Path | None:
    global _loaded
    if _loaded and not force:
        return None

    config_path = os.environ.get("NMR_RUNTIME_CONFIG", "").strip()
    candidates = [Path(config_path)] if config_path else []
    candidates.extend(DEFAULT_CONFIG_CANDIDATES)

    chosen = None
    for candidate in candidates:
        if candidate and candidate.exists():
            chosen = candidate
            break

    if chosen is None:
        _loaded = True
        return None

    payload = _load_payload(chosen)
    env_map = payload.get("env", {})
    for key, value in env_map.items():
        if value is None or os.environ.get(key):
            continue
        if isinstance(value, str) and ("/" in value or value.startswith(".")):
            os.environ[key] = _resolve_path(value, REPO_ROOT)
        else:
            os.environ[key] = str(value)

    _loaded = True
    return chosen
