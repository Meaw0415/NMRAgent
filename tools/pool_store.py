from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .path_utils import artifact_path


POOL_DIR = Path(os.environ.get("NMR_POOL_DIR", artifact_path("pools")))


def _ensure_pool_dir() -> Path:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    return POOL_DIR


def _json_default(obj: Any):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def make_pool_path(prefix: str = "pool") -> Path:
    pool_dir = _ensure_pool_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return pool_dir / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}.json"


def save_pool(candidates: list[dict], *, prefix: str = "pool", query: dict | None = None, metadata: dict | None = None, path: str = "") -> str:
    out_path = Path(path) if path else make_pool_path(prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"query": query or {}, "metadata": metadata or {}, "count": len(candidates), "candidates": candidates}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return str(out_path)


def load_pool(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_pool_candidates(path: str) -> list[dict]:
    return list(load_pool(path).get("candidates", []))
