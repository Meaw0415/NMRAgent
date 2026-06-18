from __future__ import annotations

import os
from pathlib import Path

from configs.runtime_assets import load_runtime_assets


load_runtime_assets()
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("NMR_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))


def artifact_path(*parts: str) -> Path:
    return ARTIFACTS_DIR.joinpath(*parts)
