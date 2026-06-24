"""HTTP client tool for the persistent NMR retrieval service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from .decorator import tool

DEFAULT_URL = os.environ.get("NMR_RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8011").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("NMR_RETRIEVAL_SERVICE_TIMEOUT_S", "120"))


def _post_json(url: str, payload: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {"valid": 0, "error": f"HTTP {exc.code}: {body}", "results": [], "candidates": [], "count": 0}
    except Exception as exc:
        return {"valid": 0, "error": str(exc), "results": [], "candidates": [], "count": 0, "observation": f"Retrieval service unavailable at {url}: {exc}"}


def nmr_retrieve_service_impl(
    h_shifts: str = "",
    c_shifts: str = "",
    formula: str = "",
    top_k: int = 20,
    save_pool_file: bool = True,
    pool_path: str = "",
    query_smiles: str = "",
    nprobe: int = 128,
    retrieval_mode: str = "auto",
    backend_mode: str = "embedding",
    service_url: str = "",
) -> Dict[str, Any]:
    base = (service_url or DEFAULT_URL).rstrip("/")
    payload = {
        "h_shifts": h_shifts,
        "c_shifts": c_shifts,
        "formula": formula,
        "query_smiles": query_smiles,
        "top_k": int(top_k or 20),
        "nprobe": int(nprobe or 128),
        "retrieval_mode": retrieval_mode,
        "backend_mode": backend_mode,
        "save_pool_file": bool(save_pool_file),
        "pool_path": pool_path,
    }
    result = _post_json(base + "/retrieve", payload)
    if isinstance(result, dict):
        result.setdefault("service_url", base)
        result.setdefault("source", "retrieval_service")
    return result


@tool(name="nmr_retrieve_service", description="Call a persistent Flask NMR retrieval service instead of initializing retrieval inside the agent process.")
def nmr_retrieve_service(h_shifts: str = "", c_shifts: str = "", formula: str = "", top_k: int = 20, save_pool_file: bool = True, pool_path: str = "") -> Dict[str, Any]:
    return nmr_retrieve_service_impl(h_shifts=h_shifts, c_shifts=c_shifts, formula=formula, top_k=top_k, save_pool_file=save_pool_file, pool_path=pool_path)
