"""HTTP client tool for a persistent NMR de novo generation service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from .decorator import tool

DEFAULT_URL = os.environ.get("NMR_DENOVO_SERVICE_URL", "http://127.0.0.1:8012").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("NMR_DENOVO_SERVICE_TIMEOUT_S", "300"))


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
        return {"valid": 0, "error": str(exc), "results": [], "candidates": [], "count": 0, "observation": f"Denovo service unavailable at {url}: {exc}"}


def nmr_denovo_service_impl(
    h_shifts: str = "",
    c_shifts: str = "",
    formula: str = "",
    top_k: int = 20,
    save_pool_file: bool = True,
    pool_path: str = "",
    temperature: float = 1.0,
    service_url: str = "",
) -> Dict[str, Any]:
    base = (service_url or DEFAULT_URL).rstrip("/")
    payload = {
        "h_shifts": h_shifts,
        "c_shifts": c_shifts,
        "formula": formula,
        "top_k": int(top_k or 20),
        "save_pool_file": bool(save_pool_file),
        "temperature": float(temperature or 1.0),
    }
    result = _post_json(base + "/denovo", payload)
    if isinstance(result, dict):
        result.setdefault("service_url", base)
        result.setdefault("source", "denovo_service")
    return result


@tool(name="nmr_denovo_service", description="Call a persistent Flask NMR de novo generation service instead of initializing denovo inside the agent process.")
def nmr_denovo_service(h_shifts: str = "", c_shifts: str = "", formula: str = "", top_k: int = 20, save_pool_file: bool = True, pool_path: str = "") -> Dict[str, Any]:
    return nmr_denovo_service_impl(h_shifts=h_shifts, c_shifts=c_shifts, formula=formula, top_k=top_k, save_pool_file=save_pool_file, pool_path=pool_path)
