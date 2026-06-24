#!/usr/bin/env python3
"""Persistent denovo service for NMRAgent."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from flask import Flask, jsonify, request

from configs.runtime_assets import load_runtime_assets

load_runtime_assets()
from tools.nmr_denovo_tool import nmr_denovo

SERVICE_HOST = os.environ.get("NMR_DENOVO_SERVICE_HOST", "127.0.0.1")
SERVICE_PORT = int(os.environ.get("NMR_DENOVO_SERVICE_PORT", "8012"))
SERVICE_GPU = os.environ.get("NMR_DENOVO_CUDA_VISIBLE_DEVICES", "0")
PRELOAD = os.environ.get("NMR_DENOVO_PRELOAD", "0").strip().lower() in {"1", "true", "yes"}

os.environ.setdefault("CUDA_VISIBLE_DEVICES", SERVICE_GPU)

app = Flask(__name__)
_LOCK = threading.Lock()
_WARMED = False


def _warmup() -> dict[str, Any]:
    global _WARMED
    t0 = time.time()
    # A full denovo warmup may be expensive; import already initializes most module-level code.
    _WARMED = True
    return {"ok": True, "elapsed_s": round(time.time() - t0, 3), "warmed": _WARMED}


@app.get("/health")
def health():
    return jsonify({"ok": True, "host": SERVICE_HOST, "port": SERVICE_PORT, "gpu": SERVICE_GPU, "preload": PRELOAD, "pid": os.getpid(), "warmed": _WARMED})


@app.post("/warmup")
def warmup():
    with _LOCK:
        payload = _warmup()
    return jsonify(payload)


@app.post("/denovo")
def denovo():
    payload = request.get_json(force=True) or {}
    with _LOCK:
        result = nmr_denovo(
            h_shifts=payload.get("h_shifts", ""),
            c_shifts=payload.get("c_shifts", ""),
            formula=payload.get("formula", ""),
            top_k=int(payload.get("top_k", 20)),
            save_pool_file=bool(payload.get("save_pool_file", True)),
            pool_path=payload.get("pool_path", ""),
        )
    return jsonify(result)


def main() -> int:
    if PRELOAD:
        with _LOCK:
            info = _warmup()
        print(f"[Denovo Service] Warmed up: {info}", flush=True)
    app.run(host=SERVICE_HOST, port=SERVICE_PORT, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
