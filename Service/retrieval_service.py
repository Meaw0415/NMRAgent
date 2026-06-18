#!/usr/bin/env python3
"""
Persistent retrieval service for NMRSearcher.

This keeps the retrieval model, FAISS index, ID mapping, and formula index
alive inside one process so debugging does not pay repeated cold-start cost.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from flask import Flask, jsonify, request

from configs.runtime_assets import load_runtime_assets

load_runtime_assets()
os.environ.setdefault("NMR_RETRIEVAL_DISABLE_SERVICE", "1")
from tools.nmr_retrieval_tool import get_searcher, nmr_retrieve_tool


SERVICE_HOST = os.environ.get("NMR_RETRIEVAL_SERVICE_HOST", "127.0.0.1")
SERVICE_PORT = int(os.environ.get("NMR_RETRIEVAL_SERVICE_PORT", "8011"))
SERVICE_GPU = os.environ.get("NMR_RETRIEVAL_CUDA_VISIBLE_DEVICES", "0")
PRELOAD = os.environ.get("NMR_RETRIEVAL_PRELOAD", "1").strip().lower() not in {"0", "false", "no"}

os.environ.setdefault("CUDA_VISIBLE_DEVICES", SERVICE_GPU)

app = Flask(__name__)
_LOCK = threading.Lock()


def _warmup_searcher() -> dict[str, Any]:
    t0 = time.time()
    searcher = get_searcher()
    t1 = time.time()
    ntotal = None
    if getattr(searcher, "index", None) is not None:
        try:
            ntotal = int(searcher.index.ntotal)
        except Exception:
            ntotal = None
    formula_count = None
    try:
        formula_count = len(getattr(searcher, "formula_to_ids", {}) or {})
    except Exception:
        formula_count = None
    return {
        "ok": True,
        "elapsed_s": round(t1 - t0, 3),
        "index_ntotal": ntotal,
        "formula_count": formula_count,
    }


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "host": SERVICE_HOST,
            "port": SERVICE_PORT,
            "gpu": SERVICE_GPU,
            "preload": PRELOAD,
            "pid": os.getpid(),
        }
    )


@app.post("/warmup")
def warmup():
    with _LOCK:
        payload = _warmup_searcher()
    return jsonify(payload)


@app.post("/retrieve")
def retrieve():
    payload = request.get_json(force=True) or {}
    with _LOCK:
        result = nmr_retrieve_tool(
            h_shifts=payload.get("h_shifts", ""),
            c_shifts=payload.get("c_shifts", ""),
            formula=payload.get("formula", ""),
            query_smiles=payload.get("query_smiles", ""),
            top_k=int(payload.get("top_k", 10)),
            nprobe=int(payload.get("nprobe", 128)),
            retrieval_mode=payload.get("retrieval_mode", "auto"),
            backend_mode=payload.get("backend_mode", "embedding"),
        )
    return jsonify(result)


@app.post("/retrieve_mixed")
def retrieve_mixed():
    payload = request.get_json(force=True) or {}
    base = dict(payload)
    formula_topk = int(base.pop("formula_topk", base.get("top_k", 500)))
    nonformula_topk = int(base.pop("nonformula_topk", base.get("top_k", 500)))
    backend_mode = base.get("backend_mode", "embedding")

    with _LOCK:
        formula_ret = nmr_retrieve_tool(
            h_shifts=base.get("h_shifts", ""),
            c_shifts=base.get("c_shifts", ""),
            formula=base.get("formula", ""),
            query_smiles=base.get("query_smiles", ""),
            top_k=formula_topk,
            nprobe=int(base.get("nprobe", 128)),
            retrieval_mode="formula_only",
            backend_mode=backend_mode,
        )
        nonformula_ret = nmr_retrieve_tool(
            h_shifts=base.get("h_shifts", ""),
            c_shifts=base.get("c_shifts", ""),
            formula="",
            query_smiles="",
            top_k=nonformula_topk,
            nprobe=int(base.get("nprobe", 128)),
            retrieval_mode="non_formula",
            backend_mode=backend_mode,
        )

    return jsonify(
        {
            "ok": True,
            "formula_retrieval": formula_ret.get("results", []),
            "nonformula_retrieval": nonformula_ret.get("results", []),
            "formula_count": len(formula_ret.get("results", [])),
            "nonformula_count": len(nonformula_ret.get("results", [])),
        }
    )


def main() -> int:
    if PRELOAD:
        with _LOCK:
            info = _warmup_searcher()
        print(f"[Retrieval Service] Warmed up: {info}", flush=True)

    app.run(host=SERVICE_HOST, port=SERVICE_PORT, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
