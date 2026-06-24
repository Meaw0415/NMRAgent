#!/usr/bin/env python3
"""Smoke-test RAG search tools on a real NMR-style case."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(_repo_root()))

    formula = "C20H30O3"
    h_shifts = "5.68 3.19 2.98 2.68 2.59 1.86 1.77 1.69 1.57 1.56 1.44 1.32 1.23 1.20 1.11 1.02 0.94"
    c_shifts = "210.38 182.46 172.23 111.19 86.84 56.39 46.03 42.31 41.72 40.55 40.20 37.25 34.24 33.51 21.71 19.45 18.34 18.07 17.90 17.51"
    query = "C20H30O3 natural product 1H 13C NMR ketone lactone ester oxygenated carbon alkene"

    results = {}

    from tools.textbook_rag_tool import textbook_nmr_search_impl

    results["textbook_nmr_search"] = textbook_nmr_search_impl(
        query=query,
        formula=formula,
        h_shifts=h_shifts,
        c_shifts=c_shifts,
        top_k=3,
    )

    from tools.web_rag_tool import web_nmr_search_impl

    results["web_nmr_search"] = web_nmr_search_impl(
        query=query,
        formula=formula,
        h_shifts=h_shifts,
        c_shifts=c_shifts,
        top_k=3,
    )

    try:
        from tools.kg_rag_tool import kg_graph_rag_search_impl

        results["kg_graph_rag_search"] = kg_graph_rag_search_impl(
            query=query,
            formula=formula,
            top_k=3,
            neighbor_limit=0,
        )
    except Exception as exc:
        results["kg_graph_rag_search"] = {
            "valid": 0,
            "error": str(exc),
            "evidence_pack": [],
            "observation": "KG index is not ready or not configured.",
        }

    summary = {}
    for name, result in results.items():
        evidence = result.get("evidence_pack", []) if isinstance(result, dict) else []
        summary[name] = {
            "valid": result.get("valid") if isinstance(result, dict) else 0,
            "count": result.get("count", len(evidence)) if isinstance(result, dict) else 0,
            "first_claim": evidence[0].get("claim") if evidence else None,
            "error": result.get("error") if isinstance(result, dict) else None,
            "observation": (result.get("observation", "")[:240] if isinstance(result, dict) else ""),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
