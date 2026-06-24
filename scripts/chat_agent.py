#!/usr/bin/env python3
"""Interactive terminal chat wrapper for MultiAgentNMR."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


REAL_CASE = {
    "formula": "C20H30O3",
    "c_shifts": [210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51],
    "h_shifts": [5.68, 3.19, 2.98, 2.68, 2.59, 1.86, 1.77, 1.69, 1.57, 1.56, 1.44, 1.32, 1.23, 1.20, 1.11, 1.02, 0.94],
    "note": "1H 400 MHz, CDCl3; 13C 101 MHz, CDCl3",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _numbers(text: str) -> List[float]:
    vals: List[float] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?(?:\s*-\s*-?\d+(?:\.\d+)?)?", text):
        if "-" in match.strip()[1:]:
            parts = [x.strip() for x in match.split("-") if x.strip()]
            vals.extend(float(x) for x in parts[:2])
        else:
            vals.append(float(match))
    return vals


def _parse_case_text(text: str, state: Dict[str, Any]) -> None:
    formula_match = re.search(r"\bC\d+[A-Z][A-Za-z0-9]*\b", text)
    if formula_match:
        state["formula"] = formula_match.group(0)
    lower = text.lower()
    if "13c" in lower or "c nmr" in lower:
        state["c_shifts"] = _numbers(text)
    elif "1h" in lower or "h nmr" in lower:
        state["h_shifts"] = _numbers(text)
    elif "note" in lower or "cdcl" in lower or "mhz" in lower:
        state["note"] = text.strip()


def _build_agent(args: argparse.Namespace):
    sys.path.insert(0, str(_repo_root()))
    from agents.multi_agent_nmr import MultiAgentNMR
    from configs.multi_agent_config import load_multi_agent_config
    from tools.nmr_text_parser import parse_h_nmr_text, parse_c_nmr_text

    config = load_multi_agent_config(args.config)
    overrides: Dict[str, Any] = {"trace_log_path": args.trace_log, "dry_run_tools": args.dry_run_tools}
    if args.model:
        overrides["model"] = args.model
    if args.api_key:
        overrides["api_key"] = args.api_key
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.max_turns:
        overrides["max_iterations"] = args.max_turns
    if args.textbook_rag:
        overrides["enable_textbook_rag"] = True
        overrides["textbook_rag_top_k"] = args.textbook_rag_top_k
    if args.kg_rag:
        overrides["enable_kg_rag"] = True
        overrides["kg_rag_top_k"] = args.kg_rag_top_k
        overrides["kg_rag_neighbor_limit"] = args.kg_rag_neighbor_limit
    if args.web_rag:
        overrides["enable_web_rag"] = True
        overrides["web_rag_top_k"] = args.web_rag_top_k
    if args.use_service_tools:
        overrides["executor_tools"] = ["nmr_retrieve_service", "nmr_denovo_service", "nmr_merge_pools", "nmr_optimize"]
    return MultiAgentNMR.from_config(config, **overrides)


def _rag_preview(state: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    query = (
        f"{state.get('formula', '')} natural product NMR structure elucidation "
        "ketone lactone ester oxygenated carbon alkene"
    )
    out: Dict[str, Any] = {}
    if args.textbook_rag:
        from tools.textbook_rag_tool import textbook_nmr_search_impl

        out["textbook_nmr_search"] = textbook_nmr_search_impl(
            query=query,
            formula=state.get("formula", ""),
            h_shifts=" ".join(str(x) for x in state.get("h_shifts", [])),
            c_shifts=" ".join(str(x) for x in state.get("c_shifts", [])),
            top_k=args.textbook_rag_top_k,
        )
    if args.web_rag:
        from tools.web_rag_tool import web_nmr_search_impl

        out["web_nmr_search"] = web_nmr_search_impl(
            query=query,
            formula=state.get("formula", ""),
            h_shifts=" ".join(str(x) for x in state.get("h_shifts", [])),
            c_shifts=" ".join(str(x) for x in state.get("c_shifts", [])),
            top_k=args.web_rag_top_k,
        )
    if args.kg_rag:
        try:
            from tools.kg_rag_tool import kg_graph_rag_search_impl

            out["kg_graph_rag_search"] = kg_graph_rag_search_impl(
                query=query,
                formula=state.get("formula", ""),
                top_k=args.kg_rag_top_k,
                neighbor_limit=args.kg_rag_neighbor_limit,
            )
        except Exception as exc:
            out["kg_graph_rag_search"] = {"valid": 0, "count": 0, "error": str(exc), "evidence_pack": []}
    summary: Dict[str, Any] = {}
    for name, result in out.items():
        evidence = result.get("evidence_pack", []) if isinstance(result, dict) else []
        summary[name] = {
            "valid": result.get("valid") if isinstance(result, dict) else 0,
            "count": result.get("count", len(evidence)) if isinstance(result, dict) else 0,
            "first_claim": evidence[0].get("claim") if evidence else None,
            "observation": result.get("observation", "")[:500] if isinstance(result, dict) else "",
            "error": result.get("error") if isinstance(result, dict) else None,
        }
    return summary


def _solve(agent: Any, state: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    task = (
        "Solve this NMR structure elucidation task in chat mode. "
        "Use the provided formula, 1H/13C NMR shifts, solvent/instrument notes, RAG evidence if enabled, "
        "formula-constrained candidate generation, and verifier rerank evidence."
    )
    if state.get("note"):
        task += f" Experimental note: {state['note']}"
    return agent.run(
        task=task,
        formula=state.get("formula", ""),
        h_shifts=state.get("h_shifts", []),
        c_shifts=state.get("c_shifts", []),
        max_iterations=args.max_turns or None,
        dry_run_tools=args.dry_run_tools,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive chat wrapper for MultiAgentNMR.")
    parser.add_argument("--config", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--dry-run-tools", action="store_true")
    parser.add_argument("--textbook-rag", action="store_true")
    parser.add_argument("--textbook-rag-top-k", type=int, default=5)
    parser.add_argument("--kg-rag", action="store_true")
    parser.add_argument("--kg-rag-top-k", type=int, default=5)
    parser.add_argument("--kg-rag-neighbor-limit", type=int, default=0)
    parser.add_argument("--web-rag", action="store_true")
    parser.add_argument("--web-rag-top-k", type=int, default=5)
    parser.add_argument("--use-service-tools", action="store_true", help="Use HTTP retrieval/denovo service tools instead of in-process model initialization.")
    parser.add_argument("--trace-log", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--demo-real-case", action="store_true")
    parser.add_argument("--demo-rag-only", action="store_true", help="Run the built-in real case through enabled RAG tools without calling the LLM solver.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    agent = _build_agent(args)
    state: Dict[str, Any] = {"formula": "", "h_shifts": [], "c_shifts": [], "note": ""}
    if args.h_nmr_text:
        state["h_shifts"] = parse_h_nmr_text(args.h_nmr_text)
    if args.c_nmr_text:
        state["c_shifts"] = parse_c_nmr_text(args.c_nmr_text)

    if args.demo_real_case:
        state.update(REAL_CASE)
        if args.demo_rag_only:
            result = _rag_preview(state, args)
        else:
            result = _solve(agent, state, args)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        print(rendered)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        return 0

    print("MultiAgentNMR chat. Commands: /show, /rag, /solve, /reset, /quit")
    print("Paste lines like: Formula: C20H30O3, 13C NMR: ..., 1H NMR: ..., Note: ...")
    while True:
        try:
            line = input("nmr> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            break
        if line == "/reset":
            state = {"formula": "", "h_shifts": [], "c_shifts": [], "note": ""}
            print("reset")
            continue
        if line == "/show":
            print(json.dumps(state, ensure_ascii=False, indent=2))
            continue
        if line == "/rag":
            if not state.get("formula") or not state.get("c_shifts"):
                print("Need at least formula and 13C shifts before /rag.")
                continue
            result = _rag_preview(state, args)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            continue
        if line == "/solve":
            if not state.get("formula") or not state.get("c_shifts"):
                print("Need at least formula and 13C shifts before /solve.")
                continue
            result = _solve(agent, state, args)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            continue
        _parse_case_text(line, state)
        print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
