#!/usr/bin/env python3
"""Minimal single-sample entrypoint for the standalone NMRAgent repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from configs.runtime_assets import load_runtime_assets

load_runtime_assets()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NMRAgent on one NMR sample")
    parser.add_argument("--formula", required=True, help="Molecular formula, e.g. C8H10N4O2")
    parser.add_argument("--h-shifts", nargs="*", type=float, default=[], help="1H shifts in ppm")
    parser.add_argument("--c-shifts", nargs="*", type=float, default=[], help="13C shifts in ppm")
    parser.add_argument("--backend", default="openai", choices=["openai", "vllm", "transformers"])
    parser.add_argument("--model", default=os.environ.get("NMR_MODEL", "gpt-4o-mini"))
    parser.add_argument("--tools", nargs="+", default=["nmr_retrieve", "nmr_denovo", "nmr_rerank"], help="Enabled tool names from tools.get_tools_by_names")
    parser.add_argument("--force-kg-rag", action="store_true")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from agents import NMRAgent
    from tools import get_tools_by_names

    enabled_tools = get_tools_by_names(args.tools)
    agent = NMRAgent(
        model_name_or_path=args.model,
        tools=enabled_tools,
        backend=args.backend,
        force_kg_rag=args.force_kg_rag,
        max_iterations=args.max_turns,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    prompt = (
        "Given the following NMR spectroscopy data, predict the molecular structure (SMILES):\n\n"
        f"Molecular Formula: {args.formula}\n"
        f"H-NMR shifts (ppm): {' '.join(str(x) for x in args.h_shifts)}\n"
        f"C-NMR shifts (ppm): {' '.join(str(x) for x in args.c_shifts)}\n\n"
        "Use the available tools when needed. Provide the final answer as: Answer: <SMILES>\n"
    )

    result = agent.run(task=prompt, formula=args.formula, h_shifts=args.h_shifts, c_shifts=args.c_shifts)
    print(result.get("final_answer") or result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
