#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_seed_candidates(values: List[str]) -> List[Dict[str, Any]]:
    out = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        out.append({"smiles": item, "source": "seed"})
                    elif isinstance(item, dict) and item.get("smiles"):
                        out.append(item)
            elif isinstance(parsed, dict) and parsed.get("smiles"):
                out.append(parsed)
            elif isinstance(parsed, str):
                out.append({"smiles": parsed, "source": "seed"})
        except Exception:
            out.append({"smiles": value, "source": "seed"})
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stable v2 three-role multi-agent NMRAgent")
    parser.add_argument("--config", default="", help="YAML config path.")
    parser.add_argument("--formula", required=True)
    parser.add_argument("--h-shifts", nargs="*", type=float, default=[])
    parser.add_argument("--c-shifts", nargs="*", type=float, default=[])
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--dry-run-tools", action="store_true")
    parser.add_argument("--memory-top-k", type=int, default=0)
    parser.add_argument("--auto-remember-accepted", action="store_true")
    parser.add_argument("--seed-candidate", action="append", default=[])
    parser.add_argument("--trace-log", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--print-trace-tail", type=int, default=0)
    return parser


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    from agents.multi_agent_nmr_v2 import MultiAgentNMRV2
    from configs.multi_agent_config import load_multi_agent_config

    args = build_parser().parse_args()
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
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_tokens:
        overrides["max_tokens"] = args.max_tokens
    if args.memory_top_k:
        overrides["memory_top_k"] = args.memory_top_k
    if args.auto_remember_accepted:
        overrides["auto_remember_accepted"] = True

    agent = MultiAgentNMRV2.from_config(config, **overrides)
    task = "Solve this NMR structure elucidation task with a planner, executor, and peak-atom verifier. Use formula-constrained candidate generation, pool passing, rerank alignment, and escalation when evidence is weak."
    result = agent.run(task=task, formula=args.formula, h_shifts=args.h_shifts, c_shifts=args.c_shifts, max_iterations=args.max_turns or None, dry_run_tools=args.dry_run_tools, seed_candidates=_parse_seed_candidates(args.seed_candidate))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    if args.print_trace_tail:
        trace_path = Path(result["trace_log_path"])
        if trace_path.exists():
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            print("\nTRACE_TAIL")
            for line in lines[-args.print_trace_tail:]:
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
