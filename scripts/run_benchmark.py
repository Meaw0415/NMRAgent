#!/usr/bin/env python3
"""Run the released multi-agent workflow on NMRGym, nmrshiftdb, or Exp450."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get("samples", rows.get("data", [rows]))
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Input must contain a JSON list or JSONL records.")
    return rows


def numeric_shifts(value, name):
    if value is None:
        return []
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value.strip()) if v]
    out = [float(v) for v in value]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} contains non-finite shifts")
    return out


def normalize_record(row: dict, position: int, benchmark: str) -> dict:
    extra = row.get("extra_info") or {}
    formula = extra.get("formula") or row.get("formula") or ""
    if not formula:
        raise ValueError(f"Record {position}: molecular formula is required as input")
    h = extra.get("h_shifts", row.get("h_shifts", row.get("h_shift", [])))
    c = extra.get("c_shifts", row.get("c_shifts", row.get("c_shift", [])))
    h = numeric_shifts(h, "h_shifts")
    c = numeric_shifts(c, "c_shifts")
    if not h and not c:
        raise ValueError(f"Record {position}: at least one NMR spectrum is required")
    return {
        "idx": row.get("idx", row.get("id", position)), "benchmark": benchmark,
        "formula": str(formula), "h_shifts": h, "c_shifts": c,
        "gt_smiles": row.get("answer") or row.get("gt_smiles") or row.get("reference_smiles") or row.get("smiles") or "",
    }


def agent_input(row):
    # Do not reuse dataset prompts, which may contain answer-bearing metadata.
    return {
        "task": "Determine the molecular structure from the molecular formula and NMR shifts. Use the planner, executor, and peak-atom verifier to evaluate candidates.",
        "formula": row["formula"], "h_shifts": row["h_shifts"], "c_shifts": row["c_shifts"],
    }


def canonical(smiles):
    from rdkit import Chem
    if not smiles:
        return None
    smiles = str(smiles).strip()
    if smiles.startswith("Answer:"):
        smiles = smiles.partition(":")[2].strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def score_records(records):
    """Final answers and verifier candidate lists have separate denominators/labels."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    seen, labeled, correct, similarity, candidates = set(), 0, 0, 0.0, {1: 0, 5: 0, 10: 0}
    for rec in records:
        key = (rec.get("benchmark"), str(rec["idx"]))
        if key in seen:
            raise ValueError(f"Duplicate prediction ID: {key}")
        seen.add(key)
        gt = canonical(rec.get("gt_smiles"))
        if not rec.get("gt_smiles"):
            continue
        if gt is None:
            raise ValueError(f"Invalid reference SMILES for {key}")
        labeled += 1
        pred = canonical(rec.get("pred_smiles"))
        correct += pred is not None and pred == gt
        if pred:
            similarity += DataStructs.TanimotoSimilarity(fp.GetFingerprint(Chem.MolFromSmiles(gt)), fp.GetFingerprint(Chem.MolFromSmiles(pred)))
        ranked = rec.get("verifier_candidates") or []
        ranked = [canonical(r.get("smiles", "") if isinstance(r, dict) else r) for r in ranked]
        for k in candidates:
            candidates[k] += gt in ranked[:k]
    return {
        "records": len(records), "labeled_records": labeled,
        "final_answer_top1_hits": correct,
        "final_answer_top1_accuracy": correct / labeled if labeled else None,
        "final_answer_mean_morgan_tanimoto": similarity / labeled if labeled else None,
        "verifier_candidate_topk_accuracy": {str(k): v / labeled if labeled else None for k, v in candidates.items()},
        "normalization": "RDKit canonical SMILES, isomericSmiles=False; isotope labels also omitted",
        "ranking_note": "Verifier candidate Top-k is a separate diagnostic; it is not final-answer Top-k.",
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, choices=["nmrgym", "nmrshiftdb", "exp450", "case_study"])
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/benchmark"))
    p.add_argument("--config", default="")
    p.add_argument("--model", default="")
    p.add_argument("--max-turns", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true", help="Skip IDs with an existing output, including errors; use a new directory to retry.")
    p.add_argument("--validate-only", action="store_true", help="Validate/export inputs without API or model calls.")
    p.add_argument("--use-service-tools", action="store_true")
    p.add_argument("--kg-rag", action="store_true")
    p.add_argument("--textbook-rag", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.start < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError("--start and --limit must be nonnegative")
    raw = load_records(args.input)
    selected = raw[args.start:None if args.limit is None else args.start + args.limit]
    rows = [normalize_record(r, args.start + i, args.benchmark) for i, r in enumerate(selected)]
    ids = [str(r["idx"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Input contains duplicate sample IDs")
    score_records(rows)  # Validate references before any model/API calls.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "predictions.jsonl"
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; choose a new directory or --resume")
    manifest = {
        "benchmark": args.benchmark, "input_file": str(args.input.resolve()),
        "selected_records": len(rows), "sample_ids": ids, "start": args.start,
        "workflow": "agents.multi_agent_nmr_v2.MultiAgentNMRV2",
        "validation_only": args.validate_only,
        "reference_policy": "gt_smiles is used only by score_records; it is never passed to agent.run",
    }
    existing = load_records(output) if output.exists() else []
    # Catch duplicate outputs or invalid references before resuming any API calls.
    score_records(existing)
    expected = {str(r["idx"]): r for r in rows}
    for old in existing:
        source = expected.get(str(old["idx"]))
        if source is None or any(old.get(k) != source[k] for k in ["benchmark", "formula", "h_shifts", "c_shifts", "gt_smiles"]):
            raise ValueError("Resume output does not match the selected input records")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "inputs.json").write_text(json.dumps([{"idx": r["idx"], **agent_input(r)} for r in rows], indent=2))
    if args.validate_only:
        print(json.dumps({"validated_records": len(rows), "output_dir": str(args.output_dir), "model_calls": 0}))
        return 0

    from agents.multi_agent_nmr_v2 import MultiAgentNMRV2
    from configs.multi_agent_config import load_multi_agent_config
    config = load_multi_agent_config(args.config)
    manifest["model"] = args.model or config.get("openai_compatible", {}).get("model", "gpt-5.4")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    done = {str(r["idx"]) for r in existing}
    records = existing[:]
    for position, row in enumerate(rows):
        if str(row["idx"]) in done:
            continue
        case_dir = args.output_dir / "cases" / f"sample_{args.start + position:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        overrides = {"trace_log_path": str(case_dir / "trace.jsonl"), "trace_full_prompts": True}
        if args.model:
            overrides["model"] = args.model
        if args.max_turns:
            overrides["max_iterations"] = args.max_turns
        if args.use_service_tools:
            overrides["executor_tools"] = ["nmr_retrieve_service", "nmr_denovo_service", "nmr_merge_pools", "nmr_optimize"]
        for flag in ["kg_rag", "textbook_rag"]:
            if getattr(args, flag):
                overrides["enable_" + flag] = True
        started = time.time()
        rec = {**row, "pred_smiles": "", "verifier_candidates": [], "error": None,
               "trace_path": str(case_dir / "trace.jsonl"), "state_path": str(case_dir / "state.json")}
        try:
            agent = MultiAgentNMRV2.from_config(config, **overrides)
            state = agent.run(**agent_input(row))
            (case_dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))
            rec["pred_smiles"] = state.get("final_answer") or ""
            rec["verifier_candidates"] = state.get("cand_list") or (state.get("rerank_output") or {}).get("candidates", [])
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["elapsed_seconds"] = round(time.time() - started, 3)
        with output.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        records.append(rec)
        (args.output_dir / "summary.json").write_text(json.dumps(score_records(records), indent=2))
        print(json.dumps({"idx": row["idx"], "completed": len(records), "error": rec["error"]}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(score_records(records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
