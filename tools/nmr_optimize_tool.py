"""
Pool-only NMRAgent optimization tool.

This tool is designed for agent use after upstream candidate generation.
It never calls retrieval or denovo internally. Instead, it consumes a pool of
candidate molecules directly or via a saved pool file.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from .decorator import tool
from .nmr_mol_edit_tool import _cut_non_ring, _cut_ring, _get_complement_cut_type, _stitch
from .nmr_rerank_tool import _set_match_score, nmr_rerank
from .nmr_solver_style_core import (
    BricsParams,
    SolverStyleParams,
    assemble_brics_fragments,
    build_brics_fragment_pool as build_proxy_brics_fragment_pool,
    build_final_rerank_input,
    ensure_opt_ready_candidates,
    expand_fragment_choices,
    merge_branch_pools,
    run_solver_opt_stage,
    search_formula_bucket_combinations,
    select_fragment_pool,
    vector_rerank_candidates,
)
from .pool_store import load_pool, save_pool


def _canon_no_stereo(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

rdBase.DisableLog("rdApp.error")


def parse_formula(formula: str) -> Dict[str, int]:
    import re
    out: Dict[str, int] = {}
    for elem, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        out[elem] = out.get(elem, 0) + int(cnt or 1)
    return out


def add_formula(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = defaultdict(int)
    for key, val in a.items():
        out[key] += int(val)
    for key, val in b.items():
        out[key] += int(val)
    return dict(out)


def get_canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _cut_brics(mol):
    from rdkit.Chem import BRICS
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def _get_fragment_formula(frag_smi: str) -> Optional[Dict[str, int]]:
    mol = Chem.MolFromSmiles(frag_smi)
    if not mol:
        return None
    formula_str = rdMolDescriptors.CalcMolFormula(mol)
    fd = parse_formula(formula_str)
    for dummy in ("R", "*"):
        fd.pop(dummy, None)
    return fd


def build_brics_fragment_pool(candidates: List[str], max_candidates: int = 100) -> List[Dict]:
    from rdkit.Chem import BRICS
    pool: List[Dict] = []
    seen: Set[str] = set()
    for i, smi in enumerate(candidates[:max_candidates]):
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        for frag_smi in _cut_brics(mol):
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if not frag_mol:
                continue
            canon = get_canonical(frag_smi)
            if not canon or canon in seen:
                continue
            fd = _get_fragment_formula(frag_smi)
            if not fd:
                continue
            pool.append({"id": canon, "smiles": frag_smi, "formula": fd, "mol": frag_mol, "source_idx": i})
            seen.add(canon)
    return pool


def _assemble_combination(frag_mols: List, target_formula_str: str, max_out: int = 10, max_iter: int = 5000) -> List[str]:
    from rdkit.Chem import BRICS
    try:
        gen = BRICS.BRICSBuild(frag_mols)
        results: List[str] = []
        for idx, mol in enumerate(gen):
            if idx >= max_iter:
                break
            try:
                if rdMolDescriptors.CalcMolFormula(mol) == target_formula_str:
                    canon = get_canonical(Chem.MolToSmiles(mol))
                    if canon:
                        results.append(canon)
                    if len(results) >= max_out:
                        break
            except Exception:
                continue
        return results
    except Exception:
        return []


def _parse_shift_string(values) -> List[float]:
    if isinstance(values, list):
        return [float(x) for x in values]
    return [float(x.strip()) for x in str(values).split(",") if x.strip()]


def _canonicalize(smiles: str) -> Optional[str]:
    return _canon_no_stereo(smiles)


def _normalize_candidate_rows(candidates_raw) -> List[dict]:
    out: List[dict] = []
    seen: Set[str] = set()
    for idx, row in enumerate(candidates_raw, start=1):
        if isinstance(row, dict):
            smi = row.get("smiles", "")
            base = dict(row)
        else:
            smi = str(row)
            base = {"smiles": smi}
        canon = _canonicalize(smi)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        base["smiles"] = canon
        base.setdefault("source", "upstream_pool")
        base.setdefault("score", float(base.get("vector_similarity", base.get("score", 0.0))))
        base.setdefault("rank", idx)
        out.append(base)
    return out


def _split_formula_pool(candidates: List[dict], formula: str) -> Tuple[List[dict], List[dict]]:
    exact: List[dict] = []
    non_exact: List[dict] = []
    target = formula.strip()
    for cand in candidates:
        smi = cand.get("smiles", "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = dict(cand)
        row["formula"] = rdMolDescriptors.CalcMolFormula(mol)
        if row["formula"] == target:
            exact.append(row)
        else:
            non_exact.append(row)
    return exact, non_exact


def _prepare_ranked_pool(candidates_raw, formula: str, query_h: List[float], query_c: List[float], keep_formula: int, keep_nonformula: int, pool_limit: int) -> dict:
    normalized = _normalize_candidate_rows(candidates_raw)
    ready = ensure_opt_ready_candidates(normalized)
    exact_ready, non_exact_ready = _split_formula_pool(ready, formula)
    exact_ranked = vector_rerank_candidates(query_h, query_c, exact_ready)
    non_exact_ranked = vector_rerank_candidates(query_h, query_c, non_exact_ready)
    merged = merge_branch_pools(exact_ranked, non_exact_ranked, keep_formula, keep_nonformula)
    merged = ensure_opt_ready_candidates(merged[:pool_limit])
    return {"normalized": normalized, "exact_ranked": exact_ranked, "non_exact_ranked": non_exact_ranked, "merged_pool": merged}


def _build_rec(formula: str, h_shifts: List[float], c_shifts: List[float]) -> dict:
    return {"idx": -1, "formula": formula, "h_shifts": h_shifts, "c_shifts": c_shifts, "gt_smiles": ""}


def _run_final_rerank(formula: str, query_h: List[float], query_c: List[float], originals: List[dict], new_rows: List[dict], keep_original: int, keep_new: int, top_k: int) -> dict:
    rerank_input = build_final_rerank_input(originals, new_rows, keep_original, keep_new)
    return nmr_rerank(h_shifts=query_h, c_shifts=query_c, candidates=json.dumps(rerank_input, ensure_ascii=False), top_k=top_k, formula=formula)


def _run_solver_style_mode(formula: str, query_h: List[float], query_c: List[float], pool: List[dict], top_k: int, topk_per_fragment: int, num_filter_pair: int, max_new: int, rerank_keep_original: int, rerank_keep_new: int, disable_formula_pair_prune: bool) -> dict:
    params = SolverStyleParams(topk_per_fragment=topk_per_fragment, num_filter_pair=num_filter_pair, max_new=max_new, rerank_keep_original=rerank_keep_original, rerank_keep_new=rerank_keep_new, rerank_top_k=top_k, disable_formula_pair_prune=disable_formula_pair_prune)
    rec = _build_rec(formula, query_h, query_c)
    out = run_solver_opt_stage(rec, pool, params)
    return {"final_candidates": out.get("final_candidates", out.get("top10", []))[:top_k], "assembled": out.get("top_assembled_before_rerank", []), "details": out}


def _run_brics_mode(formula: str, query_h: List[float], query_c: List[float], pool: List[dict], top_k: int, shallow: bool, max_fragments: int, max_per_formula: int, beam_size: int, max_combos: int, max_choice_variants: int, max_products_per_combo: int) -> dict:
    brics_params = BricsParams(max_fragments=max_fragments if not shallow else min(max_fragments, 400), max_per_formula=max_per_formula if not shallow else min(max_per_formula, 6), min_k=3 if not shallow else 2, max_k=6 if not shallow else 3, beam_size=beam_size if not shallow else min(beam_size, 80), max_combos=max_combos if not shallow else min(max_combos, 60), max_choice_variants=max_choice_variants if not shallow else min(max_choice_variants, 3), max_products_per_combo=max_products_per_combo if not shallow else min(max_products_per_combo, 4), rerank_top_k=top_k)
    frag_pool = build_proxy_brics_fragment_pool(pool, query_h, query_c)
    frag_pool = select_fragment_pool(frag_pool, brics_params.max_per_formula, brics_params.max_fragments)
    combos = search_formula_bucket_combinations(frag_pool, target_formula=Counter(parse_formula(formula)), min_k=brics_params.min_k, max_k=brics_params.max_k, beam_size=brics_params.beam_size, max_candidates=brics_params.max_combos)
    assembled: List[dict] = []
    seen_smiles: Set[str] = set()
    for combo in combos:
        for variant in expand_fragment_choices(combo, frag_pool, brics_params.max_choice_variants):
            for smi in assemble_brics_fragments(variant, brics_params.max_products_per_combo):
                if smi in seen_smiles:
                    continue
                seen_smiles.add(smi)
                mol = Chem.MolFromSmiles(smi)
                if mol is None or rdMolDescriptors.CalcMolFormula(mol) != formula:
                    continue
                assembled.append({"smiles": smi, "source": "shallow_brics" if shallow else "brics", "score": float(sum(x.get("frag_score", 0.0) for x in variant))})
    assembled.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    rerank_result = _run_final_rerank(formula, query_h, query_c, pool, assembled, keep_original=min(40, len(pool)), keep_new=top_k, top_k=top_k)
    return {"final_candidates": rerank_result.get("candidates", [])[:top_k], "assembled": assembled, "details": {"fragment_pool_count": len(frag_pool), "combo_count": len(combos), "assembled_count": len(assembled), "rerank_result": rerank_result, "mode": "shallow_brics" if shallow else "brics"}}


def clear_atom_maps_and_canon(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return _canonicalize(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return _canonicalize(Chem.MolToSmiles(mol, canonical=True))


def greedy_set_match_score(query: List[float], observed: List[float], sigma: float) -> float:
    if not query or not observed:
        return 0.0
    remaining = list(float(x) for x in observed)
    score = 0.0
    for q in sorted(float(x) for x in query):
        if not remaining:
            break
        best_idx = min(range(len(remaining)), key=lambda i: abs(remaining[i] - q))
        diff = abs(remaining.pop(best_idx) - q)
        score += math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    return float(score) / float(max(len(query), len(observed)))


def fragment_nmr_score(query_h: List[float], query_c: List[float], frag_h: List[float], frag_c: List[float]) -> float:
    return greedy_set_match_score(query_h, frag_h, 1.0) + greedy_set_match_score(query_c, frag_c, 10.0)


def _solver_fragment_heavy_count(frag_mol: Chem.Mol) -> int:
    return sum(1 for atom in frag_mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _recursive_solver_cuts(root_mol: Chem.Mol, min_heavy_atoms: int, max_depth: int):
    out = []
    seen: set[tuple[str, str]] = set()
    stack: List[tuple[Chem.Mol, int]] = [(root_mol, 1)]
    while stack:
        mol, depth = stack.pop()
        cuts = _cut_non_ring(mol) + _cut_ring(mol)
        for frag_mol, cut_type in cuts:
            heavy = _solver_fragment_heavy_count(frag_mol)
            if heavy < min_heavy_atoms:
                continue
            frag_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
            key = (frag_smiles, repr(cut_type))
            if key in seen:
                continue
            seen.add(key)
            out.append((frag_mol, cut_type, depth))
            if depth < max_depth and heavy > min_heavy_atoms:
                stack.append((frag_mol, depth + 1))
    return out


def fragment_atom_indices(frag_mol: Chem.Mol) -> List[int]:
    idxs: List[int] = []
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        amap = atom.GetAtomMapNum()
        if amap > 0:
            idxs.append(amap - 1)
    return idxs


def fragment_formula_from_source(source_mol: Chem.Mol, frag_mol: Chem.Mol) -> Counter[str]:
    counts: Counter[str] = Counter()
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        amap = atom.GetAtomMapNum()
        if amap <= 0:
            continue
        orig_idx = amap - 1
        if 0 <= orig_idx < source_mol.GetNumAtoms():
            orig = source_mol.GetAtomWithIdx(orig_idx)
            counts[orig.GetSymbol()] += 1
            h_count = int(orig.GetTotalNumHs())
            if h_count > 0:
                counts["H"] += h_count
    return counts


def extract_fragment_shift_lists(candidate: dict, frag_mol: Chem.Mol) -> tuple[List[float], List[float]]:
    atom_data = candidate.get("atom_data") or []
    frag_idxs = set(fragment_atom_indices(frag_mol))
    h_vals = sorted(float(x["shift"]) for x in atom_data if x.get("element") == "H" and int(x.get("parent_idx", -1)) in frag_idxs)
    c_vals = sorted(float(x["shift"]) for x in atom_data if x.get("element") == "C" and int(x.get("heavy_idx", -1)) in frag_idxs)
    return h_vals, c_vals


def extract_solver_fragments_for_candidate(candidate: dict, min_heavy_atoms: int, max_depth: int, query_h: List[float], query_c: List[float]):
    smi = candidate.get("smiles", "")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    rows = []
    seen = set()
    for frag_mol, cut_type, depth in _recursive_solver_cuts(mapped, min_heavy_atoms=min_heavy_atoms, max_depth=max_depth):
        frag_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
        key = (frag_smiles, repr(cut_type))
        if key in seen:
            continue
        seen.add(key)
        frag_h, frag_c = extract_fragment_shift_lists(candidate, frag_mol)
        rows.append({"frag_smiles": frag_smiles, "frag_formula": fragment_formula_from_source(mol, frag_mol), "source_smiles": smi, "source_score": float(candidate.get("vector_similarity", candidate.get("score", 0.0))), "frag_h": frag_h, "frag_c": frag_c, "frag_score": fragment_nmr_score(query_h, query_c, frag_h, frag_c), "frag_mol": frag_mol, "cut_type": cut_type, "cut_type_key": repr(cut_type), "complement_cut_type_key": repr(_get_complement_cut_type(cut_type)), "solver_depth": depth})
    return rows


def build_solver_fragment_pool(candidates: List[dict], query_h: List[float], query_c: List[float], min_heavy_atoms: int, max_depth: int) -> List[dict]:
    rows: List[dict] = []
    for cand in candidates:
        rows.extend(extract_solver_fragments_for_candidate(cand, min_heavy_atoms, max_depth, query_h, query_c))
    rows.sort(key=lambda x: (x.get("frag_score", 0.0), x.get("source_score", 0.0)), reverse=True)
    for idx, row in enumerate(rows):
        row["frag_id"] = idx
    return rows


def filter_solver_fragment_pool(rows: List[dict], max_per_formula: int, max_total: int, min_frag_score: float) -> List[dict]:
    by_formula = defaultdict(int)
    out = []
    for row in rows:
        if float(row.get("frag_score", 0.0)) < min_frag_score:
            continue
        key = "".join(f"{k}{'' if int(v)==1 else int(v)}" for k, v in sorted(row["frag_formula"].items()))
        if by_formula[key] >= max_per_formula:
            continue
        by_formula[key] += 1
        out.append(row)
        if len(out) >= max_total:
            break
    return out


def search_solver_pairs(fragments: List[dict], target_formula: Counter[str], max_pairs: int, enforce_formula_match: bool) -> List[dict]:
    by_cut = defaultdict(list)
    for row in fragments:
        by_cut[row["cut_type_key"]].append(row)
    out = []
    seen = set()
    for left in fragments:
        for right in by_cut.get(left["complement_cut_type_key"], []):
            if left["source_smiles"] == right["source_smiles"]:
                continue
            pair_formula = add_formula(left["frag_formula"], right["frag_formula"])
            if enforce_formula_match and pair_formula != target_formula:
                continue
            key = tuple(sorted((left["frag_id"], right["frag_id"])))
            if key in seen:
                continue
            seen.add(key)
            out.append({"left_frag_id": left["frag_id"], "right_frag_id": right["frag_id"], "score": float(left["frag_score"] + right["frag_score"]), "pair_formula": pair_formula})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:max_pairs]


def assemble_solver_pairs(pair_rows: List[dict], frag_rows: List[dict], target_formula: Counter[str], existing_smiles: set[str], max_new: int, enforce_formula_match: bool, target_heavy_atoms: int, max_heavy_atom_excess: int):
    frag_by_id = {row["frag_id"]: row for row in frag_rows}
    new_rows = []
    seen = set(existing_smiles)
    for pair in pair_rows:
        left = frag_by_id[pair["left_frag_id"]]
        right = frag_by_id[pair["right_frag_id"]]
        mol = _stitch(left["frag_mol"], right["frag_mol"], left["cut_type"][1])
        if mol is None:
            continue
        try:
            smi = clear_atom_maps_and_canon(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            continue
        if not smi or smi in seen:
            continue
        mol2 = Chem.MolFromSmiles(smi)
        if mol2 is None or "*" in smi:
            continue
        heavy_atoms = mol2.GetNumHeavyAtoms()
        if heavy_atoms > target_heavy_atoms + max_heavy_atom_excess:
            continue
        if enforce_formula_match and parse_formula(rdMolDescriptors.CalcMolFormula(mol2)) != target_formula:
            continue
        seen.add(smi)
        new_rows.append({"smiles": smi, "source": "fast_crossover", "score": float(pair["score"])})
        if len(new_rows) >= max_new:
            break
    return new_rows


def merge_unique(existing: List[dict], new_rows: List[dict]) -> List[dict]:
    by_smiles: dict[str, dict] = {}
    for row in existing + new_rows:
        smi = clear_atom_maps_and_canon(row.get("smiles", ""))
        if not smi:
            continue
        prev = by_smiles.get(smi)
        if prev is None or float(row.get("score", 0.0)) > float(prev.get("score", 0.0)):
            new_row = dict(row)
            new_row["smiles"] = smi
            by_smiles[smi] = new_row
    out = list(by_smiles.values())
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out


def _run_fast_crossover_mode(formula: str, query_h: List[float], query_c: List[float], pool: List[dict], top_k: int, keep_original: int, keep_new: int, rounds: int, max_fragments: int, max_per_formula: int, max_pairs: int, max_new_per_round: int, min_heavy_atoms: int, recursive_max_depth: int, min_frag_score: float, max_heavy_atom_excess: int, allow_off_formula_intermediate: bool) -> dict:
    current_pool = list(pool)
    target_formula = Counter(parse_formula(formula))
    target_heavy_atoms = 999
    for row in current_pool:
        mol = Chem.MolFromSmiles(row.get("smiles", ""))
        if mol is not None:
            target_heavy_atoms = min(target_heavy_atoms, mol.GetNumHeavyAtoms())
    all_new_rows: List[dict] = []
    rounds_meta: List[dict] = []
    for round_idx in range(1, rounds + 1):
        enforce_formula_match = (not allow_off_formula_intermediate) or (round_idx == rounds)
        frag_pool = build_solver_fragment_pool(current_pool, query_h, query_c, min_heavy_atoms, recursive_max_depth)
        frag_pool = filter_solver_fragment_pool(frag_pool, max_per_formula, max_fragments, min_frag_score)
        pair_rows = search_solver_pairs(frag_pool, target_formula, max_pairs, enforce_formula_match)
        new_rows = assemble_solver_pairs(pair_rows, frag_pool, target_formula, existing_smiles={clear_atom_maps_and_canon(x.get("smiles", "")) for x in current_pool if x.get("smiles")}, max_new=max_new_per_round, enforce_formula_match=enforce_formula_match, target_heavy_atoms=target_heavy_atoms, max_heavy_atom_excess=max_heavy_atom_excess)
        current_pool = merge_unique(current_pool, new_rows)
        all_new_rows = merge_unique(all_new_rows, new_rows)
        rounds_meta.append({"round": round_idx, "fragment_pool_count": len(frag_pool), "pair_count": len(pair_rows), "new_count": len(new_rows), "enforce_formula_match": enforce_formula_match})
        if not new_rows:
            break
    rerank_result = _run_final_rerank(formula, query_h, query_c, pool, all_new_rows, keep_original=keep_original, keep_new=keep_new, top_k=top_k)
    return {"final_candidates": rerank_result.get("candidates", [])[:top_k], "assembled": all_new_rows, "details": {"rounds": rounds_meta, "assembled_count": len(all_new_rows), "rerank_result": rerank_result}}


def _build_observation(mode: str, prepared: dict, result: dict, elapsed_s: float, pool_path: str = "") -> str:
    lines = [f"Pool-only optimize (mode={mode})", f"  Upstream candidates: {len(prepared['normalized'])}", f"  Exact-formula branch: {len(prepared['exact_ranked'])}", f"  Non-formula branch: {len(prepared['non_exact_ranked'])}", f"  Prepared merged pool: {len(prepared['merged_pool'])}", f"  Final candidates: {len(result.get('final_candidates', []))}"]
    details = result.get("details", {})
    if "fragment_pool_count" in details:
        lines.append(f"  Fragment pool: {details['fragment_pool_count']}")
    if "coarse_pair_count" in details:
        lines.append(f"  Coarse pair count: {details['coarse_pair_count']}")
    if "pair_candidate_count" in details:
        lines.append(f"  Pair rerank count: {details['pair_candidate_count']}")
    if "assembled_count" in details:
        lines.append(f"  New assembled: {details['assembled_count']}")
    if pool_path:
        lines.append(f"  Saved final pool: {pool_path}")
    lines.append("")
    lines.append("Top candidates:")
    for idx, row in enumerate(result.get("final_candidates", [])[:10], start=1):
        smi = row.get("smiles", "")
        if "nmr_similarity" in row:
            lines.append(f"  {idx}. {smi} (NMR sim: {row['nmr_similarity']:.4f})")
        elif "score" in row:
            lines.append(f"  {idx}. {smi} (score: {float(row['score']):.4f})")
        else:
            lines.append(f"  {idx}. {smi}")
    lines.append(f"\nTime: {elapsed_s:.1f}s")
    return "\n".join(lines)


@tool(
    name="nmr_optimize",
    description="Pool-only downstream optimization for NMRAgent. Consumes an upstream candidate pool from retrieval / denovo and never calls retrieval or denovo internally.",
)
def nmr_optimize(candidates: str = "", pool_path: str = "", formula: str = "", h_shifts: str = "", c_shifts: str = "", mode: str = "hybrid", top_k: int = 20, keep_formula: int = 300, keep_nonformula: int = 300, pool_limit: int = 800, topk_per_fragment: int = 1000, num_filter_pair: int = 200000, max_new: int = 500, rerank_keep_original: int = 40, rerank_keep_new: int = 100, disable_formula_pair_prune: bool = False, brics_max_fragments: int = 1200, brics_max_per_formula: int = 12, brics_beam_size: int = 300, brics_max_combos: int = 200, brics_max_choice_variants: int = 6, brics_max_products_per_combo: int = 10, fast_rounds: int = 1, fast_max_fragments: int = 1000, fast_max_per_formula: int = 10, fast_max_pairs: int = 500, fast_max_new_per_round: int = 100, fast_min_heavy_atoms: int = 5, fast_recursive_max_depth: int = 1, fast_min_frag_score: float = 0.0, fast_max_heavy_atom_excess: int = 12, allow_off_formula_intermediate: bool = False, save_pool_file: bool = False, output_pool_path: str = "") -> dict:
    t0 = time.time()
    if pool_path:
        payload = load_pool(pool_path)
        candidates_raw = payload.get("candidates", [])
        if not formula:
            formula = str((payload.get("query") or {}).get("formula", ""))
        if not h_shifts:
            h_shifts = (payload.get("query") or {}).get("h_shifts", [])
        if not c_shifts:
            c_shifts = (payload.get("query") or {}).get("c_shifts", [])
    else:
        try:
            candidates_raw = json.loads(candidates) if isinstance(candidates, str) and candidates else list(candidates)
        except Exception:
            return {"observation": "Error: invalid candidates JSON.", "results": [], "count": 0}
    query_h = _parse_shift_string(h_shifts) if h_shifts else []
    query_c = _parse_shift_string(c_shifts) if c_shifts else []
    if not formula:
        return {"observation": "Error: formula is required.", "results": [], "count": 0}
    prepared = _prepare_ranked_pool(candidates_raw, formula, query_h, query_c, keep_formula, keep_nonformula, pool_limit)
    merged_pool = prepared["merged_pool"]
    if not merged_pool:
        return {"observation": "Error: no valid candidates remained after pool preparation.", "results": [], "count": 0}
    if mode == "solver_style":
        result = _run_solver_style_mode(formula, query_h, query_c, merged_pool, top_k, topk_per_fragment, num_filter_pair, max_new, rerank_keep_original, rerank_keep_new, disable_formula_pair_prune)
    elif mode == "fast_crossover":
        result = _run_fast_crossover_mode(formula, query_h, query_c, merged_pool, top_k, rerank_keep_original, rerank_keep_new, fast_rounds, fast_max_fragments, fast_max_per_formula, fast_max_pairs, fast_max_new_per_round, fast_min_heavy_atoms, fast_recursive_max_depth, fast_min_frag_score, fast_max_heavy_atom_excess, allow_off_formula_intermediate)
    elif mode == "brics":
        result = _run_brics_mode(formula, query_h, query_c, merged_pool, top_k, False, brics_max_fragments, brics_max_per_formula, brics_beam_size, brics_max_combos, brics_max_choice_variants, brics_max_products_per_combo)
    elif mode == "shallow_brics":
        result = _run_brics_mode(formula, query_h, query_c, merged_pool, top_k, True, brics_max_fragments, brics_max_per_formula, brics_beam_size, brics_max_combos, brics_max_choice_variants, brics_max_products_per_combo)
    elif mode == "hybrid":
        solver_result = _run_solver_style_mode(formula, query_h, query_c, merged_pool, top_k, topk_per_fragment, num_filter_pair, max_new, rerank_keep_original, rerank_keep_new, disable_formula_pair_prune)
        augmented_pool = merge_unique(merged_pool, solver_result.get("assembled", []) + solver_result.get("final_candidates", []))
        fast_result = _run_fast_crossover_mode(formula, query_h, query_c, augmented_pool, top_k, rerank_keep_original, rerank_keep_new, fast_rounds, fast_max_fragments, fast_max_per_formula, fast_max_pairs, fast_max_new_per_round, fast_min_heavy_atoms, fast_recursive_max_depth, fast_min_frag_score, fast_max_heavy_atom_excess, allow_off_formula_intermediate)
        combined_new = merge_unique(solver_result.get("assembled", []), fast_result.get("assembled", []))
        rerank_result = _run_final_rerank(formula, query_h, query_c, merged_pool, combined_new, rerank_keep_original, rerank_keep_new, top_k)
        result = {"final_candidates": rerank_result.get("candidates", [])[:top_k], "assembled": combined_new, "details": {"solver_style": solver_result.get("details", {}), "fast_crossover": fast_result.get("details", {}), "assembled_count": len(combined_new), "rerank_result": rerank_result}}
    else:
        return {"observation": f"Error: unsupported mode '{mode}'.", "results": [], "count": 0}
    saved_pool_path = ""
    if save_pool_file:
        saved_pool_path = save_pool(result.get("final_candidates", []), prefix="opt_pool", query={"formula": formula, "h_shifts": query_h, "c_shifts": query_c}, metadata={"tool": "nmr_optimize", "mode": mode, "input_pool_path": pool_path}, path=output_pool_path)
    observation = _build_observation(mode, prepared, result, time.time() - t0, saved_pool_path)
    response = {"observation": observation, "results": result.get("final_candidates", []), "count": len(result.get("final_candidates", [])), "prepared_pool_count": len(merged_pool), "assembled_count": len(result.get("assembled", [])), "details": result.get("details", {})}
    if saved_pool_path:
        response["pool_path"] = saved_pool_path
    return response


nmr_ffa_optimize = nmr_optimize
