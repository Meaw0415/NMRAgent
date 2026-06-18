from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, rdMolDescriptors

from .fast_opt_tools import canon_no_stereo, enrich_candidate_pool
from .nmr_mol_edit_tool import _cut_non_ring, _cut_ring, _get_complement_cut_type, _stitch
from .nmr_rerank_tool import _load_nmrnet_backend, _set_match_score, nmr_rerank
from .nmr_retrieval_tool import nmr_retrieve_tool

OPTIONAL_HALOGENS = {"F", "Cl", "Br", "I"}
INVALID_PATTERNS = ["[O][O]", "[R]=[R]=[R]", "[r3,r4]=[r3,r4]", "[O][F,Cl,Br,I]"]


@dataclass
class SolverStyleParams:
    formula_topk: int = 500
    nonformula_topk: int = 500
    keep_formula: int = 500
    keep_nonformula: int = 500
    nprobe: int = 128
    topk_per_fragment: int = 1000
    num_filter_pair: int = 200000
    max_new: int = 1000
    rerank_keep_original: int = 40
    rerank_keep_new: int = 100
    rerank_top_k: int = 20
    disable_formula_pair_prune: bool = False


@dataclass
class BricsParams:
    max_fragments: int = 1200
    max_per_formula: int = 12
    min_k: int = 3
    max_k: int = 6
    beam_size: int = 300
    max_combos: int = 200
    max_choice_variants: int = 6
    max_products_per_combo: int = 10
    rerank_top_k: int = 20


def parse_formula(formula: str) -> Dict[str, int]:
    import re

    out: Dict[str, int] = {}
    for elem, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[elem] = out.get(elem, 0) + int(cnt or 1)
    return out


def allowed_elements_from_formula(formula: str) -> set[str]:
    return set(parse_formula(formula).keys())


def add_formula_dict(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = dict(a)
    for key, val in b.items():
        out[key] = out.get(key, 0) + int(val)
    return {key: val for key, val in out.items() if val}


def counter_to_formula_str(counter: Counter[str]) -> str:
    parts: List[str] = []
    for elem in sorted(counter.keys()):
        val = int(counter[elem])
        if val <= 0:
            continue
        parts.append(elem if val == 1 else f"{elem}{val}")
    return "".join(parts)


def mol_ok(mol: Chem.Mol, max_cycle_length: Optional[int] = None, invalid_patterns: Optional[List[str]] = None) -> bool:
    invalid_patterns = invalid_patterns or []
    for pattern in invalid_patterns:
        patt = Chem.MolFromSmarts(pattern)
        if patt is not None and mol.HasSubstructMatch(patt):
            return False
    if max_cycle_length:
        cycle_list = mol.GetRingInfo().AtomRings()
        if cycle_list and max(len(r) for r in cycle_list) > max_cycle_length:
            return False
    return True


def infer_max_cycle_length(candidates: List[dict], floor: int = 6) -> int:
    best = floor
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand.get("smiles", ""))
        if mol is None:
            continue
        cycle_list = mol.GetRingInfo().AtomRings()
        if cycle_list:
            best = max(best, max(len(r) for r in cycle_list))
    return best


def contains_smiles(candidates: List[dict], target_smiles: str) -> bool:
    target = canon_no_stereo(target_smiles)
    if not target:
        return False
    for cand in candidates:
        smi = canon_no_stereo(cand.get("smiles", ""))
        if smi == target:
            return True
    return False


def find_smiles_rank(candidates: List[dict], target_smiles: str) -> Optional[int]:
    target = canon_no_stereo(target_smiles)
    if not target:
        return None
    for idx, cand in enumerate(candidates, start=1):
        smi = canon_no_stereo(cand.get("smiles", ""))
        if smi == target:
            return idx
    return None


def evaluate_topk(candidates: List[dict], gt_smiles: str, ks: Tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    gt = canon_no_stereo(gt_smiles)
    ordered = [canon_no_stereo(c.get("smiles", "")) for c in candidates]
    ordered = [x for x in ordered if x]
    return {f"top{k}": gt in ordered[:k] for k in ks}


def _as_float_list(values) -> List[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [float(x) for x in values.tolist()]
    if isinstance(values, list):
        return [float(x) for x in values]
    return [float(x) for x in values]


def _sorted_unique_float_list(values: Iterable[float]) -> List[float]:
    return sorted(float(x) for x in values)


def _build_atom_data_from_rich(cand: dict) -> List[dict]:
    if cand.get("atom_data"):
        return cand["atom_data"]

    mol = Chem.MolFromSmiles(cand.get("smiles", ""))
    atom_index = cand.get("atom_index")
    nmr_predict = cand.get("nmr_predict")
    if mol is None or atom_index is None or nmr_predict is None:
        return []

    atom_data: List[dict] = []
    num_atoms = mol.GetNumAtoms()
    max_len = min(num_atoms, len(atom_index), len(nmr_predict))
    for idx in range(max_len):
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = int(atom_index[idx])
        shift = float(nmr_predict[idx])
        if np.isnan(shift):
            continue
        if atomic_num == 6:
            atom_data.append({"element": "C", "heavy_idx": idx, "shift": shift})
        elif atomic_num == 1:
            parents = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            atom_data.append({"element": "H", "parent_idx": parents[0] if parents else -1, "shift": shift})
    return atom_data


def _set_atom_map_numbers(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    return mol


def _clear_atom_map_numbers(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _fill_shift_lists_from_atom_data(candidate: dict) -> None:
    atom_data = candidate.get("atom_data") or []
    if atom_data:
        if candidate.get("H_nmr") is None:
            h_vals = [float(x["shift"]) for x in atom_data if x.get("element") == "H"]
            candidate["H_nmr"] = _sorted_unique_float_list(h_vals)
        if candidate.get("C_nmr") is None:
            c_vals = [float(x["shift"]) for x in atom_data if x.get("element") == "C"]
            candidate["C_nmr"] = _sorted_unique_float_list(c_vals)


def _candidate_is_opt_ready(candidate: dict) -> bool:
    return bool(candidate.get("atom_data")) and candidate.get("H_nmr") is not None and candidate.get("C_nmr") is not None


def ensure_opt_ready_candidates(candidates: List[dict]) -> List[dict]:
    enriched = enrich_candidate_pool(candidates)
    prepared: List[dict] = []
    missing: List[Tuple[int, str]] = []

    for idx, cand in enumerate(enriched):
        row = dict(cand)
        if not row.get("atom_data"):
            row["atom_data"] = _build_atom_data_from_rich(row)
        _fill_shift_lists_from_atom_data(row)
        row["spectra_source"] = row.get("rich_source") or ("missing" if not row.get("rich_found") else "rich")
        prepared.append(row)
        if not _candidate_is_opt_ready(row):
            missing.append((idx, row["smiles"]))

    if missing:
        backend = _load_nmrnet_backend()
        if backend is None:
            raise RuntimeError("NMRNet backend is unavailable, but some candidates still need fallback prediction.")
        pred_results = backend.predict_nmr_batch([smiles for _, smiles in missing], fast=True, include_atom_data=True)
        for (row_idx, _), pred in zip(missing, pred_results):
            if not pred.get("valid", False):
                continue
            row = prepared[row_idx]
            row["H_nmr"] = _as_float_list(pred.get("H_shifts", []))
            row["C_nmr"] = _as_float_list(pred.get("C_shifts", []))
            row["atom_data"] = pred.get("atom_data", []) or row.get("atom_data", [])
            row["nmr_fallback_used"] = True
            row["spectra_source"] = "nmrnet_fallback_after_rich" if row.get("rich_found") else "nmrnet_fallback"

    for row in prepared:
        row["nmr_fallback_used"] = bool(row.get("nmr_fallback_used", False))
        row["opt_ready"] = _candidate_is_opt_ready(row)
    return prepared


def gaussian_set2vec(values: List[float], nmr_type: str, dim: int = 128, sigma: Optional[float] = None) -> np.ndarray:
    arr = np.array(_as_float_list(values), dtype=np.float32)
    if arr.size == 0:
        return np.zeros(dim, dtype=np.float32)
    if nmr_type == "H":
        nmr_range = (-1.0, 15.0)
        sigma = sigma or 0.3
    else:
        nmr_range = (-10.0, 230.0)
        sigma = sigma or 2.0
    grid = np.linspace(nmr_range[0], nmr_range[1], dim, dtype=np.float32)
    interval = float(grid[1] - grid[0])
    coef = interval / (np.sqrt(2.0 * np.pi) * sigma)
    vec = coef * np.exp(-((np.abs(arr[:, None] - grid[None, :]) / sigma) ** 2) / 2.0).sum(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def vector_encoding(
    h_shifts: Optional[np.ndarray | List[np.ndarray]] = None,
    c_shifts: Optional[np.ndarray | List[np.ndarray]] = None,
    dim: int = 128,
    normalize: bool = True,
    padding: bool = False,
) -> np.ndarray:
    if h_shifts is None and c_shifts is None:
        raise ValueError("Both h_shifts and c_shifts are None")

    inputs = h_shifts if h_shifts is not None else c_shifts
    dim_pad = dim if padding else 0
    if isinstance(inputs, np.ndarray):
        h_vec = gaussian_set2vec(h_shifts.tolist(), "H", dim=dim, sigma=0.3) if h_shifts is not None else np.zeros((dim_pad,), dtype=np.float32)
        c_vec = gaussian_set2vec(c_shifts.tolist(), "C", dim=dim, sigma=2.0) if c_shifts is not None else np.zeros((dim_pad,), dtype=np.float32)
        if normalize:
            if h_vec.ndim == 1:
                h_norm = float(np.linalg.norm(h_vec))
                if h_norm > 0:
                    h_vec = h_vec / h_norm
            if c_vec.ndim == 1:
                c_norm = float(np.linalg.norm(c_vec))
                if c_norm > 0:
                    c_vec = c_vec / c_norm
        return np.concatenate([h_vec, c_vec], axis=0).astype(np.float32)[None, :]

    if isinstance(inputs, list):
        h_rows = []
        c_rows = []
        total = len(inputs)
        for idx in range(total):
            h_item = h_shifts[idx] if h_shifts is not None else np.array([], dtype=np.float32)
            c_item = c_shifts[idx] if c_shifts is not None else np.array([], dtype=np.float32)
            h_vec = gaussian_set2vec(h_item.tolist() if isinstance(h_item, np.ndarray) else list(h_item), "H", dim=dim, sigma=0.3)
            c_vec = gaussian_set2vec(c_item.tolist() if isinstance(c_item, np.ndarray) else list(c_item), "C", dim=dim, sigma=2.0)
            if normalize:
                h_norm = float(np.linalg.norm(h_vec))
                c_norm = float(np.linalg.norm(c_vec))
                if h_norm > 0:
                    h_vec = h_vec / h_norm
                if c_norm > 0:
                    c_vec = c_vec / c_norm
            h_rows.append(h_vec)
            c_rows.append(c_vec)
        return np.concatenate([np.stack(h_rows, axis=0), np.stack(c_rows, axis=0)], axis=1).astype(np.float32)

    raise TypeError("Unexpected input type for vector_encoding")


def build_query_vector(h_shifts: List[float], c_shifts: List[float]) -> np.ndarray:
    return np.concatenate([gaussian_set2vec(h_shifts, "H"), gaussian_set2vec(c_shifts, "C")], axis=0).astype(np.float32)


def build_candidate_vector(candidate: dict) -> Optional[np.ndarray]:
    h_values = _as_float_list(candidate.get("H_nmr"))
    c_values = _as_float_list(candidate.get("C_nmr"))
    if not h_values and not c_values:
        return None
    return np.concatenate([gaussian_set2vec(h_values, "H"), gaussian_set2vec(c_values, "C")], axis=0).astype(np.float32)


def vector_rerank_candidates(query_h: List[float], query_c: List[float], candidates: List[dict]) -> List[dict]:
    query_vec = build_query_vector(query_h, query_c)
    reranked: List[dict] = []
    for cand in candidates:
        cand_vec = build_candidate_vector(cand)
        if cand_vec is None:
            continue
        row = dict(cand)
        row["vector_similarity"] = float(np.dot(query_vec, cand_vec))
        reranked.append(row)
    reranked.sort(key=lambda x: x["vector_similarity"], reverse=True)
    for idx, row in enumerate(reranked, start=1):
        row["vector_rank"] = idx
    return reranked


def normalize_retrieval_candidates(entries: List[dict], limit: int, branch: str) -> List[dict]:
    rows: List[dict] = []
    seen: set[str] = set()
    for idx, cand in enumerate(entries[:limit], start=1):
        smiles = canon_no_stereo(cand.get("smiles", ""))
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        rows.append(
            {
                "smiles": smiles,
                "formula": cand.get("formula", ""),
                "source": f"retrieve_{branch}",
                "retrieval_branch": branch,
                "retrieval_rank": idx,
                "score": float(cand.get("similarity", cand.get("score", 0.0))),
                "retrieval_similarity": float(cand.get("similarity", cand.get("score", 0.0))),
            }
        )
    return rows


def merge_branch_pools(formula_rows: List[dict], nonformula_rows: List[dict], keep_formula: int, keep_nonformula: int) -> List[dict]:
    best_by_smiles: Dict[str, dict] = {}
    for cand in formula_rows[:keep_formula] + nonformula_rows[:keep_nonformula]:
        smiles = canon_no_stereo(cand.get("smiles", ""))
        if not smiles:
            continue
        row = dict(cand)
        row["smiles"] = smiles
        row.setdefault("pool_sources", [])
        row["pool_sources"] = sorted(set(list(row["pool_sources"]) + [row.get("source", "unknown")]))
        prev = best_by_smiles.get(smiles)
        if prev is None:
            best_by_smiles[smiles] = row
            continue
        prev["pool_sources"] = sorted(set(prev.get("pool_sources", []) + row["pool_sources"]))
        if float(row.get("vector_similarity", 0.0)) > float(prev.get("vector_similarity", 0.0)):
            row["pool_sources"] = prev["pool_sources"]
            best_by_smiles[smiles] = row
    merged = list(best_by_smiles.values())
    merged.sort(key=lambda x: float(x.get("vector_similarity", x.get("score", 0.0))), reverse=True)
    for idx, row in enumerate(merged, start=1):
        row["merged_rank"] = idx
    return merged


def run_dual_retrieval(rec: dict, params: SolverStyleParams) -> Tuple[List[dict], List[dict]]:
    formula_ret = nmr_retrieve_tool(
        h_shifts=rec["h_shifts"],
        c_shifts=rec["c_shifts"],
        formula=rec["formula"],
        query_smiles="",
        top_k=params.formula_topk,
        nprobe=params.nprobe,
        retrieval_mode="formula_only",
    )
    nonformula_ret = nmr_retrieve_tool(
        h_shifts=rec["h_shifts"],
        c_shifts=rec["c_shifts"],
        formula="",
        query_smiles="",
        top_k=params.nonformula_topk,
        nprobe=params.nprobe,
        retrieval_mode="non_formula",
    )
    formula_rows_raw = normalize_retrieval_candidates(formula_ret.get("results", []), params.formula_topk, "formula")
    nonformula_rows_raw = normalize_retrieval_candidates(nonformula_ret.get("results", []), params.nonformula_topk, "nonformula")
    return formula_rows_raw, nonformula_rows_raw


def build_clean_style_pool(rec: dict, params: SolverStyleParams) -> dict:
    formula_rows_raw, nonformula_rows_raw = run_dual_retrieval(rec, params)
    formula_rows_ready = ensure_opt_ready_candidates(formula_rows_raw)
    nonformula_rows_ready = ensure_opt_ready_candidates(nonformula_rows_raw)
    formula_rows = vector_rerank_candidates(rec["h_shifts"], rec["c_shifts"], formula_rows_ready)
    nonformula_rows = vector_rerank_candidates(rec["h_shifts"], rec["c_shifts"], nonformula_rows_ready)
    merged_pool = merge_branch_pools(formula_rows, nonformula_rows, params.keep_formula, params.keep_nonformula)
    merged_pool = ensure_opt_ready_candidates(merged_pool)
    return {
        "formula_rows_raw": formula_rows_raw,
        "nonformula_rows_raw": nonformula_rows_raw,
        "formula_rows": formula_rows,
        "nonformula_rows": nonformula_rows,
        "merged_pool": merged_pool,
    }


def _extract_h_map(cand: dict) -> Dict[int, List[float]]:
    atom_data = cand.get("atom_data") or _build_atom_data_from_rich(cand)
    h_map: Dict[int, List[float]] = {}
    for entry in atom_data:
        if entry.get("element") == "H":
            idx = int(entry.get("parent_idx", -1))
            if idx >= 0:
                h_map.setdefault(idx, []).append(float(entry["shift"]))
    for entry in cand.get("H_assignments") or []:
        shift = float(entry.get("shift"))
        for idx in entry.get("parent_idxs", []):
            h_map.setdefault(int(idx), []).append(shift)
    return h_map


def _extract_c_map(cand: dict) -> Tuple[Dict[int, float], Dict[int, float]]:
    atom_data = cand.get("atom_data") or _build_atom_data_from_rich(cand)
    c_map: Dict[int, float] = {}
    c_raw: Dict[int, float] = {}
    for entry in atom_data:
        if entry.get("element") == "C":
            idx = int(entry.get("heavy_idx", -1))
            if idx >= 0:
                c_raw[idx] = float(entry["shift"])
    equi_class = cand.get("equi_class")
    atom_index = cand.get("atom_index")
    if equi_class is not None and atom_index is not None:
        for idx, shift in c_raw.items():
            if idx < len(equi_class) and idx < len(atom_index) and int(atom_index[idx]) == 6:
                c_map[int(equi_class[idx])] = shift
    else:
        c_map = dict(c_raw)
    for entry in cand.get("C_assignments") or []:
        shift = float(entry.get("shift"))
        for idx in entry.get("atom_idxs", []):
            c_raw[int(idx)] = shift
    return c_map, c_raw


def _fragment_elements(frag_mol: Chem.Mol) -> set[str]:
    return {atom.GetSymbol() for atom in frag_mol.GetAtoms() if atom.GetSymbol()}


def _fragment_formula_dict(frag_mol: Chem.Mol) -> Dict[str, int]:
    smi = Chem.MolToSmiles(frag_mol)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    fd = parse_formula(rdMolDescriptors.CalcMolFormula(mol))
    fd.pop("*", None)
    fd.pop("R", None)
    return fd


def _fragment_atom_indices(frag_mol: Chem.Mol) -> List[int]:
    idxs: List[int] = []
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        orig_idx = atom.GetAtomMapNum()
        if orig_idx > 0:
            idxs.append(orig_idx - 1)
    return idxs


def _get_fragment_nmr_from_candidate(cand: dict, frag_mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    h_map = _extract_h_map(cand)
    c_eq_map, c_raw_map = _extract_c_map(cand)
    atom_index = cand.get("atom_index")
    equi_class = cand.get("equi_class")
    frag_idxs = _fragment_atom_indices(frag_mol)

    h_vals: List[float] = []
    for idx in frag_idxs:
        h_vals.extend(h_map.get(idx, []))
    h_arr = np.sort(np.array(h_vals, dtype=np.float32)) if h_vals else np.array([], dtype=np.float32)

    if equi_class is not None and atom_index is not None:
        eq_to_shift: Dict[int, float] = {}
        for idx in frag_idxs:
            if idx < len(atom_index) and int(atom_index[idx]) == 6 and idx < len(equi_class):
                eq_to_shift[int(equi_class[idx])] = c_raw_map.get(idx, c_eq_map.get(int(equi_class[idx]), np.nan))
        c_vals = [float(v) for v in eq_to_shift.values() if not np.isnan(float(v))]
    else:
        c_vals = [float(c_raw_map[idx]) for idx in frag_idxs if idx in c_raw_map]
    c_arr = np.sort(np.array(c_vals, dtype=np.float32)) if c_vals else np.array([], dtype=np.float32)
    return h_arr, c_arr


def build_fragment_df(candidates: List[dict], allowed_elements: Optional[set[str]] = None) -> List[dict]:
    rows: List[dict] = []
    allowed = set(allowed_elements or set())
    element_filter = set(allowed) | {"*"}
    if OPTIONAL_HALOGENS & allowed:
        element_filter |= OPTIONAL_HALOGENS

    for mol_id, cand in enumerate(candidates):
        smi = cand.get("smiles", "")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mapped = _set_atom_map_numbers(mol)
        fragment_pairs = _cut_non_ring(mapped) + _cut_ring(mapped)
        for frag_mol, cut_type in fragment_pairs:
            elems = _fragment_elements(frag_mol)
            if allowed and not elems.issubset(element_filter):
                continue
            h_arr, c_arr = _get_fragment_nmr_from_candidate(cand, frag_mol)
            rows.append(
                {
                    "fragment": frag_mol,
                    "cut_type": cut_type,
                    "smiles": Chem.MolToSmiles(frag_mol),
                    "frag_formula": _fragment_formula_dict(frag_mol),
                    "elements": sorted(elems),
                    "mol_id": mol_id,
                    "source_smiles": smi,
                    "source_score": float(cand.get("vector_similarity", cand.get("score", 0.0))),
                    "h_nmr": h_arr,
                    "c_nmr": c_arr,
                }
            )

    dedup: Dict[Tuple[str, str, int], dict] = {}
    for row in rows:
        key = (row["smiles"], repr(row["cut_type"]), int(row["mol_id"]))
        prev = dedup.get(key)
        if prev is None or row["source_score"] > prev["source_score"]:
            dedup[key] = row
    rows = list(dedup.values())

    if OPTIONAL_HALOGENS:
        for atom in sorted(OPTIONAL_HALOGENS & allowed if allowed else OPTIONAL_HALOGENS):
            rows.append(
                {
                    "fragment": Chem.MolFromSmarts(f"[1*][{atom}]"),
                    "cut_type": ((atom, "C"), "-"),
                    "smiles": f"[1*][{atom}]",
                    "frag_formula": {atom: 1},
                    "elements": [atom, "*"],
                    "mol_id": -1,
                    "source_smiles": f"[1*][{atom}]",
                    "source_score": 0.0,
                    "h_nmr": np.array([], dtype=np.float32),
                    "c_nmr": np.array([], dtype=np.float32),
                }
            )

    rows.sort(key=lambda x: repr(x["cut_type"]))
    for idx, row in enumerate(rows):
        row["frag_id"] = idx
    return rows


def _map_query(cut_type) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    atoms, bonds = cut_type
    mapped_atoms = tuple(atoms[i + 1] if atoms[i] == "C" else "*" for i in range(0, len(atoms), 2))
    return mapped_atoms, tuple(bonds)


def _map_key(cut_type) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    atoms, bonds = cut_type
    mapped_atoms = tuple(atoms[i] for i in range(0, len(atoms), 2))
    outs = []
    for mask in product([0, 1], repeat=len(mapped_atoms)):
        combo = tuple(atom if mask[i] == 0 else "*" for i, atom in enumerate(mapped_atoms))
        outs.append((combo, tuple(bonds)))
    return outs


def _build_query_key_clusters(frag_rows: List[dict]) -> List[Tuple[List[int], List[int]]]:
    frag_ids_dict: Dict[Tuple, List[int]] = defaultdict(list)
    for row in frag_rows:
        frag_ids_dict[row["cut_type"]].append(row["frag_id"])

    query_cluster = defaultdict(list)
    key_cluster = defaultdict(list)
    for cut_type in frag_ids_dict:
        query_cluster[_map_query(cut_type)].append(cut_type)
        for key in _map_key(cut_type):
            key_cluster[key].append(cut_type)

    clusters = []
    for key, query_cut_types in query_cluster.items():
        key_cut_types = key_cluster.get(key, [])
        if not key_cut_types:
            continue
        query_ids: List[int] = []
        key_ids: List[int] = []
        for ct in query_cut_types:
            query_ids += frag_ids_dict[ct]
        for ct in key_cut_types:
            key_ids += frag_ids_dict[ct]
        clusters.append((sorted(set(query_ids)), sorted(set(key_ids))))
    return clusters


def search_fragment_pairs(frag_rows: List[dict], query_h: List[float], query_c: List[float], topk: int) -> List[dict]:
    h_lists = [row["h_nmr"] for row in frag_rows]
    c_lists = [row["c_nmr"] for row in frag_rows]
    vec_frags = vector_encoding(h_lists, c_lists, normalize=False)
    vec_target = vector_encoding(
        h_shifts=np.array(query_h, dtype=np.float32) if query_h else None,
        c_shifts=np.array(query_c, dtype=np.float32) if query_c else None,
        normalize=False,
    )[0]

    pair_rows: List[dict] = []
    for query_ids, key_ids in _build_query_key_clusters(frag_rows):
        if not query_ids or not key_ids:
            continue
        query_vecs = vec_frags[query_ids]
        key_vecs = vec_frags[key_ids]
        residual = vec_target[None, :] - query_vecs
        distances = ((residual[:, None, :] - key_vecs[None, :, :]) ** 2).sum(axis=2)
        local_k = min(topk, len(key_ids))
        top_cols = np.argpartition(distances, kth=local_k - 1, axis=1)[:, :local_k]
        for row_pos, frag_i in enumerate(query_ids):
            for col_pos in top_cols[row_pos].tolist():
                frag_j = key_ids[col_pos]
                if frag_i == frag_j:
                    continue
                pair_rows.append(
                    {
                        "left_idx": min(frag_i, frag_j),
                        "right_idx": max(frag_i, frag_j),
                        "coarse_score": -float(distances[row_pos, col_pos]),
                    }
                )

    dedup: Dict[Tuple[int, int], dict] = {}
    for row in pair_rows:
        key = (row["left_idx"], row["right_idx"])
        prev = dedup.get(key)
        if prev is None or row["coarse_score"] > prev["coarse_score"]:
            dedup[key] = row
    return list(dedup.values())


def rerank_fragment_pairs(pair_rows: List[dict], frag_rows: List[dict], query_h: List[float], query_c: List[float]) -> List[dict]:
    qh = np.array(query_h, dtype=np.float32) if query_h else np.array([], dtype=np.float32)
    qc = np.array(query_c, dtype=np.float32) if query_c else np.array([], dtype=np.float32)
    out: List[dict] = []
    for row in pair_rows:
        left = frag_rows[row["left_idx"]]
        right = frag_rows[row["right_idx"]]
        h_merged = np.sort(np.concatenate([left["h_nmr"], right["h_nmr"]], axis=0)).astype(np.float32)
        c_merged = np.sort(np.concatenate([left["c_nmr"], right["c_nmr"]], axis=0)).astype(np.float32)
        h_score = _set_match_score(qh, h_merged, sigma=1.0) if len(qh) and len(h_merged) else 0.0
        c_score = _set_match_score(qc, c_merged, sigma=10.0) if len(qc) and len(c_merged) else 0.0
        score = h_score + c_score
        if score <= 0:
            continue
        new_row = dict(row)
        new_row["merged_h_score"] = float(h_score)
        new_row["merged_c_score"] = float(c_score)
        new_row["pair_score"] = float(score)
        out.append(new_row)
    out.sort(key=lambda x: x["pair_score"], reverse=True)
    return out


def filter_pairs_by_formula(pair_rows: List[dict], frag_rows: List[dict], target_formula: str) -> List[dict]:
    target = parse_formula(target_formula)
    kept = []
    for row in pair_rows:
        left = frag_rows[row["left_idx"]]
        right = frag_rows[row["right_idx"]]
        if add_formula_dict(left.get("frag_formula", {}), right.get("frag_formula", {})) == target:
            kept.append(row)
    return kept


def try_stitch_pair(left_frag: dict, right_frag: dict) -> Optional[str]:
    cut_type = left_frag.get("cut_type")
    if cut_type is None:
        return None
    mol = _stitch(left_frag["fragment"], right_frag["fragment"], cut_type[1] if isinstance(cut_type, tuple) else cut_type)
    if mol is None:
        return None
    mol = _clear_atom_map_numbers(mol)
    return canon_no_stereo(Chem.MolToSmiles(mol))


def stitch_pairs(pair_rows: List[dict], frag_rows: List[dict], target_formula: str, existing_smiles: set[str], max_cycle_length: int, max_new: int) -> List[dict]:
    assembled: List[dict] = []
    seen: set[str] = set()
    for row in pair_rows:
        left = frag_rows[row["left_idx"]]
        right = frag_rows[row["right_idx"]]
        smi = try_stitch_pair(left, right)
        if not smi or smi in seen or smi in existing_smiles:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if rdMolDescriptors.CalcMolFormula(mol) != target_formula:
            continue
        if not mol_ok(mol, max_cycle_length=max_cycle_length, invalid_patterns=INVALID_PATTERNS):
            continue
        seen.add(smi)
        assembled.append(
            {
                "smiles": smi,
                "source": "solver_style_local_core",
                "score": float(row["pair_score"]),
                "coarse_score": float(row["coarse_score"]),
                "merged_h_score": float(row["merged_h_score"]),
                "merged_c_score": float(row["merged_c_score"]),
                "pair_left": left["source_smiles"],
                "pair_right": right["source_smiles"],
                "pair_left_frag_smiles": left["smiles"],
                "pair_right_frag_smiles": right["smiles"],
                "pair_left_cut_type": repr(left["cut_type"]),
                "pair_right_cut_type": repr(right["cut_type"]),
            }
        )
        if len(assembled) >= max_new:
            break
    return assembled


def build_final_rerank_input(pool: List[dict], assembled: List[dict], keep_original: int, keep_new: int) -> List[dict]:
    originals = sorted(pool, key=lambda x: float(x.get("vector_similarity", x.get("score", 0.0))), reverse=True)[:keep_original]
    merged: List[dict] = []
    seen: set[str] = set()
    for cand in originals + assembled[:keep_new]:
        smi = canon_no_stereo(cand.get("smiles", ""))
        if not smi or smi in seen:
            continue
        seen.add(smi)
        merged.append(
            {
                "smiles": smi,
                "source": cand.get("source", "unknown"),
                "score": float(cand.get("vector_similarity", cand.get("score", 0.0))),
                "rank": int(cand.get("merged_rank", cand.get("vector_rank", cand.get("retrieval_rank", 0)))),
            }
        )
    return merged


def run_solver_opt_stage(rec: dict, pool: List[dict], params: SolverStyleParams) -> dict:
    allowed_elements = allowed_elements_from_formula(rec["formula"])
    max_cycle_length = infer_max_cycle_length(pool, floor=6)
    frag_rows = build_fragment_df(pool, allowed_elements=allowed_elements)
    coarse_pairs = search_fragment_pairs(frag_rows, rec["h_shifts"], rec["c_shifts"], topk=params.topk_per_fragment)
    coarse_pairs.sort(key=lambda x: x["coarse_score"], reverse=True)
    coarse_pairs = coarse_pairs[: params.num_filter_pair]
    coarse_pair_count_pre_formula = len(coarse_pairs)
    if not params.disable_formula_pair_prune:
        coarse_pairs = filter_pairs_by_formula(coarse_pairs, frag_rows, rec["formula"])
    pair_rows = rerank_fragment_pairs(coarse_pairs, frag_rows, rec["h_shifts"], rec["c_shifts"])
    assembled = stitch_pairs(
        pair_rows=pair_rows,
        frag_rows=frag_rows,
        target_formula=rec["formula"],
        existing_smiles={x["smiles"] for x in pool},
        max_cycle_length=max_cycle_length,
        max_new=params.max_new,
    )
    rerank_input = build_final_rerank_input(pool, assembled, params.rerank_keep_original, params.rerank_keep_new)
    rerank_result = nmr_rerank(
        h_shifts=rec["h_shifts"],
        c_shifts=rec["c_shifts"],
        candidates=json.dumps(rerank_input, ensure_ascii=False),
        top_k=params.rerank_top_k,
        formula=rec["formula"],
    )
    out = {
        "full_pool_count": len(pool),
        "fragment_pool_count": len(frag_rows),
        "coarse_pair_count_pre_formula": coarse_pair_count_pre_formula,
        "coarse_pair_count": len(coarse_pairs),
        "pair_candidate_count": len(pair_rows),
        "assembled_count": len(assembled),
        "rerank_input_count": len(rerank_input),
        "assembled": assembled,
        "rerank_result": rerank_result,
    }
    gt_smiles = rec.get("gt_smiles", "")
    if gt_smiles:
        out.update(
            {
                "baseline_hits": evaluate_topk(pool, gt_smiles),
                "post_hits": evaluate_topk(rerank_result.get("candidates", []), gt_smiles),
                "gt_in_full_pool": contains_smiles(pool, gt_smiles),
                "gt_full_pool_rank": find_smiles_rank(pool, gt_smiles),
                "gt_in_assembled": contains_smiles(assembled, gt_smiles),
                "gt_assembled_rank": find_smiles_rank(assembled, gt_smiles),
                "gt_in_rerank_topk": contains_smiles(rerank_result.get("candidates", []), gt_smiles),
                "gt_rerank_rank": find_smiles_rank(rerank_result.get("candidates", []), gt_smiles),
            }
        )
    return out


def run_solver_style_only(rec: dict, params: Optional[SolverStyleParams] = None) -> dict:
    params = params or SolverStyleParams()
    clean_pool = build_clean_style_pool(rec, params)
    merged_pool = clean_pool["merged_pool"]
    opt_result = run_solver_opt_stage(rec, merged_pool, params)
    return {
        "idx": rec.get("idx", -1),
        "formula": rec["formula"],
        "gt_smiles": rec.get("gt_smiles", ""),
        "formula_retrieval_count": len(clean_pool["formula_rows_raw"]),
        "nonformula_retrieval_count": len(clean_pool["nonformula_rows_raw"]),
        "formula_vector_count": len(clean_pool["formula_rows"]),
        "nonformula_vector_count": len(clean_pool["nonformula_rows"]),
        "merged_pool_count": len(merged_pool),
        "merged_pool": merged_pool,
        "formula_vector_top10": clean_pool["formula_rows"][:10],
        "nonformula_vector_top10": clean_pool["nonformula_rows"][:10],
        "merged_pool_top20": merged_pool[:20],
        **opt_result,
    }


def strip_atom_maps(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def fragment_atom_indices(frag_mol: Chem.Mol) -> List[int]:
    idxs: List[int] = []
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        atom_map = atom.GetAtomMapNum()
        if atom_map > 0:
            idxs.append(atom_map - 1)
    return idxs


def extract_fragment_shift_lists(candidate: dict, frag_mol: Chem.Mol) -> Tuple[List[float], List[float]]:
    atom_data = candidate.get("atom_data") or _build_atom_data_from_rich(candidate)
    frag_idxs = set(fragment_atom_indices(frag_mol))
    h_vals = sorted(float(x["shift"]) for x in atom_data if x.get("element") == "H" and int(x.get("parent_idx", -1)) in frag_idxs)
    c_vals = sorted(float(x["shift"]) for x in atom_data if x.get("element") == "C" and int(x.get("heavy_idx", -1)) in frag_idxs)
    return h_vals, c_vals


def fragment_formula_from_source(source_mol: Chem.Mol, frag_mol: Chem.Mol) -> Counter[str]:
    counts: Counter[str] = Counter()
    for orig_idx in fragment_atom_indices(frag_mol):
        if 0 <= orig_idx < source_mol.GetNumAtoms():
            atom = source_mol.GetAtomWithIdx(orig_idx)
            counts[atom.GetSymbol()] += 1
            h_count = int(atom.GetTotalNumHs())
            if h_count > 0:
                counts["H"] += h_count
    return counts


def greedy_set_match_score(query: List[float], observed: List[float], sigma: float) -> float:
    if not query or not observed:
        return 0.0
    remaining = list(observed)
    score = 0.0
    for q in sorted(query):
        if not remaining:
            break
        best_idx = min(range(len(remaining)), key=lambda i: abs(remaining[i] - q))
        diff = abs(remaining.pop(best_idx) - q)
        score += math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    return float(score) / float(max(len(query), len(observed)))


def fragment_nmr_score(query_h: List[float], query_c: List[float], frag_h: List[float], frag_c: List[float]) -> float:
    return greedy_set_match_score(query_h, frag_h, 1.0) + greedy_set_match_score(query_c, frag_c, 10.0)


def build_brics_fragment_pool(candidates: List[dict], query_h: List[float], query_c: List[float]) -> List[dict]:
    rows: List[dict] = []
    seen: set[Tuple[str, str]] = set()
    for cand_idx, cand in enumerate(candidates):
        smi = cand.get("smiles", "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mapped = _set_atom_map_numbers(Chem.Mol(mol))
        bonds = list(BRICS.FindBRICSBonds(mapped))
        if not bonds:
            frag_mols = [mapped]
        else:
            broken = BRICS.BreakBRICSBonds(mapped, bonds=bonds)
            frag_mols = list(Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True))
        for frag_mol in frag_mols:
            frag_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
            key = (frag_smiles, smi)
            if key in seen:
                continue
            seen.add(key)
            frag_formula = fragment_formula_from_source(mol, frag_mol)
            frag_h, frag_c = extract_fragment_shift_lists(cand, frag_mol)
            rows.append(
                {
                    "frag_smiles": frag_smiles,
                    "frag_formula": frag_formula,
                    "frag_formula_str": counter_to_formula_str(frag_formula),
                    "source_smiles": smi,
                    "source_rank": cand_idx + 1,
                    "source_score": float(cand.get("vector_similarity", cand.get("score", 0.0))),
                    "frag_score": float(fragment_nmr_score(query_h, query_c, frag_h, frag_c)),
                    "frag_h_count": len(frag_h),
                    "frag_c_count": len(frag_c),
                }
            )
    rows.sort(key=lambda x: (x["frag_score"], x["source_score"]), reverse=True)
    for idx, row in enumerate(rows):
        row["frag_id"] = idx
    return rows


def select_fragment_pool(rows: List[dict], max_per_formula: int, max_total: int) -> List[dict]:
    per_formula: Dict[str, int] = defaultdict(int)
    out: List[dict] = []
    for row in rows:
        key = row["frag_formula_str"]
        if per_formula[key] >= max_per_formula:
            continue
        per_formula[key] += 1
        out.append(row)
        if len(out) >= max_total:
            break
    return out


def is_formula_subcount(partial: Counter[str], target: Counter[str]) -> bool:
    for key, val in partial.items():
        if val > target.get(key, 0):
            return False
    return True


def search_formula_bucket_combinations(fragments: List[dict], target_formula: Counter[str], min_k: int, max_k: int, beam_size: int, max_candidates: int) -> List[dict]:
    by_formula: Dict[str, List[dict]] = defaultdict(list)
    for row in fragments:
        by_formula[row["frag_formula_str"]].append(row)

    bucket_rows: List[dict] = []
    for key, rows in by_formula.items():
        rows = sorted(rows, key=lambda x: (x["frag_score"], x["source_score"]), reverse=True)
        by_formula[key] = rows
        bucket_rows.append(
            {
                "formula_key": key,
                "formula": rows[0]["frag_formula"],
                "best_score": float(rows[0]["frag_score"]),
                "count": len(rows),
            }
        )
    bucket_rows.sort(key=lambda x: x["best_score"], reverse=True)

    complete: List[dict] = []

    def dfs(start: int, depth: int, current_formula: Counter[str], current_keys: List[str], current_score: float) -> None:
        if len(complete) >= max_candidates * 4:
            return
        if current_formula == target_formula:
            if min_k <= depth <= max_k:
                complete.append({"formula": Counter(current_formula), "frag_keys": list(current_keys), "score": float(current_score)})
            return
        if depth >= max_k:
            return

        frontier: List[Tuple[float, int, dict, Counter[str]]] = []
        for pos in range(start, len(bucket_rows)):
            bucket = bucket_rows[pos]
            new_formula = Counter(current_formula)
            new_formula.update(bucket["formula"])
            if not is_formula_subcount(new_formula, target_formula):
                continue
            frontier.append((current_score + bucket["best_score"], pos, bucket, new_formula))
        frontier.sort(key=lambda x: x[0], reverse=True)
        for _, pos, bucket, new_formula in frontier[:beam_size]:
            current_keys.append(bucket["formula_key"])
            dfs(pos, depth + 1, new_formula, current_keys, current_score + bucket["best_score"])
            current_keys.pop()

    dfs(0, 0, Counter(), [], 0.0)
    dedup: Dict[Tuple[str, ...], dict] = {}
    for row in complete:
        key = tuple(row["frag_keys"])
        prev = dedup.get(key)
        if prev is None or row["score"] > prev["score"]:
            dedup[key] = row
    return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:max_candidates]


def expand_fragment_choices(combo: dict, fragments: List[dict], max_variants: int) -> List[List[dict]]:
    by_formula: Dict[str, List[dict]] = defaultdict(list)
    for row in fragments:
        by_formula[row["frag_formula_str"]].append(row)
    pools = [by_formula[key][:max_variants] for key in combo["frag_keys"]]
    if not pools:
        return []
    out: List[List[dict]] = [[]]
    for pool in pools:
        next_out: List[List[dict]] = []
        for prefix in out:
            used_sources = {x["source_smiles"] for x in prefix}
            for row in pool:
                if row["source_smiles"] in used_sources:
                    continue
                next_out.append(prefix + [row])
        out = next_out[: max(1, max_variants * 4)]
        if not out:
            break
    return out[:max_variants]


def assemble_brics_fragments(fragment_rows: List[dict], max_products: int) -> List[str]:
    mols = []
    for row in fragment_rows:
        mol = Chem.MolFromSmiles(row["frag_smiles"])
        if mol is None:
            return []
        mols.append(mol)
    out: List[str] = []
    seen: set[str] = set()
    for prod in BRICS.BRICSBuild(mols, onlyCompleteMols=True, uniquify=True):
        smi = canon_no_stereo(Chem.MolToSmiles(prod, canonical=True))
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
        if len(out) >= max_products:
            break
    return out


def augment_pool_with_solver_style(pool: List[dict], solver_result: dict, include_sections: Optional[List[str]] = None) -> Tuple[List[dict], dict]:
    include_sections = include_sections or ["assembled", "rerank_top"]
    merged: List[dict] = [dict(row) for row in pool]
    by_smiles: Dict[str, dict] = {}
    for row in merged:
        smi = canon_no_stereo(row.get("smiles", ""))
        if smi:
            row["smiles"] = smi
            by_smiles[smi] = row

    added = 0
    added_by_section: Dict[str, int] = {}
    mapping = {
        "assembled": solver_result.get("assembled", []) or [],
        "rerank_top": solver_result.get("rerank_result", {}).get("candidates", []) or [],
    }
    for section in include_sections:
        local_added = 0
        for src_row in mapping.get(section, []):
            smi = canon_no_stereo(src_row.get("smiles", ""))
            if not smi:
                continue
            if smi in by_smiles:
                existing = by_smiles[smi]
                pool_sources = set(existing.get("pool_sources", []))
                pool_sources.add(f"solver_style:{section}")
                existing["pool_sources"] = sorted(pool_sources)
                continue
            row = dict(src_row)
            row["smiles"] = smi
            row["source"] = row.get("source") or f"solver_style_{section}"
            row["score"] = float(row.get("score", row.get("nmr_similarity", row.get("original_score", 0.0))))
            row.setdefault("pool_sources", [])
            row["pool_sources"] = sorted(set(list(row["pool_sources"]) + [f"solver_style:{section}"]))
            merged.append(row)
            by_smiles[smi] = row
            added += 1
            local_added += 1
        added_by_section[section] = local_added

    merged = ensure_opt_ready_candidates(merged)
    merged.sort(key=lambda x: float(x.get("vector_similarity", x.get("score", 0.0))), reverse=True)
    for idx, row in enumerate(merged, start=1):
        row["merged_rank"] = idx
    return merged, {"added_count": added, "added_by_section": added_by_section, "sections": include_sections}


def run_solver_style_plus_brics(rec: dict, solver_params: Optional[SolverStyleParams] = None, brics_params: Optional[BricsParams] = None, augment_sections: Optional[List[str]] = None) -> dict:
    solver_params = solver_params or SolverStyleParams()
    brics_params = brics_params or BricsParams()

    solver_result = run_solver_style_only(rec, solver_params)
    augmented_pool, augment_info = augment_pool_with_solver_style(solver_result["merged_pool"], solver_result, include_sections=augment_sections)
    frag_pool = build_brics_fragment_pool(augmented_pool, rec["h_shifts"], rec["c_shifts"])
    frag_pool = select_fragment_pool(frag_pool, brics_params.max_per_formula, brics_params.max_fragments)

    target_formula = Counter(parse_formula(rec["formula"]))
    combos = search_formula_bucket_combinations(
        frag_pool,
        target_formula=target_formula,
        min_k=brics_params.min_k,
        max_k=brics_params.max_k,
        beam_size=brics_params.beam_size,
        max_candidates=brics_params.max_combos,
    )

    assembled_candidates: List[dict] = []
    seen_smiles: set[str] = set()
    for combo in combos:
        for variant in expand_fragment_choices(combo, frag_pool, brics_params.max_choice_variants):
            for smi in assemble_brics_fragments(variant, brics_params.max_products_per_combo):
                if smi in seen_smiles:
                    continue
                seen_smiles.add(smi)
                mol = Chem.MolFromSmiles(smi)
                if mol is None or rdMolDescriptors.CalcMolFormula(mol) != rec["formula"]:
                    continue
                assembled_candidates.append(
                    {
                        "smiles": smi,
                        "source": "solver_style_plus_brics",
                        "score": float(sum(x["frag_score"] for x in variant)),
                        "fragment_formula_keys": [x["frag_formula_str"] for x in variant],
                        "fragment_smiles": [strip_atom_maps(x["frag_smiles"]) or x["frag_smiles"] for x in variant],
                        "fragment_sources": [x["source_smiles"] for x in variant],
                    }
                )

    rerank_result = {}
    if assembled_candidates:
        payload = [
            {
                "smiles": row["smiles"],
                "source": row.get("source", "solver_style_plus_brics"),
                "score": float(row.get("score", 0.0)),
                "rank": idx + 1,
            }
            for idx, row in enumerate(sorted(assembled_candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)[: brics_params.rerank_top_k])
        ]
        rerank_result = nmr_rerank(
            h_shifts=rec["h_shifts"],
            c_shifts=rec["c_shifts"],
            candidates=json.dumps(payload, ensure_ascii=False),
            top_k=brics_params.rerank_top_k,
            formula=rec["formula"],
        )

    return {
        "solver_style": solver_result,
        "augment_info": augment_info,
        "augmented_pool_count": len(augmented_pool),
        "fragment_pool_count": len(frag_pool),
        "combo_count": len(combos),
        "top_fragment_formulas": [row["frag_formula_str"] for row in frag_pool[:20]],
        "top_combos": combos[:20],
        "assembled_count": len(assembled_candidates),
        "assembled": assembled_candidates,
        "gt_in_assembled": contains_smiles(assembled_candidates, rec.get("gt_smiles", "")) if rec.get("gt_smiles") else False,
        "rerank_result": rerank_result,
    }
