"""
NMR Rerank Tool

This tool uses NMRNet to predict NMR spectra for candidate SMILES and rerank them
based on similarity to the query NMR spectra.
"""

import torch
import numpy as np
from typing import List, Dict
from .decorator import tool


# Global NMRNet backend cache
_NMRNET_BACKEND = None


def _load_nmrnet_backend():
    """Load NMRNet backend (lazy loading)."""
    global _NMRNET_BACKEND

    if _NMRNET_BACKEND is not None:
        return _NMRNET_BACKEND

    try:
        from .nmr_models import nmrnet_predict

        print("[NMR Rerank] Loading NMRNet backend...")
        _NMRNET_BACKEND = nmrnet_predict
        print("[NMR Rerank] NMRNet backend loaded successfully")
        return nmrnet_predict

    except Exception as e:
        print(f"[NMR Rerank] Failed to load NMRNet backend: {e}")
        import traceback
        traceback.print_exc()
        return None


def _peak_assign(query: np.ndarray, pred: np.ndarray, sigma: float,
                 atom_meta=None):
    """
    Hungarian matching between query peaks and predicted peaks.

    Args:
        query:     query shift array
        pred:      predicted shift array (sorted aggregate, OR raw per-atom)
        sigma:     unused here; kept for API consistency
        atom_meta: optional per-predicted-peak metadata parallel to pred.
                   If provided, matched and unused predicted peaks carry atom data.

    Returns:
        matched:          list[dict] for aligned peaks
        unmatched_query:  list[float] for query peaks with no predicted counterpart
        unused_predicted: list[dict] for predicted peaks not consumed by matching
    """
    from scipy.optimize import linear_sum_assignment

    if len(query) == 0:
        unused_predicted = []
        if len(pred) > 0:
            for idx, ppm in enumerate(pred):
                rec = {"pred_shift": float(ppm)}
                if atom_meta is not None:
                    rec.update(atom_meta[idx])
                unused_predicted.append(rec)
        return [], [], unused_predicted

    if len(pred) == 0:
        return [], [float(q) for q in query], []

    residual_matrix = np.abs(query[:, None] - pred[None, :])
    row_ind, col_ind = linear_sum_assignment(residual_matrix)

    assigned_query = set(row_ind)
    assigned_pred = set(col_ind)

    matched = []
    for r, c in zip(row_ind, col_ind):
        rec = {
            "query_shift": float(query[r]),
            "pred_shift": float(pred[c]),
            "residual": float(residual_matrix[r, c]),
        }
        if atom_meta is not None:
            rec.update(atom_meta[c])
        matched.append(rec)

    unmatched_query = [float(query[i]) for i in range(len(query)) if i not in assigned_query]

    unused_predicted = []
    for idx, ppm in enumerate(pred):
        if idx in assigned_pred:
            continue
        rec = {"pred_shift": float(ppm)}
        if atom_meta is not None:
            rec.update(atom_meta[idx])
        unused_predicted.append(rec)

    return matched, unmatched_query, unused_predicted


def _set_match_score(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """
    Set similarity S(X,Y) = (1/√mn) * max_P Σ exp(-(xi-yj)²/(2σ²))

    Uses Hungarian algorithm for optimal bipartite matching.
    Same formulation as NMR-Solver (setsim.py / nmr_match.py).
    """
    from scipy.optimize import linear_sum_assignment

    if len(x) == 0 or len(y) == 0:
        return 0.0

    score_matrix = np.exp(-(x[:, np.newaxis] - y[np.newaxis, :]) ** 2 / (2 * sigma ** 2))
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    return float(score_matrix[row_ind, col_ind].sum() / np.sqrt(len(x) * len(y)))


def _calculate_nmr_similarity(
    query_h: List[float],
    query_c: List[float],
    pred_h: List[float],
    pred_c: List[float],
    h_sigma: float = 1.0,
    c_sigma: float = 10.0,
) -> float:
    """
    Calculate NMR similarity using set matching (NMR-Solver convention).

    Args:
        query_h: Query ¹H shifts
        query_c: Query ¹³C shifts
        pred_h:  Predicted ¹H shifts
        pred_c:  Predicted ¹³C shifts
        h_sigma: Gaussian width for ¹H matching  (default 1.0 ppm)
        c_sigma: Gaussian width for ¹³C matching (default 10.0 ppm)

    Returns:
        Combined similarity score in [0, 1]
    """
    h_sim = _set_match_score(np.array(query_h, dtype=float),
                              np.array(pred_h,  dtype=float), h_sigma) \
            if len(query_h) > 0 and len(pred_h) > 0 else 0.0

    c_sim = _set_match_score(np.array(query_c, dtype=float),
                              np.array(pred_c,  dtype=float), c_sigma) \
            if len(query_c) > 0 and len(pred_c) > 0 else 0.0

    if len(query_h) > 0 and len(query_c) > 0:
        return (h_sim + c_sim) / 2.0
    return h_sim if len(query_h) > 0 else c_sim


@tool(
    name="nmr_rerank",
    description="Rerank candidate SMILES based on predicted NMR similarity to query spectra."
)
def nmr_rerank(
    h_shifts: str,
    c_shifts: str,
    candidates: str,  # JSON string of candidate list
    top_k: int = 10,
    formula: str = ""
) -> dict:
    """
    Rerank candidate SMILES using NMRNet predictions.

    This tool predicts NMR spectra for each candidate and reranks them based on
    similarity to the query NMR spectra.

    Args:
        h_shifts (str): Comma-separated H-NMR chemical shifts (ppm)
        c_shifts (str): Comma-separated C-NMR chemical shifts (ppm)
        candidates (str): JSON string of candidate list [{"smiles": "...", "score": ...}, ...]
        top_k (int): Number of top candidates to return (default: 10)
        formula (str): Molecular formula to filter candidates (e.g. "C9H12O2"). Only candidates matching this formula are kept.

    Returns:
        dict: Dictionary containing:
            - observation: Human-readable summary
            - candidates: Reranked list of candidates with NMR similarity scores
            - count: Number of candidates returned

    Example:
        >>> nmr_rerank("7.3,7.2,2.3", "138.0,129.0,21.0", '[{"smiles": "Cc1ccccc1", "score": 0.9}]', top_k=5)
    """
    try:
        import json

        # Parse inputs
        h_list = h_shifts if isinstance(h_shifts, list) else [float(x.strip()) for x in h_shifts.split(',') if x.strip()]
        c_list = c_shifts if isinstance(c_shifts, list) else [float(x.strip()) for x in c_shifts.split(',') if x.strip()]
        h_list = [float(x) for x in h_list]
        c_list = [float(x) for x in c_list]

        # Parse candidates
        try:
            if isinstance(candidates, list):
                cand_list = candidates
            else:
                try:
                    cand_list = json.loads(candidates)
                except Exception:
                    try:
                        import ast
                        cand_list = ast.literal_eval(candidates)
                    except Exception:
                        # Last resort: extract SMILES strings via regex
                        import re
                        smiles_found = re.findall(
                            r'"smiles"\s*:\s*"([^"]+)"', candidates
                        )
                        if not smiles_found:
                            smiles_found = re.findall(
                                r"'smiles'\s*:\s*'([^']+)'", candidates
                            )
                        if smiles_found:
                            import re as _re
                            def _clean_smi(s):
                                # Strip all backslash-escapes before parens/brackets
                                s = _re.sub(r'\\+\(', '(', s)
                                s = _re.sub(r'\\+\)', ')', s)
                                s = _re.sub(r'\\+\[', '[', s)
                                s = _re.sub(r'\\+\]', ']', s)
                                return s
                            cand_list = [{'smiles': _clean_smi(s)} for s in smiles_found]
                        else:
                            raise ValueError("no SMILES found")
        except Exception:
            return {
                "observation": "Error: Invalid candidates JSON format",
                "candidates": [],
                "count": 0
            }

        if not cand_list:
            return {
                "observation": "Error: No candidates provided",
                "candidates": [],
                "count": 0
            }

        # Load NMRNet backend
        backend = _load_nmrnet_backend()
        if backend is None:
            return {
                "observation": "Error: Failed to load NMRNet backend",
                "candidates": [],
                "count": 0
            }

        # Normalize candidates: accept list of strings or list of dicts
        if cand_list and isinstance(cand_list[0], str):
            cand_list = [{"smiles": s} for s in cand_list]

        # Deduplicate by canonical SMILES, filter invalid and wrong formula
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        seen = set()
        smiles_list = []
        valid_candidates = []
        for cand in cand_list:
            if not isinstance(cand, dict):
                continue
            smiles = cand.get('smiles', '')
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in seen:
                continue
            seen.add(canon)
            # Filter by formula if provided
            if formula:
                mol_formula = rdMolDescriptors.CalcMolFormula(mol)
                if mol_formula != formula.strip():
                    continue
            smiles_list.append(canon)
            valid_candidates.append(cand)

        if not smiles_list:
            return {
                "observation": "Error: No valid SMILES in candidates",
                "candidates": [],
                "count": 0
            }

        # Batch predict NMR for all candidates (with per-atom data)
        print(f"[NMR Rerank] Batch predicting for {len(smiles_list)} candidates...")
        pred_results = backend.predict_nmr_batch(smiles_list, fast=True,
                                                  include_atom_data=True)

        # Calculate similarity for each candidate
        reranked = []
        failed = 0

        for cand, pred_result in zip(valid_candidates, pred_results):
            if pred_result.get('valid', False):
                pred_h = pred_result['H_shifts']
                pred_c = pred_result['C_shifts']
                atom_data = pred_result.get('atom_data', [])

                # Calculate similarity
                nmr_sim = _calculate_nmr_similarity(h_list, c_list, pred_h, pred_c)

                # Peak assignment using per-atom predictions when available.
                # Build per-atom shift arrays + heavy_idx arrays for H and C.
                if atom_data:
                    h_meta = [
                        {
                            'element': a['element'],
                            'atom_index': a['parent_idx'],
                            'heavy_idx': a['heavy_idx'],
                            'parent_idx': a['parent_idx'],
                        }
                        for a in atom_data if a['element'] == 'H'
                    ]
                    c_meta = [
                        {
                            'element': a['element'],
                            'atom_index': a['heavy_idx'],
                            'heavy_idx': a['heavy_idx'],
                            'parent_idx': a['parent_idx'],
                        }
                        for a in atom_data if a['element'] == 'C'
                    ]
                    pa_h = np.array([a['shift'] for a in atom_data if a['element'] == 'H'], dtype=float)
                    pa_c = np.array([a['shift'] for a in atom_data if a['element'] == 'C'], dtype=float)
                    h_matched, h_unmatched, h_unused_pred = _peak_assign(
                        np.array(h_list, dtype=float), pa_h, sigma=1.0, atom_meta=h_meta)
                    c_matched, c_unmatched, c_unused_pred = _peak_assign(
                        np.array(c_list, dtype=float), pa_c, sigma=10.0, atom_meta=c_meta)
                else:
                    h_matched, h_unmatched, h_unused_pred = _peak_assign(
                        np.array(h_list, dtype=float), pred_h, sigma=1.0)
                    c_matched, c_unmatched, c_unused_pred = _peak_assign(
                        np.array(c_list, dtype=float), pred_c, sigma=10.0)

                reranked.append({
                    'smiles':         cand.get('smiles', ''),
                    'nmr_similarity': nmr_sim,
                    'original_score': cand.get('score', 0.0),
                    'source':         cand.get('source', 'unknown'),
                    'original_rank':  cand.get('rank', 0),
                    'h_assignment':   h_matched,
                    'h_unmatched':    h_unmatched,
                    'c_assignment':   c_matched,
                    'c_unmatched':    c_unmatched,
                    'matched_peaks': {
                        'H': h_matched,
                        'C': c_matched,
                    },
                    'unmatched_query_peaks': {
                        'H': h_unmatched,
                        'C': c_unmatched,
                    },
                    'unused_predicted_peaks': {
                        'H': h_unused_pred,
                        'C': c_unused_pred,
                    },
                    'atom_level_assignment_summary': {
                        'H_matched_count': len(h_matched),
                        'C_matched_count': len(c_matched),
                        'H_unmatched_query_count': len(h_unmatched),
                        'C_unmatched_query_count': len(c_unmatched),
                        'H_unused_predicted_count': len(h_unused_pred),
                        'C_unused_predicted_count': len(c_unused_pred),
                    },
                    'atom_data':      atom_data,
                })
            else:
                failed += 1

        # Sort by NMR similarity (descending)
        reranked.sort(key=lambda x: x['nmr_similarity'], reverse=True)

        # Add new ranks
        for i, cand in enumerate(reranked, 1):
            cand['rank'] = i

        # Return top_k
        top_candidates = reranked[:top_k]

        # Format observation
        obs_lines = [
            f"Reranked {len(cand_list)} candidates using NMRNet predictions:",
            f"  Input: {len(h_list)} H-NMR peaks, {len(c_list)} C-NMR peaks",
            f"  Successfully reranked: {len(reranked)}/{len(cand_list)}",
            f"  Failed: {failed}",
            "",
            "Top candidates:"
        ]

        for i, cand in enumerate(top_candidates[:5], 1):
            source_marker = "🔬" if cand['source'] == 'denovo' else "🔍"
            obs_lines.append(
                f"  {i}. {cand['smiles']} "
                f"(NMR sim: {cand['nmr_similarity']:.4f}, "
                f"orig rank: {cand['original_rank']}) {source_marker}"
            )
            # H assignment
            h_parts = [
                f"{e['query_shift']:.2f}->{e['pred_shift']:.2f}(Δ{e['residual']:.2f})"
                for e in cand['h_assignment'][:6]
            ]
            if cand['h_unmatched']:
                h_parts.append(f"unmatched:{[round(x,2) for x in cand['h_unmatched']]}")
            if cand['unused_predicted_peaks']['H']:
                h_parts.append(
                    'unused_pred:' + str([round(x['pred_shift'], 2) for x in cand['unused_predicted_peaks']['H'][:4]])
                )
            if h_parts:
                obs_lines.append(f"     H: {', '.join(h_parts)}")
            # C assignment
            c_parts = [
                f"{e['query_shift']:.1f}->{e['pred_shift']:.1f}(Δ{e['residual']:.1f})"
                for e in cand['c_assignment'][:6]
            ]
            if cand['c_unmatched']:
                c_parts.append(f"unmatched:{[round(x,1) for x in cand['c_unmatched']]}")
            if cand['unused_predicted_peaks']['C']:
                c_parts.append(
                    'unused_pred:' + str([round(x['pred_shift'], 1) for x in cand['unused_predicted_peaks']['C'][:4]])
                )
            if c_parts:
                obs_lines.append(f"     C: {', '.join(c_parts)}")

        if len(top_candidates) > 5:
            obs_lines.append(f"  ... and {len(top_candidates) - 5} more")

        obs_lines.append("")
        obs_lines.append("🔍 = retrieve, 🔬 = denovo")

        return {
            "observation": "\n".join(obs_lines),
            "candidates": top_candidates,
            "count": len(top_candidates)
        }

    except Exception as e:
        return {
            "observation": f"Error in reranking: {str(e)}",
            "candidates": [],
            "count": 0
        }
