"""
Offline ChefNMR-based de novo generation tool.

This tool reads precomputed ChefNMR candidates from a local pickle cache and
returns them in a format parallel to the existing online/denovo tools.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

from .decorator import tool
from .path_utils import artifact_path


CHEFNMR_OFFLINE_CACHE = os.environ.get(
    "CHEFNMR_OFFLINE_CACHE",
    str(artifact_path("chefnmr-checkpoints", "chefnmr_offline_cache.pkl")),
)

_CACHE: dict | None = None


def _load_cache() -> dict:
    global _CACHE
    if _CACHE is None:
        path = Path(CHEFNMR_OFFLINE_CACHE)
        if not path.exists():
            raise FileNotFoundError(f"ChefNMR offline cache not found: {path}")
        with open(path, "rb") as f:
            _CACHE = pickle.load(f)
    return _CACHE


def _fetch_record(cache: dict, dataset: str, sample_idx: int) -> dict | None:
    key = f"{dataset}:{int(sample_idx)}"

    # Flat mapping: cache["uspto:0"] -> record
    if key in cache:
        record = cache[key]
        return record if isinstance(record, dict) else {"candidates": record}

    # Nested mapping: cache["uspto"][0] or cache["uspto"]["0"]
    if dataset in cache and isinstance(cache[dataset], dict):
        sub = cache[dataset]
        if sample_idx in sub:
            record = sub[sample_idx]
            return record if isinstance(record, dict) else {"candidates": record}
        if str(sample_idx) in sub:
            record = sub[str(sample_idx)]
            return record if isinstance(record, dict) else {"candidates": record}

    return None


def _normalize_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    out = []
    seen = set()
    for item in candidates:
        smi = (item.get("smiles") or "").strip()
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(
            {
                "smiles": smi,
                "score": float(item.get("score", item.get("similarity", 0.0))),
                "source": item.get("source", "chefnmr_denovo_offline"),
                "rank": len(out) + 1,
                **({"target_smiles": item["target_smiles"]} if "target_smiles" in item else {}),
            }
        )
        if len(out) >= top_k:
            break
    return out


@tool(
    name="nmr_chefnmr_offline_denovo",
    description=(
        "Offline ChefNMR-based de novo generation tool. Reads precomputed top-k "
        "ChefNMR candidates from a local pickle cache using dataset + sample_idx."
    ),
)
def nmr_chefnmr_offline_denovo(
    dataset: str,
    sample_idx: int,
    top_k: int = 10,
) -> dict:
    """
    Return precomputed ChefNMR denovo candidates for one benchmark sample.

    Args:
        dataset: Dataset namespace, e.g. 'uspto', 'spectrabase', 'spectranp'
        sample_idx: Sample index inside the dataset split
        top_k: Number of candidates to return

    Returns:
        dict with observation, candidates, count, and metadata
    """
    try:
        dataset = (dataset or "").strip().lower()
        cache = _load_cache()
        record = _fetch_record(cache, dataset, sample_idx)

        if record is None:
            return {
                "observation": (
                    f"No offline ChefNMR record found for dataset={dataset}, "
                    f"sample_idx={sample_idx}. Use nmr_chefnmr_denovo instead."
                ),
                "candidates": [],
                "count": 0,
                "metadata": {},
            }

        candidates = _normalize_candidates(record.get("candidates", []), top_k=top_k)
        metadata = record.get("metadata", {})

        obs_lines = [
            f"ChefNMR offline cache returned {len(candidates)} candidates for dataset={dataset}, sample_idx={sample_idx}:",
        ]
        if metadata:
            for key in ("run_dir", "sample_dir", "ckpt_path", "actual_test_index"):
                if key in metadata:
                    obs_lines.append(f"  {key}: {metadata[key]}")
        obs_lines.append("")
        obs_lines.append("Top candidates:")
        for i, cand in enumerate(candidates[:10], 1):
            obs_lines.append(f"  {i}. {cand['smiles']} (score: {cand['score']:.4f})")
        if len(candidates) > 10:
            obs_lines.append(f"  ... and {len(candidates) - 10} more")

        return {
            "observation": "\n".join(obs_lines),
            "candidates": candidates,
            "count": len(candidates),
            "metadata": metadata,
        }
    except Exception as e:
        import traceback
        return {
            "observation": f"Error in ChefNMR offline denovo: {e}\n{traceback.format_exc()}",
            "candidates": [],
            "count": 0,
            "metadata": {},
        }
