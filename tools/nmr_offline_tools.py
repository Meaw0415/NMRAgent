"""
Offline NMR tools backed by precomputed benchmark caches.

These are drop-in offline counterparts of the main retrieval / denovo / rerank
tools. They read precomputed records from NMRBench LMDB caches instead of
running GPU inference.

Supported datasets:
- nmrgym
- uspto
- spectrabase
- spectranp

Environment variables:
- NMRBENCH_ROOT: root of benchmark datasets
- NMR_OFFLINE_LMDB: optional explicit LMDB path for legacy single-cache use
"""

from __future__ import annotations

import os
import pickle
import traceback
from pathlib import Path
from typing import Any

import lmdb
import msgpack

from .decorator import tool
from .path_utils import artifact_path

_DEFAULT_BENCH_ROOT = Path(
    os.environ.get(
        "NMRBENCH_ROOT",
        str(artifact_path("benchmarks")),
    )
)
_LEGACY_SINGLE_LMDB = os.environ.get("NMR_OFFLINE_LMDB", "").strip()
_SUPPORTED_DATASETS = {"nmrgym", "uspto", "spectrabase", "spectranp"}

_LMDB_ENV_CACHE: dict[str, lmdb.Environment] = {}
_DATASET_LMDB_PATH_CACHE: dict[tuple[str, str], list[Path]] = {}


def _normalize_dataset(dataset: str | None) -> str:
    value = (dataset or "nmrgym").strip().lower()
    aliases = {
        "nmrgym": "nmrgym",
        "uspto": "uspto",
        "spectrabase": "spectrabase",
        "spectranp": "spectranp",
        "np": "spectranp",
    }
    return aliases.get(value, value)


def _normalize_split(split: str | None) -> str:
    value = (split or "test").strip().lower()
    return value or "test"


def _record_key_candidates(dataset: str, split: str, sample_idx: int) -> list[bytes]:
    if dataset == "nmrgym":
        return [f"nmrgym_{sample_idx}".encode()]
    return [
        f"{dataset}_{split}_{sample_idx}".encode(),
        f"{dataset}_{sample_idx}".encode(),
    ]


def _decode_record(raw: bytes) -> dict[str, Any]:
    try:
        obj = msgpack.unpackb(raw, raw=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = pickle.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError("Unsupported offline cache record format")


def _open_env(path: Path) -> lmdb.Environment:
    cache_key = str(path)
    env = _LMDB_ENV_CACHE.get(cache_key)
    if env is None:
        if not path.exists():
            raise FileNotFoundError(f"Offline LMDB not found: {path}")
        env = lmdb.open(
            str(path),
            readonly=True,
            subdir=True,
            lock=False,
            meminit=False,
        )
        _LMDB_ENV_CACHE[cache_key] = env
    return env


def _discover_lmdb_paths(dataset: str, split: str) -> list[Path]:
    cache_key = (dataset, split)
    if cache_key in _DATASET_LMDB_PATH_CACHE:
        return _DATASET_LMDB_PATH_CACHE[cache_key]

    paths: list[Path] = []

    if _LEGACY_SINGLE_LMDB:
        legacy_path = Path(_LEGACY_SINGLE_LMDB)
        if legacy_path.exists():
            paths.append(legacy_path)

    dataset_dir = _DEFAULT_BENCH_ROOT / ("NMRGym" if dataset == "nmrgym" else dataset)
    offline_dir = dataset_dir / "offline_cache"

    if dataset == "nmrgym":
        preferred = [
            dataset_dir / "offline_cache.lmdb",
            dataset_dir / "offline_cache_gpu0.lmdb",
            dataset_dir / "offline_cache_gpu1.lmdb",
            dataset_dir / "offline_cache_gpu2.lmdb",
            dataset_dir / "offline_cache_gpu3.lmdb",
        ]
        paths.extend([p for p in preferred if p.exists()])
    else:
        preferred = [
            offline_dir / f"{split}.lmdb",
            offline_dir / f"{split}_0.lmdb",
            offline_dir / f"{split}_1.lmdb",
            offline_dir / f"{split}_2.lmdb",
            offline_dir / f"{split}_3.lmdb",
        ]
        paths.extend([p for p in preferred if p.exists()])
        if offline_dir.exists():
            extra = sorted(offline_dir.glob(f"{split}_*.lmdb"))
            for path in extra:
                if path not in paths:
                    paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No offline cache found for dataset={dataset}, split={split} under {dataset_dir}"
        )

    _DATASET_LMDB_PATH_CACHE[cache_key] = paths
    return paths


def _fetch_record(dataset: str, sample_idx: int, split: str = "test") -> dict[str, Any] | None:
    dataset = _normalize_dataset(dataset)
    split = _normalize_split(split)

    if dataset not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported: {sorted(_SUPPORTED_DATASETS)}"
        )

    keys = _record_key_candidates(dataset, split, int(sample_idx))
    for lmdb_path in _discover_lmdb_paths(dataset, split):
        env = _open_env(lmdb_path)
        with env.begin() as txn:
            for key in keys:
                raw = txn.get(key)
                if raw is not None:
                    return _decode_record(raw)
    return None


def _clip_top_k(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    top_k = max(1, int(top_k))
    return items[:top_k]


def _format_dataset_prefix(dataset: str, split: str, sample_idx: int) -> str:
    return f"dataset={dataset}, split={split}, sample_idx={sample_idx}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. nmr_offline_retrieve  —  same return format as nmr_retrieve
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nmr_offline_retrieve",
    description=(
        "Offline version of nmr_retrieve. Reads precomputed retrieval results "
        "from NMRBench LMDB caches using dataset + sample_idx."
    ),
)
def nmr_offline_retrieve(
    sample_idx: int,
    dataset: str = "nmrgym",
    split: str = "test",
    top_k: int = 10,
) -> dict:
    """
    Return precomputed retrieval results for one benchmark sample.

    Args:
        sample_idx (int): Sample index from '[Sample ID: N]' in the task prompt.
    """
    try:
        dataset = _normalize_dataset(dataset)
        split = _normalize_split(split)
        record = _fetch_record(dataset=dataset, sample_idx=sample_idx, split=split)

        if record is None:
            return {
                "observation": (
                    f"No precomputed retrieval data for {_format_dataset_prefix(dataset, split, sample_idx)}. "
                    "Use nmr_retrieve instead."
                ),
                "valid": 0,
                "results": [],
                "num_results": 0,
                "matched": 0,
                "unmatched": 0,
            }

        formula = record.get("formula", "") or ""
        raw_items = record.get("retrieval", []) or []
        items = []
        for res in _clip_top_k(raw_items, top_k):
            item = {
                "smiles": res.get("smiles", ""),
                "formula": res.get("formula", ""),
                "similarity": float(res.get("similarity", res.get("cosine_similarity", 0.0))),
            }
            item["formula_match"] = bool(formula and item["formula"] == formula)
            items.append(item)

        matched = [x for x in items if x.get("formula_match", False)]
        unmatched = [x for x in items if not x.get("formula_match", False)]

        obs_lines = []
        if formula:
            obs_lines.append(
                f"Retrieved {len(matched)} formula-matched + {len(unmatched)} similar structures:"
            )
        else:
            obs_lines.append(f"Retrieved {len(items)} similar structures:")
        obs_lines.append(_format_dataset_prefix(dataset, split, sample_idx))
        obs_lines.append(f"Formula: {formula}")
        obs_lines.append("")

        for i, res in enumerate(items, 1):
            mark = "✓" if res.get("formula_match", False) else "✗"
            obs_lines.append(f"{i}. {mark} {res['smiles']}")

        obs_lines.append("")
        obs_lines.append("✓ = formula match, ✗ = similar but different formula")

        return {
            "observation": "\n".join(obs_lines),
            "valid": 1,
            "results": items,
            "num_results": len(items),
            "matched": len(matched),
            "unmatched": len(unmatched),
        }
    except Exception as e:
        return {
            "observation": f"Error in offline retrieval: {e}\n{traceback.format_exc()}",
            "valid": 0,
            "results": [],
            "num_results": 0,
            "matched": 0,
            "unmatched": 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. nmr_offline_denovo  —  same return format as nmr_denovo
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nmr_offline_denovo",
    description=(
        "Offline version of nmr_denovo. Reads precomputed de novo results "
        "from NMRBench LMDB caches using dataset + sample_idx."
    ),
)
def nmr_offline_denovo(
    sample_idx: int,
    dataset: str = "nmrgym",
    split: str = "test",
    top_k: int = 10,
) -> dict:
    """
    Return precomputed de novo generation results for one benchmark sample.

    Args:
        sample_idx (int): Sample index from '[Sample ID: N]' in the task prompt.
    """
    try:
        dataset = _normalize_dataset(dataset)
        split = _normalize_split(split)
        record = _fetch_record(dataset=dataset, sample_idx=sample_idx, split=split)

        if record is None:
            return {
                "observation": (
                    f"No precomputed denovo data for {_format_dataset_prefix(dataset, split, sample_idx)}. "
                    "Use nmr_denovo instead."
                ),
                "candidates": [],
                "count": 0,
            }

        formula = record.get("formula", "") or ""
        candidates = []
        for cand in _clip_top_k(record.get("denovo", []) or [], top_k):
            candidates.append(
                {
                    "smiles": cand.get("smiles", ""),
                    "score": float(cand.get("score", 0.0)),
                    "source": cand.get("source", "denovo"),
                    "rank": int(cand.get("rank", len(candidates) + 1)),
                }
            )
        matched = [c for c in candidates if c.get("source") != "denovo_unmatched"]

        obs_lines = [
            f"Generated {len(candidates)} candidate structures from NMR spectra (de novo):",
            f"  {_format_dataset_prefix(dataset, split, sample_idx)}",
            f"  Formula: {formula}",
            f"  Exact matches: {len(matched)}/{len(candidates)}",
            "",
            "Top candidates:",
        ]
        for i, cand in enumerate(candidates[:5], 1):
            marker = "✓" if cand.get("source") == "denovo" else "~"
            obs_lines.append(
                f"  {i}. {cand['smiles']} (score: {cand['score']:.4f}) {marker}"
            )
        if len(candidates) > 5:
            obs_lines.append(f"  ... and {len(candidates) - 5} more")
        obs_lines.append("")
        obs_lines.append("✓ = exact formula match, ~ = approximate match")

        return {
            "observation": "\n".join(obs_lines),
            "candidates": candidates,
            "count": len(candidates),
        }
    except Exception as e:
        return {
            "observation": f"Error in offline denovo: {e}\n{traceback.format_exc()}",
            "candidates": [],
            "count": 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. nmr_offline_rerank  —  same return format as nmr_rerank
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nmr_offline_rerank",
    description=(
        "Offline version of nmr_rerank. Reads precomputed rerank results "
        "from NMRBench LMDB caches using dataset + sample_idx."
    ),
)
def nmr_offline_rerank(
    sample_idx: int,
    dataset: str = "nmrgym",
    split: str = "test",
    top_k: int = 10,
) -> dict:
    """
    Return precomputed rerank results for one benchmark sample.

    Args:
        sample_idx (int): Sample index from '[Sample ID: N]' in the task prompt.
    """
    try:
        dataset = _normalize_dataset(dataset)
        split = _normalize_split(split)
        record = _fetch_record(dataset=dataset, sample_idx=sample_idx, split=split)

        if record is None:
            return {
                "observation": (
                    f"No precomputed rerank data for {_format_dataset_prefix(dataset, split, sample_idx)}. "
                    "Use nmr_rerank instead."
                ),
                "candidates": [],
                "count": 0,
            }

        candidates = _clip_top_k(record.get("rerank", []) or [], top_k)
        if not candidates:
            err = record.get("error", "unknown error")
            return {
                "observation": (
                    f"Precomputed rerank is empty for {_format_dataset_prefix(dataset, split, sample_idx)} "
                    f"(error: {err}). Use nmr_rerank instead."
                ),
                "candidates": [],
                "count": 0,
            }

        formula = record.get("formula", "") or ""
        top5 = candidates[:5]

        obs_lines = [
            f"Reranked {len(candidates)} candidates by NMR similarity (formula: {formula}):",
            f"  {_format_dataset_prefix(dataset, split, sample_idx)}",
            f"  Successfully reranked: {len(candidates)}/{len(candidates)}",
            f"  Failed: 0",
            "",
            "Top candidates:",
        ]

        for cand in top5:
            source_marker = "🔬" if cand.get("source") == "denovo" else "🔍"
            obs_lines.append(
                f"  {cand.get('rank', '?')}. {cand.get('smiles', '')} "
                f"(NMR sim: {float(cand.get('nmr_similarity', 0.0)):.4f}, "
                f"orig rank: {cand.get('original_rank', '?')}) {source_marker}"
            )
            h_assign = cand.get("h_assignment", [])
            h_parts = [f"{e[0]:.2f}->{e[1]:.2f}(Δ{e[2]:.2f})" for e in h_assign[:6]]
            if cand.get("h_unmatched"):
                h_parts.append(f"unmatched:{[round(x, 2) for x in cand['h_unmatched']]}")
            if h_parts:
                obs_lines.append(f"     H: {', '.join(h_parts)}")

            c_assign = cand.get("c_assignment", [])
            c_parts = [f"{e[0]:.1f}->{e[1]:.1f}(Δ{e[2]:.1f})" for e in c_assign[:6]]
            if cand.get("c_unmatched"):
                c_parts.append(f"unmatched:{[round(x, 1) for x in cand['c_unmatched']]}")
            if c_parts:
                obs_lines.append(f"     C: {', '.join(c_parts)}")

        if len(candidates) > 5:
            obs_lines.append(f"  ... and {len(candidates) - 5} more")
        obs_lines.append("")
        obs_lines.append("🔍 = retrieve, 🔬 = denovo")

        return {
            "observation": "\n".join(obs_lines),
            "candidates": candidates,
            "count": len(candidates),
        }
    except Exception as e:
        return {
            "observation": f"Error in offline rerank: {e}\n{traceback.format_exc()}",
            "candidates": [],
            "count": 0,
        }
