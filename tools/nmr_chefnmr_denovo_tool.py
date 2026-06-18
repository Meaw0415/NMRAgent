"""
ChefNMR-based de novo generation tool.

This tool calls a persistent local ChefNMR service over HTTP. The service keeps
the external ChefNMR runtime isolated in its own environment while exposing a
lightweight inference endpoint to NMRAgent tools.
"""

from __future__ import annotations

import csv
import json
import os
import requests
from pathlib import Path
from typing import Dict, List

from .decorator import tool
from .path_utils import artifact_path


CHEFNMR_SERVICE_BASE = os.environ.get("CHEFNMR_SERVICE_BASE", "http://127.0.0.1:8012")
CHEFNMR_CKPT_DIR = Path(
    os.environ.get("CHEFNMR_CKPT_DIR", str(artifact_path("chefnmr-checkpoints")))
)


DEFAULT_DATA_CONFIG = {
    "uspto": "uspto",
    "spectrabase": "spectrabase",
    "spectranp": "spectranp",
}

DEFAULT_CONDITION = {
    "uspto": "h1c13nmr-10k-80",
    "spectrabase": "h1c13nmr-10k-80",
    "spectranp": "h1c13nmr-10k-10k",
}

DEFAULT_MODEL = {
    "uspto": "dit-l",
    "spectrabase": "dit-l",
    "spectranp": "dit-l",
}

DEFAULT_CKPT = {
    "uspto": CHEFNMR_CKPT_DIR / "US-H10kC80-L128-epoch3099.ckpt",
    "spectrabase": CHEFNMR_CKPT_DIR / "SB-H10kC80-L128-epoch5249.ckpt",
    "spectranp": CHEFNMR_CKPT_DIR / "NP-H10kC10k-L64-epoch18149.ckpt",
}


def _canonical_dataset_name(dataset: str) -> str:
    d = (dataset or "").strip().lower()
    aliases = {
        "uspto": "uspto",
        "uspoto": "uspto",
        "spectrabase": "spectrabase",
        "sb": "spectrabase",
        "spectranp": "spectranp",
        "np": "spectranp",
    }
    if d not in aliases:
        raise ValueError(f"Unsupported ChefNMR dataset: {dataset}")
    return aliases[d]


def _default_ckpt_for_dataset(dataset: str) -> Path:
    path = DEFAULT_CKPT[dataset]
    if not path.exists():
        raise FileNotFoundError(f"ChefNMR checkpoint not found: {path}")
    return path


def run_chefnmr_single_sample(
    dataset: str,
    sample_idx: int,
    ckpt_path: str = "",
    diffusion_samples: int = 10,
    num_sampling_steps: int = 20,
    batch_size: int = 1,
) -> Dict:
    dataset = _canonical_dataset_name(dataset)
    ckpt = str(Path(ckpt_path) if ckpt_path else _default_ckpt_for_dataset(dataset))
    payload = {
        "dataset": dataset,
        "sample_idx": int(sample_idx),
        "top_k": 10,
        "ckpt_path": ckpt,
        "diffusion_samples": int(diffusion_samples),
        "num_sampling_steps": int(num_sampling_steps),
        "batch_size": int(batch_size),
    }
    resp = requests.post(
        f"{CHEFNMR_SERVICE_BASE}/infer",
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


@tool(
    name="nmr_chefnmr_denovo",
    description=(
        "ChefNMR-based de novo generation tool. Runs an external diffusion model "
        "for a benchmark sample identified by dataset + sample_idx and returns top-k "
        "predicted candidate SMILES with similarity-derived scores."
    ),
)
def nmr_chefnmr_denovo(
    dataset: str,
    sample_idx: int,
    top_k: int = 10,
    ckpt_path: str = "",
    diffusion_samples: int = 10,
    num_sampling_steps: int = 20,
) -> dict:
    """
    Run ChefNMR standalone inference for one benchmark sample and return top-k candidates.

    Args:
        dataset: One of 'uspto', 'spectrabase', 'spectranp'
        sample_idx: Test index inside the target ChefNMR dataset split
        top_k: Number of candidates to return
        ckpt_path: Optional explicit checkpoint path
        diffusion_samples: Number of samples to generate per target
        num_sampling_steps: Reverse-diffusion steps

    Returns:
        dict with observation, candidates, count, and metadata
    """
    try:
        service_result = run_chefnmr_single_sample(
            dataset=dataset,
            sample_idx=sample_idx,
            ckpt_path=ckpt_path,
            diffusion_samples=diffusion_samples,
            num_sampling_steps=num_sampling_steps,
        )
        candidates = service_result.get("candidates", [])[:top_k]
        meta = service_result.get("metadata", {})
        obs_lines = [
            f"ChefNMR generated {len(candidates)} candidates for dataset={dataset}, sample_idx={sample_idx}:",
            f"  run_dir: {meta['run_dir']}",
            f"  sample_dir: {meta['sample_dir']}",
            "",
            "Top candidates:",
        ]
        for i, cand in enumerate(candidates[:10], 1):
            obs_lines.append(f"  {i}. {cand['smiles']} (score: {cand['score']:.4f})")
        if len(candidates) > 10:
            obs_lines.append(f"  ... and {len(candidates) - 10} more")
        return {
            "observation": "\n".join(obs_lines),
            "candidates": candidates,
            "count": len(candidates),
            "metadata": meta,
        }
    except Exception as e:
        import traceback
        return {
            "observation": f"Error in ChefNMR denovo: {e}\n{traceback.format_exc()}",
            "candidates": [],
            "count": 0,
            "metadata": {},
        }
