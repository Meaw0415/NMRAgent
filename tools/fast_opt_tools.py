"""
Fast offline optimization helpers.

This module is intentionally not an agent tool wrapper. It is a plain Python
module for offline experiments on merged retrieval + denovo candidate pools.

Core idea:
- keep the candidate pool small
- use formula-aware 2-fragment assembly only
- rerank a small merged pool instead of reproducing the full NMR-Solver search
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
from rdkit import Chem
from rdkit import RDLogger


from rdkit.Chem import BRICS, rdMolDescriptors


def parse_formula(formula: str) -> Dict[str, int]:
    import re
    out: Dict[str, int] = {}
    for elem, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        out[elem] = out.get(elem, 0) + int(cnt or 1)
    return out


def _cut_brics(mol):
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def _get_fragment_formula(frag_smi: str) -> Dict[str, int] | None:
    mol = Chem.MolFromSmiles(frag_smi)
    if not mol:
        return None
    fd = parse_formula(rdMolDescriptors.CalcMolFormula(mol))
    for dummy in ("R", "*"):
        fd.pop(dummy, None)
    return fd


def build_brics_fragment_pool(candidates: List[str], max_candidates: int = 100) -> List[Dict]:
    pool: List[Dict] = []
    seen = set()
    for i, smi in enumerate(candidates[:max_candidates]):
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        for frag_smi in _cut_brics(mol):
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if not frag_mol:
                continue
            canon = canon_no_stereo(frag_smi)
            if not canon or canon in seen:
                continue
            fd = _get_fragment_formula(frag_smi)
            if not fd:
                continue
            pool.append({"id": canon, "smiles": frag_smi, "formula": fd, "mol": frag_mol, "source_idx": i})
            seen.add(canon)
    return pool


def _assemble_combination(frag_mols: List, target_formula_str: str, max_out: int = 10, max_iter: int = 5000) -> List[str]:
    try:
        gen = BRICS.BRICSBuild(frag_mols)
        results: List[str] = []
        for idx, mol in enumerate(gen):
            if idx >= max_iter:
                break
            try:
                if rdMolDescriptors.CalcMolFormula(mol) == target_formula_str:
                    canon = canon_no_stereo(Chem.MolToSmiles(mol))
                    if canon:
                        results.append(canon)
                    if len(results) >= max_out:
                        break
            except Exception:
                continue
        return results
    except Exception:
        return []
from .nmr_rerank_tool import nmr_rerank
from .path_utils import artifact_path

RDLogger.DisableLog("rdApp.*")

_PUBSIM_REVERSE_DIR = Path(os.environ.get("NMR_PUBSIM_REVERSE_DIR", artifact_path("pubsim_rich_shards")))
_PUBSIM_META_LMDB = Path(os.environ.get("NMR_PUBSIM_META_LMDB", artifact_path("pubsim_metadata", "PubChem_merged_id.lmdb")))
_XIEHE_REVERSE_DIR = Path(os.environ.get("NMR_XIEHE_REVERSE_DIR", artifact_path("xiehe_reverse_index_shards")))
_XIEHE_RICH_DIR = Path(os.environ.get("NMR_XIEHE_RICH_DIR", artifact_path("xiehe_rich_shards")))

_PUBSIM_SHARD_ENVS: list[tuple[str, lmdb.Environment]] | None = None
_XIEHE_RAW_ENVS: list[tuple[str, lmdb.Environment]] | None = None
_XIEHE_NS_ENVS: list[tuple[str, lmdb.Environment]] | None = None
_XIEHE_RICH_ENVS: dict[str, lmdb.Environment] = {}


def _hash_key(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).hexdigest().encode("ascii")


def canon_no_stereo(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def canon_with_stereo(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _open_pubsim_reverse_envs():
    global _PUBSIM_SHARD_ENVS
    if _PUBSIM_SHARD_ENVS is not None:
        return _PUBSIM_SHARD_ENVS
    envs = []
    for p in sorted(_PUBSIM_REVERSE_DIR.glob("pubsim_rich_shard_*.lmdb")):
        envs.append((p.name, lmdb.open(str(p), readonly=True, lock=False, subdir=False, readahead=False)))
    _PUBSIM_SHARD_ENVS = envs
    return envs


def _open_xiehe_reverse_envs():
    global _XIEHE_RAW_ENVS, _XIEHE_NS_ENVS
    if _XIEHE_RAW_ENVS is None:
        _XIEHE_RAW_ENVS = []
        for p in sorted(_XIEHE_REVERSE_DIR.glob("xiehe_raw_to_id_*.lmdb")):
            _XIEHE_RAW_ENVS.append((p.name, lmdb.open(str(p), readonly=True, lock=False, subdir=False, readahead=False)))
    if _XIEHE_NS_ENVS is None:
        _XIEHE_NS_ENVS = []
        for p in sorted(_XIEHE_REVERSE_DIR.glob("xiehe_nostereo_to_id_*.lmdb")):
            _XIEHE_NS_ENVS.append((p.name, lmdb.open(str(p), readonly=True, lock=False, subdir=False, readahead=False)))
    return _XIEHE_RAW_ENVS, _XIEHE_NS_ENVS


def _open_xiehe_rich_env(shard_name: str) -> lmdb.Environment:
    env = _XIEHE_RICH_ENVS.get(shard_name)
    if env is not None:
        return env
    path = _XIEHE_RICH_DIR / f"xiehe_rich_shard_{shard_name}.lmdb"
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False, readahead=False)
    _XIEHE_RICH_ENVS[shard_name] = env
    return env


def lookup_pubsim_rich_by_smiles(smiles: str) -> tuple[str | None, int | None, dict | None]:
    cansmi = canon_no_stereo(smiles)
    if not cansmi:
        return None, None, None
    key = _hash_key(cansmi)
    for shard_name, env in _open_pubsim_reverse_envs():
        with env.begin() as txn:
            raw = txn.get(key)
            if raw is None:
                continue
            pubsim_id = int(pickle.loads(raw))
            meta_env = lmdb.open(str(_PUBSIM_META_LMDB), readonly=True, lock=False, subdir=False, readahead=False)
            with meta_env.begin() as meta_txn:
                rich_raw = meta_txn.get(str(pubsim_id).encode())
                rich = pickle.loads(rich_raw) if rich_raw is not None else None
            meta_env.close()
            return shard_name, pubsim_id, rich
    return None, None, None


def lookup_xiehe_rich_by_smiles(smiles: str) -> tuple[str | None, int | None, dict | None]:
    raw_envs, ns_envs = _open_xiehe_reverse_envs()
    raw_key = _hash_key(smiles)
    for name, env in raw_envs:
        with env.begin() as txn:
            raw = txn.get(raw_key)
            if raw is None:
                continue
            try:
                local_id = int(pickle.loads(raw))
            except Exception:
                local_id = int(raw.decode())
            shard_suffix = name.replace("xiehe_raw_to_id_", "").replace(".lmdb", "")
            rich_env = _open_xiehe_rich_env(shard_suffix)
            with rich_env.begin() as rich_txn:
                rich_raw = rich_txn.get(str(local_id).encode())
                rich = pickle.loads(rich_raw) if rich_raw is not None else None
            return shard_suffix, local_id, rich
    ns = canon_no_stereo(smiles)
    if not ns:
        return None, None, None
    ns_key = _hash_key(ns)
    for name, env in ns_envs:
        with env.begin() as txn:
            raw = txn.get(ns_key)
            if raw is None:
                continue
            try:
                local_id = int(pickle.loads(raw))
            except Exception:
                local_id = int(raw.decode())
            shard_suffix = name.replace("xiehe_nostereo_to_id_", "").replace(".lmdb", "")
            rich_env = _open_xiehe_rich_env(shard_suffix)
            with rich_env.begin() as rich_txn:
                rich_raw = rich_txn.get(str(local_id).encode())
                rich = pickle.loads(rich_raw) if rich_raw is not None else None
            return shard_suffix, local_id, rich
    return None, None, None


def enrich_candidate_pool(candidates: List[dict]) -> List[dict]:
    """
    Enrich candidate pool with PubSim/Xiehe rich info.

    Priority:
    1. PubSim reverse index by canonical_no_stereo
    2. Xiehe reverse index by raw smiles
    3. Xiehe reverse index by canonical_no_stereo
    """
    enriched = []
    for cand in candidates:
        out = dict(cand)
        smi = out.get("smiles", "")
        source = None
        shard = None
        source_id = None
        rich = None

        shard, source_id, rich = lookup_pubsim_rich_by_smiles(smi)
        if rich is not None:
            source = "pubsim"
        else:
            shard, source_id, rich = lookup_xiehe_rich_by_smiles(smi)
            if rich is not None:
                source = "xiehe"

        out["rich_source"] = source
        out["rich_found"] = rich is not None
        out["rich_shard"] = shard
        out["rich_id"] = source_id

        if rich is not None:
            out["canonical_smiles"] = rich.get("canonical_smiles", out.get("canonical_smiles"))
            out["formula"] = rich.get("formula", out.get("formula"))
            out["inchikey"] = rich.get("inchikey")
            out["nmr_predict"] = rich.get("nmr_predict")
            out["atom_index"] = rich.get("atom_index")
            out["equi_class"] = rich.get("equi_class")
            out["H_nmr"] = rich.get("H_nmr")
            out["C_nmr"] = rich.get("C_nmr")
            if "H_assignments" in rich:
                out["H_assignments"] = rich.get("H_assignments")
            if "C_assignments" in rich:
                out["C_assignments"] = rich.get("C_assignments")
            if "heavy_atom_count" in rich:
                out["heavy_atom_count"] = rich.get("heavy_atom_count")

        enriched.append(out)
    return enriched


def same_formula(fd1: Dict[str, int], fd2: Dict[str, int]) -> bool:
    keys = set(fd1) | set(fd2)
    return all(fd1.get(k, 0) == fd2.get(k, 0) for k in keys)


def add_formula(fd1: Dict[str, int], fd2: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in fd1.items():
        out[k] = out.get(k, 0) + v
    for k, v in fd2.items():
        out[k] = out.get(k, 0) + v
    return out


def _normalize_candidates(
    entries: List[dict],
    limit: int,
    source: str,
    score_key: str = "similarity",
) -> List[dict]:
    out = []
    seen = set()
    for i, cand in enumerate(entries[:limit], start=1):
        smi = canon_no_stereo(cand.get("smiles", ""))
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(
            {
                "smiles": smi,
                "source": source,
                "rank": i,
                "score": cand.get(score_key, cand.get("score", cand.get("nmr_similarity", 0.0))),
            }
        )
    return out


def build_main_and_donor_pools(
    rec: dict,
    retrieval_k: int = 300,
    denovo_k: int = 30,
    donor_k: int = 100,
) -> Tuple[List[dict], List[dict]]:
    """
    Build:
    - main_pool: candidates that are allowed to survive to final rerank
    - donor_pool: fragment donor molecules used only during assembly

    For mixed-retrieval records we keep formula retrieval in the main pool and
    non-formula retrieval in the donor pool. This avoids polluting the final
    molecule pool with structurally distant full molecules.
    """
    main_pool: List[dict] = []
    donor_pool: List[dict] = []
    seen_main = set()
    seen_donor = set()

    if rec.get("formula_retrieval") or rec.get("nonformula_retrieval"):
        formula_entries = _normalize_candidates(
            rec.get("formula_retrieval", []),
            retrieval_k,
            source="retrieve_formula",
            score_key="similarity",
        )
        donor_entries = _normalize_candidates(
            rec.get("nonformula_retrieval", []),
            donor_k,
            source="retrieve_nonformula",
            score_key="similarity",
        )
    else:
        formula_entries = _normalize_candidates(
            rec.get("retrieval", []),
            retrieval_k,
            source="retrieve",
            score_key="similarity",
        )
        donor_entries = []

    denovo_entries = _normalize_candidates(
        rec.get("denovo", []),
        denovo_k,
        source="denovo",
        score_key="score",
    )

    for cand in formula_entries + denovo_entries:
        smi = cand["smiles"]
        if smi in seen_main:
            continue
        seen_main.add(smi)
        main_pool.append(cand)

    for cand in donor_entries:
        smi = cand["smiles"]
        if smi in seen_main or smi in seen_donor:
            continue
        seen_donor.add(smi)
        donor_pool.append(cand)

    return main_pool, donor_pool


def assemble_formula_guided_pairs(
    pool_smiles: List[str],
    formula: str,
    max_candidates: int = 120,
    max_pairs: int = 500,
    max_assemblies_per_pair: int = 5,
    max_iter_per_pair: int = 2000,
) -> dict:
    target_formula = parse_formula(formula)
    fragment_pool = build_brics_fragment_pool(pool_smiles, max_candidates=max_candidates)

    pair_hits = []
    assembled = []
    seen = set()

    for i, fi in enumerate(fragment_pool):
        for j, fj in enumerate(fragment_pool[i:], start=i):
            combo_formula = add_formula(fi["formula"], fj["formula"])
            if not same_formula(combo_formula, target_formula):
                continue
            pair_hits.append((fi["id"], fj["id"]))
            if len(pair_hits) > max_pairs:
                break
            for smi in _assemble_combination(
                [fi["mol"], fj["mol"]],
                formula,
                max_out=max_assemblies_per_pair,
                max_iter=max_iter_per_pair,
            ):
                canon = canon_no_stereo(smi)
                if canon and canon not in seen:
                    seen.add(canon)
                    assembled.append(canon)
        if len(pair_hits) > max_pairs:
            break

    return {
        "fragment_pool_count": len(fragment_pool),
        "valid_pair_count": len(pair_hits),
        "assembled_count": len(assembled),
        "assembled_smiles": assembled,
    }


def merge_original_and_assembled(pool: List[dict], assembled_smiles: List[str]) -> List[dict]:
    final_pool = list(pool)
    seen = {x["smiles"] for x in pool}
    for smi in assembled_smiles:
        if smi in seen:
            continue
        seen.add(smi)
        final_pool.append(
            {
                "smiles": smi,
                "source": "assembly",
                "rank": 0,
                "score": 0.0,
            }
        )
    return final_pool


def build_rerank_input_pool(
    pool: List[dict],
    assembled_smiles: List[str],
    keep_original: int = 24,
    keep_assembled: int = 12,
) -> List[dict]:
    """
    Build a small rerank pool to keep NMR rerank cost under control.

    Strategy:
    - keep the best-scoring original retrieval/denovo candidates
    - keep a limited number of newly assembled molecules
    - deduplicate by canonical no-stereo SMILES
    """
    originals = sorted(pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:keep_original]
    assembled_entries = [{"smiles": s, "source": "assembly", "rank": 0, "score": 0.0}
                         for s in assembled_smiles[:keep_assembled]]

    merged = []
    seen = set()
    for cand in originals + assembled_entries:
        smi = canon_no_stereo(cand.get("smiles", ""))
        if not smi or smi in seen:
            continue
        seen.add(smi)
        new_cand = dict(cand)
        new_cand["smiles"] = smi
        merged.append(new_cand)
    return merged


def rerank_candidate_pool(
    h_shifts: List[float],
    c_shifts: List[float],
    candidates: List[dict],
    formula: str,
    top_k: int = 20,
) -> dict:
    return nmr_rerank(
        h_shifts=h_shifts,
        c_shifts=c_shifts,
        candidates=json.dumps(candidates, ensure_ascii=False),
        top_k=top_k,
        formula=formula,
    )


def run_fast_opt_pipeline(
    rec: dict,
    retrieval_k: int = 300,
    denovo_k: int = 30,
    donor_k: int = 100,
    max_candidates: int = 120,
    max_pairs: int = 500,
    max_assemblies_per_pair: int = 5,
    max_iter_per_pair: int = 2000,
    rerank_top_k: int = 20,
    rerank_keep_original: int = 24,
    rerank_keep_assembled: int = 12,
    enrich: bool = False,
) -> dict:
    merged_pool, donor_pool = build_main_and_donor_pools(
        rec,
        retrieval_k=retrieval_k,
        denovo_k=denovo_k,
        donor_k=donor_k,
    )
    if enrich:
        merged_pool = enrich_candidate_pool(merged_pool)
        donor_pool = enrich_candidate_pool(donor_pool)
    assembly_pool = merged_pool + donor_pool
    pool_smiles = [x["smiles"] for x in assembly_pool]
    assembly = assemble_formula_guided_pairs(
        pool_smiles=pool_smiles,
        formula=rec["formula"],
        max_candidates=max_candidates,
        max_pairs=max_pairs,
        max_assemblies_per_pair=max_assemblies_per_pair,
        max_iter_per_pair=max_iter_per_pair,
    )
    final_pool = merge_original_and_assembled(merged_pool, assembly["assembled_smiles"])
    rerank_input_pool = build_rerank_input_pool(
        merged_pool,
        assembly["assembled_smiles"],
        keep_original=rerank_keep_original,
        keep_assembled=rerank_keep_assembled,
    )
    rerank = rerank_candidate_pool(
        h_shifts=rec["h_shifts"],
        c_shifts=rec["c_shifts"],
        candidates=rerank_input_pool,
        formula=rec["formula"],
        top_k=rerank_top_k,
    )
    return {
        "merged_pool": merged_pool,
        "donor_pool": donor_pool,
        "assembly_pool": assembly_pool,
        "assembly": assembly,
        "final_pool": final_pool,
        "rerank_input_pool": rerank_input_pool,
        "rerank": rerank,
    }
