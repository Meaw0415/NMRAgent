from __future__ import annotations

import json

from rdkit import Chem

from .decorator import tool
from .pool_store import load_pool, save_pool


def _canon_no_stereo(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _merge_unique_candidates(candidates: list[dict]) -> list[dict]:
    best_by_smiles = {}
    for row in candidates:
        smi = _canon_no_stereo(row.get("smiles", ""))
        if not smi:
            continue
        new_row = dict(row)
        new_row["smiles"] = smi
        prev = best_by_smiles.get(smi)
        if prev is None or float(new_row.get("score", new_row.get("vector_similarity", 0.0))) > float(prev.get("score", prev.get("vector_similarity", 0.0))):
            sources = set(prev.get("pool_sources", [])) if prev else set()
            sources.update(new_row.get("pool_sources", []))
            source = new_row.get("source")
            if source:
                sources.add(source)
            new_row["pool_sources"] = sorted(sources)
            best_by_smiles[smi] = new_row
    merged = list(best_by_smiles.values())
    merged.sort(key=lambda x: float(x.get("score", x.get("vector_similarity", 0.0))), reverse=True)
    for idx, row in enumerate(merged, start=1):
        row.setdefault("rank", idx)
        row["merged_rank"] = idx
    return merged


@tool(
    name="nmr_merge_pools",
    description="Merge multiple saved candidate pool files, deduplicate by canonical SMILES, and save a new merged pool file.",
)
def nmr_merge_pools(pool_paths: str, output_path: str = "", top_k: int = 0) -> dict:
    try:
        paths = json.loads(pool_paths) if isinstance(pool_paths, str) else list(pool_paths)
    except Exception:
        return {"observation": "Error: invalid pool_paths JSON.", "pool_path": "", "count": 0, "preview": []}

    all_rows = []
    used_paths = []
    queries = []
    for path in paths:
        payload = load_pool(str(path))
        used_paths.append(str(path))
        if payload.get("query"):
            queries.append(payload["query"])
        all_rows.extend(payload.get("candidates", []))

    merged = _merge_unique_candidates(all_rows)
    if top_k and top_k > 0:
        merged = merged[:top_k]
    out_path = save_pool(merged, prefix="merged_pool", query=queries[0] if queries else {}, metadata={"source_pool_paths": used_paths, "merge_count": len(used_paths)}, path=output_path)
    preview = [{"smiles": row.get("smiles", ""), "source": row.get("source", ""), "score": float(row.get("score", row.get("vector_similarity", 0.0)))} for row in merged[:10]]
    return {"observation": f"Merged {len(used_paths)} pools into {out_path} with {len(merged)} unique candidates.", "pool_path": out_path, "count": len(merged), "preview": preview}
