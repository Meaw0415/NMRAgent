#!/usr/bin/env python3
"""Merge per-source unified KG JSONL parts into one directory."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def iter_jsonl(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(handle, obj: Dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def alias_key(obj: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(obj.get("node_id") or ""),
        str(obj.get("alias") or "").lower(),
        str(obj.get("alias_type") or ""),
        str(obj.get("source") or ""),
    )


def merge(parts_dir: Path, out_dir: Path, dedupe_aliases: bool = True) -> Dict[str, Any]:
    started = time.time()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dirs = sorted(p for p in parts_dir.iterdir() if p.is_dir())
    counts = {"nodes": 0, "edges": 0, "documents": 0, "aliases": 0}
    source_manifests = {}

    seen_nodes = set()
    seen_edges = set()
    seen_docs = set()
    seen_aliases = set() if dedupe_aliases else None

    with (out_dir / "nodes.jsonl").open("w", encoding="utf-8") as nodes_f:
        for part in part_dirs:
            for obj in iter_jsonl(part / "nodes.jsonl") or []:
                key = obj.get("node_id")
                if not key or key in seen_nodes:
                    continue
                seen_nodes.add(key)
                write_jsonl(nodes_f, obj)
                counts["nodes"] += 1

    with (out_dir / "edges.jsonl").open("w", encoding="utf-8") as edges_f:
        for part in part_dirs:
            for obj in iter_jsonl(part / "edges.jsonl") or []:
                key = obj.get("edge_id") or "|".join(str(obj.get(k) or "") for k in ["subject_id", "predicate", "object_id", "source"])
                if not key or key in seen_edges:
                    continue
                seen_edges.add(key)
                write_jsonl(edges_f, obj)
                counts["edges"] += 1

    with (out_dir / "documents.jsonl").open("w", encoding="utf-8") as docs_f:
        for part in part_dirs:
            for obj in iter_jsonl(part / "documents.jsonl") or []:
                key = obj.get("doc_id")
                if not key or key in seen_docs:
                    continue
                seen_docs.add(key)
                write_jsonl(docs_f, obj)
                counts["documents"] += 1

    with (out_dir / "aliases.jsonl").open("w", encoding="utf-8") as aliases_f:
        for part in part_dirs:
            for obj in iter_jsonl(part / "aliases.jsonl") or []:
                key = alias_key(obj)
                if seen_aliases is not None:
                    if key in seen_aliases:
                        continue
                    seen_aliases.add(key)
                write_jsonl(aliases_f, obj)
                counts["aliases"] += 1

    for part in part_dirs:
        mf = part / "manifest.json"
        if mf.exists():
            source_manifests[part.name] = json.loads(mf.read_text(encoding="utf-8"))

    manifest = {
        "schema_version": "ns-unified-kg-v0.1-merged",
        "parts_dir": str(parts_dir),
        "out_dir": str(out_dir),
        "sources": sorted(source_manifests),
        "counts": counts,
        "source_manifests": source_manifests,
        "dedupe_aliases": dedupe_aliases,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, default=Path("artifacts/ns_unified_kg_parts"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/ns_unified_kg_merged"))
    parser.add_argument("--no-dedupe-aliases", action="store_true")
    args = parser.parse_args()
    manifest = merge(args.parts_dir, args.out_dir, dedupe_aliases=not args.no_dedupe_aliases)
    print(json.dumps({k: manifest[k] for k in ["schema_version", "sources", "counts", "elapsed_sec", "out_dir"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
