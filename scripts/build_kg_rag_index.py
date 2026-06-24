#!/usr/bin/env python3
"""Build a fast SQLite/FTS index for unified NS KG Graph RAG."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULT_KG_DIR = Path("artifacts/ns_unified_kg_merged")
DEFAULT_OUT = Path("artifacts/ns_rag_index/kg_index.sqlite")


def iter_jsonl(path: Path, limit: int = -1):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if limit >= 0 and idx > limit:
                break
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def text_norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def json_get_formula(props: Dict[str, Any]) -> str:
    for key in ("formula", "molecular_formula", "mol_formula", "generalized_empirical_formula"):
        value = text_norm(props.get(key))
        if value:
            return value
    return ""


def json_get_smiles(props: Dict[str, Any]) -> str:
    for key in ("smiles", "canonical_smiles", "smiles_string"):
        value = text_norm(props.get(key))
        if value:
            return value
    return ""


def json_get_inchikey(obj: Dict[str, Any], props: Dict[str, Any]) -> str:
    identifiers = obj.get("identifiers") or {}
    for container in (identifiers, props):
        for key in ("inchikey", "standard_inchi_key", "inchi_key_string"):
            value = text_norm(container.get(key))
            if value:
                return value
    node_id = text_norm(obj.get("node_id"))
    if node_id.startswith("mol:inchikey:"):
        return node_id.split(":", 2)[-1]
    return ""


def label_text(labels: Any) -> str:
    if isinstance(labels, list):
        return " ".join(text_norm(x) for x in labels if text_norm(x))
    return text_norm(labels)


def setup_db(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        DROP TABLE IF EXISTS nodes;
        DROP TABLE IF EXISTS aliases;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS edges;
        DROP TABLE IF EXISTS node_neighbors;
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS documents_fts;

        CREATE TABLE nodes(
            node_id TEXT PRIMARY KEY,
            name TEXT,
            labels TEXT,
            source TEXT,
            formula TEXT,
            inchikey TEXT,
            smiles TEXT,
            identifiers_json TEXT,
            properties_json TEXT,
            provenance_json TEXT
        );
        CREATE TABLE aliases(
            alias TEXT,
            alias_norm TEXT,
            alias_type TEXT,
            node_id TEXT,
            source TEXT,
            provenance_json TEXT
        );
        CREATE TABLE documents(
            doc_id TEXT PRIMARY KEY,
            node_id TEXT,
            title TEXT,
            text TEXT,
            doc_type TEXT,
            source TEXT,
            formula TEXT,
            metadata_json TEXT,
            provenance_json TEXT
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            title,
            text,
            content='documents',
            content_rowid='rowid'
        );
        CREATE TABLE edges(
            edge_id TEXT PRIMARY KEY,
            subject_id TEXT,
            predicate TEXT,
            object_id TEXT,
            relation_label TEXT,
            source TEXT,
            properties_json TEXT,
            provenance_json TEXT
        );
        CREATE TABLE node_neighbors(
            node_id TEXT,
            edge_id TEXT,
            direction TEXT,
            neighbor_id TEXT,
            predicate TEXT,
            relation_label TEXT,
            source TEXT
        );
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        """
    )


def create_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source);
        CREATE INDEX IF NOT EXISTS idx_nodes_formula ON nodes(formula);
        CREATE INDEX IF NOT EXISTS idx_nodes_inchikey ON nodes(inchikey);
        CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
        CREATE INDEX IF NOT EXISTS idx_aliases_norm ON aliases(alias_norm);
        CREATE INDEX IF NOT EXISTS idx_aliases_node ON aliases(node_id);
        CREATE INDEX IF NOT EXISTS idx_aliases_source ON aliases(source);
        CREATE INDEX IF NOT EXISTS idx_documents_node ON documents(node_id);
        CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
        CREATE INDEX IF NOT EXISTS idx_documents_formula ON documents(formula);
        CREATE INDEX IF NOT EXISTS idx_edges_subject ON edges(subject_id);
        CREATE INDEX IF NOT EXISTS idx_edges_object ON edges(object_id);
        CREATE INDEX IF NOT EXISTS idx_edges_predicate ON edges(predicate);
        CREATE INDEX IF NOT EXISTS idx_neighbors_node ON node_neighbors(node_id);
        """
    )


def insert_nodes(con: sqlite3.Connection, kg_dir: Path, limit: int, batch_size: int) -> int:
    count = 0
    rows = []
    for idx, obj in iter_jsonl(kg_dir / "nodes.jsonl", limit) or []:
        props = obj.get("properties") or {}
        rows.append((
            text_norm(obj.get("node_id")),
            text_norm(obj.get("name")),
            label_text(obj.get("labels")),
            text_norm(obj.get("source")),
            json_get_formula(props),
            json_get_inchikey(obj, props),
            json_get_smiles(props),
            json.dumps(obj.get("identifiers") or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(props, ensure_ascii=False, sort_keys=True),
            json.dumps(obj.get("provenance") or {}, ensure_ascii=False, sort_keys=True),
        ))
        count += 1
        if len(rows) >= batch_size:
            con.executemany("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            con.commit(); rows.clear()
            print(f"nodes {count}", flush=True)
    if rows:
        con.executemany("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
    return count


def insert_aliases(con: sqlite3.Connection, kg_dir: Path, limit: int, batch_size: int) -> int:
    count = 0
    rows = []
    for idx, obj in iter_jsonl(kg_dir / "aliases.jsonl", limit) or []:
        alias = text_norm(obj.get("alias"))
        if not alias:
            continue
        rows.append((alias, alias.lower(), text_norm(obj.get("alias_type")), text_norm(obj.get("node_id")), text_norm(obj.get("source")), json.dumps(obj.get("provenance") or {}, ensure_ascii=False, sort_keys=True)))
        count += 1
        if len(rows) >= batch_size:
            con.executemany("INSERT INTO aliases VALUES (?,?,?,?,?,?)", rows)
            con.commit(); rows.clear()
            print(f"aliases {count}", flush=True)
    if rows:
        con.executemany("INSERT INTO aliases VALUES (?,?,?,?,?,?)", rows)
        con.commit()
    return count


def insert_documents(con: sqlite3.Connection, kg_dir: Path, limit: int, batch_size: int) -> int:
    count = 0
    rows = []
    for idx, obj in iter_jsonl(kg_dir / "documents.jsonl", limit) or []:
        meta = obj.get("metadata") or {}
        rows.append((
            text_norm(obj.get("doc_id")),
            text_norm(obj.get("node_id")),
            text_norm(obj.get("title")),
            text_norm(obj.get("text")),
            text_norm(obj.get("doc_type")),
            text_norm(obj.get("source")),
            json_get_formula(meta),
            json.dumps(meta, ensure_ascii=False, sort_keys=True),
            json.dumps(obj.get("provenance") or {}, ensure_ascii=False, sort_keys=True),
        ))
        count += 1
        if len(rows) >= batch_size:
            con.executemany("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?,?)", rows)
            con.commit(); rows.clear()
            print(f"documents {count}", flush=True)
    if rows:
        con.executemany("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
    con.execute("INSERT INTO documents_fts(rowid, title, text) SELECT rowid, title, text FROM documents")
    con.commit()
    return count


def insert_edges(con: sqlite3.Connection, kg_dir: Path, limit: int, batch_size: int) -> int:
    count = 0
    edge_rows = []
    neigh_rows = []
    for idx, obj in iter_jsonl(kg_dir / "edges.jsonl", limit) or []:
        edge_id = text_norm(obj.get("edge_id"))
        subject = text_norm(obj.get("subject_id"))
        object_id = text_norm(obj.get("object_id"))
        predicate = text_norm(obj.get("predicate"))
        relation_label = text_norm(obj.get("relation_label"))
        source = text_norm(obj.get("source"))
        props = json.dumps(obj.get("properties") or {}, ensure_ascii=False, sort_keys=True)
        prov = json.dumps(obj.get("provenance") or {}, ensure_ascii=False, sort_keys=True)
        edge_rows.append((edge_id, subject, predicate, object_id, relation_label, source, props, prov))
        if subject:
            neigh_rows.append((subject, edge_id, "out", object_id, predicate, relation_label, source))
        if object_id:
            neigh_rows.append((object_id, edge_id, "in", subject, predicate, relation_label, source))
        count += 1
        if len(edge_rows) >= batch_size:
            con.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?,?,?)", edge_rows)
            con.executemany("INSERT INTO node_neighbors VALUES (?,?,?,?,?,?,?)", neigh_rows)
            con.commit(); edge_rows.clear(); neigh_rows.clear()
            print(f"edges {count}", flush=True)
    if edge_rows:
        con.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?,?,?)", edge_rows)
        con.executemany("INSERT INTO node_neighbors VALUES (?,?,?,?,?,?,?)", neigh_rows)
        con.commit()
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KG RAG SQLite/FTS index from unified KG JSONL artifacts.")
    p.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit-nodes", type=int, default=-1)
    p.add_argument("--limit-aliases", type=int, default=-1)
    p.add_argument("--limit-documents", type=int, default=-1)
    p.add_argument("--limit-edges", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--skip-edges", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    con = sqlite3.connect(str(args.out))
    setup_db(con)
    counts = {}
    counts["nodes"] = insert_nodes(con, args.kg_dir, args.limit_nodes, args.batch_size)
    counts["aliases"] = insert_aliases(con, args.kg_dir, args.limit_aliases, args.batch_size)
    counts["documents"] = insert_documents(con, args.kg_dir, args.limit_documents, args.batch_size)
    counts["edges"] = 0 if args.skip_edges else insert_edges(con, args.kg_dir, args.limit_edges, args.batch_size)
    create_indexes(con)
    manifest = {
        "kg_dir": str(args.kg_dir),
        "sqlite_path": str(args.out),
        "counts": counts,
        "limits": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    con.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("manifest", json.dumps(manifest, ensure_ascii=False, sort_keys=True)))
    con.commit()
    con.close()
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
