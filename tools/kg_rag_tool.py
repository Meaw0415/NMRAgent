"""Fast Graph RAG tools over the unified NS KG SQLite index."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .decorator import tool
from .path_utils import artifact_path

DEFAULT_INDEX_PATH = artifact_path("ns_rag_index", "kg_index.sqlite")


def _index_path() -> Path:
    return Path(os.environ.get("NMR_KG_RAG_INDEX", str(DEFAULT_INDEX_PATH)))


def _connect(index_path: str = "") -> sqlite3.Connection:
    path = Path(index_path) if index_path else _index_path()
    if not path.exists():
        raise FileNotFoundError(f"KG RAG index not found: {path}. Build it with scripts/build_kg_rag_index.py")
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _rows(cur) -> List[Dict[str, Any]]:
    return [dict(row) for row in cur.fetchall()]


def _safe_json(text: Any) -> Any:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split()).strip()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_:+.-]+", query or "")
    return " OR ".join(tokens[:12]) or query


def _format_node(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["identifiers"] = _safe_json(out.pop("identifiers_json", None))
    out["properties"] = _safe_json(out.pop("properties_json", None))
    out["provenance"] = _safe_json(out.pop("provenance_json", None))
    return out


def _format_doc(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["metadata"] = _safe_json(out.pop("metadata_json", None))
    out["provenance"] = _safe_json(out.pop("provenance_json", None))
    return out


def _format_edge(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["properties"] = _safe_json(out.pop("properties_json", None))
    out["provenance"] = _safe_json(out.pop("provenance_json", None))
    return out


def kg_entity_lookup_impl(
    query: str = "",
    formula: str = "",
    inchikey: str = "",
    source: str = "",
    top_k: int = 10,
    index_path: str = "",
) -> Dict[str, Any]:
    query = _normalize_query(query)
    formula = _normalize_query(formula)
    inchikey = _normalize_query(inchikey)
    source = _normalize_query(source)
    top_k = max(1, min(int(top_k or 10), 50))
    with _connect(index_path) as con:
        hits: List[Dict[str, Any]] = []
        seen = set()

        def add_nodes(sql: str, params: tuple, reason: str) -> None:
            nonlocal hits
            for row in _rows(con.execute(sql, params)):
                node = _format_node(row)
                if node["node_id"] in seen:
                    continue
                seen.add(node["node_id"])
                node["match_reason"] = reason
                hits.append(node)
                if len(hits) >= top_k:
                    return

        if inchikey and len(hits) < top_k:
            add_nodes("SELECT * FROM nodes WHERE inchikey = ? LIMIT ?", (inchikey, top_k), "inchikey")
        if formula and len(hits) < top_k:
            if source:
                add_nodes("SELECT * FROM nodes WHERE formula = ? AND source = ? LIMIT ?", (formula, source, top_k - len(hits)), "formula+source")
            add_nodes("SELECT * FROM nodes WHERE formula = ? LIMIT ?", (formula, top_k - len(hits)), "formula")
        if query and len(hits) < top_k:
            alias_rows = _rows(con.execute(
                "SELECT node_id, alias, alias_type, source FROM aliases WHERE alias_norm = lower(?) LIMIT ?",
                (query, top_k * 3),
            ))
            for alias in alias_rows:
                if len(hits) >= top_k:
                    break
                node_rows = _rows(con.execute("SELECT * FROM nodes WHERE node_id = ?", (alias["node_id"],)))
                for row in node_rows:
                    node = _format_node(row)
                    if node["node_id"] in seen:
                        continue
                    seen.add(node["node_id"])
                    node["match_reason"] = f"alias:{alias['alias_type']}"
                    node["matched_alias"] = alias["alias"]
                    hits.append(node)
        if query and len(hits) < top_k:
            like = f"%{query}%"
            add_nodes("SELECT * FROM nodes WHERE name LIKE ? LIMIT ?", (like, top_k - len(hits)), "name_like")
    return {"valid": 1 if hits else 0, "nodes": hits, "count": len(hits)}


def kg_document_search_impl(
    query: str,
    formula: str = "",
    source: str = "",
    top_k: int = 5,
    index_path: str = "",
) -> Dict[str, Any]:
    query = _normalize_query(query)
    formula = _normalize_query(formula)
    source = _normalize_query(source)
    top_k = max(1, min(int(top_k or 5), 50))
    if not query and not formula:
        return {"valid": 0, "documents": [], "count": 0, "observation": "Empty KG document query."}
    with _connect(index_path) as con:
        params: List[Any] = []
        where = []
        if formula:
            where.append("d.formula = ?")
            params.append(formula)
        if source:
            where.append("d.source = ?")
            params.append(source)
        if query:
            match = _fts_query(query)
            where.append("documents_fts MATCH ?")
            params.append(match)
            rank_expr = "bm25(documents_fts) AS rank"
            join = "JOIN documents_fts ON documents_fts.rowid = d.rowid"
            order = "ORDER BY rank"
        else:
            rank_expr = "0.0 AS rank"
            join = ""
            order = "ORDER BY d.source, d.title"
        sql = f"""
            SELECT d.*, {rank_expr}
            FROM documents d
            {join}
            WHERE {' AND '.join(where) if where else '1=1'}
            {order}
            LIMIT ?
        """
        params.append(top_k)
        docs = [_format_doc(row) for row in _rows(con.execute(sql, tuple(params)))]
    return {"valid": 1 if docs else 0, "documents": docs, "count": len(docs)}


def kg_neighbors_impl(
    node_id: str,
    predicates: str = "",
    sources: str = "",
    limit: int = 20,
    index_path: str = "",
) -> Dict[str, Any]:
    node_id = _normalize_query(node_id)
    if not node_id:
        return {"valid": 0, "neighbors": [], "count": 0, "observation": "Empty node_id."}
    limit = max(1, min(int(limit or 20), 100))
    pred_set = {x.strip() for x in predicates.split(",") if x.strip()}
    src_set = {x.strip() for x in sources.split(",") if x.strip()}
    where = ["node_id = ?"]
    params: List[Any] = [node_id]
    if pred_set:
        where.append("predicate IN (%s)" % ",".join("?" for _ in pred_set))
        params.extend(sorted(pred_set))
    if src_set:
        where.append("source IN (%s)" % ",".join("?" for _ in src_set))
        params.extend(sorted(src_set))
    params.append(limit)
    with _connect(index_path) as con:
        rows = _rows(con.execute(
            f"SELECT * FROM node_neighbors WHERE {' AND '.join(where)} LIMIT ?",
            tuple(params),
        ))
        out = []
        for row in rows:
            item = dict(row)
            node_rows = _rows(con.execute("SELECT node_id, name, labels, source, formula, inchikey, smiles FROM nodes WHERE node_id = ?", (item["neighbor_id"],)))
            item["neighbor"] = dict(node_rows[0]) if node_rows else {"node_id": item["neighbor_id"]}
            out.append(item)
    return {"valid": 1 if out else 0, "neighbors": out, "count": len(out)}


def kg_graph_rag_search_impl(
    query: str,
    formula: str = "",
    inchikey: str = "",
    candidate_smiles: str = "",
    sources: str = "",
    top_k: int = 5,
    neighbor_limit: int = 10,
    index_path: str = "",
) -> Dict[str, Any]:
    query = _normalize_query(query)
    formula = _normalize_query(formula)
    inchikey = _normalize_query(inchikey)
    top_k = max(1, min(int(top_k or 5), 20))
    neighbor_limit = max(0, min(int(neighbor_limit or 10), 50))
    source_list = [x.strip() for x in sources.split(",") if x.strip()]
    source0 = source_list[0] if len(source_list) == 1 else ""

    entity_hits = kg_entity_lookup_impl(query=query, formula=formula, inchikey=inchikey, source=source0, top_k=top_k, index_path=index_path)
    doc_query = " ".join(x for x in [query, formula] if x)
    doc_hits = kg_document_search_impl(query=doc_query, formula="", source=source0, top_k=top_k, index_path=index_path)

    node_ids: List[str] = []
    for node in entity_hits.get("nodes", []):
        if node.get("node_id") and node["node_id"] not in node_ids:
            node_ids.append(node["node_id"])
    for doc in doc_hits.get("documents", []):
        if doc.get("node_id") and doc["node_id"] not in node_ids:
            node_ids.append(doc["node_id"])
    node_ids = node_ids[:top_k]

    neighbors = []
    if neighbor_limit:
        per_node = max(1, neighbor_limit // max(1, len(node_ids)))
        for nid in node_ids:
            res = kg_neighbors_impl(nid, limit=per_node, index_path=index_path)
            neighbors.extend(res.get("neighbors", []))
            if len(neighbors) >= neighbor_limit:
                break

    evidence = []
    for node in entity_hits.get("nodes", [])[:top_k]:
        claim = f"KG contains {node.get('name') or node.get('node_id')} from {node.get('source')}"
        bits = []
        if node.get("formula"):
            bits.append(f"formula {node['formula']}")
        if node.get("labels"):
            bits.append(f"labels {node['labels']}")
        if node.get("smiles"):
            bits.append(f"SMILES {node['smiles']}")
        evidence.append({"source_type": "graph", "claim": claim, "evidence": "; ".join(bits), "metadata": {"node_id": node.get("node_id"), "source": node.get("source"), "match_reason": node.get("match_reason")}, "confidence": "high" if node.get("match_reason") in {"inchikey", "formula"} else "medium", "provenance": node.get("provenance")})
    for doc in doc_hits.get("documents", [])[:top_k]:
        evidence.append({"source_type": "graph", "claim": doc.get("title") or doc.get("doc_id"), "evidence": (doc.get("text") or "")[:700], "metadata": {"doc_id": doc.get("doc_id"), "node_id": doc.get("node_id"), "source": doc.get("source"), "doc_type": doc.get("doc_type")}, "confidence": "medium", "provenance": doc.get("provenance")})
    for nb in neighbors[:neighbor_limit]:
        neighbor = nb.get("neighbor") or {}
        evidence.append({"source_type": "graph", "claim": f"{nb.get('direction')} relation {nb.get('predicate')}", "evidence": f"{nb.get('node_id')} --{nb.get('predicate')}--> {neighbor.get('name') or nb.get('neighbor_id')}", "metadata": {"node_id": nb.get("node_id"), "neighbor_id": nb.get("neighbor_id"), "source": nb.get("source"), "relation_label": nb.get("relation_label")}, "confidence": "medium", "provenance": {"edge_id": nb.get("edge_id")}})

    observation_lines = ["=== KG Graph RAG Evidence ==="]
    for item in evidence[: max(top_k * 3, 10)]:
        observation_lines.append(f"- [{item['source_type']}] {item['claim']}: {item['evidence'][:240]}")
    if not evidence:
        observation_lines.append("No KG evidence found.")
    return {
        "valid": 1 if evidence else 0,
        "query": query,
        "formula": formula,
        "entity_hits": entity_hits.get("nodes", []),
        "document_hits": doc_hits.get("documents", []),
        "neighbors": neighbors[:neighbor_limit],
        "evidence_pack": evidence,
        "observation": "\n".join(observation_lines),
    }


def kg_entity_lookup(query: str = "", formula: str = "", inchikey: str = "", source: str = "", top_k: int = 10) -> dict:
    return kg_entity_lookup_impl(query=query, formula=formula, inchikey=inchikey, source=source, top_k=top_k)


def kg_document_search(query: str, formula: str = "", source: str = "", top_k: int = 5) -> dict:
    return kg_document_search_impl(query=query, formula=formula, source=source, top_k=top_k)


def kg_neighbors(node_id: str, predicates: str = "", sources: str = "", limit: int = 20) -> dict:
    return kg_neighbors_impl(node_id=node_id, predicates=predicates, sources=sources, limit=limit)


@tool(
    name="kg_graph_rag_search",
    description="Fast Graph RAG over the unified chemistry KG. Returns compact evidence pack with metadata hits, KG documents, and one-hop graph context.",
)
def kg_graph_rag_search(query: str, formula: str = "", inchikey: str = "", candidate_smiles: str = "", sources: str = "", top_k: int = 5, neighbor_limit: int = 10) -> dict:
    return kg_graph_rag_search_impl(query=query, formula=formula, inchikey=inchikey, candidate_smiles=candidate_smiles, sources=sources, top_k=top_k, neighbor_limit=neighbor_limit)
