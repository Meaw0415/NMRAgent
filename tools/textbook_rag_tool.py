"""Local textbook RAG tools for NMR spectroscopy evidence."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .decorator import tool
from .path_utils import artifact_path

DEFAULT_TEXTBOOK_PATH = Path(__file__).resolve().parents[1] / "textbook" / "MinerU_markdown_202606220044465_e2317b4d.md"
DEFAULT_INDEX_PATH = artifact_path("textbook_rag", "nmr_textbook.sqlite")


def _textbook_path(path: str = "") -> Path:
    return Path(path or os.environ.get("NMR_TEXTBOOK_PATH", str(DEFAULT_TEXTBOOK_PATH)))


def _index_path(path: str = "") -> Path:
    return Path(path or os.environ.get("NMR_TEXTBOOK_RAG_INDEX", str(DEFAULT_INDEX_PATH)))


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split()).strip()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_:+.-]+", query or "")
    return " OR ".join(tokens[:16]) or query


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Textbook markdown not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def _heading_for(line: str) -> Tuple[int, str]:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not match:
        return 0, ""
    return len(match.group(1)), match.group(2).strip()


def _chunk_markdown(text: str, max_chars: int = 1800, overlap: int = 180) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    headings: Dict[int, str] = {}
    buffer: List[str] = []
    start_line = 1

    def current_heading() -> str:
        ordered = [headings[level] for level in sorted(headings) if headings[level]]
        return " / ".join(ordered[-3:])

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line
        raw = "\n".join(buffer).strip()
        if not raw:
            buffer = []
            start_line = end_line + 1
            return
        while len(raw) > max_chars:
            part = raw[:max_chars].rsplit(" ", 1)[0] or raw[:max_chars]
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):06d}",
                "heading": current_heading(),
                "text": part.strip(),
                "start_line": start_line,
                "end_line": end_line,
            })
            raw = raw[max(0, len(part) - overlap):].strip()
        if raw:
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):06d}",
                "heading": current_heading(),
                "text": raw,
                "start_line": start_line,
                "end_line": end_line,
            })
        buffer = []
        start_line = end_line + 1

    for line_no, line in enumerate(text.splitlines(), start=1):
        level, title = _heading_for(line)
        if level:
            flush(line_no - 1)
            headings[level] = title
            for deeper in [x for x in headings if x > level]:
                headings.pop(deeper, None)
            start_line = line_no
        if not buffer:
            start_line = line_no
        buffer.append(line)
        if sum(len(x) + 1 for x in buffer) >= max_chars:
            flush(line_no)
    flush(len(text.splitlines()))
    return chunks


def build_textbook_rag_index_impl(
    textbook_path: str = "",
    index_path: str = "",
    force: bool = False,
    max_chars: int = 1800,
    overlap: int = 180,
) -> Dict[str, Any]:
    book_path = _textbook_path(textbook_path)
    db_path = _index_path(index_path)
    if db_path.exists() and not force:
        return {"valid": 1, "index_path": str(db_path), "rebuilt": False}
    started = time.time()
    text = _read_text(book_path)
    chunks = _chunk_markdown(text, max_chars=max_chars, overlap=overlap)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(str(db_path)) as con:
        con.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, heading TEXT, text TEXT, start_line INTEGER, end_line INTEGER, source_path TEXT)")
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, heading, text, tokenize='unicode61')")
        con.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            [(row["chunk_id"], row["heading"], row["text"], row["start_line"], row["end_line"], str(book_path)) for row in chunks],
        )
        con.executemany(
            "INSERT INTO chunks_fts (chunk_id, heading, text) VALUES (?, ?, ?)",
            [(row["chunk_id"], row["heading"], row["text"]) for row in chunks],
        )
        con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("source_path", str(book_path)),
                ("chunk_count", str(len(chunks))),
                ("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            ],
        )
    return {"valid": 1, "index_path": str(db_path), "source_path": str(book_path), "chunk_count": len(chunks), "elapsed_sec": round(time.time() - started, 3), "rebuilt": True}


def _ensure_index(index_path: str = "", textbook_path: str = "") -> Path:
    path = _index_path(index_path)
    if not path.exists():
        build_textbook_rag_index_impl(textbook_path=textbook_path, index_path=str(path), force=True)
    return path


def textbook_nmr_search_impl(
    query: str,
    formula: str = "",
    h_shifts: str = "",
    c_shifts: str = "",
    top_k: int = 5,
    index_path: str = "",
    textbook_path: str = "",
) -> Dict[str, Any]:
    query = _normalize_query(query)
    formula = _normalize_query(formula)
    top_k = max(1, min(int(top_k or 5), 20))
    if not query and not formula:
        return {"valid": 0, "documents": [], "evidence_pack": [], "observation": "Empty textbook query."}
    terms = [query, formula]
    if c_shifts:
        terms.append("13C chemical shifts carbonyl alkene aromatic oxygenated")
    if h_shifts:
        terms.append("1H chemical shifts integration coupling")
    search_text = _normalize_query(" ".join(x for x in terms if x))
    db_path = _ensure_index(index_path=index_path, textbook_path=textbook_path)
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        match = _fts_query(search_text)
        rows = [dict(row) for row in con.execute(
            """
            SELECT c.*, bm25(chunks_fts) AS rank
            FROM chunks c
            JOIN chunks_fts ON chunks_fts.chunk_id = c.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, top_k),
        ).fetchall()]
    evidence = []
    for row in rows:
        evidence.append({
            "source_type": "textbook",
            "claim": row.get("heading") or row.get("chunk_id"),
            "evidence": (row.get("text") or "")[:900],
            "metadata": {
                "chunk_id": row.get("chunk_id"),
                "source_path": row.get("source_path"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "rank": row.get("rank"),
            },
            "confidence": "medium",
        })
    observation = ["=== Textbook NMR Evidence ==="]
    observation.extend(f"- {item['claim']}: {item['evidence'][:240]}" for item in evidence)
    if not evidence:
        observation.append("No textbook evidence found.")
    return {
        "valid": 1 if evidence else 0,
        "query": search_text,
        "documents": rows,
        "evidence_pack": evidence,
        "observation": "\n".join(observation),
        "count": len(evidence),
        "index_path": str(db_path),
    }


@tool(name="textbook_nmr_search", description="Search local NMR textbook chunks for formula/NMR interpretation evidence.")
def textbook_nmr_search(query: str, formula: str = "", h_shifts: str = "", c_shifts: str = "", top_k: int = 5, index_path: str = "") -> Dict[str, Any]:
    return textbook_nmr_search_impl(query=query, formula=formula, h_shifts=h_shifts, c_shifts=c_shifts, top_k=top_k, index_path=index_path)
