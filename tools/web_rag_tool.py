"""Configurable web-search evidence tool for NMR literature/context."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from urllib import parse, request

from .decorator import tool


def _clean(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _http_json(url: str, headers: Dict[str, str], data: Dict[str, Any] | None = None, timeout: int = 20) -> Dict[str, Any]:
    payload = None if data is None else json.dumps(data).encode("utf-8")
    req = request.Request(url, data=payload, headers=headers, method="POST" if data is not None else "GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _tavily_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": os.environ.get("NMR_WEB_SEARCH_DEPTH", "basic"),
        "max_results": top_k,
        "include_answer": False,
        "include_raw_content": False,
    }
    data = _http_json("https://api.tavily.com/search", {"Content-Type": "application/json"}, payload)
    out = []
    for item in data.get("results", [])[:top_k]:
        out.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "score": item.get("score"),
            "provider": "tavily",
        })
    return out


def _serper_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return []
    data = _http_json(
        "https://google.serper.dev/search",
        {"Content-Type": "application/json", "X-API-KEY": api_key},
        {"q": query, "num": top_k},
    )
    out = []
    for item in data.get("organic", [])[:top_k]:
        out.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "score": None,
            "provider": "serper",
        })
    return out


def _brave_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return []
    url = "https://api.search.brave.com/res/v1/web/search?" + parse.urlencode({"q": query, "count": top_k})
    data = _http_json(url, {"Accept": "application/json", "X-Subscription-Token": api_key}, None)
    out = []
    for item in (data.get("web") or {}).get("results", [])[:top_k]:
        out.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "score": None,
            "provider": "brave",
        })
    return out


def web_nmr_search_impl(
    query: str,
    formula: str = "",
    h_shifts: str = "",
    c_shifts: str = "",
    top_k: int = 5,
    provider: str = "",
) -> Dict[str, Any]:
    query = _clean(query)
    formula = _clean(formula)
    top_k = max(1, min(int(top_k or 5), 10))
    search_query = _clean(" ".join(x for x in [
        query,
        formula,
        "NMR spectroscopy",
        "chemical shift",
        "structure elucidation",
        "paper OR DOI OR PubMed",
    ] if x))
    if not search_query:
        return {"valid": 0, "results": [], "evidence_pack": [], "observation": "Empty web search query."}

    providers = [provider] if provider else ["tavily", "serper", "brave"]
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    for name in providers:
        try:
            if name == "tavily":
                results = _tavily_search(search_query, top_k)
            elif name == "serper":
                results = _serper_search(search_query, top_k)
            elif name == "brave":
                results = _brave_search(search_query, top_k)
            else:
                errors.append(f"unknown provider: {name}")
            if results:
                break
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    evidence = []
    for row in results[:top_k]:
        evidence.append({
            "source_type": "web",
            "claim": row.get("title") or row.get("url"),
            "evidence": row.get("snippet") or "",
            "metadata": {"url": row.get("url"), "provider": row.get("provider"), "score": row.get("score")},
            "confidence": "low",
        })
    if not evidence:
        configured = [name for name, env in [("tavily", "TAVILY_API_KEY"), ("serper", "SERPER_API_KEY"), ("brave", "BRAVE_SEARCH_API_KEY")] if os.environ.get(env)]
        msg = "No configured web search provider. Set TAVILY_API_KEY, SERPER_API_KEY, or BRAVE_SEARCH_API_KEY." if not configured else "No web evidence found."
        if errors:
            msg += " Errors: " + "; ".join(errors[:3])
        return {"valid": 0, "query": search_query, "results": [], "evidence_pack": [], "observation": msg, "errors": errors, "count": 0}

    observation = ["=== Web NMR Evidence ==="]
    observation.extend(f"- {item['claim']}: {item['evidence'][:240]}" for item in evidence)
    return {
        "valid": 1,
        "query": search_query,
        "results": results,
        "evidence_pack": evidence,
        "observation": "\n".join(observation),
        "count": len(evidence),
        "errors": errors,
    }


@tool(name="web_nmr_search", description="Search web/literature snippets for NMR evidence when a search API key is configured.")
def web_nmr_search(query: str, formula: str = "", h_shifts: str = "", c_shifts: str = "", top_k: int = 5, provider: str = "") -> Dict[str, Any]:
    return web_nmr_search_impl(query=query, formula=formula, h_shifts=h_shifts, c_shifts=c_shifts, top_k=top_k, provider=provider)
