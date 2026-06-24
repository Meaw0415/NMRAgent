"""
Stable three-role multi-agent framework for NMR structure elucidation.

Roles:
- Planner: LLM-only NMR reasoning and execution planning.
- Executor: deterministic, tool-bounded candidate/pool generation and optimize.
- Verifier: rerank/alignment tool use plus LLM verdict normalization.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from openai import OpenAI

from configs.runtime_assets import load_runtime_assets
from .prompt import (
    DEFAULT_TOOL_DESCRIPTIONS,
    build_executor_prompt,
    build_planner_prompt,
    build_verifier_prompt,
)

load_runtime_assets()
DEFAULT_EXECUTOR_TOOLS = ["nmr_retrieve", "nmr_denovo", "nmr_merge_pools", "nmr_optimize"]
DEFAULT_VERIFIER_TOOLS = ["nmr_rerank"]
MEMORY_NAMESPACE = ("nmr_agent", "confirmed_cases")


class MultiAgentState(TypedDict, total=False):
    task: str
    formula: str
    h_shifts: List[float]
    c_shifts: List[float]
    sample_idx: Optional[int]
    planner_output: Dict[str, Any]
    executor_output: Dict[str, Any]
    verifier_output: Dict[str, Any]
    rerank_output: Dict[str, Any]
    pooled_paths: List[str]
    merged_pool_path: Optional[str]
    candidate_buffer: List[Dict[str, Any]]
    final_answer: Optional[str]
    next_stage: str
    iteration: int
    max_iterations: int
    trace_log_path: str
    dry_run_tools: bool
    seed_candidates: List[Dict[str, Any]]
    kg_rag_output: Dict[str, Any]
    textbook_rag_output: Dict[str, Any]
    web_rag_output: Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _csv(values: List[float]) -> str:
    return ", ".join(str(float(x)) for x in values)


def _normalize_base_url(base_url: str) -> str:
    value = (base_url or "").rstrip("/")
    if not value:
        return value
    return value if value.endswith("/v1") else value + "/v1"


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    return value if len(value) <= 12 else value[:6] + "..." + value[-4:]


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {"raw": parsed}
            except Exception:
                pass
    return {"raw": text}


class MultiAgentNMRV2:
    def __init__(
        self,
        model_name_or_path: str,
        backend: str = "openai",
        planner_tool_descriptions: Optional[List[str]] = None,
        executor_tools: Optional[List[str]] = None,
        verifier_tools: Optional[List[str]] = None,
        max_iterations: int = 4,
        trace_log_path: str = "",
        dry_run_tools: bool = False,
        memory_store: Optional[InMemoryStore] = None,
        memory_top_k: int = 5,
        auto_remember_accepted: bool = False,
        enable_kg_rag: bool = False,
        kg_rag_top_k: int = 5,
        kg_rag_neighbor_limit: int = 0,
        enable_textbook_rag: bool = False,
        textbook_rag_top_k: int = 5,
        enable_web_rag: bool = False,
        web_rag_top_k: int = 5,
        **backend_kwargs: Any,
    ) -> None:
        if backend != "openai":
            raise ValueError("MultiAgentNMR supports OpenAI-compatible backends only.")
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.max_iterations = max_iterations
        self.executor_tool_names = executor_tools or list(DEFAULT_EXECUTOR_TOOLS)
        self.verifier_tool_names = verifier_tools or list(DEFAULT_VERIFIER_TOOLS)
        self.planner_tool_descriptions = planner_tool_descriptions or list(DEFAULT_TOOL_DESCRIPTIONS)
        self.dry_run_tools = dry_run_tools
        self.verifier_rerank_top_k = int(backend_kwargs.pop("verifier_rerank_top_k", 10))
        self.verifier_rerank_candidate_limit = int(backend_kwargs.pop("verifier_rerank_candidate_limit", 120))
        self.memory_store = memory_store or InMemoryStore()
        self.memory_top_k = int(backend_kwargs.pop("memory_top_k", memory_top_k))
        self.auto_remember_accepted = bool(backend_kwargs.pop("auto_remember_accepted", auto_remember_accepted))
        self.enable_kg_rag = bool(backend_kwargs.pop("enable_kg_rag", enable_kg_rag))
        self.kg_rag_top_k = int(backend_kwargs.pop("kg_rag_top_k", kg_rag_top_k))
        self.kg_rag_neighbor_limit = int(backend_kwargs.pop("kg_rag_neighbor_limit", kg_rag_neighbor_limit))
        self.enable_textbook_rag = bool(backend_kwargs.pop("enable_textbook_rag", enable_textbook_rag))
        self.textbook_rag_top_k = int(backend_kwargs.pop("textbook_rag_top_k", textbook_rag_top_k))
        self.enable_web_rag = bool(backend_kwargs.pop("enable_web_rag", enable_web_rag))
        self.web_rag_top_k = int(backend_kwargs.pop("web_rag_top_k", web_rag_top_k))
        self.trace_log_path = self._init_trace_log_path(trace_log_path)
        self.nmr_skill_text = _read_text(_repo_root() / "NMR_SKILL.md")
        self.planner_prompt = build_planner_prompt(self.nmr_skill_text, self.planner_tool_descriptions)
        self.executor_prompt = build_executor_prompt(self.nmr_skill_text, self.planner_tool_descriptions)
        self.verifier_prompt = build_verifier_prompt(self.nmr_skill_text, self.planner_tool_descriptions)
        self.graph = self._build_graph()
        self._trace("agent_init", {
            "backend": backend,
            "model": model_name_or_path,
            "base_url": self._effective_base_url(),
            "api_key": _mask_secret(self._api_key()),
            "executor_tools": self.executor_tool_names,
            "verifier_tools": self.verifier_tool_names,
            "max_iterations": max_iterations,
            "dry_run_tools": dry_run_tools,
            "memory_top_k": self.memory_top_k,
            "auto_remember_accepted": self.auto_remember_accepted,
            "enable_kg_rag": self.enable_kg_rag,
            "kg_rag_top_k": self.kg_rag_top_k,
            "kg_rag_neighbor_limit": self.kg_rag_neighbor_limit,
            "enable_textbook_rag": self.enable_textbook_rag,
            "textbook_rag_top_k": self.textbook_rag_top_k,
            "enable_web_rag": self.enable_web_rag,
            "web_rag_top_k": self.web_rag_top_k,
        })

    @classmethod
    def from_config(cls, config: Dict[str, Any], **overrides: Any) -> "MultiAgentNMRV2":
        openai_cfg = dict(config.get("openai_compatible", {}) or {})
        ma_cfg = dict(config.get("multi_agent", {}) or {})
        api_key = overrides.pop("api_key", None)
        if api_key is None:
            api_key = openai_cfg.get("api_key") or os.environ.get(openai_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        base_url = overrides.pop("base_url", None)
        if base_url is None:
            base_url = openai_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
        model = overrides.pop("model_name_or_path", None) or overrides.pop("model", None) or openai_cfg.get("model", "gpt-5.4")
        return cls(
            model_name_or_path=model,
            backend=overrides.pop("backend", openai_cfg.get("backend", "openai")),
            planner_tool_descriptions=overrides.pop("planner_tool_descriptions", ma_cfg.get("planner_tool_descriptions") or DEFAULT_TOOL_DESCRIPTIONS),
            executor_tools=overrides.pop("executor_tools", ma_cfg.get("executor_tools") or DEFAULT_EXECUTOR_TOOLS),
            verifier_tools=overrides.pop("verifier_tools", ma_cfg.get("verifier_tools") or DEFAULT_VERIFIER_TOOLS),
            max_iterations=int(overrides.pop("max_iterations", ma_cfg.get("max_iterations", 4))),
            api_key=api_key,
            base_url=base_url,
            temperature=float(overrides.pop("temperature", openai_cfg.get("temperature", 0.0))),
            max_tokens=int(overrides.pop("max_tokens", openai_cfg.get("max_tokens", 4096))),
            trace_log_path=overrides.pop("trace_log_path", ""),
            dry_run_tools=bool(overrides.pop("dry_run_tools", False)),
            memory_top_k=int(overrides.pop("memory_top_k", ma_cfg.get("memory_top_k", 5))),
            auto_remember_accepted=bool(overrides.pop("auto_remember_accepted", ma_cfg.get("auto_remember_accepted", False))),
            enable_kg_rag=bool(overrides.pop("enable_kg_rag", ma_cfg.get("enable_kg_rag", False))),
            kg_rag_top_k=int(overrides.pop("kg_rag_top_k", ma_cfg.get("kg_rag_top_k", 5))),
            kg_rag_neighbor_limit=int(overrides.pop("kg_rag_neighbor_limit", ma_cfg.get("kg_rag_neighbor_limit", 0))),
            enable_textbook_rag=bool(overrides.pop("enable_textbook_rag", ma_cfg.get("enable_textbook_rag", False))),
            textbook_rag_top_k=int(overrides.pop("textbook_rag_top_k", ma_cfg.get("textbook_rag_top_k", 5))),
            enable_web_rag=bool(overrides.pop("enable_web_rag", ma_cfg.get("enable_web_rag", False))),
            web_rag_top_k=int(overrides.pop("web_rag_top_k", ma_cfg.get("web_rag_top_k", 5))),
            verifier_rerank_top_k=int(overrides.pop("verifier_rerank_top_k", ma_cfg.get("verifier_rerank_top_k", 10))),
            verifier_rerank_candidate_limit=int(overrides.pop("verifier_rerank_candidate_limit", ma_cfg.get("verifier_rerank_candidate_limit", 120))),
            **overrides,
        )

    def _init_trace_log_path(self, trace_log_path: str) -> str:
        if trace_log_path:
            path = Path(trace_log_path)
        else:
            log_dir = _repo_root() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = log_dir / f"multi_agent_trace_{stamp}_{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **payload}
        with open(self.trace_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def _api_key(self) -> str:
        return self.backend_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY", "")

    def _effective_base_url(self) -> str:
        return _normalize_base_url(self.backend_kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL", ""))

    def _client(self) -> OpenAI:
        kwargs: Dict[str, Any] = {"api_key": self._api_key()}
        base_url = self._effective_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _peak_distance_score(self, query: List[float], memory: List[float], tolerance: float) -> Dict[str, Any]:
        query_vals = [float(x) for x in query or []]
        memory_vals = [float(x) for x in memory or []]
        if not query_vals or not memory_vals:
            return {"matched": 0, "total": len(query_vals), "mean_abs_delta": None, "score": 0.0}
        used = set()
        deltas: List[float] = []
        for q in query_vals:
            best_idx = None
            best_delta = tolerance
            for idx, m in enumerate(memory_vals):
                if idx in used:
                    continue
                delta = abs(q - m)
                if delta <= best_delta:
                    best_idx = idx
                    best_delta = delta
            if best_idx is not None:
                used.add(best_idx)
                deltas.append(best_delta)
        matched = len(deltas)
        coverage = matched / max(1, len(query_vals))
        closeness = 1.0 - min(1.0, (sum(deltas) / max(1, matched)) / tolerance) if matched else 0.0
        return {
            "matched": matched,
            "total": len(query_vals),
            "mean_abs_delta": round(sum(deltas) / matched, 4) if matched else None,
            "score": round(0.7 * coverage + 0.3 * closeness, 4),
        }

    def _memory_key(self, formula: str, final_smiles: str) -> str:
        canonical = self._canonical_key(final_smiles or "") or "unknown"
        slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{formula}_{canonical}").strip("_")[:120]
        return f"{slug}_{uuid.uuid4().hex[:8]}"

    def remember_confirmed_case(
        self,
        *,
        formula: str,
        h_shifts: List[float],
        c_shifts: List[float],
        final_smiles: str,
        peak_atom_assignments: Optional[Dict[str, Any]] = None,
        diagnostic_notes: Optional[List[str]] = None,
        source: str = "user_confirmed",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a user-confirmed NMR case as structured JSON memory."""
        canonical = self._canonical_key(final_smiles or "")
        key = self._memory_key(formula, canonical or final_smiles)
        value = {
            "formula": formula,
            "h_shifts": [float(x) for x in h_shifts or []],
            "c_shifts": [float(x) for x in c_shifts or []],
            "final_smiles": final_smiles,
            "canonical_smiles": canonical,
            "peak_atom_assignments": peak_atom_assignments or {},
            "diagnostic_notes": diagnostic_notes or [],
            "source": source,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            value["extra"] = extra
        self.memory_store.put(MEMORY_NAMESPACE, key, value, index=False)
        self._trace("memory_put", {"key": key, "value": value})
        return key

    def _get_relevant_memories(self, formula: str, h_shifts: List[float], c_shifts: List[float], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = self.memory_top_k if limit is None else int(limit)
        if limit <= 0:
            return []
        try:
            items = self.memory_store.search(MEMORY_NAMESPACE, query=formula or None, limit=200)
        except Exception as exc:
            self._trace("memory_search_failed", {"error": str(exc)})
            return []
        ranked: List[Dict[str, Any]] = []
        for item in items:
            value = dict(getattr(item, "value", {}) or {})
            h_score = self._peak_distance_score(h_shifts, value.get("h_shifts", []), tolerance=0.08)
            c_score = self._peak_distance_score(c_shifts, value.get("c_shifts", []), tolerance=2.0)
            formula_bonus = 0.15 if value.get("formula") == formula else 0.0
            score = round(formula_bonus + 0.35 * h_score["score"] + 0.5 * c_score["score"], 4)
            ranked.append({
                "key": getattr(item, "key", ""),
                "similarity_score": score,
                "formula_match": value.get("formula") == formula,
                "h_peak_match": h_score,
                "c_peak_match": c_score,
                "memory": value,
            })
        ranked.sort(key=lambda row: row["similarity_score"], reverse=True)
        selected = ranked[:limit]
        if selected:
            self._trace("memory_recall", {"formula": formula, "count": len(selected), "memories": selected})
        return selected

    def _auto_remember_result(self, result: Dict[str, Any]) -> None:
        if not self.auto_remember_accepted or not result.get("final_answer"):
            return
        verdict = result.get("verifier_output") or {}
        if verdict.get("verdict") != "accept":
            return
        existing = self._get_relevant_memories(result.get("formula", ""), result.get("h_shifts", []), result.get("c_shifts", []), limit=20)
        final_key = self._canonical_key(result.get("final_answer", ""))
        for row in existing:
            memory = row.get("memory", {})
            if memory.get("canonical_smiles") == final_key and memory.get("formula") == result.get("formula"):
                return
        rerank_output = result.get("rerank_output") or {}
        self.remember_confirmed_case(
            formula=result.get("formula", ""),
            h_shifts=result.get("h_shifts", []),
            c_shifts=result.get("c_shifts", []),
            final_smiles=result.get("final_answer", ""),
            peak_atom_assignments={"verifier_output": verdict},
            diagnostic_notes=["auto remembered from accepted multi-agent run"],
            source="auto_accepted",
            extra={"rerank_top_candidate": (rerank_output.get("candidates") or [None])[0] if isinstance(rerank_output, dict) else None},
        )

    def export_confirmed_memory_json(self, path: str) -> int:
        """Export confirmed-case memory as a JSON list."""
        items = self.memory_store.search(MEMORY_NAMESPACE, limit=10000)
        records = [{"key": getattr(item, "key", ""), "value": getattr(item, "value", {})} for item in items]
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"namespace": list(MEMORY_NAMESPACE), "confirmed_cases": records}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        self._trace("memory_export", {"path": str(out_path), "count": len(records)})
        return len(records)

    def import_confirmed_memory_json(self, path: str) -> int:
        """Import confirmed-case memory from a JSON list or {confirmed_cases: [...]} object."""
        in_path = Path(path)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        records = payload.get("confirmed_cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("Memory JSON must be a list or contain a confirmed_cases list.")
        count = 0
        for idx, record in enumerate(records):
            if isinstance(record, dict) and "value" in record:
                value = dict(record.get("value") or {})
                key = str(record.get("key") or "")
            elif isinstance(record, dict):
                value = dict(record)
                key = ""
            else:
                continue
            formula = str(value.get("formula") or "")
            final_smiles = str(value.get("canonical_smiles") or value.get("final_smiles") or "")
            if not key:
                key = self._memory_key(formula, final_smiles or f"record_{idx}")
            if final_smiles and not value.get("canonical_smiles"):
                value["canonical_smiles"] = self._canonical_key(final_smiles)
            self.memory_store.put(MEMORY_NAMESPACE, key, value, index=False)
            count += 1
        self._trace("memory_import", {"path": str(in_path), "count": count})
        return count

    def _invoke_json_llm(self, node: str, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=_json_default)},
        ]
        self._trace("llm_request", {"node": node, "model": self.model_name_or_path, "payload_keys": sorted(payload.keys())})
        started = time.time()
        try:
            response = self._client().chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=float(self.backend_kwargs.get("temperature", 0.0)),
                max_tokens=int(self.backend_kwargs.get("max_tokens", 4096)),
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self._trace("llm_json_mode_failed", {"node": node, "error": str(exc)})
            response = self._client().chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=float(self.backend_kwargs.get("temperature", 0.0)),
                max_tokens=int(self.backend_kwargs.get("max_tokens", 4096)),
            )
        content = response.choices[0].message.content or ""
        parsed = _parse_json_object(content)
        self._trace("llm_response", {
            "node": node,
            "elapsed_sec": round(time.time() - started, 3),
            "finish_reason": response.choices[0].finish_reason,
            "content": content[:4000],
            "parsed": parsed,
        })
        return parsed

    def _kg_rag_context(self, formula: str, h_shifts: List[float], c_shifts: List[float], task: str = "") -> Dict[str, Any]:
        if not self.enable_kg_rag:
            return {}
        query_parts = [formula, task]
        if c_shifts:
            c_sorted = sorted(float(x) for x in c_shifts)
            high_c = [x for x in c_sorted if x >= 160]
            if high_c:
                query_parts.append("downfield carbonyl lactone ester ketone")
            if any(100 <= x <= 160 for x in c_sorted):
                query_parts.append("alkene aromatic vinylic")
            if any(50 <= x <= 90 for x in c_sorted):
                query_parts.append("oxygenated sp3 carbon")
        query = " ".join(str(x) for x in query_parts if x)
        try:
            from tools.kg_rag_tool import kg_graph_rag_search_impl

            result = kg_graph_rag_search_impl(
                query=query,
                formula=formula,
                top_k=self.kg_rag_top_k,
                neighbor_limit=self.kg_rag_neighbor_limit,
            )
            self._trace("kg_rag", {"query": query, "valid": result.get("valid"), "evidence_count": len(result.get("evidence_pack", []))})
            return result if isinstance(result, dict) else {"raw": result}
        except Exception as exc:
            self._trace("kg_rag_failed", {"query": query, "error": str(exc)})
            return {"valid": 0, "error": str(exc), "evidence_pack": []}

    def _evidence_query(self, formula: str, h_shifts: List[float], c_shifts: List[float], task: str = "") -> str:
        query_parts = [formula, task]
        if c_shifts:
            c_sorted = sorted(float(x) for x in c_shifts)
            if any(x >= 160 for x in c_sorted):
                query_parts.append("13C downfield carbonyl lactone ester ketone")
            if any(100 <= x <= 160 for x in c_sorted):
                query_parts.append("13C alkene aromatic vinylic")
            if any(50 <= x <= 90 for x in c_sorted):
                query_parts.append("13C oxygenated sp3 carbon")
        if h_shifts:
            query_parts.append("1H chemical shift integration coupling")
        return " ".join(str(x) for x in query_parts if x)

    def _textbook_rag_context(self, formula: str, h_shifts: List[float], c_shifts: List[float], task: str = "") -> Dict[str, Any]:
        if not self.enable_textbook_rag:
            return {}
        query = self._evidence_query(formula, h_shifts, c_shifts, task)
        try:
            from tools.textbook_rag_tool import textbook_nmr_search_impl

            result = textbook_nmr_search_impl(
                query=query,
                formula=formula,
                h_shifts=_csv(h_shifts),
                c_shifts=_csv(c_shifts),
                top_k=self.textbook_rag_top_k,
            )
            self._trace("textbook_rag", {"query": query, "valid": result.get("valid"), "evidence_count": len(result.get("evidence_pack", []))})
            return result if isinstance(result, dict) else {"raw": result}
        except Exception as exc:
            self._trace("textbook_rag_failed", {"query": query, "error": str(exc)})
            return {"valid": 0, "error": str(exc), "evidence_pack": []}

    def _web_rag_context(self, formula: str, h_shifts: List[float], c_shifts: List[float], task: str = "") -> Dict[str, Any]:
        if not self.enable_web_rag:
            return {}
        query = self._evidence_query(formula, h_shifts, c_shifts, task)
        try:
            from tools.web_rag_tool import web_nmr_search_impl

            result = web_nmr_search_impl(
                query=query,
                formula=formula,
                h_shifts=_csv(h_shifts),
                c_shifts=_csv(c_shifts),
                top_k=self.web_rag_top_k,
            )
            self._trace("web_rag", {"query": query, "valid": result.get("valid"), "evidence_count": len(result.get("evidence_pack", []))})
            return result if isinstance(result, dict) else {"raw": result}
        except Exception as exc:
            self._trace("web_rag_failed", {"query": query, "error": str(exc)})
            return {"valid": 0, "error": str(exc), "evidence_pack": []}

    def _planner_node(self, state: MultiAgentState) -> Dict[str, Any]:
        formula = state.get("formula", "")
        h_shifts = state.get("h_shifts", [])
        c_shifts = state.get("c_shifts", [])
        payload = {
            "task": state.get("task", ""),
            "formula": formula,
            "h_shifts": h_shifts,
            "c_shifts": c_shifts,
            "iteration": state.get("iteration", 0),
            "existing_pool_paths": state.get("pooled_paths", []),
            "merged_pool_path": state.get("merged_pool_path"),
            "previous_executor_output": state.get("executor_output", {}),
            "previous_verifier_output": state.get("verifier_output", {}),
            "relevant_confirmed_memories": self._get_relevant_memories(formula, h_shifts, c_shifts),
            "retrieved_kg_evidence": state.get("kg_rag_output", {}),
            "retrieved_textbook_evidence": state.get("textbook_rag_output", {}),
            "retrieved_web_evidence": state.get("web_rag_output", {}),
        }
        plan = self._invoke_json_llm("planner", self.planner_prompt, payload)
        previous_verdict = (state.get("verifier_output") or {}).get("verdict")
        if previous_verdict == "need_bigger_pool":
            plan["need_large_pool"] = True
            plan["use_retrieval"] = True
            plan["use_denovo"] = True
            plan["retrieval_top_k"] = max(int(plan.get("retrieval_top_k", 0) or 0), 100)
            plan["denovo_top_k"] = max(int(plan.get("denovo_top_k", 0) or 0), 20)
            plan["save_pool_file"] = True
        if previous_verdict == "need_opt":
            plan["need_opt_after_generation"] = True
            plan["save_pool_file"] = True
        return {"planner_output": plan, "next_stage": "executor"}

    def _tool_map(self, names: List[str]) -> Dict[str, Any]:
        from tools import get_tools_by_names

        tools = get_tools_by_names(names)
        out: Dict[str, Any] = {}
        for tool_fn in tools:
            out[getattr(tool_fn, "name", getattr(tool_fn, "__name__", ""))] = tool_fn
        return out

    def _rows(self, value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, dict):
            value = value.get("results") or value.get("candidates") or value.get("preview") or []
        if not isinstance(value, list):
            return []
        rows: List[Dict[str, Any]] = []
        for row in value:
            if isinstance(row, str):
                rows.append({"smiles": row})
            elif isinstance(row, dict) and row.get("smiles"):
                rows.append(dict(row))
        return rows

    def _load_pool_preview(self, pool_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not pool_path:
            return []
        try:
            from tools.pool_store import load_pool_candidates

            rows = load_pool_candidates(pool_path)
            return rows if limit is None else rows[:limit]
        except Exception as exc:
            self._trace("pool_preview_failed", {"pool_path": pool_path, "error": str(exc)})
            return []

    def _canonical_key(self, smiles: str) -> str:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles or "")
            if mol is None:
                return smiles or ""
            Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            return smiles or ""

    def _dedupe_candidates(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for row in rows:
            smiles = row.get("smiles")
            key = self._canonical_key(smiles or "")
            if not key or key in seen:
                continue
            seen.add(key)
            new_row = dict(row)
            new_row.setdefault("canonical_smiles", key)
            deduped.append(new_row)
        return deduped

    def _select_rerank_candidates(self, candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return list(candidates)
        buckets = {"denovo": [], "retrieval": [], "optimize": [], "merged": [], "other": []}
        for row in candidates:
            source = str(row.get("source") or "").lower()
            pool_sources = [str(x).lower() for x in row.get("pool_sources", [])] if isinstance(row.get("pool_sources"), list) else []
            if source == "denovo" or "denovo" in pool_sources:
                buckets["denovo"].append(row)
            elif source in {"retrieval", "retrieve"} or "retrieval" in pool_sources or "retrieve" in pool_sources:
                buckets["retrieval"].append(row)
            elif source in {"optimize", "opt"} or "optimize" in pool_sources or "opt" in pool_sources:
                buckets["optimize"].append(row)
            elif source == "merged" or "merged" in pool_sources:
                buckets["merged"].append(row)
            else:
                buckets["other"].append(row)

        selected: List[Dict[str, Any]] = []
        per_source_floor = max(5, min(30, limit // 4))
        for name in ["denovo", "retrieval", "optimize", "merged", "other"]:
            selected.extend(buckets[name][:per_source_floor])
        selected.extend(candidates)
        return self._dedupe_candidates(selected)[:limit]

    def _executor_node(self, state: MultiAgentState) -> Dict[str, Any]:
        plan = state.get("planner_output", {}) or {}
        actions: List[str] = []
        pool_paths = list(state.get("pooled_paths", []) or [])
        candidate_buffer = list(state.get("candidate_buffer", []) or [])
        formula = state.get("formula", "")
        h_values = list(state.get("h_shifts", []) or [])
        # Denovo must preserve repeated 1H shifts from integration expansion.
        # Retrieval can use unique representative shifts to avoid overweighting integrations.
        h_csv_denovo = _csv(h_values)
        h_csv_retrieval = _csv(list(dict.fromkeys(float(x) for x in h_values)))
        c_csv = _csv(state.get("c_shifts", []) or [])
        merged_pool_path = state.get("merged_pool_path")
        dry_run = bool(state.get("dry_run_tools", self.dry_run_tools))
        tools = {name: None for name in self.executor_tool_names} if dry_run else self._tool_map(self.executor_tool_names)

        def call_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
            self._trace("tool_start", {"node": "executor", "tool": name, "args": kwargs})
            if dry_run:
                result = {"observation": f"Dry-run skipped {name}.", "results": [], "candidates": [], "count": 0}
            else:
                try:
                    result = tools[name](**kwargs)
                except Exception as exc:
                    result = {"observation": f"Error in {name}: {exc}", "results": [], "candidates": [], "count": 0, "error": str(exc)}
            self._trace("tool_end", {"node": "executor", "tool": name, "result": result})
            return result if isinstance(result, dict) else {"raw": result}

        save_pool_file = bool(plan.get("save_pool_file", True))
        retrieval_top_k = int(plan.get("retrieval_top_k", 20) or 20)
        denovo_top_k = int(plan.get("denovo_top_k", 20) or 20)
        if plan.get("need_large_pool"):
            retrieval_top_k = max(retrieval_top_k, 100)
            denovo_top_k = max(denovo_top_k, 20)

        retrieval_tool = "nmr_retrieve" if "nmr_retrieve" in tools else "nmr_retrieve_service" if "nmr_retrieve_service" in tools else ""
        denovo_tool = "nmr_denovo" if "nmr_denovo" in tools else "nmr_denovo_service" if "nmr_denovo_service" in tools else ""

        if plan.get("use_retrieval") and retrieval_tool:
            res = call_tool(retrieval_tool, h_shifts=h_csv_retrieval, c_shifts=c_csv, formula=formula, top_k=retrieval_top_k, save_pool_file=save_pool_file)
            actions.append("retrieval")
            rows = self._rows(res)
            for row in rows:
                row.setdefault("source", "retrieval")
            candidate_buffer.extend(rows)
            if res.get("pool_path"):
                pool_paths.append(str(res["pool_path"]))

        if plan.get("use_denovo") and denovo_tool:
            res = call_tool(denovo_tool, h_shifts=h_csv_denovo, c_shifts=c_csv, formula=formula, top_k=denovo_top_k, save_pool_file=save_pool_file)
            actions.append("denovo")
            rows = self._rows(res)
            for row in rows:
                row.setdefault("source", "denovo")
            candidate_buffer.extend(rows)
            if res.get("pool_path"):
                pool_paths.append(str(res["pool_path"]))

        pool_paths = list(dict.fromkeys([p for p in pool_paths if p]))
        if len(pool_paths) >= 2 and "nmr_merge_pools" in tools:
            res = call_tool("nmr_merge_pools", pool_paths=json.dumps(pool_paths), top_k=0)
            actions.append("merge_pools")
            if res.get("pool_path"):
                merged_pool_path = str(res["pool_path"])
                candidate_buffer.extend(self._load_pool_preview(merged_pool_path))
            candidate_buffer.extend(self._rows(res))
        elif pool_paths and not merged_pool_path:
            merged_pool_path = pool_paths[0]
            candidate_buffer.extend(self._load_pool_preview(merged_pool_path))

        optimize_attempted = False
        if plan.get("need_opt_after_generation") and merged_pool_path and "nmr_optimize" in tools:
            res = call_tool("nmr_optimize", pool_path=merged_pool_path, formula=formula, h_shifts=h_csv_denovo, c_shifts=c_csv, mode="hybrid", top_k=10, save_pool_file=True)
            actions.append("optimize")
            optimize_attempted = True
            candidate_buffer.extend(self._rows(res))
            if res.get("pool_path"):
                pool_paths.append(str(res["pool_path"]))

        deduped_candidates = self._dedupe_candidates(candidate_buffer)

        executor_output = {
            "actions_taken": actions,
            "pool_paths": pool_paths,
            "merged_pool_path": merged_pool_path or "",
            "optimize_attempted": optimize_attempted,
            "candidate_count": len(deduped_candidates),
            "top_candidates": deduped_candidates[:20],
            "notes_for_verifier": plan.get("notes_for_executor", ""),
            "dry_run_tools": dry_run,
        }
        self._trace("executor_output", executor_output)
        return {
            "executor_output": executor_output,
            "pooled_paths": pool_paths,
            "merged_pool_path": merged_pool_path,
            "candidate_buffer": deduped_candidates,
            "next_stage": "verifier",
        }

    def _verifier_node(self, state: MultiAgentState) -> Dict[str, Any]:
        candidates = list(state.get("candidate_buffer", []) or [])
        formula = state.get("formula", "")
        h_values = list(state.get("h_shifts", []) or [])
        # Denovo must preserve repeated 1H shifts from integration expansion.
        # Retrieval can use unique representative shifts to avoid overweighting integrations.
        h_csv_denovo = _csv(h_values)
        h_csv_retrieval = _csv(list(dict.fromkeys(float(x) for x in h_values)))
        c_csv = _csv(state.get("c_shifts", []) or [])
        dry_run = bool(state.get("dry_run_tools", self.dry_run_tools))
        rerank_output: Dict[str, Any] = {}
        verifier_tools = {name: None for name in self.verifier_tool_names} if dry_run else self._tool_map(self.verifier_tool_names)
        rerank_candidates = self._select_rerank_candidates(candidates, self.verifier_rerank_candidate_limit)
        if rerank_candidates and "nmr_rerank" in verifier_tools:
            args = {"h_shifts": h_csv_denovo, "c_shifts": c_csv, "candidates": json.dumps(rerank_candidates, ensure_ascii=False, default=_json_default), "top_k": self.verifier_rerank_top_k, "formula": formula}
            self._trace("tool_start", {"node": "verifier", "tool": "nmr_rerank", "args": {"h_shifts": h_csv_denovo, "c_shifts": c_csv, "top_k": self.verifier_rerank_top_k, "formula": formula, "candidates_count": len(rerank_candidates), "total_candidate_count": len(candidates)}})
            if dry_run:
                rerank_output = {"observation": "Dry-run skipped nmr_rerank.", "candidates": rerank_candidates[: self.verifier_rerank_top_k], "count": min(len(rerank_candidates), self.verifier_rerank_top_k)}
            else:
                try:
                    raw = verifier_tools["nmr_rerank"](**args)
                    rerank_output = raw if isinstance(raw, dict) else {"raw": raw}
                except Exception as exc:
                    rerank_output = {"observation": f"Error in nmr_rerank: {exc}", "candidates": [], "count": 0, "error": str(exc)}
            self._trace("tool_end", {"node": "verifier", "tool": "nmr_rerank", "result": rerank_output})

        verdict = self._invoke_json_llm("verifier", self.verifier_prompt, {
            "task": state.get("task", ""),
            "formula": formula,
            "h_shifts": state.get("h_shifts", []),
            "c_shifts": state.get("c_shifts", []),
            "iteration": state.get("iteration", 0),
            "executor_output": state.get("executor_output", {}),
            "rerank_output": rerank_output,
            "relevant_confirmed_memories": self._get_relevant_memories(formula, state.get("h_shifts", []), state.get("c_shifts", [])),
            "retrieved_kg_evidence": state.get("kg_rag_output", {}),
            "retrieved_textbook_evidence": state.get("textbook_rag_output", {}),
            "retrieved_web_evidence": state.get("web_rag_output", {}),
        })
        if verdict.get("verdict") not in {"accept", "need_opt", "need_bigger_pool", "need_retry"}:
            verdict["verdict"] = "need_retry"
        iteration = int(state.get("iteration", 0) or 0) + 1
        final_answer = verdict.get("top_candidate") if verdict.get("verdict") == "accept" else None
        if verdict.get("verdict") == "accept":
            next_stage = "end"
        elif iteration >= int(state.get("max_iterations", self.max_iterations) or self.max_iterations):
            next_stage = "end"
        else:
            next_stage = "planner"
        update = {
            "verifier_output": verdict,
            "rerank_output": rerank_output,
            "cand_list": rerank_output.get("candidates", rerank_candidates[: self.verifier_rerank_top_k]) if isinstance(rerank_output, dict) else rerank_candidates[: self.verifier_rerank_top_k],
            "final_answer": final_answer,
            "next_stage": next_stage,
            "iteration": iteration,
        }
        self._trace("verifier_output", update)
        return update

    def _route_after_verifier(self, state: MultiAgentState) -> str:
        return "planner" if state.get("next_stage") == "planner" else END

    def _build_graph(self):
        workflow = StateGraph(MultiAgentState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("verifier", self._verifier_node)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "verifier")
        workflow.add_conditional_edges("verifier", self._route_after_verifier, {"planner": "planner", END: END})
        return workflow.compile()

    def run(self, task: str, *, formula: str, h_shifts: List[float], c_shifts: List[float], sample_idx: Optional[int] = None, max_iterations: Optional[int] = None, dry_run_tools: Optional[bool] = None, seed_candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        kg_rag_output = self._kg_rag_context(formula, h_shifts, c_shifts, task)
        textbook_rag_output = self._textbook_rag_context(formula, h_shifts, c_shifts, task)
        web_rag_output = self._web_rag_context(formula, h_shifts, c_shifts, task)
        initial: MultiAgentState = {
            "task": task,
            "formula": formula,
            "h_shifts": h_shifts,
            "c_shifts": c_shifts,
            "sample_idx": sample_idx,
            "planner_output": {},
            "executor_output": {},
            "verifier_output": {},
            "rerank_output": {},
            "pooled_paths": [],
            "merged_pool_path": None,
            "candidate_buffer": list(seed_candidates or []),
            "final_answer": None,
            "next_stage": "planner",
            "iteration": 0,
            "max_iterations": max_iterations or self.max_iterations,
            "trace_log_path": self.trace_log_path,
            "dry_run_tools": self.dry_run_tools if dry_run_tools is None else dry_run_tools,
            "seed_candidates": list(seed_candidates or []),
            "kg_rag_output": kg_rag_output,
            "textbook_rag_output": textbook_rag_output,
            "web_rag_output": web_rag_output,
        }
        result = self.graph.invoke(initial)
        if "cand_list" not in result:
            rerank_output = result.get("rerank_output") or {}
            if isinstance(rerank_output, dict):
                result["cand_list"] = rerank_output.get("candidates", [])
            else:
                result["cand_list"] = []
        result["trace_log_path"] = self.trace_log_path
        self._auto_remember_result(result)
        return result


MultiAgentNMR = MultiAgentNMRV2
