#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

REAL_CASE = {
    "formula": "C20H30O3",
    "c_shifts": [210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51],
    "h_shifts": [5.68, 3.085, 3.085, 2.68, 2.59, 1.86, 1.86, 1.77, 1.63, 1.63, 1.5, 1.5, 1.5, 1.275, 1.20, 1.20, 1.20, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.02, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94],
    "h_nmr_text": "1H NMR (400 MHz, Chloroform-d) δ 5.68 (s, 1H), 3.19 - 2.98 (m, 2H), 2.68 (p, J = 6.9 Hz, 1H), 2.59 (dt, J = 9.1, 2.4 Hz, 1H), 1.86 (dt, J = 13.7, 4.1 Hz, 2H), 1.77 (tt, J = 14.5, 3.9 Hz, 1H), 1.69 - 1.57 (m, 2H), 1.56 - 1.44 (m, 3H), 1.32 - 1.23 (m, 1H), 1.20 (s, 3H), 1.11 (dd, J = 6.9, 2.9 Hz, 6H), 1.02 (td, J = 8.1, 3.8 Hz, 1H), 0.94 (d, J = 3.7 Hz, 6H).",
    "c_nmr_text": "13C NMR (101 MHz, CDCl3) δ 210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51.",
    "note": "1H 400 MHz, CDCl3; 13C 101 MHz, CDCl3",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _empty_state() -> Dict[str, Any]:
    return {"formula": "", "h_shifts": [], "c_shifts": [], "note": ""}


def _numbers(text: Any) -> List[float]:
    if isinstance(text, list):
        return [float(x) for x in text]
    vals: List[float] = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", str(text or "")):
        vals.append(float(token))
    return vals


def _parse_h_input(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    text = str(value or "").strip()
    if not text:
        return []
    if "δ" in text or re.search(r"\b\d+(?:\.\d+)?\s*H\b", text, flags=re.I):
        sys.path.insert(0, str(_repo_root()))
        from tools.nmr_text_parser import parse_h_nmr_text

        parsed = parse_h_nmr_text(text)
        return parsed or _numbers(text)
    return _numbers(text)


def _parse_c_input(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    text = str(value or "").strip()
    if not text:
        return []
    if "δ" in text or re.search(r"\b13C\b|NMR|MHz|CDCl3|Chloroform", text, flags=re.I):
        sys.path.insert(0, str(_repo_root()))
        from tools.nmr_text_parser import parse_c_nmr_text

        parsed = parse_c_nmr_text(text)
        return parsed or _numbers(text)
    return _numbers(text)


def _state(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "formula": str(payload.get("formula") or "").strip(),
        "h_shifts": _parse_h_input(payload.get("h_shifts")),
        "c_shifts": _parse_c_input(payload.get("c_shifts")),
        "note": str(payload.get("note") or "").strip(),
    }


def _new_session() -> Dict[str, Any]:
    return {"id": uuid.uuid4().hex, "state": _empty_state(), "messages": [], "last_result": None}



def _merge_payload_state(state: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    changed: List[str] = []
    if "formula" in payload and str(payload.get("formula") or "").strip():
        state["formula"] = str(payload["formula"]).strip()
        changed.append("formula")
    if "h_shifts" in payload and payload.get("h_shifts") not in (None, ""):
        state["h_shifts"] = _parse_h_input(payload.get("h_shifts"))
        changed.append("1H")
    if "c_shifts" in payload and payload.get("c_shifts") not in (None, ""):
        state["c_shifts"] = _parse_c_input(payload.get("c_shifts"))
        changed.append("13C")
    if "note" in payload and str(payload.get("note") or "").strip():
        state["note"] = str(payload["note"]).strip()
        changed.append("note")
    return changed


def _extract_formula(text: str) -> str:
    labeled = re.search(r"(?:formula|分子式)\s*[:=：]\s*([A-Z][A-Za-z0-9]*)", text, flags=re.I)
    if labeled:
        return labeled.group(1).strip()
    match = re.search(r"\bC\d+(?:H\d+)?(?:[A-Z][a-z]?\d*)*\b", text)
    return match.group(0) if match else ""


def _parse_message_into_state(text: str, state: Dict[str, Any]) -> List[str]:
    changed: List[str] = []
    formula = _extract_formula(text)
    if formula:
        state["formula"] = formula
        changed.append("formula")

    h_chunks: List[str] = []
    c_chunks: List[str] = []
    note_chunks: List[str] = []
    for raw_line in re.split(r"[\r\n]+", text):
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if re.search(r"\b13\s*c\b|13c|carbon", lower):
            c_chunks.append(line)
        elif re.search(r"\b1\s*h\b|1h|proton", lower):
            h_chunks.append(line)
        elif any(key in lower for key in ["note", "solvent", "cdcl", "dmso", "mhz", "备注"]):
            note_chunks.append(line)

    lower_all = text.lower()
    if not c_chunks and re.search(r"\b13\s*c\b|13c", lower_all):
        c_chunks.append(text)
    if not h_chunks and re.search(r"\b1\s*h\b|1h", lower_all):
        h_chunks.append(text)

    if c_chunks:
        state["c_shifts"] = _parse_c_input("\n".join(c_chunks))
        changed.append("13C")
    if h_chunks:
        state["h_shifts"] = _parse_h_input("\n".join(h_chunks))
        changed.append("1H")
    if note_chunks:
        state["note"] = "\n".join(note_chunks)
        changed.append("note")
    return list(dict.fromkeys(changed))


def _missing_required(state: Dict[str, Any]) -> List[str]:
    missing = []
    if not state.get("formula"):
        missing.append("formula")
    if not state.get("c_shifts"):
        missing.append("13C NMR shifts")
    return missing


def _should_preview_rag(message: str) -> bool:
    lower = message.lower()
    return "rag" in lower or "evidence" in lower or "检索" in lower or "证据" in lower


def _should_solve(message: str) -> bool:
    lower = message.lower()
    words = ["solve", "run", "agent", "elucidate", "structure", "answer", "解析", "求解", "推理", "结构", "跑", "开始"]
    return any(word in lower for word in words)


def _is_greeting(message: str) -> bool:
    lower = message.strip().lower()
    greetings = {"hi", "hello", "hey", "你好", "您好", "嗨", "哈喽", "hello!", "hi!"}
    return lower in greetings


def _chat_help_text() -> str:
    return "你好。我可以按多轮对话累积 NMR case。你可以直接贴 Formula、13C NMR、1H NMR；也可以发 demo、rag 或 solve。"


def _rag_preview(state: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))
    query = f"{state.get('formula', '')} natural product NMR structure elucidation ketone lactone ester oxygenated carbon alkene"
    out: Dict[str, Any] = {}
    if opts.get("textbook_rag", True):
        from tools.textbook_rag_tool import textbook_nmr_search_impl

        out["textbook_nmr_search"] = textbook_nmr_search_impl(query=query, formula=state.get("formula", ""), h_shifts=" ".join(map(str, state.get("h_shifts", []))), c_shifts=" ".join(map(str, state.get("c_shifts", []))), top_k=int(opts.get("textbook_rag_top_k", 5)))
    if opts.get("web_rag"):
        from tools.web_rag_tool import web_nmr_search_impl

        out["web_nmr_search"] = web_nmr_search_impl(query=query, formula=state.get("formula", ""), h_shifts=" ".join(map(str, state.get("h_shifts", []))), c_shifts=" ".join(map(str, state.get("c_shifts", []))), top_k=int(opts.get("web_rag_top_k", 5)))
    if opts.get("kg_rag", True):
        try:
            from tools.kg_rag_tool import kg_graph_rag_search_impl

            out["kg_graph_rag_search"] = kg_graph_rag_search_impl(query=query, formula=state.get("formula", ""), top_k=int(opts.get("kg_rag_top_k", 5)), neighbor_limit=int(opts.get("kg_rag_neighbor_limit", 0)))
        except Exception as exc:
            out["kg_graph_rag_search"] = {"valid": 0, "count": 0, "error": str(exc), "evidence_pack": []}
    return out


def _summarize_rag(raw: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, result in raw.items():
        evidence = result.get("evidence_pack", []) if isinstance(result, dict) else []
        summary[name] = {
            "valid": result.get("valid") if isinstance(result, dict) else 0,
            "count": result.get("count", len(evidence)) if isinstance(result, dict) else 0,
            "first_claim": evidence[0].get("claim") if evidence else None,
            "observation": result.get("observation", "")[:1200] if isinstance(result, dict) else "",
            "error": result.get("error") if isinstance(result, dict) else None,
            "evidence_pack": evidence[:5],
        }
    return summary


def _build_agent(payload: Dict[str, Any]):
    sys.path.insert(0, str(_repo_root()))
    from agents.multi_agent_nmr import MultiAgentNMR
    from configs.multi_agent_config import load_multi_agent_config

    config = load_multi_agent_config(str(payload.get("config") or ""))
    overrides: Dict[str, Any] = {"trace_log_path": "", "dry_run_tools": bool(payload.get("dry_run_tools", False))}
    for src, dst in [("model", "model"), ("api_key", "api_key"), ("base_url", "base_url")]:
        if payload.get(src):
            overrides[dst] = str(payload[src]).strip()
    if payload.get("max_turns"):
        overrides["max_iterations"] = int(payload["max_turns"])
    if payload.get("temperature") is not None:
        overrides["temperature"] = float(payload.get("temperature") or 0.0)
    if payload.get("max_tokens"):
        overrides["max_tokens"] = int(payload["max_tokens"])
    if payload.get("textbook_rag", True):
        overrides["enable_textbook_rag"] = True
        overrides["textbook_rag_top_k"] = int(payload.get("textbook_rag_top_k", 5))
    if payload.get("kg_rag", True):
        overrides["enable_kg_rag"] = True
        overrides["kg_rag_top_k"] = int(payload.get("kg_rag_top_k", 5))
        overrides["kg_rag_neighbor_limit"] = int(payload.get("kg_rag_neighbor_limit", 0))
    if payload.get("web_rag"):
        overrides["enable_web_rag"] = True
        overrides["web_rag_top_k"] = int(payload.get("web_rag_top_k", 5))
    if payload.get("use_service_tools"):
        overrides["executor_tools"] = ["nmr_retrieve_service", "nmr_denovo_service", "nmr_merge_pools", "nmr_optimize"]
    return MultiAgentNMR.from_config(config, **overrides)



def _normalize_chat_base_url(base_url: str) -> str:
    value = (base_url or "").rstrip("/")
    if not value:
        return value
    return value if value.endswith("/v1") else value + "/v1"


def _chat_backend_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))
    from configs.multi_agent_config import load_multi_agent_config

    config = load_multi_agent_config(str(payload.get("config") or ""))
    openai_cfg = dict(config.get("openai_compatible", {}) or {})
    api_key = str(payload.get("api_key") or "").strip()
    if not api_key:
        api_key = openai_cfg.get("api_key") or os.environ.get(openai_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
    base_url = str(payload.get("base_url") or "").strip() or openai_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
    model = str(payload.get("model") or "").strip() or openai_cfg.get("model", "gpt-5.4")
    return {
        "api_key": api_key,
        "base_url": _normalize_chat_base_url(base_url),
        "model": model,
        "temperature": float(payload.get("temperature") if payload.get("temperature") is not None else openai_cfg.get("temperature", 0.2)),
        "max_tokens": int(payload.get("max_tokens") or min(int(openai_cfg.get("max_tokens", 2048)), 2048)),
    }


def _llm_chat_response(state: ChatState, case_state: Dict[str, Any], payload: Dict[str, Any], changed: List[str]) -> tuple[str, Dict[str, Any]]:
    settings = _chat_backend_settings(payload)
    if not settings["api_key"]:
        return "OpenAI API key is not configured. Fill API Key in the left panel or set OPENAI_API_KEY, then send the message again.", {"chat_error": "missing_api_key"}

    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": settings["api_key"]}
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    client = OpenAI(**client_kwargs)

    case_summary = {
        "formula": case_state.get("formula", ""),
        "h_shift_count": len(case_state.get("h_shifts", []) or []),
        "c_shift_count": len(case_state.get("c_shifts", []) or []),
        "note": case_state.get("note", ""),
        "changed_this_turn": changed,
    }
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are NMRAgent, a multi-turn chat assistant for NMR structure elucidation. "
                "Answer ordinary user messages naturally. When the user provides NMR case data, acknowledge what was captured and tell them they can ask for rag or solve. "
                "Do not run structure solving in this free-chat response; solving is handled by the explicit solve command. "
                "Keep answers concise and useful. You can speak Chinese or English matching the user."
            ),
        },
        {"role": "system", "content": "Current accumulated NMR case state: " + json.dumps(case_summary, ensure_ascii=False, default=str)},
    ]
    for msg in state.get("messages", [])[-12:]:
        role = _message_role(msg)
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": _message_content(msg)})

    started = time.time()
    try:
        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )
        content = response.choices[0].message.content or ""
        meta = {
            "chat_model": settings["model"],
            "chat_base_url": settings["base_url"],
            "chat_elapsed_sec": round(time.time() - started, 3),
            "finish_reason": response.choices[0].finish_reason,
        }
        return content.strip() or "I did not receive a text response from the chat model.", meta
    except Exception as exc:
        return f"Chat model call failed: {exc}", {"chat_error": str(exc), "chat_model": settings["model"], "chat_base_url": settings["base_url"]}


def _trace_tail(path: str, limit: int = 80) -> List[Dict[str, Any]]:
    trace_path = Path(path or "")
    if not trace_path.exists():
        return []
    rows = []
    for line in trace_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"raw": line})
    return rows


def _process_view(result: Dict[str, Any]) -> Dict[str, Any]:
    trace_tail = _trace_tail(result.get("trace_log_path", ""))
    tool_calls = [row for row in trace_tail if row.get("event") in {"tool_start", "tool_end"}]
    return {
        "planner": (result.get("planner_output") or {}).get("analysis"),
        "planner_output": result.get("planner_output") or {},
        "executor_output": result.get("executor_output") or {},
        "executor_actions": (result.get("executor_output") or {}).get("actions_taken"),
        "verifier": (result.get("verifier_output") or {}).get("analysis"),
        "verifier_output": result.get("verifier_output") or {},
        "verdict": (result.get("verifier_output") or {}).get("verdict"),
        "tool_calls": tool_calls,
        "trace_log_path": result.get("trace_log_path"),
        "trace_tail": trace_tail,
    }



class ChatState(MessagesState, total=False):
    case_state: Dict[str, Any]
    options: Dict[str, Any]
    last_result: Dict[str, Any] | None
    last_meta: Dict[str, Any]
    changed: List[str]


def _message_role(message: BaseMessage) -> str:
    role = getattr(message, "type", "") or message.__class__.__name__.lower()
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return role or "message"


def _message_content(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _messages_for_client(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages[-80:]:
        meta = dict(getattr(message, "additional_kwargs", {}) or {}).get("meta")
        row = {"id": getattr(message, "id", "") or uuid.uuid4().hex[:12], "role": _message_role(message), "content": _message_content(message)}
        if meta:
            row["meta"] = meta
        out.append(row)
    return out


def _last_human_message(messages: List[BaseMessage]) -> str:
    for message in reversed(messages or []):
        if _message_role(message) == "user":
            return _message_content(message).strip()
    return ""


def _conversation_task_from_graph(state: ChatState, message: str) -> str:
    messages = state.get("messages", [])[-8:]
    transcript = "\n".join(f"{_message_role(m)}: {_message_content(m)}" for m in messages)
    task = (
        "Solve this NMR structure elucidation task as a multi-turn LangGraph chat agent. "
        "Use the accumulated case state, the latest user request, RAG evidence if enabled, "
        "formula-constrained candidate generation, and verifier rerank evidence. "
        f"Latest user request: {message.strip()}"
    )
    if transcript:
        task += f"\nRecent conversation in the same LangGraph thread:\n{transcript}"
    return task


def _run_solver_graph(state: ChatState, payload: Dict[str, Any], message: str = "") -> tuple[Dict[str, Any], Dict[str, Any]]:
    case_state = state.get("case_state", _empty_state())
    agent = _build_agent(payload)
    task = _conversation_task_from_graph(state, message or "solve")
    if case_state.get("note"):
        task += f"\nExperimental note: {case_state['note']}"
    result = agent.run(
        task=task,
        formula=case_state["formula"],
        h_shifts=case_state.get("h_shifts", []),
        c_shifts=case_state.get("c_shifts", []),
        max_iterations=int(payload.get("max_turns") or 1),
        dry_run_tools=bool(payload.get("dry_run_tools", False)),
    )
    return result, _process_view(result)


def _chat_node(state: ChatState) -> Dict[str, Any]:
    payload = dict(state.get("options") or {})
    case_state = dict(state.get("case_state") or _empty_state())
    message = _last_human_message(state.get("messages", []))
    started = time.time()

    changed = _merge_payload_state(case_state, payload)
    changed.extend(_parse_message_into_state(message, case_state))
    changed = list(dict.fromkeys(changed))

    lower = message.lower()
    meta: Dict[str, Any] = {}
    last_result = state.get("last_result")
    if lower in {"/reset", "reset", "清空", "重置"}:
        case_state = _empty_state()
        last_result = None
        assistant_text = "Session reset."
    elif "demo" in lower or "示例" in lower:
        case_state = {"formula": REAL_CASE["formula"], "h_shifts": list(REAL_CASE["h_shifts"]), "c_shifts": list(REAL_CASE["c_shifts"]), "note": REAL_CASE["note"]}
        assistant_text = _state_update_text(case_state, ["demo"])
    elif lower in {"/show", "show", "state", "状态"}:
        assistant_text = _state_update_text(case_state, [])
    elif _should_preview_rag(message):
        missing = _missing_required(case_state)
        if missing:
            assistant_text = "Missing required input: " + ", ".join(missing) + "."
        else:
            raw = _rag_preview(case_state, payload)
            meta["rag"] = _summarize_rag(raw)
            assistant_text = "RAG preview complete."
    elif _should_solve(message):
        missing = _missing_required(case_state)
        if missing:
            assistant_text = "Missing required input: " + ", ".join(missing) + "."
        else:
            graph_state: ChatState = dict(state)
            graph_state["case_state"] = case_state
            result, process = _run_solver_graph(graph_state, payload, message)
            meta.update({"result": result, "process": process})
            last_result = {"result": result, "process": process}
            assistant_text = _assistant_result_text(result, process, time.time() - started)
    else:
        assistant_text, chat_meta = _llm_chat_response(state, case_state, payload, changed)
        meta.update(chat_meta)

    return {
        "case_state": case_state,
        "changed": changed,
        "last_meta": meta,
        "last_result": last_result,
        "messages": [AIMessage(content=assistant_text, additional_kwargs={"meta": meta} if meta else {})],
    }


def _build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat", _chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=MemorySaver())


def _public_graph_session(chat_graph: Any, thread_id: str) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    values = chat_graph.get_state(config).values or {}
    return {
        "id": thread_id,
        "state": values.get("case_state") or _empty_state(),
        "messages": _messages_for_client(values.get("messages", [])),
        "last_result": values.get("last_result"),
    }


def _legacy_conversation_task(session: Dict[str, Any], message: str) -> str:
    recent = session.get("messages", [])[-8:]
    transcript = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in recent)
    task = (
        "Solve this NMR structure elucidation task as a multi-turn chat agent. "
        "Use the accumulated case state, the latest user request, RAG evidence if enabled, "
        "formula-constrained candidate generation, and verifier rerank evidence. "
        f"Latest user request: {message.strip()}"
    )
    if transcript:
        task += f"\nRecent conversation:\n{transcript}"
    return task


def _run_solver(session: Dict[str, Any], payload: Dict[str, Any], message: str = "") -> tuple[Dict[str, Any], Dict[str, Any]]:
    state = session.get("state", _empty_state())
    agent = _build_agent(payload)
    task = _legacy_conversation_task(session, message or "solve")
    if state.get("note"):
        task += f"\nExperimental note: {state['note']}"
    result = agent.run(
        task=task,
        formula=state["formula"],
        h_shifts=state.get("h_shifts", []),
        c_shifts=state.get("c_shifts", []),
        max_iterations=int(payload.get("max_turns") or 1),
        dry_run_tools=bool(payload.get("dry_run_tools", False)),
    )
    process = _process_view(result)
    session["last_result"] = {"result": result, "process": process}
    return result, process


def _assistant_result_text(result: Dict[str, Any], process: Dict[str, Any], elapsed: float) -> str:
    verifier = result.get("verifier_output") or {}
    executor = result.get("executor_output") or {}
    rerank = result.get("rerank_output") or {}
    verdict = verifier.get("verdict") or "unknown"
    final_answer = result.get("final_answer") or verifier.get("top_candidate")
    lines = [f"Agent run finished in {elapsed:.1f}s. Verdict: {verdict}."]
    if final_answer:
        lines.append(f"Top structure: {final_answer}")
    elif rerank.get("candidates"):
        top = rerank["candidates"][0]
        lines.append(f"Top reranked candidate: {top.get('smiles', top)}")
    else:
        lines.append("No accepted final structure yet.")
    actions = executor.get("actions_taken") or []
    if actions:
        lines.append("Tools: " + ", ".join(str(x) for x in actions))
    if executor.get("candidate_count") is not None:
        lines.append(f"Candidate count: {executor.get('candidate_count')}")
    if verifier.get("analysis"):
        lines.append("Verifier summary: " + str(verifier.get("analysis"))[:900])
    if process.get("trace_log_path"):
        lines.append(f"Trace: {process['trace_log_path']}")
    return "\n".join(lines)


def _state_update_text(state: Dict[str, Any], changed: List[str]) -> str:
    parts = []
    if state.get("formula"):
        parts.append(f"formula {state['formula']}")
    if state.get("c_shifts"):
        parts.append(f"{len(state['c_shifts'])} 13C shifts")
    if state.get("h_shifts"):
        parts.append(f"{len(state['h_shifts'])} 1H shifts")
    if state.get("note"):
        parts.append("experimental note")
    if changed:
        return "Captured " + ", ".join(parts or changed) + "."
    return "Current case: " + (", ".join(parts) if parts else "empty.")


def create_app() -> Flask:
    app = Flask(__name__)
    chat_graph = _build_chat_graph()

    @app.errorhandler(Exception)
    def handle_exception(exc: Exception):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": str(exc), "type": exc.__class__.__name__}), 500
        raise exc

    @app.get("/")
    def index():
        return HTML

    @app.post("/api/session")
    def api_session():
        payload = request.get_json(silent=True) or {}
        thread_id = str(payload.get("session_id") or uuid.uuid4().hex)
        return jsonify({"ok": True, "session": _public_graph_session(chat_graph, thread_id)})

    @app.get("/api/demo-case")
    def demo_case():
        return jsonify(REAL_CASE)

    @app.post("/api/rag")
    def api_rag():
        payload = request.get_json(silent=True) or {}
        state = _state(payload)
        started = time.time()
        raw = _rag_preview(state, payload)
        return jsonify({"ok": True, "elapsed_sec": round(time.time() - started, 3), "state": state, "rag": _summarize_rag(raw)})

    @app.post("/api/solve")
    def api_solve():
        payload = request.get_json(silent=True) or {}
        session = _new_session()
        session["state"] = _state(payload)
        state = session["state"]
        if not state["formula"] or not state["c_shifts"]:
            return jsonify({"ok": False, "error": "Formula and 13C NMR shifts are required."}), 400
        started = time.time()
        result, process = _run_solver(session, payload, "solve")
        return jsonify({"ok": True, "elapsed_sec": round(time.time() - started, 3), "state": state, "process": process, "result": result})

    @app.post("/api/chat")
    def api_chat():
        payload = request.get_json(silent=True) or {}
        thread_id = str(payload.get("session_id") or uuid.uuid4().hex)
        message = str(payload.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Message is empty."}), 400
        started = time.time()
        config = {"configurable": {"thread_id": thread_id}}
        result_state = chat_graph.invoke(
            {"messages": [HumanMessage(content=message)], "options": payload},
            config=config,
        )
        session = _public_graph_session(chat_graph, thread_id)
        meta = result_state.get("last_meta") or {}
        return jsonify({
            "ok": True,
            "elapsed_sec": round(time.time() - started, 3),
            "session": session,
            "state": session["state"],
            "changed": result_state.get("changed", []),
            "meta": meta,
        })

    return app

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NMRAgent Chat</title>
<style>
:root{--ink:#17202c;--muted:#667085;--line:#d7dee8;--panel:#ffffff;--bg:#eef2f5;--accent:#0f6b68;--accent2:#6f5b24;--danger:#a33b3b;--code:#101722}*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}header{height:58px;padding:0 20px;border-bottom:1px solid var(--line);background:#fff;display:flex;align-items:center;justify-content:space-between;gap:16px}h1{font-size:19px;margin:0;letter-spacing:0}.status{font-size:13px;color:var(--muted)}main{display:grid;grid-template-columns:360px minmax(0,1fr);height:calc(100vh - 58px)}aside{border-right:1px solid var(--line);background:#fff;overflow:auto}.settings{padding:14px;display:grid;gap:12px}.group{display:grid;gap:9px;padding-bottom:12px;border-bottom:1px solid var(--line)}.group:last-child{border-bottom:0}label{display:grid;gap:5px;font-size:12px;color:var(--muted)}input,textarea{width:100%;border:1px solid var(--line);border-radius:6px;padding:9px 10px;font:inherit;color:var(--ink);background:#fff}textarea{min-height:90px;resize:vertical;line-height:1.35}.row{display:grid;grid-template-columns:1fr 1fr;gap:9px}.checks{display:grid;grid-template-columns:1fr 1fr;gap:8px}.check{display:flex;align-items:center;gap:7px;min-height:34px;padding:7px 8px;border:1px solid var(--line);border-radius:6px;color:var(--ink);font-size:13px}.check input{width:auto}.chat{display:grid;grid-template-rows:1fr auto;min-width:0}.messages{overflow:auto;padding:18px;display:flex;flex-direction:column;gap:12px}.msg{max-width:min(820px,92%);border:1px solid var(--line);border-radius:8px;padding:11px 12px;background:#fff;white-space:pre-wrap;line-height:1.45}.msg.user{align-self:flex-end;background:#e7f3f2;border-color:#bddbd8}.msg.assistant{align-self:flex-start}.msg .meta{margin-top:8px;color:var(--muted);font-size:12px}.composer{border-top:1px solid var(--line);background:#fff;padding:12px;display:grid;grid-template-columns:1fr auto;gap:10px}.composer textarea{min-height:58px;max-height:180px}.buttons{display:flex;gap:8px;align-items:end;flex-wrap:wrap}button{border:0;border-radius:6px;padding:10px 12px;color:#fff;background:var(--accent);font-weight:650;cursor:pointer;min-height:38px}button.secondary{background:var(--accent2)}button.ghost{background:#546171}button.danger{background:var(--danger)}button:disabled{opacity:.58;cursor:not-allowed}.state{font-size:12px;color:var(--muted);line-height:1.45}.details{display:grid;gap:10px;padding:0 18px 18px}.panel{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.panel summary{cursor:pointer;padding:10px 12px;font-weight:650}.panel pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.45;background:var(--code);color:#e8eef7;padding:12px;max-height:260px;overflow:auto}.pill{display:inline-flex;align-items:center;min-height:22px;padding:2px 7px;border-radius:999px;background:#e7f3f2;color:#0d5967;font-size:12px;margin-right:5px}@media(max-width:920px){main{grid-template-columns:1fr;height:auto;min-height:calc(100vh - 58px)}aside{border-right:0;border-bottom:1px solid var(--line)}.messages{min-height:55vh}.composer{grid-template-columns:1fr}.buttons{justify-content:flex-end}}
</style>
</head>
<body>
<header><h1>NMRAgent Chat</h1><div class="status" id="status">idle</div></header>
<main>
  <aside>
    <div class="settings">
      <div class="group">
        <div class="row"><label>Base URL<input id="base_url" placeholder="OpenAI-compatible endpoint"></label><label>API Key<input id="api_key" type="password" placeholder="sk-..."></label></div>
        <div class="row"><label>Model<input id="model" placeholder="gpt-4o-mini / gpt-5.4"></label><label>Max turns<input id="max_turns" type="number" min="1" value="1"></label></div>
      </div>
      <div class="group">
        <div class="checks"><label class="check"><input id="textbook_rag" type="checkbox" checked>Textbook</label><label class="check"><input id="kg_rag" type="checkbox" checked>Graph</label><label class="check"><input id="web_rag" type="checkbox">Web</label><label class="check"><input id="dry_run_tools" type="checkbox">Dry run</label></div>
        <label class="check"><input id="use_service_tools" type="checkbox">Use retrieval/denovo services</label>
      </div>
      <div class="group">
        <div class="buttons"><button class="ghost" id="demoBtn">Demo</button><button class="secondary" id="ragBtn">RAG</button><button class="danger" id="resetBtn">Reset</button></div>
        <div class="state" id="caseState"></div>
      </div>
    </div>
  </aside>
  <section class="chat">
    <div>
      <div class="messages" id="messages"></div>
      <div class="details">
        <details class="panel" id="processPanel"><summary>Reasoning / trace summary</summary><pre id="process"></pre></details>
        <details class="panel" id="rawPanel"><summary>Raw result</summary><pre id="raw"></pre></details>
      </div>
    </div>
    <div class="composer"><textarea id="message" placeholder="Paste NMR data or ask the agent"></textarea><div class="buttons"><button id="sendBtn">Send</button><button class="secondary" id="solveBtn">Solve</button></div></div>
  </section>
</main>
<script>
let sessionId = localStorage.getItem('nmragent_session_id') || '';
const messagesEl = document.getElementById('messages');
const statusEl = document.getElementById('status');
const processEl = document.getElementById('process');
const rawEl = document.getElementById('raw');
const stateEl = document.getElementById('caseState');
const messageEl = document.getElementById('message');

function v(id){return document.getElementById(id).value}
function ck(id){return document.getElementById(id).checked}
function opts(){return{session_id:sessionId,base_url:v('base_url'),api_key:v('api_key'),model:v('model'),max_turns:Number(v('max_turns')||1),textbook_rag:ck('textbook_rag'),kg_rag:ck('kg_rag'),web_rag:ck('web_rag'),dry_run_tools:ck('dry_run_tools'),use_service_tools:ck('use_service_tools'),kg_rag_neighbor_limit:0}}
function setStatus(text){statusEl.textContent=text}
function show(obj,el){el.textContent=typeof obj==='string'?obj:JSON.stringify(obj,null,2)}
function esc(text){return String(text||'').replace(/[&<>"']/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]))}
function renderMessages(list){messagesEl.innerHTML='';(list||[]).forEach(m=>{const div=document.createElement('div');div.className='msg '+m.role;div.innerHTML=esc(m.content)+(m.meta?'<div class="meta">'+esc(metaLine(m.meta))+'</div>':'');messagesEl.appendChild(div)});messagesEl.scrollTop=messagesEl.scrollHeight}
function metaLine(meta){const bits=[];if(meta.process?.trace_log_path)bits.push('trace '+meta.process.trace_log_path);if(meta.rag)bits.push('rag sources '+Object.keys(meta.rag).length);return bits.join(' | ')}
function renderState(state){const parts=[];if(state?.formula)parts.push('<span class="pill">'+esc(state.formula)+'</span>');if(state?.c_shifts?.length)parts.push('<span class="pill">13C '+state.c_shifts.length+'</span>');if(state?.h_shifts?.length)parts.push('<span class="pill">1H '+state.h_shifts.length+'</span>');if(state?.note)parts.push('<span class="pill">note</span>');stateEl.innerHTML=parts.join(' ')||'No case loaded';}
function renderSession(session){if(!session)return;sessionId=session.id;localStorage.setItem('nmragent_session_id',sessionId);renderMessages(session.messages);renderState(session.state);if(session.last_result){show(session.last_result.process,processEl);show(session.last_result.result,rawEl)}}
async function apiPost(url,body){const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','Accept':'application/json'},body:JSON.stringify(body)});const ct=res.headers.get('content-type')||'';if(!ct.includes('application/json')){const text=await res.text();throw new Error('Expected JSON from '+url+', got HTTP '+res.status+': '+text.slice(0,160));}const data=await res.json();if(!res.ok||data.ok===false)throw new Error(data.error||res.statusText);return data}
async function init(){try{const data=await apiPost('/api/session',{session_id:sessionId});renderSession(data.session)}catch(e){setStatus('error: '+e.message)}}
async function send(text){const message=(text===undefined?messageEl.value:text).trim();if(!message)return;messageEl.value='';setStatus('running');disable(true);try{const data=await apiPost('/api/chat',{...opts(),message});renderSession(data.session);if(data.meta?.process)show(data.meta.process,processEl);if(data.meta?.result)show(data.meta.result,rawEl);else if(data.meta?.rag){show(data.meta.rag,processEl);show(data.meta.rag,rawEl)}setStatus('done '+data.elapsed_sec+'s')}catch(e){setStatus('error');appendLocal('assistant',e.message)}finally{disable(false)}}
function appendLocal(role,content){const div=document.createElement('div');div.className='msg '+role;div.textContent=content;messagesEl.appendChild(div);messagesEl.scrollTop=messagesEl.scrollHeight}
function disable(flag){['sendBtn','solveBtn','demoBtn','ragBtn','resetBtn'].forEach(id=>document.getElementById(id).disabled=flag)}
document.getElementById('sendBtn').addEventListener('click',()=>send());
document.getElementById('solveBtn').addEventListener('click',()=>send('solve'));
document.getElementById('demoBtn').addEventListener('click',()=>send('demo'));
document.getElementById('ragBtn').addEventListener('click',()=>send('rag'));
document.getElementById('resetBtn').addEventListener('click',()=>send('reset'));
messageEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();send()}});
init();
</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NMRAgent web chat frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
