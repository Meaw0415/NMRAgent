"""
LangGraph-based NMR Agent with support for local models and OpenAI.

For OpenAI-compatible backends, this file now prefers the official LangGraph
tool-calling pattern:
- ChatOpenAI.bind_tools(...)
- ToolNode(tools)
- tools_condition

Local vLLM / transformers backends still use the legacy text-ReAct path.
"""

import json
import inspect
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Sequence
import operator
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State for the NMR Agent."""
    messages: Annotated[Sequence[dict], operator.add]
    task: str
    formula: Optional[str]
    gt_smiles: Optional[str]
    h_shifts: Optional[List[float]]
    c_shifts: Optional[List[float]]
    sample_idx: Optional[int]
    iteration: int
    max_iterations: int
    final_answer: Optional[str]
    tool_results: List[Dict]
    kg_context: Optional[str]
    kg_injected: bool


# ──────────────────────────────────────────────────────────────────────────────
# ReAct Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_react_step(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a single ReAct-style step.

    Supports:
    1. Classic: Thought: ... / Action: ... / Input: {...}
    2. Qwen3 native: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    """
    result = {"thought": None, "action": None, "input": None}

    # Strip <think>...</think>
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Branch 1: <tool_call> format
    tc = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", clean, re.DOTALL)
    if tc:
        raw = tc.group(1).strip()
        try:
            parsed = json.loads(raw)
            result["action"] = parsed.get("name")
            args = parsed.get("arguments", {})
            result["input"] = json.dumps(args) if isinstance(args, dict) else str(args)
        except Exception:
            result["input"] = raw
        before = clean[:tc.start()].strip()
        tm = re.search(r"Thought:\s*(.*)", before, re.IGNORECASE | re.DOTALL)
        if tm:
            result["thought"] = tm.group(1).strip()
        elif before:
            result["thought"] = before
        return result

    # Branch 2: Classic ReAct
    tm = re.search(r"Thought:\s*(.*?)(?=\s*(?:Action:|Input:|$))", clean, re.IGNORECASE | re.DOTALL)
    if tm:
        result["thought"] = tm.group(1).strip()

    am = re.search(r"Action:\s*(.*?)(?=\s*(?:Thought:|Input:|$))", clean, re.IGNORECASE | re.DOTALL)
    if am:
        result["action"] = am.group(1).strip()

    im = re.search(r"Input:\s*(.*?)(?=\s*(?:Thought:|Action:|$))", clean, re.IGNORECASE | re.DOTALL)
    if im:
        result["input"] = im.group(1).strip()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an expert NMR structure elucidation specialist.
You are given a molecular formula together with the corresponding 1H NMR and 13C NMR chemical shifts.
Your goal is to determine the single most likely molecular structure consistent with the formula and spectra.

When you receive a query, follow this ReAct process:

1. Think briefly about the next best step.
   - Prefix each reasoning step with `Thought:`.
2. If needed, call exactly one available tool.
   - Prefix with `Action:` and the exact tool name.
   - Prefix with `Input:` and a valid JSON object for the tool arguments.
3. Read the tool result carefully.
   - The result appears as `Observation: ...`.
4. Repeat until you have enough evidence to identify the best structure.
5. When finished, output the final answer exactly as:
   `Answer: <SMILES>`

Guidelines:
- Treat the molecular formula as a hard constraint.
- Use both 1H and 13C NMR signals as primary evidence.
- Prefer the single best-supported structure over a list of possibilities.
- Use retrieval, de novo generation, optimization, and reranking tools only when they help narrow the answer.
- Keep tool calls valid, minimal, and evidence-driven.

{tool_descriptions}

{task_info}
"""

STRUCTURED_SYSTEM_PROMPT = """You are an expert NMR structure elucidation specialist.
You are given a molecular formula together with the corresponding 1H NMR and 13C NMR chemical shifts.
Your goal is to determine the single most likely molecular structure consistent with the formula and spectra.

Rules:
- Treat the molecular formula as a hard constraint.
- Use both 1H and 13C NMR signals as primary evidence.
- Use the provided tools when needed to retrieve, generate, optimize, or rerank candidates.
- Retrieval results are already ranked by similarity and may be sufficient without immediate reranking.
- Prefer the single best-supported final structure.
- If you need tools, call them using the provided tool-calling interface.
- When you are ready to finish, answer with plain text in this exact format:
  Answer: <SMILES>

{tool_descriptions}

{task_info}
"""


# ──────────────────────────────────────────────────────────────────────────────
# NMR Agent
# ──────────────────────────────────────────────────────────────────────────────

class NMRAgent:
    """
    LangGraph-based NMR Agent for molecular structure elucidation.

    Supports:
    - Local models (via vLLM, transformers)
    - OpenAI-compatible models (via API)
    - Knowledge Graph RAG pre-injection
    - Retrieval, De Novo, Reranking, Fragment, Mol Edit tools
    - Online and Offline modes
    """

    def __init__(
        self,
        model_name_or_path: str,
        tools: List,
        backend: str = "openai",
        task_info: str = None,
        force_kg_rag: bool = False,
        max_iterations: int = 6,
        **backend_kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.tools = tools
        self.backend = backend
        self.force_kg_rag = force_kg_rag
        self.max_iterations = max_iterations
        self.backend_kwargs = backend_kwargs
        self.trace_log_path = self._init_trace_log_path()
        self._trace_context: Dict[str, Any] = {}

        # Build tool name map
        self.tool_map = {}
        for t in tools:
            name = getattr(t, "name", t.__name__)
            self.tool_map[name] = t

        # Task info for system prompt
        self.task_info = task_info or self._default_task_info()

        # Build tool descriptions
        self.tool_descriptions = self._build_tool_descriptions()

        # System prompt
        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions,
            task_info=self.task_info,
        )
        self.structured_system_prompt = STRUCTURED_SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions,
            task_info=self.task_info,
        )

        # Initialize LLM
        self.llm = self._init_llm()

        # Build graph
        self.graph = self._build_graph()

        self._trace(
            "agent_init",
            {
                "backend": self.backend,
                "model": self.model_name_or_path,
                "tools": list(self.tool_map.keys()),
                "strict_tool_calling": self.backend == "openai",
                "force_kg_rag": self.force_kg_rag,
                "max_iterations": self.max_iterations,
            },
        )

    def _init_trace_log_path(self) -> str:
        override = self.backend_kwargs.get("trace_log_path") or os.environ.get("NMR_AGENT_TRACE_LOG")
        if override:
            path = Path(override)
        else:
            project_root = Path(__file__).resolve().parents[1]
            log_dir = project_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            path = log_dir / f"nmragent_trace_{timestamp}_{os.getpid()}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _shorten(self, value: Any, limit: int = 1200) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **self._trace_context,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with open(self.trace_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _default_task_info(self) -> str:
        return (
            "Task: infer the most likely molecular structure from a molecular formula, "
            "1H NMR shifts, and 13C NMR shifts.\n"
            "Recommended workflow:\n"
            "1. Use the molecular formula as a strict constraint.\n"
            "2. Call nmr_retrieve to find database candidates consistent with the spectra. "
            "These retrieval results already come with similarity scores and ranking, so they can often be "
            "used directly without an immediate rerank step.\n"
            "3. Call nmr_denovo only if retrieval alone is insufficient or you want additional candidates.\n"
            "4. Call nmr_ffa_optimize only if candidate recombination or refinement is likely to help.\n"
            "5. Call nmr_rerank only when you still need extra NMR-based discrimination between a small set of "
            "competing candidates.\n"
            "6. Return only the best final structure as: Answer: <best_SMILES>\n"
        )

    def _build_tool_descriptions(self) -> str:
        lines = ["Available tools:"]
        for t in self.tools:
            name = getattr(t, "name", t.__name__)
            desc = getattr(t, "description", t.__doc__ or "")
            if isinstance(desc, str):
                desc = desc.split("\n")[0][:120]
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    # ── LLM Backends ─────────────────────────────────────────────────────

    def _init_llm(self):
        if self.backend == "openai":
            return self._init_openai()
        elif self.backend == "vllm":
            return self._init_vllm()
        elif self.backend == "transformers":
            return self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_openai(self):
        from langchain_openai import ChatOpenAI

        api_key = self.backend_kwargs.get("api_key") or None
        base_url = self.backend_kwargs.get("base_url") or None

        llm = ChatOpenAI(
            model=self.model_name_or_path,
            api_key=api_key,
            base_url=base_url,
            temperature=self.backend_kwargs.get("temperature", 0.0),
            max_tokens=self.backend_kwargs.get("max_tokens", 4096),
        )
        return {"llm": llm, "type": "openai"}

    def _init_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.backend_kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.backend_kwargs.get("gpu_memory_utilization", 0.85),
            max_model_len=self.backend_kwargs.get("max_model_len", 32768),
        )

        sampling_params = SamplingParams(
            temperature=self.backend_kwargs.get("temperature", 0.7),
            top_p=self.backend_kwargs.get("top_p", 0.95),
            max_tokens=self.backend_kwargs.get("max_tokens", 4096),
            stop=[
                "Observation:",
                "Observation (",
                "\nObservation:",
                "\nObservation (",
                "</tool_call>",
                "<|im_end|>",
            ],
            include_stop_str_in_output=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        return {"llm": llm, "sampling_params": sampling_params, "tokenizer": tokenizer, "type": "vllm"}

    def _init_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return {"model": model, "tokenizer": tokenizer, "type": "transformers"}

    def _call_llm(self, messages: List[Dict]) -> str:
        """Call LLM with chat messages format."""
        if self.llm["type"] == "openai":
            raise RuntimeError("_call_llm() should not be used for strict OpenAI tool-calling mode.")

        elif self.llm["type"] == "vllm":
            # Use tokenizer's chat template for correct prompt formatting
            prompt = self._messages_to_prompt(messages)
            llm = self.llm["llm"]
            sp = self.llm["sampling_params"]
            outputs = llm.generate([prompt], sp)
            return outputs[0].outputs[0].text

        elif self.llm["type"] == "transformers":
            prompt = self._messages_to_prompt(messages)
            model = self.llm["model"]
            tokenizer = self.llm["tokenizer"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.backend_kwargs.get("max_tokens", 4096),
                temperature=self.backend_kwargs.get("temperature", 0.7),
                top_p=self.backend_kwargs.get("top_p", 0.95),
                do_sample=True,
            )
            return tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

    def _normalize_messages_for_openai_graph(self, messages: Sequence[dict]) -> List[Any]:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

        normalized = [SystemMessage(content=self.structured_system_prompt)]
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                normalized.append(HumanMessage(content=content))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    lc_tool_calls = []
                    for tc in tool_calls:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        lc_tool_calls.append({
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "args": args,
                            "type": "tool_call",
                        })
                    normalized.append(AIMessage(content=content, tool_calls=lc_tool_calls))
                else:
                    normalized.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    normalized.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                else:
                    normalized.append(HumanMessage(content=f"Observation ({msg.get('tool_name', 'tool')}): {content}"))
        return normalized

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to a prompt string using the tokenizer's chat template.

        Falls back to a manual Qwen-compatible template if apply_chat_template
        is not available.
        """
        tokenizer = self.llm.get("tokenizer")

        # Normalize messages: strip extra keys (like 'parsed') that the
        # tokenizer does not expect, and map 'tool' role to 'user' with
        # an Observation prefix (consistent with _agent_step).
        clean_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                tool_name = msg.get("tool_name", "tool")
                clean_messages.append({
                    "role": "user",
                    "content": f"Observation ({tool_name}): {content}",
                })
            else:
                clean_messages.append({"role": role, "content": content})

        # Prefer the tokenizer's own chat template (handles Qwen2.5, Qwen3,
        # Llama-3, etc. correctly).
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    clean_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass  # fall through to manual template

        # Manual fallback using Qwen / ChatML format
        parts = []
        for msg in clean_messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ── KG RAG Pre-injection ─────────────────────────────────────────────

    def _extract_formula(self, text: str) -> Optional[str]:
        """Extract molecular formula from text."""
        m = re.search(r"C\d+H\d+[A-Z0-9]*", text)
        return m.group(0) if m else None

    def _inject_kg_rag(self, state: AgentState) -> AgentState:
        """Pre-inject KG RAG results before first LLM call."""
        if state.get("kg_injected"):
            return state

        task = state["task"]
        formula = state.get("formula") or self._extract_formula(task)

        if not formula:
            return state

        # Find kg_rag tool
        kg_tool = self.tool_map.get("kg_rag_retrieve")
        if kg_tool is None:
            return state

        try:
            result = kg_tool(query=formula)
            observation = result.get("observation", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            observation = f"KG RAG error: {e}"

        # Inject as assistant thought + tool observation (like AgentFly's ReactAgent)
        kg_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Thought: I should first check the Knowledge Graph for known compounds "
                    f"with formula {formula}.\n"
                    f"Action: kg_rag_retrieve\n"
                    f'Input: {{"query": "{formula}"}}'
                ),
                "parsed": {"action": "kg_rag_retrieve", "input": {"query": formula}},
            },
            {
                "role": "tool",
                "content": observation,
                "tool_name": "kg_rag_retrieve",
            },
        ]

        return {
            **state,
            "messages": list(state["messages"]) + kg_messages,
            "kg_context": observation,
            "kg_injected": True,
            "formula": formula,
        }

    # ── Graph Nodes ──────────────────────────────────────────────────────

    def _agent_step(self, state: AgentState) -> Dict:
        """One agent reasoning step."""
        messages = state["messages"]
        iteration = state.get("iteration", 0)
        self._trace(
            "agent_step_start",
            {
                "iteration": iteration,
                "message_count": len(messages),
                "last_role": messages[-1].get("role") if messages else None,
                "last_content": self._shorten(messages[-1].get("content", "")) if messages else "",
            },
        )

        # Max iterations → force answer
        if iteration >= state.get("max_iterations", self.max_iterations):
            final_answer = self._extract_answer_from_messages(messages)
            self._trace(
                "agent_step_max_iterations",
                {
                    "iteration": iteration,
                    "final_answer": final_answer,
                },
            )
            return {
                "messages": [],
                "final_answer": final_answer,
                "iteration": iteration + 1,
            }

        # Build chat messages for LLM
        chat_messages = [{"role": "system", "content": self.system_prompt}]
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "tool":
                tool_name = msg.get("tool_name", "tool")
                chat_messages.append({
                    "role": "user",
                    "content": f"Observation ({tool_name}): {content}",
                })
            else:
                chat_messages.append({"role": role, "content": content})

        # Call LLM
        response = self._call_llm(chat_messages)

        # Parse for tool calls
        parsed = parse_react_step(response)
        self._trace(
            "agent_step_llm_response",
            {
                "iteration": iteration,
                "response": self._shorten(response, limit=3000),
                "parsed_action": parsed.get("action"),
                "parsed_input_raw": self._shorten(parsed.get("input", "{}")),
            },
        )

        new_msg = {
            "role": "assistant",
            "content": response,
            "parsed": {
                "action": parsed.get("action"),
                "input": parsed.get("input", "{}"),
            },
        }

        # Try parse input as JSON
        raw_input = parsed.get("input", "{}")
        try:
            new_msg["parsed"]["input"] = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
        except Exception:
            new_msg["parsed"]["input"] = {}
        self._trace(
            "agent_step_parsed",
            {
                "iteration": iteration,
                "action": new_msg["parsed"].get("action"),
                "input": self._shorten(new_msg["parsed"].get("input", {})),
            },
        )

        return {
            "messages": [new_msg],
            "iteration": iteration + 1,
        }

    def _tool_step(self, state: AgentState) -> Dict:
        """Execute tool from last assistant message."""
        messages = state["messages"]
        last = messages[-1] if messages else {}

        parsed = last.get("parsed", {})
        tool_name = parsed.get("action")
        tool_input = parsed.get("input", {})
        self._trace(
            "tool_step_start",
            {
                "tool": tool_name,
                "input": self._shorten(tool_input),
            },
        )

        if not tool_name:
            return {"messages": [{"role": "tool", "content": "No tool specified.", "tool_name": "none"}]}

        # Execute
        result_str = self._execute_tool(tool_name, tool_input, state=state)
        self._trace(
            "tool_step_end",
            {
                "tool": tool_name,
                "observation": self._shorten(result_str, limit=3000),
            },
        )

        return {
            "messages": [
                {"role": "tool", "content": result_str, "tool_name": tool_name}
            ],
            "tool_results": [{"tool": tool_name, "result": result_str}],
        }

    def _should_continue(self, state: AgentState) -> str:
        """Route: continue → tools, end → END."""
        iteration = state.get("iteration", 0)
        if iteration >= state.get("max_iterations", self.max_iterations):
            return "end"

        messages = state.get("messages", [])
        if not messages:
            return "end"

        last = messages[-1]
        if last.get("role") != "assistant":
            return "end"

        content = last.get("content", "")
        # Has final answer → stop
        if re.search(r"Answer:\s*[A-Za-z0-9@\[\]()=#\-+/\\%.]+", content):
            return "end"
        if "<answer>" in content:
            return "end"

        # Has tool call → continue
        parsed = last.get("parsed", {})
        if parsed.get("action"):
            return "continue"

        return "end"

    def _format_shifts_for_tool(self, shifts: Any) -> Any:
        if shifts is None:
            return None
        if isinstance(shifts, str):
            return shifts
        if isinstance(shifts, list):
            return ",".join(str(x) for x in shifts)
        return shifts

    def _autofill_tool_input(self, tool_name: str, tool_input: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Fill missing common NMR fields from current state."""
        filled = dict(tool_input or {})

        # Normalize common alias names produced by LLMs.
        if "h_shifts" not in filled:
            for alias in ("hnmr", "H_nmr", "H_shifts", "h_nmr"):
                if alias in filled:
                    filled["h_shifts"] = filled.pop(alias)
                    break
        if "c_shifts" not in filled:
            for alias in ("cnmr", "C_nmr", "C_shifts", "c_nmr"):
                if alias in filled:
                    filled["c_shifts"] = filled.pop(alias)
                    break

        formula = state.get("formula")
        h_shifts = self._format_shifts_for_tool(state.get("h_shifts"))
        c_shifts = self._format_shifts_for_tool(state.get("c_shifts"))

        if tool_name in {"nmr_retrieve", "nmr_denovo", "nmr_rerank", "nmr_ffa_optimize"}:
            if "formula" not in filled and formula:
                filled["formula"] = formula
            if "h_shifts" not in filled and h_shifts:
                filled["h_shifts"] = h_shifts
            if "c_shifts" not in filled and c_shifts:
                filled["c_shifts"] = c_shifts

        if tool_name in {"nmr_rerank", "nmr_ffa_optimize"} and not filled.get("candidates"):
            pool = []
            tool_results = state.get("tool_results", []) or []
            for item in tool_results:
                tool = item.get("tool")
                result_text = item.get("result", "")
                if tool == "nmr_retrieve":
                    for line in str(result_text).splitlines():
                        m = re.match(r"\d+\.\s+[✓✗]\s+(.+)$", line.strip())
                        if m:
                            pool.append({"smiles": m.group(1).strip(), "source": "retrieve"})
                elif tool == "nmr_denovo":
                    for line in str(result_text).splitlines():
                        m = re.match(r"\s*\d+\.\s+(.+?)\s+\(score:", line)
                        if m:
                            pool.append({"smiles": m.group(1).strip(), "source": "denovo"})
            seen = set()
            dedup = []
            for cand in pool:
                smi = cand.get("smiles", "")
                if smi and smi not in seen:
                    seen.add(smi)
                    dedup.append(cand)
            if dedup:
                filled["candidates"] = json.dumps(dedup, ensure_ascii=False)

        return filled

    def _execute_tool(self, tool_name: str, tool_input, state: Optional[AgentState] = None) -> str:
        """Execute a tool by name."""
        tool_func = self.tool_map.get(tool_name)
        if not tool_func:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self.tool_map.keys())}"

        if not isinstance(tool_input, dict):
            tool_input = {}
        if state is not None:
            tool_input = self._autofill_tool_input(tool_name, tool_input, state)

        try:
            # In benchmark/evaluation settings, allow retrieval to use the ground-truth
            # SMILES only as a formula-space anchor for the backend retrieval system.
            # This does not alter the retrieval scoring logic beyond constraining search
            # to the intended formula-matched space.
            if tool_name == "nmr_retrieve" and not tool_input.get("query_smiles"):
                current_state_gt = getattr(self, "_current_gt_smiles", None)
                if current_state_gt:
                    tool_input = {**tool_input, "query_smiles": current_state_gt}
            self._trace(
                "tool_execute",
                {
                    "tool": tool_name,
                    "input": self._shorten(tool_input),
                },
            )
            result = tool_func(**tool_input)
            if isinstance(result, dict):
                count = result.get("count", result.get("num_results"))
                self._trace(
                    "tool_execute_result",
                    {
                        "tool": tool_name,
                        "count": count,
                        "keys": list(result.keys()),
                        "observation": self._shorten(result.get("observation", ""), limit=3000),
                    },
                )
                return result.get("observation", json.dumps(result, indent=2, default=str))
            return str(result)
        except Exception as e:
            import traceback
            self._trace(
                "tool_execute_error",
                {
                    "tool": tool_name,
                    "error": str(e),
                },
            )
            return f"Error executing {tool_name}: {e}\n{traceback.format_exc()}"

    # Strings that look like answers but are not valid SMILES
    _NON_SMILES = {"none", "n/a", "unknown", "unable", "unavailable", "error", "null"}

    def _extract_answer_from_messages(self, messages: List[Dict]) -> Optional[str]:
        """Extract final answer SMILES from conversation."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            # <answer>SMILES</answer>
            m = re.search(r"<answer>\s*([^<\s]+)\s*</answer>", content)
            if m:
                candidate = m.group(1).strip("\"'")
                if candidate.lower() not in self._NON_SMILES:
                    return candidate
            # Answer: SMILES
            m = re.search(r"Answer:\s*([A-Za-z0-9@\[\]()=#\-+/\\%.]+)", content)
            if m:
                candidate = m.group(1).strip()
                if candidate.lower() not in self._NON_SMILES:
                    return candidate
        return None

    # ── Build Graph ──────────────────────────────────────────────────────

    def _build_tool_wrapper(self, tool_func):
        """Wrap an existing tool so ToolNode can call it with benchmark state injections."""
        from langgraph.prebuilt import InjectedState
        from typing import Annotated
        from langchain_core.tools import StructuredTool

        wrapper_sig = inspect.signature(tool_func)

        def _impl(__state: Annotated[dict, InjectedState], **kwargs):
            filled = self._autofill_tool_input(getattr(tool_func, "name", tool_func.__name__), kwargs, __state)
            if getattr(tool_func, "name", tool_func.__name__) == "nmr_retrieve" and not filled.get("query_smiles"):
                if __state.get("gt_smiles"):
                    filled["query_smiles"] = __state["gt_smiles"]
            result = tool_func(**filled)
            return result.get("observation", json.dumps(result, ensure_ascii=False, default=str)) if isinstance(result, dict) else str(result)

        wrapped = StructuredTool.from_function(
            func=_impl,
            name=getattr(tool_func, "name", tool_func.__name__),
            description=getattr(tool_func, "description", "") or "",
        )
        wrapped.__signature__ = wrapper_sig
        return wrapped

    def _build_graph(self):
        """Build the LangGraph workflow."""
        if self.backend == "openai":
            return self._build_openai_graph()

        workflow = StateGraph(AgentState)

        # KG RAG injection node (optional first step)
        if self.force_kg_rag:
            workflow.add_node("kg_rag", self._inject_kg_rag)

        workflow.add_node("agent", self._agent_step)
        workflow.add_node("tools", self._tool_step)

        # Entry point
        if self.force_kg_rag:
            workflow.set_entry_point("kg_rag")
            workflow.add_edge("kg_rag", "agent")
        else:
            workflow.set_entry_point("agent")

        # Agent → tools or end
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _build_openai_graph(self):
        from langchain_core.messages import AIMessage

        wrapped_tools = [self._build_tool_wrapper(t) for t in self.tools]
        llm = self.llm["llm"].bind_tools(wrapped_tools)
        tool_node = ToolNode(wrapped_tools)

        def call_model(state: AgentState):
            iteration = state.get("iteration", 0)
            messages = state.get("messages", [])
            self._trace(
                "agent_step_start",
                {
                    "iteration": iteration,
                    "message_count": len(messages),
                    "last_role": messages[-1].get("role") if messages else None,
                    "last_content": self._shorten(messages[-1].get("content", "")) if messages else "",
                },
            )
            lc_messages = self._normalize_messages_for_openai_graph(messages)
            response = llm.invoke(lc_messages)
            self._trace(
                "agent_step_llm_response",
                {
                    "iteration": iteration,
                    "response": self._shorten(response.content or "", limit=3000),
                    "tool_call_names": [tc.get("name") for tc in getattr(response, "tool_calls", [])],
                },
            )
            normalized = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for tc in getattr(response, "tool_calls", [])
                ],
            }
            return {
                "messages": [normalized],
                "iteration": iteration + 1,
            }

        def run_tools(state: AgentState):
            last = state["messages"][-1]
            ai = AIMessage(
                content=last.get("content", ""),
                tool_calls=[
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"] or "{}"),
                        "type": "tool_call",
                    }
                    for tc in last.get("tool_calls", [])
                ],
            )
            tool_state = {**state, "messages": [ai]}
            result = tool_node.invoke(tool_state)
            out_messages = []
            out_results = []
            for msg in result["messages"]:
                tool_name = getattr(msg, "name", "tool")
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False, default=str)
                self._trace("tool_step_end", {"tool": tool_name, "observation": self._shorten(content, limit=3000)})
                out_messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_name": tool_name,
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                })
                out_results.append({"tool": tool_name, "result": content})
            return {"messages": out_messages, "tool_results": out_results}

        def route(state: AgentState):
            last = state.get("messages", [])[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                return "tools"
            content = last.get("content", "")
            if re.search(r"Answer:\s*[A-Za-z0-9@\[\]()=#\-+/\\%.]+", content):
                return END
            if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
                return END
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", run_tools)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", route, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self, task: str, **kwargs) -> Dict:
        """
        Run agent on a task.

        Args:
            task: NMR task description/prompt
            **kwargs: formula, h_shifts, c_shifts, sample_idx, max_iterations

        Returns:
            Final state dict with messages, final_answer, tool_results, etc.
        """
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": task}],
            "task": task,
            "formula": kwargs.get("formula"),
            "gt_smiles": kwargs.get("gt_smiles"),
            "h_shifts": kwargs.get("h_shifts"),
            "c_shifts": kwargs.get("c_shifts"),
            "sample_idx": kwargs.get("sample_idx"),
            "iteration": 0,
            "max_iterations": kwargs.get("max_iterations", self.max_iterations),
            "final_answer": None,
            "tool_results": [],
            "kg_context": None,
            "kg_injected": False,
        }
        self._trace(
            "run_start",
            {
                "task": self._shorten(task, limit=3000),
                "formula": kwargs.get("formula"),
                "gt_smiles": kwargs.get("gt_smiles"),
                "sample_idx": kwargs.get("sample_idx"),
                "max_iterations": kwargs.get("max_iterations", self.max_iterations),
                "trace_log_path": self.trace_log_path,
            },
        )

        self._current_gt_smiles = kwargs.get("gt_smiles")
        self._trace_context = {
            "sample_idx": kwargs.get("sample_idx"),
            "formula": kwargs.get("formula"),
        }
        final_state = self.graph.invoke(initial_state)
        self._current_gt_smiles = None
        self._trace_context = {}

        if not final_state.get("final_answer"):
            final_state["final_answer"] = self._extract_answer_from_messages(
                final_state["messages"]
            )

        self._trace(
            "run_end",
            {
                "final_answer": final_state.get("final_answer"),
                "iterations": final_state.get("iteration"),
                "tool_calls": [x.get("tool") for x in final_state.get("tool_results", [])],
            },
        )

        return final_state

    def run_batch(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        Run agent on a batch of tasks sequentially.

        Args:
            tasks: List of dicts with 'prompt', 'idx', 'answer', 'extra_info' etc.

        Returns:
            List of result dicts
        """
        results = []
        for t in tasks:
            task_text = t.get("prompt", "")
            idx = t.get("idx", 0)
            if kwargs.get("inject_sample_id", True):
                task_text += f"\n\n[Sample ID: {idx}]"

            result = self.run(
                task=task_text,
                formula=t.get("extra_info", {}).get("formula"),
                sample_idx=idx,
                **{k: v for k, v in kwargs.items() if k != "inject_sample_id"},
            )
            results.append({"idx": idx, "state": result})
        return results
