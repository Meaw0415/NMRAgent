"""
NMRAgent tools package exports.

Tool modules are imported lazily so lightweight RAG utilities do not trigger
retrieval/denovo/rerank model initialization during simple imports.
"""

from __future__ import annotations

from .decorator import tool, convert_to_langgraph_tool

_TOOL_SPECS = {
    "kg_graph_rag_search": (".kg_rag_tool", "kg_graph_rag_search"),
    "textbook_nmr_search": (".textbook_rag_tool", "textbook_nmr_search"),
    "web_nmr_search": (".web_rag_tool", "web_nmr_search"),
    "nmr_retrieve": (".nmr_retrieval_tool", "nmr_retrieve_tool"),
    "nmr_retrieve_tool": (".nmr_retrieval_tool", "nmr_retrieve_tool"),
    "nmr_retrieve_service": (".nmr_retrieval_service_tool", "nmr_retrieve_service"),
    "get_nmr_searcher": (".nmr_retrieval_tool", "get_searcher"),
    "nmr_denovo": (".nmr_denovo_tool", "nmr_denovo"),
    "nmr_denovo_service": (".nmr_denovo_service_tool", "nmr_denovo_service"),
    "nmr_rerank": (".nmr_rerank_tool", "nmr_rerank"),
    "nmr_canonicalize_smiles": (".nmr_inplace_edit_tool", "nmr_canonicalize_smiles"),
    "nmr_replace_atom": (".nmr_inplace_edit_tool", "nmr_replace_atom"),
    "nmr_delete_atom": (".nmr_inplace_edit_tool", "nmr_delete_atom"),
    "nmr_optimize": (".nmr_optimize_tool", "nmr_optimize"),
    "nmr_merge_pools": (".nmr_pool_tools", "nmr_merge_pools"),
}

__all__ = ["tool", "convert_to_langgraph_tool", *_TOOL_SPECS.keys(), "get_all_tools", "get_offline_tools", "get_tools_by_names"]


def _load_tool(public_name: str):
    spec = _TOOL_SPECS.get(public_name)
    if spec is None:
        raise AttributeError(public_name)
    module_name, attr_name = spec
    try:
        module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
        tool_fn = getattr(module, attr_name)
    except Exception as exc:
        print(f"[tools] Optional import failed for {module_name}.{attr_name}: {exc}")
        tool_fn = None
    globals()[public_name] = tool_fn
    return tool_fn


def __getattr__(name: str):
    if name in _TOOL_SPECS:
        return _load_tool(name)
    raise AttributeError(name)


def _available(*tools):
    return [tool_fn for tool_fn in tools if tool_fn is not None]


def get_all_tools():
    """Get the active tool set used by the multi-agent runner."""
    return get_tools_by_names([
        "kg_graph_rag_search",
        "textbook_nmr_search",
        "web_nmr_search",
        "nmr_retrieve",
        "nmr_denovo",
        "nmr_rerank",
        "nmr_canonicalize_smiles",
        "nmr_replace_atom",
        "nmr_delete_atom",
        "nmr_optimize",
        "nmr_merge_pools",
    ])


def get_offline_tools():
    """No legacy offline tool set is exported by the focused package API."""
    return []


def get_tools_by_names(names):
    """Get tools by their registered names, importing modules only when requested."""
    tools = []
    for name in names:
        tool_fn = _load_tool(name) if name in _TOOL_SPECS else None
        if tool_fn is not None:
            tools.append(tool_fn)
        else:
            print(f"Warning: Tool {name} not found or unavailable")
    return tools
