"""
NMRAgent tools package exports.

Offline tools are kept importable even when heavy online dependencies such as
torch are unavailable in the current environment.
"""

from __future__ import annotations

from .decorator import tool, convert_to_langgraph_tool
from .kg_tools import kg_get_description, kg_search_triples
from .nmr_chefnmr_offline_denovo_tool import nmr_chefnmr_offline_denovo
from .nmr_offline_tools import (
    nmr_offline_denovo,
    nmr_offline_rerank,
    nmr_offline_retrieve,
)


def _safe_import(module_name: str, names: list[str]) -> dict[str, object]:
    try:
        module = __import__(f"{__name__}{module_name}", fromlist=names)
        return {name: getattr(module, name) for name in names}
    except Exception as exc:
        print(f"[tools] Optional import failed for {module_name}: {exc}")
        return {name: None for name in names}


_kg_rag = _safe_import(".nmr_kg_rag_tool", ["kg_rag_retrieve", "_run_kg_rag_impl"])
kg_rag_retrieve = _kg_rag["kg_rag_retrieve"]
_run_kg_rag_impl = _kg_rag["_run_kg_rag_impl"]

_retrieval = _safe_import(
    ".nmr_retrieval_tool",
    ["nmr_retrieve_tool", "nmr_retrieve", "get_searcher"],
)
nmr_retrieve_tool = _retrieval["nmr_retrieve_tool"]
nmr_retrieve = _retrieval["nmr_retrieve"]
get_nmr_searcher = _retrieval["get_searcher"]

_denovo = _safe_import(".nmr_denovo_tool", ["nmr_denovo"])
nmr_denovo = _denovo["nmr_denovo"]

_chefnmr_online = _safe_import(".nmr_chefnmr_denovo_tool", ["nmr_chefnmr_denovo"])
nmr_chefnmr_denovo = _chefnmr_online["nmr_chefnmr_denovo"]

_rerank = _safe_import(".nmr_rerank_tool", ["nmr_rerank"])
nmr_rerank = _rerank["nmr_rerank"]

_mol_edit = _safe_import(
    ".nmr_mol_edit_tool",
    ["nmr_mol_edit", "nmr_fragment_edit", "nmr_refine"],
)
nmr_mol_edit = _mol_edit["nmr_mol_edit"]
nmr_fragment_edit = _mol_edit["nmr_fragment_edit"]
nmr_refine = _mol_edit["nmr_refine"]

_inplace_edit = _safe_import(
    ".nmr_inplace_edit_tool",
    ["nmr_canonicalize_smiles", "nmr_replace_atom", "nmr_delete_atom"],
)
nmr_canonicalize_smiles = _inplace_edit["nmr_canonicalize_smiles"]
nmr_replace_atom = _inplace_edit["nmr_replace_atom"]
nmr_delete_atom = _inplace_edit["nmr_delete_atom"]

_optimize = _safe_import(".nmr_optimize_tool", ["nmr_optimize", "nmr_ffa_optimize"])
nmr_optimize = _optimize["nmr_optimize"]
nmr_ffa_optimize = _optimize["nmr_ffa_optimize"]

_pool_tools = _safe_import(".nmr_pool_tools", ["nmr_merge_pools"])
nmr_merge_pools = _pool_tools["nmr_merge_pools"]

_fragment_search = _safe_import(".nmr_fragment_search_tool", ["nmr_fragment_search"])
nmr_fragment_search = _fragment_search["nmr_fragment_search"]

__all__ = [
    "tool",
    "convert_to_langgraph_tool",
    # KG Tools
    "kg_rag_retrieve",
    "_run_kg_rag_impl",
    "kg_search_triples",
    "kg_get_description",
    # Retrieval
    "nmr_retrieve_tool",
    "nmr_retrieve",
    "get_nmr_searcher",
    # De Novo Generation
    "nmr_denovo",
    "nmr_chefnmr_denovo",
    # Reranking
    "nmr_rerank",
    # Molecular Editing
    "nmr_mol_edit",
    "nmr_fragment_edit",
    "nmr_refine",
    "nmr_canonicalize_smiles",
    "nmr_replace_atom",
    "nmr_delete_atom",
    # FFA Optimization
    "nmr_optimize",
    "nmr_ffa_optimize",
    "nmr_merge_pools",
    # Fragment Search
    "nmr_fragment_search",
    # Offline Tools
    "nmr_offline_retrieve",
    "nmr_offline_denovo",
    "nmr_offline_rerank",
    "nmr_chefnmr_offline_denovo",
]
def _available(*tools):
    return [tool_fn for tool_fn in tools if tool_fn is not None]


def get_all_tools():
    """Get all available NMR tools as a list."""
    return _available(
        kg_rag_retrieve,
        nmr_retrieve_tool,
        nmr_denovo,
        nmr_rerank,
        nmr_mol_edit,
        nmr_fragment_edit,
        nmr_refine,
        nmr_canonicalize_smiles,
        nmr_replace_atom,
        nmr_delete_atom,
        nmr_optimize,
        nmr_merge_pools,
        nmr_fragment_search,
        kg_search_triples,
        kg_get_description,
    )


def get_offline_tools():
    """Get offline tools that use precomputed caches."""
    return [
        nmr_offline_retrieve,
        nmr_offline_denovo,
        nmr_offline_rerank,
        nmr_chefnmr_offline_denovo,
    ]


def get_tools_by_names(names):
    """
    Get tools by their names.

    Args:
        names: List of tool names (strings)

    Returns:
        List of tool functions
    """
    tool_map = {
        "kg_rag_retrieve": kg_rag_retrieve,
        "nmr_retrieve": nmr_retrieve_tool,
        "nmr_denovo": nmr_denovo,
        "nmr_chefnmr_denovo": nmr_chefnmr_denovo,
        "nmr_rerank": nmr_rerank,
        "nmr_mol_edit": nmr_mol_edit,
        "nmr_fragment_edit": nmr_fragment_edit,
        "nmr_refine": nmr_refine,
        "nmr_canonicalize_smiles": nmr_canonicalize_smiles,
        "nmr_replace_atom": nmr_replace_atom,
        "nmr_delete_atom": nmr_delete_atom,
        "nmr_optimize": nmr_optimize,
        "nmr_ffa_optimize": nmr_ffa_optimize,
        "nmr_merge_pools": nmr_merge_pools,
        "nmr_fragment_search": nmr_fragment_search,
        "kg_search_triples": kg_search_triples,
        "kg_get_description": kg_get_description,
        "nmr_offline_retrieve": nmr_offline_retrieve,
        "nmr_offline_denovo": nmr_offline_denovo,
        "nmr_offline_rerank": nmr_offline_rerank,
        "nmr_chefnmr_offline_denovo": nmr_chefnmr_offline_denovo,
    }

    tools = []
    for name in names:
        tool_fn = tool_map.get(name)
        if tool_fn is not None:
            tools.append(tool_fn)
        else:
            print(f"Warning: Tool {name} not found or unavailable")

    return tools
