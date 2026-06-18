import json
from typing import List

from .decorator import tool
from .nmr_solver_style_core import SolverStyleParams, run_solver_style_only


def _parse_shift_input(values) -> List[float]:
    if isinstance(values, list):
        return [float(x) for x in values]
    if isinstance(values, str):
        return [float(x.strip()) for x in values.split(",") if x.strip()]
    return [float(x) for x in values]


@tool(
    name="nmr_solver_style",
    description=(
        "Run the local solver-style retrieval + vector-rerank + fragment-pair + stitch pipeline. "
        "This is the retrieval-only solver-style mainline rebuilt from local tools code, without "
        "depending on external NMRSolver script paths."
    ),
)
def nmr_solver_style_tool(
    h_shifts,
    c_shifts,
    formula: str,
    gt_smiles: str = "",
    formula_topk: int = 500,
    nonformula_topk: int = 500,
    keep_formula: int = 500,
    keep_nonformula: int = 500,
    nprobe: int = 128,
    topk_per_fragment: int = 1000,
    num_filter_pair: int = 200000,
    max_new: int = 1000,
    rerank_keep_original: int = 40,
    rerank_keep_new: int = 100,
    rerank_top_k: int = 20,
    disable_formula_pair_prune: bool = False,
) -> dict:
    rec = {
        "idx": 0,
        "formula": formula,
        "gt_smiles": gt_smiles,
        "h_shifts": _parse_shift_input(h_shifts),
        "c_shifts": _parse_shift_input(c_shifts),
    }
    params = SolverStyleParams(
        formula_topk=formula_topk,
        nonformula_topk=nonformula_topk,
        keep_formula=keep_formula,
        keep_nonformula=keep_nonformula,
        nprobe=nprobe,
        topk_per_fragment=topk_per_fragment,
        num_filter_pair=num_filter_pair,
        max_new=max_new,
        rerank_keep_original=rerank_keep_original,
        rerank_keep_new=rerank_keep_new,
        rerank_top_k=rerank_top_k,
        disable_formula_pair_prune=disable_formula_pair_prune,
    )
    result = run_solver_style_only(rec, params)
    return {
        "observation": json.dumps(
            {
                "formula_retrieval_count": result["formula_retrieval_count"],
                "nonformula_retrieval_count": result["nonformula_retrieval_count"],
                "merged_pool_count": result["merged_pool_count"],
                "assembled_count": result["assembled_count"],
                "gt_in_full_pool": result.get("gt_in_full_pool", False),
                "gt_in_assembled": result.get("gt_in_assembled", False),
                "gt_in_rerank_topk": result.get("gt_in_rerank_topk", False),
            },
            ensure_ascii=False,
        ),
        "valid": 1,
        "result": result,
    }

