import json
from typing import List

from .decorator import tool
from .nmr_solver_style_core import BricsParams, SolverStyleParams, run_solver_style_plus_brics


def _parse_shift_input(values) -> List[float]:
    if isinstance(values, list):
        return [float(x) for x in values]
    if isinstance(values, str):
        return [float(x.strip()) for x in values.split(",") if x.strip()]
    return [float(x) for x in values]


@tool(
    name="nmr_solver_style_brics",
    description=(
        "Run solver-style retrieval/vector-rerank first, then expand the molecule pool with local "
        "solver-style assembled/reranked molecules and attempt BRICS-based formula-constrained reassembly. "
        "This is the local 'solver-style + BRICS' path without depending on external NMRSolver script paths."
    ),
)
def nmr_solver_style_brics_tool(
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
    max_fragments: int = 1200,
    max_per_formula: int = 12,
    min_k: int = 3,
    max_k: int = 6,
    beam_size: int = 300,
    max_combos: int = 200,
    max_choice_variants: int = 6,
    max_products_per_combo: int = 10,
    brics_rerank_top_k: int = 20,
    augment_sections: str = "assembled,rerank_top",
) -> dict:
    rec = {
        "idx": 0,
        "formula": formula,
        "gt_smiles": gt_smiles,
        "h_shifts": _parse_shift_input(h_shifts),
        "c_shifts": _parse_shift_input(c_shifts),
    }
    solver_params = SolverStyleParams(
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
    brics_params = BricsParams(
        max_fragments=max_fragments,
        max_per_formula=max_per_formula,
        min_k=min_k,
        max_k=max_k,
        beam_size=beam_size,
        max_combos=max_combos,
        max_choice_variants=max_choice_variants,
        max_products_per_combo=max_products_per_combo,
        rerank_top_k=brics_rerank_top_k,
    )
    sections = [x.strip() for x in augment_sections.split(",") if x.strip()]
    result = run_solver_style_plus_brics(rec, solver_params, brics_params, augment_sections=sections)
    return {
        "observation": json.dumps(
            {
                "solver_merged_pool_count": result["solver_style"]["merged_pool_count"],
                "augmented_pool_count": result["augmented_pool_count"],
                "fragment_pool_count": result["fragment_pool_count"],
                "combo_count": result["combo_count"],
                "assembled_count": result["assembled_count"],
                "gt_in_assembled": result.get("gt_in_assembled", False),
            },
            ensure_ascii=False,
        ),
        "valid": 1,
        "result": result,
    }

