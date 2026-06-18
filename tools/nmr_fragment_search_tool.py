"""
nmr_fragment_search: Search static NMR fragment library by C-shift range.

The static library was built from 158M molecules in embeddings_with_smiles.lmdb.
Each fragment entry stores: smiles, cut_types, c_median, c_p25, c_p75, count.

The shift index maps 5-ppm bins → [fragment_smiles] for fast lookup.
"""

import os
import pickle
import lmdb
from .decorator import tool
from .path_utils import artifact_path

LIBRARY_DIR = os.environ.get("NMR_FRAGMENT_LIB_DIR", str(artifact_path("fragments")))
LMDB_PATH = os.path.join(LIBRARY_DIR, "fragment_library.lmdb")
INDEX_PATH = os.path.join(LIBRARY_DIR, "fragment_shift_index.pkl")
SHIFT_BIN    = 5.0

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_shift_index = None
_frag_db     = None

def _load_resources():
    global _shift_index, _frag_db
    if _shift_index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"Shift index not found: {INDEX_PATH}")
        with open(INDEX_PATH, 'rb') as f:
            _shift_index = pickle.load(f)
    if _frag_db is None:
        if not os.path.exists(LMDB_PATH):
            raise FileNotFoundError(f"Fragment library not found: {LMDB_PATH}")
        _frag_db = lmdb.open(LMDB_PATH, readonly=True, lock=False, subdir=False,
                              max_readers=64)
    return _shift_index, _frag_db


def _get_bins(shift_min: float, shift_max: float):
    """Return all 5-ppm bin keys covered by [shift_min, shift_max]."""
    lo = int(shift_min // SHIFT_BIN) * int(SHIFT_BIN)
    hi = int(shift_max // SHIFT_BIN) * int(SHIFT_BIN)
    return list(range(int(lo), int(hi) + int(SHIFT_BIN), int(SHIFT_BIN)))


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool(
    name="nmr_fragment_search",
    description=(
        "Search a static NMR fragment library (built from 158M molecules) for fragments "
        "whose median C-shift falls within [shift_min, shift_max] ppm. "
        "Use this when nmr_rerank shows a poorly-matched C peak (large residual or unmatched) "
        "to find replacement fragments with the correct chemical environment. "
        "Returns fragment SMILES, typical C-shift range, occurrence count, and compatible cut types. "
        "Pass the returned fragment SMILES to nmr_mol_edit (action='replace') to swap it in."
    ),
)
def nmr_fragment_search(
    shift_min: float,
    shift_max: float,
    cut_type: str = "",
    top_k: int = 10,
) -> dict:
    """
    Search static fragment library for fragments matching a C-shift range.

    Args:
        shift_min:  Lower bound of target C-shift range (ppm).
        shift_max:  Upper bound of target C-shift range (ppm).
        cut_type:   Optional filter, e.g. "C-C:-" or "C-O:-".
                    Only return fragments compatible with this bond type.
        top_k:      Maximum number of fragments to return (default 10).

    Returns:
        dict with 'observation' (formatted results) and 'fragments' (list of dicts).

    Example:
        # Looking for a fragment with ~198 ppm C (ketone/aldehyde)
        nmr_fragment_search(shift_min=185, shift_max=215)

        # Looking for aromatic C with C-C single bond cut site
        nmr_fragment_search(shift_min=120, shift_max=145, cut_type="C-C:-")
    """
    try:
        shift_index, frag_db = _load_resources()
    except FileNotFoundError as e:
        return {
            "observation": f"Fragment library not available: {e}",
            "fragments": [],
        }

    if shift_min > shift_max:
        shift_min, shift_max = shift_max, shift_min

    # Collect candidate fragment SMILES from shift index
    bins = _get_bins(shift_min, shift_max)
    candidate_smiles = set()
    for b in bins:
        candidate_smiles.update(shift_index.get(b, []))

    if not candidate_smiles:
        return {
            "observation": (f"No fragments found in shift range [{shift_min}, {shift_max}] ppm. "
                            f"Checked {len(bins)} bins: {bins}."),
            "fragments": [],
        }

    # Look up full records from LMDB
    results = []
    with frag_db.begin() as txn:
        for smi in candidate_smiles:
            raw = txn.get(smi.encode())
            if raw is None:
                continue
            rec = pickle.loads(raw)

            c_med = rec.get('c_median')
            if c_med is None or not (shift_min <= c_med <= shift_max):
                continue

            # Optional cut_type filter
            if cut_type:
                ct_norm = cut_type.strip()
                if not any(ct_norm in ct for ct in rec.get('cut_types', [])):
                    continue

            results.append({
                'smiles':    smi,
                'c_median':  round(c_med, 1),
                'c_p25':     round(rec['c_p25'], 1) if rec.get('c_p25') is not None else None,
                'c_p75':     round(rec['c_p75'], 1) if rec.get('c_p75') is not None else None,
                'count':     rec.get('count', 0),
                'cut_types': rec.get('cut_types', []),
            })

    # Sort by count (most common first), then by proximity to center of range
    center = (shift_min + shift_max) / 2.0
    results.sort(key=lambda r: (-r['count'], abs(r['c_median'] - center)))
    results = results[:top_k]

    if not results:
        return {
            "observation": (f"No fragments with median C-shift in [{shift_min}, {shift_max}] ppm "
                            f"after filtering{' by cut_type=' + cut_type if cut_type else ''}."),
            "fragments": [],
        }

    # Format observation
    lines = [
        f"Found {len(results)} fragments with C-shift in [{shift_min}, {shift_max}] ppm"
        + (f" (cut_type filter: {cut_type})" if cut_type else "") + ":",
        "",
    ]
    for i, r in enumerate(results, 1):
        c_range = (f"C: {r['c_p25']}–{r['c_p75']} ppm (median {r['c_median']})"
                   if r['c_p25'] is not None else f"C: {r['c_median']} ppm")
        lines.append(f"  {i:2d}. {r['smiles'][:60]}")
        lines.append(f"      {c_range}, occurs {r['count']:,}x")
        cts = r['cut_types'][:4]
        lines.append(f"      cut_types: {', '.join(cts)}" +
                     (f" +{len(r['cut_types'])-4} more" if len(r['cut_types']) > 4 else ""))
        lines.append("")

    lines.append("To use: pass a fragment SMILES to nmr_mol_edit(action='replace', "
                 "group='<old_fragment>><new_fragment>').")

    return {
        "observation": "\n".join(lines),
        "fragments":   results,
    }
