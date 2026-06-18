"""
NMR-Based Molecular Structure Retrieval Tool (Updated with Formula Filter).

This tool uses NMRSearcher from search_library.py to retrieve similar structures
with optional molecular formula filtering.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional
import warnings
import json
import urllib.request
import urllib.error

from .decorator import tool
from .path_utils import artifact_path
from .pool_store import save_pool

# Suppress warnings
warnings.filterwarnings('ignore')

# ============== NMRSearcher Backend ==============

_SEARCHER = None
_GAUSSIAN_SEARCHER = None
_SEARCHER_AVAILABLE = False
_GAUSSIAN_SEARCHER_AVAILABLE = False
_SERVICE_URL = os.environ.get("NMR_RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8011").rstrip("/")
_SERVICE_TIMEOUT_S = float(os.environ.get("NMR_RETRIEVAL_SERVICE_TIMEOUT_S", "120.0"))
_DISABLE_SERVICE = os.environ.get("NMR_RETRIEVAL_DISABLE_SERVICE", "").strip().lower() in {"1", "true", "yes"}
_ENABLE_VECTOR_RERANK = os.environ.get("NMR_RETRIEVAL_VECTOR_RERANK", "1").strip().lower() not in {"0", "false", "no"}
_RERANK_POOL_FACTOR = max(1, int(os.environ.get("NMR_RETRIEVAL_RERANK_POOL_FACTOR", "3")))
_RERANK_POOL_CAP = max(1, int(os.environ.get("NMR_RETRIEVAL_RERANK_POOL_CAP", "3000")))
_GAUSSIAN_EMBEDDING_BACKENDS = {"gaussian_then_embedding", "hybrid_gaussian_embedding"}

try:
    # Import NMRSearcher from local nmr_models package
    from .nmr_models.search_library import NMRSearcher

    _SEARCHER_AVAILABLE = True
    print("[NMR Retrieval Tool] NMRSearcher available")
except Exception as e:
    print(f"[NMR Retrieval Tool] NMRSearcher not available: {e}")
    import traceback
    traceback.print_exc()

try:
    from .gaussian_search_library import GaussianSearcher

    _GAUSSIAN_SEARCHER_AVAILABLE = True
    print("[NMR Retrieval Tool] GaussianSearcher available")
except Exception as e:
    print(f"[NMR Retrieval Tool] GaussianSearcher not available: {e}")


def _set2vec(values, nmr_type: str, dim: int = 128, normalize: bool = True, sigma: float | None = None) -> np.ndarray:
    if isinstance(values, np.ndarray):
        values = [values]
    assert isinstance(values, list)
    if nmr_type == "H":
        nmr_range = (-1.0, 15.0)
        sigma = sigma or 0.3
    else:
        nmr_range = (-10.0, 230.0)
        sigma = sigma or 2.0
    ni = np.linspace(nmr_range[0], nmr_range[1], dim)
    interval = ni[1] - ni[0]
    coef = interval / (np.sqrt(2 * np.pi) * sigma)
    result = [coef * np.exp(-(np.abs(item[:, np.newaxis] - ni) / sigma) ** 2 / 2).sum(axis=0) for item in values]
    result = np.array(result, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = result / norms
    return result.astype(np.float32)


def _vector_encoding(h_shifts=None, c_shifts=None, dim: int = 128, normalize: bool = True, padding: bool = True) -> np.ndarray:
    if h_shifts is None and c_shifts is None:
        raise ValueError("Both h_shifts and c_shifts are None")
    dim_pad = dim if padding else 0
    if isinstance(h_shifts, np.ndarray) or isinstance(c_shifts, np.ndarray):
        h_vec = _set2vec(h_shifts, nmr_type="H", dim=dim, normalize=normalize) if h_shifts is not None else np.zeros((1, dim_pad), dtype=np.float32)
        c_vec = _set2vec(c_shifts, nmr_type="C", dim=dim, normalize=normalize) if c_shifts is not None else np.zeros((1, dim_pad), dtype=np.float32)
    else:
        raise TypeError("Expected numpy arrays for h_shifts/c_shifts")
    return np.concatenate([h_vec, c_vec], axis=1).astype(np.float32)


def _extract_h_values(h_obj) -> np.ndarray:
    if isinstance(h_obj, np.ndarray):
        return h_obj.astype(np.float32)
    if isinstance(h_obj, list):
        if not h_obj:
            return np.array([], dtype=np.float32)
        if isinstance(h_obj[0], dict):
            vals = [float(x["shift"]) for x in h_obj if "shift" in x]
            return np.array(vals, dtype=np.float32)
        return np.array([float(x) for x in h_obj], dtype=np.float32)
    return np.array([], dtype=np.float32)


def _extract_c_values(c_obj) -> np.ndarray:
    if isinstance(c_obj, np.ndarray):
        return c_obj.astype(np.float32)
    if isinstance(c_obj, list):
        if not c_obj:
            return np.array([], dtype=np.float32)
        return np.array([float(x) for x in c_obj], dtype=np.float32)
    return np.array([], dtype=np.float32)


def _compute_cosine_similarity(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if candidate_vecs.ndim == 1:
        candidate_vecs = candidate_vecs.reshape(1, -1)

    query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
    candidate_norm = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-8)
    return np.dot(query_norm, candidate_norm.T).flatten()


def _expand_rerank_pool_top_k(top_k: int) -> int:
    return min(max(top_k, top_k * _RERANK_POOL_FACTOR), _RERANK_POOL_CAP)


def _uses_gaussian_then_embedding_backend(backend_mode: str) -> bool:
    return backend_mode in _GAUSSIAN_EMBEDDING_BACKENDS


def _vector_rerank_results(results: list[dict], h_list: list[float], c_list: list[float]) -> list[dict]:
    if not results or (not h_list and not c_list):
        return results

    query_vec = _vector_encoding(
        h_shifts=np.array(h_list, dtype=np.float32) if h_list else np.array([], dtype=np.float32),
        c_shifts=np.array(c_list, dtype=np.float32) if c_list else np.array([], dtype=np.float32),
        normalize=True,
        padding=True,
    )[0]

    reranked = []
    for row in results:
        h = _extract_h_values(row.get("H_nmr", []))
        c = _extract_c_values(row.get("C_nmr", []))

        if h.size == 0 and c.size == 0:
            vector_score = float(row.get("cosine_similarity", 0.0))
            fallback = True
        else:
            cand_vec = _vector_encoding(h_shifts=h, c_shifts=c, normalize=True, padding=True)[0]
            vector_score = float(np.dot(query_vec, cand_vec))
            fallback = False

        new_row = dict(row)
        new_row["retrieval_similarity"] = float(row.get("cosine_similarity", 0.0))
        new_row["vector_similarity"] = vector_score
        new_row["vector_rerank_fallback_to_retrieval"] = fallback
        reranked.append(new_row)

    reranked.sort(key=lambda x: x["vector_similarity"], reverse=True)
    for i, row in enumerate(reranked, start=1):
        row["vector_rank"] = i
    return reranked


def _embedding_rerank_results(results: list[dict], h_list: list[float], c_list: list[float]) -> list[dict]:
    if not results or (not h_list and not c_list):
        return results

    query_spectrum = {
        "H_nmr": h_list,
        "C_nmr": c_list,
    }
    searcher = get_searcher()
    query_embedding = searcher.encode_spectrum(query_spectrum).astype(np.float32)

    reranked = []
    with searcher.lmdb_env.begin() as txn:
        for row in results:
            retrieval_score = float(row.get("retrieval_similarity", row.get("cosine_similarity", 0.0)))
            candidate_payload = None
            candidate_keys = []

            db_key = row.get("db_key")
            if db_key is not None and str(db_key).strip():
                candidate_keys.append(db_key if isinstance(db_key, bytes) else str(db_key).encode())

            db_id = row.get("db_id")
            if db_id is not None:
                try:
                    lmdb_key = searcher.id_to_key.get(int(db_id))
                except Exception:
                    lmdb_key = None
                if lmdb_key is not None and lmdb_key not in candidate_keys:
                    candidate_keys.append(lmdb_key)

            for candidate_key in candidate_keys:
                raw = txn.get(candidate_key)
                if raw is None:
                    continue
                try:
                    candidate_payload = pickle.loads(raw)
                    break
                except Exception:
                    continue

            embedding_score = retrieval_score
            fallback = True
            if candidate_payload is not None:
                candidate_embedding = candidate_payload.get("embedding")
                if candidate_embedding is not None:
                    candidate_embedding = np.asarray(candidate_embedding, dtype=np.float32)
                    if candidate_embedding.ndim == 1 and candidate_embedding.size > 0:
                        embedding_score = float(_compute_cosine_similarity(query_embedding, candidate_embedding)[0])
                        fallback = False

            new_row = dict(row)
            new_row["retrieval_similarity"] = retrieval_score
            new_row["gaussian_similarity"] = retrieval_score
            new_row["embedding_similarity"] = embedding_score
            new_row["embedding_rerank_fallback_to_retrieval"] = fallback
            reranked.append(new_row)

    reranked.sort(key=lambda x: x["embedding_similarity"], reverse=True)
    for i, row in enumerate(reranked, start=1):
        row["embedding_rank"] = i
    return reranked


def get_searcher():
    """Get or initialize the NMRSearcher instance."""
    global _SEARCHER

    if not _SEARCHER_AVAILABLE:
        raise RuntimeError("NMRSearcher not available")

    if _SEARCHER is not None:
        return _SEARCHER

    try:
        # Initialize NMRSearcher with UltraNMR checkpoint
        retrieval_dir = os.environ.get(
            "NMR_CKPT_DIR",
            str(artifact_path("checkpoints")),
        )
        model_checkpoint = os.environ.get(
            "NMR_SEARCH_CKPT",
            os.path.join(retrieval_dir, "search.pth"),
        )

        print(f"[NMR Retrieval Tool] Initializing NMRSearcher...")
        print(f"  Model: {model_checkpoint}")
        print(f"  Data: {retrieval_dir}")

        _SEARCHER = NMRSearcher(
            model_checkpoint=model_checkpoint,
            retrieval_dir=retrieval_dir,
            embedding_dim=768,
            enable_formula_filter=True  # Enable formula filter
        )

        print("[NMR Retrieval Tool] NMRSearcher initialized successfully")
        return _SEARCHER

    except Exception as e:
        print(f"[NMR Retrieval Tool] Failed to initialize NMRSearcher: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_gaussian_searcher():
    """Get or initialize the GaussianSearcher instance."""
    global _GAUSSIAN_SEARCHER

    if not _GAUSSIAN_SEARCHER_AVAILABLE:
        raise RuntimeError("GaussianSearcher not available")

    if _GAUSSIAN_SEARCHER is not None:
        return _GAUSSIAN_SEARCHER

    retrieval_dir = os.environ.get(
        "NMR_CKPT_DIR",
        str(artifact_path("checkpoints")),
    )
    _GAUSSIAN_SEARCHER = GaussianSearcher(
        retrieval_dir=retrieval_dir,
        enable_formula_filter=True,
    )
    return _GAUSSIAN_SEARCHER


def _call_retrieval_service(payload: dict) -> Optional[dict]:
    """Call the persistent retrieval service when available."""
    if not _SERVICE_URL or _DISABLE_SERVICE:
        return None
    url = f"{_SERVICE_URL}/retrieve"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_SERVICE_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
        out = json.loads(raw)
        if isinstance(out, dict) and "results" in out:
            return out
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    except Exception:
        return None
    return None


@tool(
    name="nmr_retrieve",
    description=(
        "Retrieve similar molecular structures from the NMR database with optional formula filtering. "
        "Returns candidates already ranked by retrieval similarity score. "
        "Use these ranked results directly as a primary candidate list; do not automatically call rerank "
        "unless additional NMR-based discrimination is still needed."
    )
)
def nmr_retrieve_tool(
    h_shifts: str,
    c_shifts: str,
    formula: str = "",
    query_smiles: str = "",
    top_k: int = 10,
    nprobe: int = 128,
    retrieval_mode: str = "auto",
    backend_mode: str = "embedding",
    save_pool_file: bool = False,
    pool_path: str = "",
) -> dict:
    """
    Retrieve similar molecular structures from a large database (158M+ structures).

    This tool:
    1. Encodes the input NMR spectra using a pretrained NMR-BERT encoder
    2. Searches a database of 158 million molecular structures
    3. Returns the top-k most similar structures already ranked by cosine similarity
    4. Optionally filters by molecular formula for more accurate results

    Args:
        h_shifts (str): Comma-separated H-NMR chemical shifts (ppm), e.g., "7.3, 7.2, 2.3"
        c_shifts (str): Comma-separated C-NMR chemical shifts (ppm), e.g., "138.0, 129.0, 21.0"
        formula (str): Molecular formula constraint, e.g., "C9H12O2". If provided,
                       only structures with this exact formula will be returned. Leave empty for no constraint.
        query_smiles (str): Optional anchor SMILES used only to derive the formula-space
                            in the backend retrieval system. This is useful in benchmark
                            settings where a ground-truth structure is available and does
                            not change the ranking logic beyond selecting the formula-matched
                            search space.
        top_k (int): Number of top similar structures to return (default: 10, max: 20)
        nprobe (int): Search parameter for accuracy vs speed tradeoff (default: 128)
                      Higher values = more accurate but slower. Only used when formula is not provided.
        retrieval_mode (str): Retrieval mode - "auto" (default), "formula_only", "non_formula", or "mixed"
                             - "auto": Use formula filter if provided, otherwise search all
                             - "formula_only": Only return formula-matched candidates
                             - "non_formula": Ignore formula, return top-k from all database
                             - "mixed": Return top-k formula-matched + top-k non-formula
        backend_mode (str): Retrieval backend mode.
                            - "embedding": embedding retrieval, optionally followed by gaussian vector rerank
                            - "gaussian": gaussian / nmr2vector retrieval only
                            - "gaussian_then_embedding": gaussian retrieval followed by embedding rerank

    Returns:
        dict: Dictionary containing:
            - observation: Human-readable summary of results
            - valid: 1 if successful, 0 if failed
            - results: List of similar structures, each with:
                * smiles: SMILES string
                * formula: Molecular formula
                * similarity: Cosine similarity score (0-1, higher is better)
              The list is already sorted by retrieval score, so it can often be
              used directly without an immediate rerank step.
            - num_results: Number of results returned
            - scanned: Number of database entries scanned
            - formula_filter: Whether formula filter was used

    Example:
        >>> nmr_retrieve_tool(
        ...     h_shifts="7.3, 7.2, 2.3",
        ...     c_shifts="138.0, 129.0, 21.0",
        ...     formula="C9H12O2",
        ...     top_k=5
        ... )
        {
            "observation": "Retrieved 5 structures with formula C9H12O2...",
            "valid": 1,
            "results": [
                {"smiles": "COCCOc1ccccc1", "formula": "C9H12O2", "similarity": 0.95},
                ...
            ],
            "num_results": 5,
            "formula_filter": True
        }
    """
    try:
        # Parse input (accept list or comma-separated string)
        h_list = h_shifts if isinstance(h_shifts, list) else [float(x.strip()) for x in h_shifts.split(',') if x.strip()]
        c_list = c_shifts if isinstance(c_shifts, list) else [float(x.strip()) for x in c_shifts.split(',') if x.strip()]
        h_list = [float(x) for x in h_list]
        c_list = [float(x) for x in c_list]

        if not h_list and not c_list:
            return {
                "observation": "Error: At least one of H-NMR or C-NMR shifts must be provided.",
                "valid": 0,
                "results": [],
                "num_results": 0
            }

        # Limit top_k (increased to 1000 for fragment assembly)
        top_k = min(max(1, top_k), 1000)
        rerank_pool_top_k = _expand_rerank_pool_top_k(top_k) if _uses_gaussian_then_embedding_backend(backend_mode) else top_k
        search_backend_mode = "gaussian" if _uses_gaussian_then_embedding_backend(backend_mode) else backend_mode

        if search_backend_mode == "embedding":
            service_result = _call_retrieval_service(
                {
                    "h_shifts": h_list,
                    "c_shifts": c_list,
                    "formula": formula,
                    "query_smiles": query_smiles,
                    "top_k": top_k,
                    "nprobe": nprobe,
                    "retrieval_mode": retrieval_mode,
                }
            )
            if service_result is not None:
                return service_result

        # Get searcher
        if search_backend_mode == "gaussian":
            searcher = get_gaussian_searcher()
        else:
            searcher = get_searcher()

        # Prepare query spectrum
        query_spectrum = {
            'H_nmr': h_list,
            'C_nmr': c_list
        }

        # Search strategy based on retrieval_mode
        if retrieval_mode == "mixed":
            # Mixed mode: get formula-matched + non-formula
            results_formula = []
            results_no_formula = []
            formula_search_k = max(top_k * 3, rerank_pool_top_k)
            nonformula_search_k = rerank_pool_top_k

            # Try formula search
            try:
                query_smiles = None
                if formula and formula.strip():
                    query_smiles = _get_smiles_for_formula(searcher, formula)

                if query_smiles:
                    print(f"[NMR Retrieval Tool] Mixed mode: searching {formula_search_k} with formula filter")
                    results_formula, _ = searcher.search(
                        query_spectrum=query_spectrum,
                        k=formula_search_k,
                        nprobe=nprobe,
                        query_smiles=query_smiles
                    )
            except:
                pass

            # Always do non-formula search
            print(f"[NMR Retrieval Tool] Mixed mode: searching {nonformula_search_k} without formula filter")
            results_no_formula, _ = searcher.search(
                query_spectrum=query_spectrum,
                k=nonformula_search_k,
                nprobe=nprobe,
                query_smiles=None,
                disable_formula_filter=True
            )

            # Combine results
            results = results_formula + results_no_formula

        elif retrieval_mode == "non_formula":
            # Non-formula mode: ignore formula completely
            print(f"[NMR Retrieval Tool] Non-formula mode: searching {rerank_pool_top_k} candidates")
            results, _ = searcher.search(
                query_spectrum=query_spectrum,
                k=rerank_pool_top_k,
                nprobe=nprobe,
                query_smiles=None,
                disable_formula_filter=True
            )

        else:
            # Auto or formula_only mode.
            # Keep formula as a hard constraint when provided. We only search in
            # the formula-compatible space. If that space cannot be resolved, we
            # return no formula-matched candidates instead of falling back to a
            # broader non-formula search.
            search_k = top_k * 3 if formula and formula.strip() else top_k
            search_k = max(search_k, rerank_pool_top_k)
            if formula and formula.strip():
                anchor_smiles = query_smiles.strip() if isinstance(query_smiles, str) else ""
                if not anchor_smiles:
                    anchor_smiles = _get_smiles_for_formula(searcher, formula)
                if anchor_smiles is None or not str(anchor_smiles).strip():
                    return {
                        "observation": (
                            f"Retrieved 0 formula-matched + 0 similar structures:\n"
                            f"Query: {len(h_list)} H peaks, {len(c_list)} C peaks\n\n"
                            f"No retrieval anchor was found for formula {formula}. "
                            "The formula-constrained search space returned no candidates."
                        ),
                        "valid": 1,
                        "results": [],
                        "num_results": 0,
                        "matched": 0,
                        "unmatched": 0,
                    }

            print(f"[NMR Retrieval Tool] Searching {search_k} candidates")
            results, target_in_library = searcher.search(
                query_spectrum=query_spectrum,
                k=search_k,
                nprobe=nprobe,
                query_smiles=anchor_smiles if formula and formula.strip() else None
            )

        if _uses_gaussian_then_embedding_backend(backend_mode):
            try:
                results = _embedding_rerank_results(results, h_list, c_list)
            except Exception as rerank_err:
                print(f"[NMR Retrieval Tool] Embedding rerank failed after gaussian retrieval, falling back to retrieval order: {rerank_err}")
        elif _ENABLE_VECTOR_RERANK and search_backend_mode != "gaussian":
            try:
                results = _vector_rerank_results(results, h_list, c_list)
            except Exception as rerank_err:
                print(f"[NMR Retrieval Tool] Vector rerank failed, falling back to retrieval order: {rerank_err}")

        # Format and classify results
        matched, unmatched = [], []
        for res in results:
            item = {
                'smiles': res['smiles'],
                'canonical_smiles': res.get('canonical_smiles', ''),
                'formula': res.get('formula', ''),
                'similarity': float(res.get('embedding_similarity', res.get('vector_similarity', res['cosine_similarity']))),
                'retrieval_similarity': float(res.get('retrieval_similarity', res['cosine_similarity'])),
                'vector_similarity': float(res.get('vector_similarity', res['cosine_similarity'])),
                'gaussian_similarity': float(res.get('gaussian_similarity', res.get('retrieval_similarity', res['cosine_similarity']))),
                'embedding_similarity': float(res.get('embedding_similarity', res.get('retrieval_similarity', res['cosine_similarity']))),
                'vector_rank': res.get('vector_rank'),
                'embedding_rank': res.get('embedding_rank'),
                'vector_rerank_fallback_to_retrieval': bool(res.get('vector_rerank_fallback_to_retrieval', False)),
                'embedding_rerank_fallback_to_retrieval': bool(res.get('embedding_rerank_fallback_to_retrieval', False)),
                'db_id': res.get('db_id'),
                'db_key': res.get('db_key'),
                'H_nmr': res.get('H_nmr', []),
                'C_nmr': res.get('C_nmr', []),
            }
            if formula and formula.strip():
                if item['formula'] == formula:
                    matched.append(item)
                else:
                    unmatched.append(item)
            else:
                matched.append(item)

        # Prioritize matched, fallback to unmatched if needed
        if retrieval_mode == "mixed":
            # Mixed mode: return all matched + top_k unmatched
            output_items = matched + unmatched[:top_k]
        elif retrieval_mode == "formula_only":
            # Hard formula-only mode: never backfill with formula-mismatched hits.
            output_items = matched[:top_k]
        else:
            # Auto/formula_only: prioritize matched, fill with unmatched
            output_items = matched[:top_k]
            if len(output_items) < top_k:
                output_items.extend(unmatched[:top_k - len(output_items)])

        # Format observation
        obs_lines = []
        if formula:
            obs_lines.append(f"Retrieved {len(matched)} formula-matched + {len(unmatched)} similar structures:")
        else:
            obs_lines.append(f"Retrieved {len(output_items)} similar structures:")
        obs_lines.append(f"Query: {len(h_list)} H peaks, {len(c_list)} C peaks")
        obs_lines.append("")

        for i, res in enumerate(output_items, 1):
            mark = "✓" if res in matched else "✗"
            obs_lines.append(f"{i}. {mark} {res['smiles']}")

        obs_lines.append("")
        obs_lines.append(f"✓ = formula match, ✗ = similar but different formula")

        response = {
            "observation": "\n".join(obs_lines),
            "valid": 1,
            "results": output_items,
            "num_results": len(output_items),
            "matched": len(matched),
            "unmatched": len(unmatched)
        }
        if save_pool_file:
            response["pool_path"] = save_pool(
                output_items,
                prefix="retrieve_pool",
                query={"formula": formula, "h_shifts": h_list, "c_shifts": c_list},
                metadata={"tool": "nmr_retrieve", "retrieval_mode": retrieval_mode, "backend_mode": backend_mode},
                path=pool_path,
            )
        return response

    except Exception as e:
        import traceback
        return {
            "observation": f"Error in NMR retrieval: {str(e)}\n{traceback.format_exc()}",
            "valid": 0,
            "results": [],
            "num_results": 0
        }


def _get_smiles_for_formula(searcher, formula: str) -> Optional[str]:
    """
    Get any SMILES from the database that has the given formula.
    This is used to enable formula filtering.

    Args:
        searcher: NMRSearcher instance
        formula: Molecular formula like "C9H12O2"

    Returns:
        A SMILES string with that formula from the database, or None if not found
    """
    try:
        # Check if this formula exists in the database
        if not hasattr(searcher, 'formula_to_ids'):
            return None

        candidate_ids = searcher.formula_to_ids.get(formula, [])
        if len(candidate_ids) == 0:
            return None

        # Get the first candidate's SMILES
        import pickle
        with searcher.lmdb_env.begin() as txn:
            for cand_id in candidate_ids[:10]:  # Try first 10
                try:
                    lmdb_key = searcher.id_to_key.get(cand_id)
                    if lmdb_key is None:
                        continue

                    value = txn.get(lmdb_key)
                    if value is None:
                        continue

                    data = pickle.loads(value)
                    smiles = data.get('smiles')
                    if smiles:
                        return smiles
                except:
                    continue

        return None

    except Exception as e:
        print(f"[NMR Retrieval Tool] Failed to get SMILES for formula: {e}")
        return None


def _generate_dummy_smiles_for_formula(formula: str) -> Optional[str]:
    """
    Generate a dummy SMILES string that has the given molecular formula.
    This is used to enable formula filtering in the searcher.

    Args:
        formula: Molecular formula like "C9H12O2"

    Returns:
        A valid SMILES string with that formula, or None if invalid
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        # Simple approach: just return a basic alkane and let the searcher extract formula
        # The searcher only needs the formula, not the actual structure
        # We can create a simple molecule and add hydrogens to match

        # Parse formula to get carbon count
        import re
        c_match = re.search(r'C(\d*)', formula)
        if c_match:
            c_count = int(c_match.group(1)) if c_match.group(1) else 1
        else:
            c_count = 0

        # Create a simple alkane chain
        if c_count > 0:
            dummy_smiles = 'C' * c_count
            mol = Chem.MolFromSmiles(dummy_smiles)
            if mol:
                # Add explicit hydrogens
                mol = Chem.AddHs(mol)
                # Get the formula
                actual_formula = rdMolDescriptors.CalcMolFormula(mol)

                # If it matches, great! Otherwise just return the chain
                # The searcher will extract the formula from query_smiles
                return dummy_smiles

        # Fallback: just return methane
        return "C"

    except Exception as e:
        print(f"[NMR Retrieval Tool] Failed to generate dummy SMILES: {e}")
        return "C"  # Return methane as fallback


# For backward compatibility
nmr_retrieve = nmr_retrieve_tool
