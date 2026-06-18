from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import lmdb
import numpy as np

from .nmr_models.search_library import compute_cosine_similarity, get_canonical_smiles, smiles_to_formula
from .shared_lmdb import open_shared_lmdb

logger = logging.getLogger(__name__)


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


class GaussianSearcher:
    """
    Retrieval backend that works directly in the gaussian / nmr2vector space.

    - non-formula: global FAISS ANN on gaussian vectors
    - formula: exact scan inside one formula bucket using gaussian vectors already
      stored in the LMDB
    """

    def __init__(self, retrieval_dir: str | Path, enable_formula_filter: bool = False):
        self.retrieval_dir = Path(retrieval_dir)
        self.enable_formula_filter = enable_formula_filter

        self.lmdb_path = os.environ.get(
            "NMR_RETRIEVAL_LMDB",
            str(self.retrieval_dir / "embeddings_with_smiles.lmdb"),
        )
        if not Path(self.lmdb_path).exists():
            alt_lmdb = self.retrieval_dir / "embeddings_with_smiles_gaussian.lmdb"
            if alt_lmdb.exists():
                self.lmdb_path = str(alt_lmdb)

        self.faiss_index_path = os.environ.get(
            "NMR_GAUSSIAN_FAISS_INDEX_PATH",
            str(self.retrieval_dir / "gaussian_faiss.index"),
        )
        self.id_map_path = os.environ.get(
            "NMR_GAUSSIAN_ID_MAP_PATH",
            str(self.retrieval_dir / "gaussian_id_mapping.pkl"),
        )
        self.library_id_map_path = os.environ.get(
            "NMR_ID_MAP_PATH",
            str(self.retrieval_dir / "id_mapping.pkl"),
        )
        self.formula_map_path = os.environ.get(
            "NMR_FORMULA_MAP_PATH",
            str(self.retrieval_dir / "formula_to_ids_new.pkl"),
        )

        for path, name in [
            (self.lmdb_path, "LMDB"),
            (self.id_map_path, "Gaussian ID mapping"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        self.has_faiss_index = Path(self.faiss_index_path).exists()
        if not self.has_faiss_index and not enable_formula_filter:
            raise FileNotFoundError(f"Gaussian Faiss index not found: {self.faiss_index_path}")

        self.index = None
        with open(self.id_map_path, "rb") as f:
            self.id_to_raw_id = pickle.load(f)
        if Path(self.library_id_map_path).exists():
            with open(self.library_id_map_path, "rb") as f:
                self.id_to_key = pickle.load(f)
        else:
            self.id_to_key = {}
        self.lmdb_env = open_shared_lmdb(self.lmdb_path, readonly=True, lock=False, readahead=False, subdir=False)

        self.formula_to_ids = None
        if enable_formula_filter:
            if not Path(self.formula_map_path).exists():
                raise FileNotFoundError(f"Formula mapping not found: {self.formula_map_path}")
            with open(self.formula_map_path, "rb") as f:
                self.formula_to_ids = pickle.load(f)

    def _ensure_index(self):
        if self.index is None:
            if not self.has_faiss_index:
                raise RuntimeError("Gaussian Faiss index is not available.")
            self.index = faiss.read_index(self.faiss_index_path)
        return self.index

    def encode_spectrum(self, spectrum: dict) -> np.ndarray:
        h = spectrum.get("H_nmr") or []
        c = spectrum.get("C_nmr") or []
        q = _vector_encoding(
            h_shifts=np.array(h, dtype=np.float32) if h else np.array([], dtype=np.float32),
            c_shifts=np.array(c, dtype=np.float32) if c else np.array([], dtype=np.float32),
            normalize=True,
            padding=True,
        )[0]
        return q.reshape(1, -1).astype(np.float32)

    def _candidate_keys(self, library_id: int) -> list[bytes]:
        keys: list[bytes] = []

        mapped_key = self.id_to_key.get(int(library_id))
        if mapped_key is not None:
            key_bytes = mapped_key if isinstance(mapped_key, bytes) else str(mapped_key).encode()
            if key_bytes not in keys:
                keys.append(key_bytes)

        fallback_key = str(int(library_id)).encode()
        if fallback_key not in keys:
            keys.append(fallback_key)

        return keys

    def _read_record(self, library_id: int, txn=None) -> tuple[Optional[dict], Optional[str]]:
        manage_txn = txn is None
        if manage_txn:
            txn = self.lmdb_env.begin()

        try:
            for key in self._candidate_keys(library_id):
                raw = txn.get(key)
                if raw is None:
                    continue
                try:
                    return pickle.loads(raw), key.decode() if isinstance(key, bytes) else str(key)
                except Exception:
                    continue
            return None, None
        finally:
            if manage_txn:
                txn.abort()

    def search(self, query_spectrum, k=10, nprobe=128, query_smiles=None, disable_formula_filter=False):
        query_vec = self.encode_spectrum(query_spectrum)

        if self.enable_formula_filter and not disable_formula_filter:
            if query_smiles is None:
                raise ValueError("query_smiles is required when formula filter is enabled")
            query_formula = smiles_to_formula(query_smiles)
            if query_formula is None:
                raise ValueError(f"Failed to parse SMILES: {query_smiles}")

            candidate_ids = self.formula_to_ids.get(query_formula, [])
            if len(candidate_ids) == 0:
                return [], False

            query_canonical = get_canonical_smiles(query_smiles)
            candidate_vectors = []
            candidate_data = []
            target_in_library = False
            with self.lmdb_env.begin() as txn:
                for library_id in candidate_ids:
                    data, resolved_key = self._read_record(int(library_id), txn=txn)
                    if data is None:
                        continue
                    vec = data.get("gaussian_vector")
                    if vec is None:
                        continue
                    arr = np.asarray(vec, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    candidate_vectors.append(arr)
                    candidate_data.append({"data": data, "db_id": int(library_id), "db_key": resolved_key or str(int(library_id))})

                    cand_smiles = data.get("smiles")
                    if cand_smiles:
                        cand_can = get_canonical_smiles(cand_smiles)
                        if cand_can and cand_can == query_canonical:
                            target_in_library = True

            if not candidate_vectors:
                return [], target_in_library

            candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)
            similarities = compute_cosine_similarity(query_vec, candidate_vectors)
            top_indices = np.argsort(similarities)[::-1][:k]
            results = []
            for idx in top_indices:
                row = candidate_data[idx]
                data = row["data"]
                results.append(
                    {
                        "cosine_similarity": float(similarities[idx]),
                        "H_nmr": data.get("H_nmr", []),
                        "C_nmr": data.get("C_nmr", []),
                        "smiles": data.get("smiles", None),
                        "canonical_smiles": data.get("canonical_smiles", get_canonical_smiles(data.get("smiles"))),
                        "formula": data.get("formula", ""),
                        "db_id": row["db_id"],
                        "db_key": row["db_key"],
                    }
                )
            return results, target_in_library

        index = self._ensure_index()

        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        search_k = min(k * 10, index.ntotal)
        distances, indices = index.search(query_vec, search_k)

        results = []
        for rank_pos, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            library_id = int(self.id_to_raw_id[int(idx)])
            data, resolved_key = self._read_record(library_id)
            if data is None:
                continue
            results.append(
                {
                    "cosine_similarity": float(distances[0][rank_pos]),
                    "H_nmr": data.get("H_nmr", []),
                    "C_nmr": data.get("C_nmr", []),
                        "smiles": data.get("smiles", None),
                        "canonical_smiles": data.get("canonical_smiles", get_canonical_smiles(data.get("smiles"))),
                        "formula": data.get("formula", ""),
                        "db_id": library_id,
                        "db_key": resolved_key or str(library_id),
                    }
                )
            if len(results) >= k:
                break
        return results, None
