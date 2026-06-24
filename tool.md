# Tools

This file documents the two currently validated tools in this repository:
- `nmr_retrieve`
- `nmr_denovo`

Both tools were debugged successfully in the standalone `NMRAgent` repository with the centralized runtime asset config.

## 1. `nmr_retrieve`

Purpose:
Retrieve similar molecular structures from the NMR retrieval database. This is a database-backed tool and is usually the first high-recall candidate generator.

Implementation:
- Source: `tools/nmr_retrieval_tool.py`
- Exported tool name: `nmr_retrieve`
- Python function: `nmr_retrieve_tool(...)`

### Parameters

- `h_shifts: str`
  1H NMR chemical shifts as a comma-separated string.
  Example: `"7.3, 7.2, 2.3"`

- `c_shifts: str`
  13C NMR chemical shifts as a comma-separated string.
  Example: `"138.0, 129.0, 21.0"`

- `formula: str = ""`
  Optional molecular formula constraint.
  If provided, retrieval stays in the exact formula space when possible.
  Example: `"C20H30O3"`

- `query_smiles: str = ""`
  Optional anchor SMILES. Mainly useful in benchmark or controlled settings where a known formula-space anchor is available.
  In ordinary inference, this is usually left empty.

- `top_k: int = 10`
  Number of candidates to return.
  Internal code currently clamps this to `1..1000`.

- `nprobe: int = 128`
  FAISS search parameter.
  Higher values can improve recall but increase latency.
  Mainly relevant in non-formula retrieval.

- `retrieval_mode: str = "auto"`
  Supported values:
  - `"auto"`: use formula constraint if available
  - `"formula_only"`: only formula-matched candidates
  - `"non_formula"`: ignore formula and search globally
  - `"mixed"`: combine formula and non-formula retrieval

- `backend_mode: str = "embedding"`
  Supported values:
  - `"embedding"`: embedding-based retrieval
  - `"gaussian"`: gaussian / nmr2vector retrieval only
  - `"gaussian_then_embedding"`: gaussian retrieval followed by embedding rerank

### Returns

Returns a `dict` with these main fields:

- `observation: str`
  Human-readable summary of retrieved results.

- `valid: int`
  `1` means success, `0` means failure.

- `results: list[dict]`
  Candidate list. Common fields include:
  - `smiles`
  - `canonical_smiles`
  - `formula`
  - `similarity`
  - `retrieval_similarity`
  - `vector_similarity`
  - `db_id`
  - `db_key`
  - `H_nmr`
  - `C_nmr`

- `num_results: int`
  Number of results returned.

### Notes

- This tool is heavy on first cold start.
- In the current local setup it loads a large FAISS index, ID mapping, LMDB, and formula index.
- For formula-constrained structure elucidation, `retrieval_mode="auto"` with `formula` filled is the normal setting.

### Example

```python
from tools.nmr_retrieval_tool import nmr_retrieve_tool

result = nmr_retrieve_tool(
    h_shifts="5.68, 3.19, 2.98, 2.68, 2.59, 1.86, 1.77, 1.69, 1.57, 1.56, 1.44, 1.32, 1.23, 1.20, 1.11, 1.02, 0.94",
    c_shifts="210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51",
    formula="C20H30O3",
    top_k=10,
    retrieval_mode="auto",
    backend_mode="embedding",
)
```

## 2. `nmr_denovo`

Purpose:
Generate candidate SMILES directly from NMR spectra and formula using the de novo model. This is the model-based candidate generator and complements retrieval.

Implementation:
- Source: `tools/nmr_denovo_tool.py`
- Exported tool name: `nmr_denovo`
- Python function: `nmr_denovo(...)`

### Parameters

- `h_shifts: str`
  1H NMR chemical shifts as a comma-separated string.
  Example: `"7.3,7.2,2.3"`

- `c_shifts: str`
  13C NMR chemical shifts as a comma-separated string.
  Example: `"138.0,129.0,21.0"`

- `formula: str`
  Required molecular formula.
  Example: `"C20H30O3"`

- `top_k: int = 10`
  Number of top candidates to return.
  Internal code clamps this to `1..20`.

### Returns

Returns a `dict` with these main fields:

- `observation: str`
  Human-readable generation summary.

- `candidates: list[dict]`
  Candidate list. Common fields include:
  - `smiles`
  - `score`
  - `source`
  - `rank`

- `count: int`
  Number of final candidates returned.

### Notes

- This tool requires the formula.
- Internally it uses beam search.
- It prioritizes exact formula matches and then fills remaining slots with approximate candidates if needed.
- First cold start is also heavy because the model checkpoint is large.

### Example

```python
from tools.nmr_denovo_tool import nmr_denovo

result = nmr_denovo(
    h_shifts="5.68, 3.19, 2.98, 2.68, 2.59, 1.86, 1.77, 1.69, 1.57, 1.56, 1.44, 1.32, 1.23, 1.20, 1.11, 1.02, 0.94",
    c_shifts="210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51",
    formula="C20H30O3",
    top_k=10,
)
```

## 3. `nmr_rerank`

Purpose:
Rerank candidate SMILES by predicting their NMR spectra and comparing those predictions against the original query 1H / 13C shifts.

Implementation:
- Source: `tools/nmr_rerank_tool.py`
- Exported tool name: `nmr_rerank`
- Python function: `nmr_rerank(...)`

### What It Reranks

This tool is intended to rerank the outputs of:
- `nmr_retrieve`
- `nmr_denovo`

In practice, you usually:
1. run `nmr_retrieve` and/or `nmr_denovo`
2. collect the candidate `SMILES`
3. pass those candidates into `nmr_rerank`
4. let `nmr_rerank` predict candidate NMR shifts and compare them with the original query spectrum

Important:
- `nmr_rerank` does **not** need the internal hidden states or intermediate outputs of `nmr_denovo`
- `nmr_rerank` **does** need the original query `h_shifts` and `c_shifts`
- `nmr_rerank` takes candidate molecules from retrieval / denovo and rescores them against the query spectrum

### Required Inputs

- `h_shifts: str`
  Query 1H NMR shifts.

- `c_shifts: str`
  Query 13C NMR shifts.

- `candidates: str`
  JSON string of candidate molecules, usually produced by `nmr_retrieve` or `nmr_denovo`.
  Example:
  ```json
  [{"smiles": "...", "score": 0.91, "source": "retrieve"}]
  ```

### Optional Inputs

- `top_k: int = 10`
- `formula: str = ""`

### Returns

Main fields:
- `observation`
- `candidates`
- `count`

Per-candidate diagnostic fields now include:
- `nmr_similarity`
- `matched_peaks`
- `unmatched_query_peaks`
- `unused_predicted_peaks`
- `atom_level_assignment_summary`
- `atom_data`

### Dependency Relationship

Use this mental model:
- `nmr_retrieve`: database candidate generator
- `nmr_denovo`: model-based candidate generator
- `nmr_rerank`: candidate rescorer over retrieval / denovo outputs

So yes: `nmr_rerank` is the stage that reranks the results from `retrieval` and `denovo`.

## Current Debug Status

Validated from the repository root:
- `nmr_retrieve`: runnable
- `nmr_denovo`: runnable

Validated sample:
- Formula: `C20H30O3`
- 13C NMR: `210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51`
- 1H NMR: `5.68, 3.19-2.98, 2.68, 2.59, 1.86, 1.77, 1.69-1.57, 1.56-1.44, 1.32-1.23, 1.20, 1.11, 1.02, 0.94`

Observed outcomes:
- `nmr_retrieve` returned formula-matched candidates successfully.
- `nmr_denovo` returned valid de novo candidates successfully.
