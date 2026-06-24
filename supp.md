# Appendix: Implementation Details

## A. Multi-Agent System Overview

NMRAgent uses a three-role workflow for NMR structure elucidation:

1. Planner agent.
   The Planner is an LLM-only reasoning module. It reads the molecular formula, expanded 1H shifts, 13C shifts, previous executor/verifier outputs, and optional recalled memory. It does not call tools directly. Its output is a JSON execution plan.

2. Executor agent.
   The Executor is a deterministic tool runner. It performs retrieval, denovo generation, pool merging, and optional optimization according to the Planner JSON. It preserves candidate provenance so that retrieval and denovo candidates remain comparable downstream.

3. Peak-Atom Verifier agent.
   The Verifier runs NMR reranking and asks an LLM to normalize the final verdict. It focuses on formula consistency, NMRNet forward-predicted shifts, matched/unmatched peaks, residuals, and atom-level assignment diagnostics.

The stable implementation is in `agents/multi_agent_nmr_v2.py`. The shared prompts are defined in `agents/prompt.py`.

## B. System Prompts

The prompts are separated into one source file so the agent behavior can be audited and revised without changing graph logic.

### B.1 Planner Prompt

The Planner prompt defines the agent as an expert organic chemistry NMR planner. Its main constraints are:

- Treat molecular formula as a hard constraint.
- Parse 1H integration carefully; repeated 1H shifts from integration expansion must not be collapsed.
- Use 13C regions as strong evidence for carbonyls, alkene/aromatic carbons, O-bearing carbons, and saturated carbons.
- Use both retrieval and denovo when compact natural-product-like, fused-ring, lactone, enone, or unusual oxygenated scaffolds are plausible.
- Do not let retrieval rank suppress denovo candidates.
- Use recalled confirmed memories only as analogies, not as proof.
- Return JSON only.

Required Planner JSON keys:

```json
{
  "analysis": "short reasoning about formula and NMR evidence",
  "use_retrieval": true,
  "use_denovo": true,
  "retrieval_top_k": 100,
  "denovo_top_k": 20,
  "save_pool_file": true,
  "need_large_pool": false,
  "need_opt_after_generation": false,
  "notes_for_executor": "concrete execution notes"
}
```

### B.2 Executor Prompt

The Executor prompt is currently used as documentation/future LLM execution policy; the v2 Executor itself is deterministic Python code. The policy says:

- Do not decide the final answer.
- Preserve SMILES, source, rank, score, pool path, formula, and NMR metadata.
- Keep retrieval and denovo candidates visible for verifier reranking.
- Deduplicate by canonical non-isomeric SMILES, while preserving source provenance.
- Run `nmr_optimize` only over an existing merged/provided pool.
- Late-stage RDKit in-place edit tools are allowed only when there is a specific high-confidence local-edit hypothesis.
- Keep the unedited parent candidate after any edit.

### B.3 Verifier Prompt

The Verifier prompt defines the agent as a peak-atom NMR assignment expert. It uses `nmr_rerank` as the main quantitative evidence and then emits a normalized verdict.

The Verifier must inspect:

- `nmr_similarity`.
- Matched query peaks.
- Unmatched query peaks.
- Unused predicted peaks.
- 1H and 13C residuals.
- `atom_level_assignment_summary`.
- Formula consistency.
- Candidate provenance.

Allowed verdicts:

```json
{
  "verdict": "accept | need_opt | need_bigger_pool | need_retry",
  "analysis": "evidence-grounded summary",
  "top_candidate": "SMILES or null",
  "retry_recommendation": "concrete next action"
}
```

Acceptance requires coherent H/C alignment and no major unexplained diagnostic peaks. Denovo candidates can beat retrieval candidates when rerank evidence is stronger.

## C. Memory Module

The memory module lets NMRAgent reuse previously confirmed NMR cases in a controlled way. It is not a separate structure predictor and it does not directly choose the final molecule. Instead, it gives the Planner and Verifier a small set of similar past cases that can help them recognize recurring NMR patterns.

A memory record is written only for a confirmed case. Each record keeps the molecular formula, expanded 1H shifts, 13C shifts, final SMILES, canonical SMILES, optional peak-to-atom assignments, and short diagnostic notes. For example, a confirmed natural-product-like case may store that a carbon peak near 210 ppm corresponded to a ketone carbonyl and that a proton peak near 5.7 ppm was vinylic.

During a new run, the agent searches memory for cases with similar formula and similar 1H/13C peak positions. The similarity score is simple and interpretable: exact formula match receives a bonus, 1H peaks are matched within a narrow ppm window, and 13C peaks are matched within a wider ppm window. Peaks are matched one-to-one so that one remembered peak cannot explain multiple query peaks.

The retrieved memories are passed to both the Planner and the Verifier as `relevant_confirmed_memories`. The Planner may use them to decide whether retrieval, denovo generation, or a larger candidate pool is likely needed. The Verifier may use them as analogy when checking motifs and shift neighborhoods. However, memory is always advisory: the current molecular formula, candidate validity, NMR rerank score, matched/unmatched peaks, and atom-level assignment diagnostics remain the deciding evidence.

By default, accepted outputs are not automatically stored. User-confirmed cases should be added explicitly with `remember_confirmed_case(...)`. Confirmed memories can also be saved and reloaded with `export_confirmed_memory_json(path)` and `import_confirmed_memory_json(path)`.

## D. Candidate Generation And Reranking

### D.1 Source-Aware Candidate Fusion

The Executor collects candidates from multiple sources:

- `retrieval`.
- `denovo`.
- `merged`.
- `optimize`.
- `seed`.

Candidates are deduplicated by canonical non-isomeric SMILES. Each row keeps source information where possible.

The Verifier does not simply take the first N candidates. It uses source-aware sampling:

- Denovo candidates get a protected slice.
- Retrieval candidates get a protected slice.
- Optimized and merged candidates get protected slices.
- The full candidate order is appended afterward.
- The final rerank list is deduplicated and truncated to the configured limit.

This prevents the failure mode where high-volume retrieval candidates push out a correct denovo candidate before NMRNet rerank.

### D.2 NMR Rerank

`nmr_rerank` performs forward spectral prediction and alignment. Important output fields:

- `candidates`: ranked candidate list.
- `nmr_similarity`: overall NMR alignment score.
- `matched_peaks`: query-to-predicted peak matches.
- `unmatched_query_peaks`: query peaks not explained by the candidate.
- `unused_predicted_peaks`: predicted peaks not used by matching.
- `atom_level_assignment_summary`: matched/unmatched counts for H and C.
- `atom_data`: predicted atom-level shifts.

The final result includes `cand_list`, so downstream analysis can inspect the ranked list rather than only one final answer.

## E. Tool Descriptions

### E.1 Candidate Tools

`nmr_retrieve`

- Database-backed candidate retrieval.
- Uses formula and NMR peaks.
- Can save a candidate pool file.
- Best for known or database-near structures.

`nmr_denovo`

- Model-based structure generation from formula and query peaks.
- Uses formula embedding when available.
- Can generate structures absent from retrieval.
- Can save a candidate pool file.

`nmr_merge_pools`

- Merges retrieval/denovo/other pool files.
- Deduplicates candidates.
- Preserves candidate provenance.

`nmr_optimize`

- Pool-only downstream optimizer.
- Takes existing candidates and attempts local/global improvements.
- Should not secretly call retrieval or denovo.

### E.2 Verifier Tool

`nmr_rerank`

- Predicts candidate shifts.
- Aligns predicted and query peaks.
- Produces matched/unmatched peak diagnostics.
- Main quantitative signal for the Verifier.

### E.3 Late-Stage RDKit In-Place Edit Tools

These tools are intentionally late-stage. They should be used only when the Verifier has a specific high-confidence edit hypothesis.

`nmr_canonicalize_smiles`

- Canonicalizes a SMILES string with RDKit.
- Returns canonical non-isomeric SMILES and validity.

`nmr_replace_atom`

- Replaces one atom by atom index.
- Sanitizes the molecule after replacement.
- Returns the edited canonical SMILES or an error object.

`nmr_delete_atom`

- Deletes one atom by atom index.
- Sanitizes the resulting molecule.
- Returns the edited canonical SMILES or an error object.

### E.4 Graph RAG Tools

NMRAgent uses a lightweight Graph RAG layer as advisory chemistry context for the Planner and Verifier. The Graph RAG layer is not a structure predictor and is not allowed to override molecular formula constraints, NMR rerank evidence, or atom-level peak assignment diagnostics. Its role is to provide chemical priors, source metadata, ontology context, and analog examples before candidate generation or verification.

The active KG tools are:

`kg_graph_rag_search`

- Main Graph RAG entry point.
- Searches the unified KG text index and optional metadata fields.
- Returns a compact `evidence_pack` containing chemistry context, matched documents, and optional one-hop graph context.
- Used by the v2 workflow only when KG RAG is enabled.

`kg_entity_lookup`

- Exact or near-exact metadata lookup.
- Supports alias/name, formula, InChIKey, source, and source identifier queries.
- Best for resolving a molecule, class, organism, ontology node, or database entry to a KG `node_id`.

`kg_document_search`

- BM25/SQLite FTS search over KG documents.
- Searches molecule profiles, ontology profiles, relation glossary documents, and synthetic knowledge cards.
- Returns short text evidence with source and provenance metadata.

`kg_neighbors`

- Bounded one-hop graph expansion for a known `node_id`.
- Can filter by relation predicate or source.
- Disabled by default in fast mode by setting `kg_rag_neighbor_limit=0`.

The active v2 command-line runner is `scripts/run_multi_agent_nmr_v2.py`. KG RAG is optional and enabled with `--kg-rag`. The default fast mode uses document and metadata retrieval only; graph-neighbor expansion can be enabled separately with `--kg-rag-neighbor-limit`.

## F. Multi-Source Knowledge Graph Construction

The KG is built from heterogeneous public chemistry and biomedical resources under a unified intermediate schema. The goal is not to treat every file as free text. Instead, entity records, relation edges, retrievable documents, and aliases are stored separately so the retrieval layer can combine exact metadata lookup, text search, and graph expansion.

### F.1 Source Types

The current KG integrates nine source groups:

| Source | Original form | Unified role |
| --- | --- | --- |
| COCONUT | natural-product CSV table | molecule and natural-product nodes, aliases, molecule documents |
| NP Atlas | JSON natural-product records | molecule, organism, and chemical-class nodes; `produced_by` and `has_class` edges |
| LOTUS | SMILES-to-LOTUS-ID table | molecule nodes and minimal SMILES documents |
| TTD approved drugs | block key-value text | approved-drug molecule nodes and drug documents |
| ChEMBL | SQLite database | ChEMBL compound nodes and molecule documents |
| ChEBI | OBO Graph JSON | ontology nodes, synonyms/xrefs, and ontology edges |
| PrimeKG | CSV edge table | biomedical entity nodes and typed relation edges |
| DRKG | TSV triples plus relation glossary | biomedical entity nodes, typed triples, and relation glossary documents |
| ElementKG synthetic corpus | JSON text cards | auxiliary synthetic knowledge documents |

Large database-like sources are handled with source-specific adapters rather than generic text parsing. For example, ChEMBL is extracted through SQL joins between `molecule_dictionary` and `compound_structures`, while ChEBI is parsed as ontology nodes and edges.

### F.2 Unified KG Schema

The KG builder writes four JSONL files and one manifest:

```text
nodes.jsonl       entity records
edges.jsonl       directed graph relations
documents.jsonl   retrievable text units
aliases.jsonl     name/synonym/source-ID to node mappings
manifest.json     source list, limits, counts, and provenance
```

A node stores a stable `node_id`, labels, name, source identifiers, properties, source, and provenance. Molecules with an InChIKey use the global identifier `mol:inchikey:{InChIKey}` so that molecule records from different sources can be aligned. Source-only molecules use source-prefixed IDs such as `lotus:{LOTUS_ID}` or `ttd:{DRUG_ID}`. Ontology and biomedical nodes use source-specific namespaces such as `chebi:CHEBI_...`, `primekg:...`, and `drkg:...`.

An edge stores `subject_id`, `predicate`, `object_id`, relation label, source, optional properties, and provenance. Examples include `produced_by`, `has_class`, ChEBI ontology relations, PrimeKG biomedical relations, and DRKG triples.

A document stores a short retrievable text unit linked to a `node_id` when possible. Documents are generated from structured molecule metadata, ontology profiles, relation glossaries, and synthetic knowledge cards. Raw graph edges are not treated as ordinary documents.

An alias maps a name, synonym, source ID, xref, or database identifier back to a `node_id`. This supports fast exact lookup before text retrieval.

### F.3 KG Scale

The current merged fast KG contains:

```text
nodes      4,275,055
edges      2,516,946
documents  4,337,742
aliases   10,432,630
```

PrimeKG and DRKG were also rebuilt without edge caps for the full-edge KG parts:

```text
PrimeKG edges  8,100,498
DRKG edges     5,874,261
```

The full-edge artifacts are stored as separate KG parts so that the RAG index can choose whether to include all biomedical edges. The default fast RAG index skips full graph-neighbor materialization and prioritizes metadata plus document retrieval.

### F.4 KG RAG Index

Runtime RAG does not scan the JSONL artifacts. The JSONL files are converted into a SQLite/FTS index with:

```text
nodes(node_id, name, labels, source, formula, inchikey, smiles, identifiers_json, properties_json, provenance_json)
aliases(alias, alias_norm, alias_type, node_id, source, provenance_json)
documents(doc_id, node_id, title, text, doc_type, source, formula, metadata_json, provenance_json)
documents_fts(title, text)
edges(edge_id, subject_id, predicate, object_id, relation_label, source, properties_json, provenance_json)
node_neighbors(node_id, edge_id, direction, neighbor_id, predicate, relation_label, source)
```

The fast index used by default builds `nodes`, `aliases`, `documents`, and `documents_fts`, while skipping `edges` and `node_neighbors`. This avoids materializing tens of millions of neighbor rows during ordinary NMR runs. Full graph expansion remains available as an offline or hard-case mode.

### F.5 Planner Integration

In the v2 multi-agent workflow, KG RAG is optional. When enabled, the agent performs a single fast Graph RAG query before the first Planner call. The returned object is passed into both Planner and Verifier payloads as `retrieved_kg_evidence`.

The Planner may use this evidence to recognize likely chemical classes, relevant natural-product sources, diagnostic motifs, and analog examples. The Verifier may use it as background context when interpreting candidate provenance and NMR motifs. Both agents are instructed that KG evidence is advisory only. The final decision remains controlled by formula consistency, candidate validity, NMR rerank alignment, matched/unmatched peaks, and atom-level assignment diagnostics.

## G. C20H30O3 Validation Case

Input:

```text
Formula: C20H30O3
13C NMR: 210.38, 182.46, 172.23, 111.19, 86.84, 56.39, 46.03, 42.31, 41.72, 40.55, 40.20, 37.25, 34.24, 33.51, 21.71, 19.45, 18.34, 18.07, 17.90, 17.51
1H NMR: expanded repeated shifts totaling 30 H
```

User-confirmed answer:

```text
CC12C(CCC(O3)(CC(C(C)C)=O)C2=CC3=O)C(C)(C)CCC1
```

RDKit canonical non-isomeric SMILES:

```text
CC(C)C(=O)CC12CCC3C(C)(C)CCCC3(C)C1=CC(=O)O2
```

Observed behavior after source-aware rerank:

- Denovo produced the correct candidate.
- The Verifier reranked denovo together with retrieval candidates.
- Final accepted answer matched the user-confirmed canonical structure.

## H. Reproducibility Notes

Recommended runtime:

```bash
$NMR_SOLVER_PYTHON or the `python` executable from the activated conda environment
```

Important dependencies in the solver environment:

- `langgraph`.
- `langchain-core`.
- `openai`.
- RDKit.
- UltraNMR denovo backend.
- NMRNet rerank backend.

The main NMR environment previously showed `transformers/scipy` metadata issues, so the solver conda environment is the current stable runtime.
