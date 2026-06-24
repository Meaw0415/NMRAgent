# Fast Chemistry RAG Plan

## Goal

Build a fast retrieval layer for NMRAgent that gives the Planner and Verifier useful chemistry context without turning every run into a long database scan or a long LLM context. The RAG layer has three sources only:

1. Textbook RAG for stable chemistry and NMR rules.
2. Graph RAG for chemical priors, molecule/source metadata, ontology context, and analog examples.
3. Web Search RAG for missing or current external information, disabled by default.

The system should return compact evidence, not raw dumps. Graph RAG is not meant only for exact ID lookup. It should provide chemistry knowledge context such as likely classes, natural-product priors, analog molecules, source metadata, and nearby ontology or biomedical relations.

## High-Level Flow

```text
Input formula / NMR peaks / candidate / agent output
        |
Retrieval Planner
        |
        +-- Textbook RAG query plan
        +-- Graph RAG query plan
        +-- Optional Web Search plan
        |
RAG Executors
        |
Evidence Pack Builder
        |
NMR Planner / Verifier / Final Answer
```

The Retrieval Planner should be cheap. It extracts query facets and decides which RAG source to call. It should not read large files and should not make final structure decisions.

## RAG Source Roles

### Textbook RAG

Use for stable rules and interpretation heuristics:

- 1H/13C chemical shift ranges.
- Functional-group diagnostics such as ketone, aldehyde, ester, lactone, enone, alkene, aromatic, O-bearing carbon, methyl/isopropyl patterns.
- Integration interpretation and repeated peak handling.
- General NMR assignment principles.

Textbook RAG should answer: "What rule helps interpret this spectrum?"

### Graph RAG

Use for structured chemistry context:

- Exact or near metadata matches: formula, InChIKey, source ID, name, alias.
- Similar natural-product classes and analog molecules.
- Source provenance from COCONUT, NP Atlas, LOTUS, ChEMBL, TTD, ChEBI, PrimeKG, and DRKG.
- ChEBI ontology parents/synonyms/xrefs.
- Graph neighbors such as `has_class`, `produced_by`, ontology `is_a`, and selected biomedical relations.

Graph RAG should answer: "What does the existing chemical knowledge base suggest is plausible or relevant?"

It must not directly decide the final structure. It supplies priors and analog evidence for the normal NMR candidate generation and verifier stages.

### Web Search RAG

Use only when local sources are insufficient:

- Missing source metadata.
- A term or compound is absent from the KG and textbook.
- User explicitly asks for latest/current information.
- External validation is needed.

Default mode keeps Web Search off to avoid latency and unstable results.

## Textbook Chunking

Do not let GPT read an entire textbook directly during planning. Ingest the textbook once, then query indexed chunks.

Recommended chunking:

```text
chunk_size: 600-1000 tokens
overlap: 100-150 tokens
split_priority: chapter -> section -> paragraph -> page window
```

Each chunk should store:

```json
{
  "doc_id": "textbook:...",
  "book_title": "...",
  "chapter": "...",
  "section": "...",
  "page_start": 123,
  "page_end": 124,
  "text": "...",
  "topic_tags": ["13C NMR", "carbonyl", "lactone"]
}
```

Runtime retrieval should return only top 3-5 chunks in normal mode.

## Fast-First Policy

Default retrieval budgets:

```json
{
  "textbook_top_k": 3,
  "graph_doc_top_k": 5,
  "graph_neighbor_limit": 10,
  "graph_expand_depth": 1,
  "web_top_k": 0,
  "max_evidence_tokens": 3000
}
```

Hard-case budgets:

```json
{
  "textbook_top_k": 5,
  "graph_doc_top_k": 10,
  "graph_neighbor_limit": 25,
  "graph_expand_depth": 1,
  "web_top_k": 3,
  "max_evidence_tokens": 5000
}
```

Avoid depth-2 graph expansion by default. Use it only when a specific node or class needs more context.

## Planner Interface

Input:

```json
{
  "task_type": "nmr_structure_elucidation | verification | explanation",
  "formula": "C20H30O3",
  "h_shifts": [5.68, 3.08],
  "c_shifts": [210.38, 182.46],
  "candidate_smiles": "optional",
  "agent_output": "optional",
  "user_question": "optional"
}
```

Output:

```json
{
  "use_textbook_rag": true,
  "textbook_queries": [
    "13C NMR ketone lactone enone chemical shift ranges"
  ],
  "use_graph_rag": true,
  "graph_queries": [
    "C20H30O3 oxygenated natural product diterpenoid lactone ketone analogs"
  ],
  "graph_filters": {
    "formula": "C20H30O3",
    "sources": ["coconut", "npatlas", "chembl", "chebi"],
    "labels": ["molecule", "natural_product", "ontology_node"]
  },
  "use_web_rag": false,
  "web_queries": [],
  "budget": {
    "textbook_top_k": 3,
    "graph_doc_top_k": 5,
    "graph_neighbor_limit": 10,
    "max_evidence_tokens": 3000
  }
}
```

## Graph RAG Execution

Graph RAG should be deterministic and indexed. It should not scan JSONL files at runtime.

Execution steps:

1. Exact metadata lookup.
   - Formula, InChIKey, source ID, name, alias, candidate SMILES if available.
2. Text-like KG document search.
   - Search molecule profiles, ontology profiles, and synthetic cards with BM25 first.
3. Node scoring.
   - Combine exact-match score, document score, source priority, label priority, and formula/class match.
4. One-hop graph expansion.
   - Fetch selected relations for top nodes only.
5. Context summarization.
   - Return concise priors, analog examples, relation snippets, and provenance.

Recommended source priority for NMR natural-product tasks:

```text
exact InChIKey/source ID > formula match > NP Atlas/COCONUT/LOTUS natural product metadata > ChEBI ontology > ChEMBL properties > PrimeKG/DRKG biomedical context > ElementKG synthetic text
```

## Evidence Pack Schema

All RAG outputs should be converted to the same compact evidence format:

```json
{
  "source_type": "textbook | graph | web",
  "claim": "13C shifts near 200-220 ppm support ketone-like carbonyls.",
  "evidence": "Short quoted or paraphrased evidence.",
  "metadata": {
    "node_id": "optional",
    "doc_id": "optional",
    "page": "optional",
    "source": "optional",
    "url": "optional"
  },
  "confidence": "high | medium | low",
  "provenance": "stable pointer to source"
}
```

The Evidence Pack Builder should deduplicate claims and cap the final text length before sending it to any LLM.

## Index Plan

First implementation should use SQLite FTS5/BM25, not embeddings. This is faster to build, easier to debug, and enough for formula/class/source metadata plus short chemistry text.

Target directory:

```text
artifacts/ns_rag_index/
├── kg_index.sqlite
├── textbook_index.sqlite
└── manifest.json
```

`kg_index.sqlite` tables:

```text
nodes(node_id primary key, name, labels_json, source, identifiers_json, properties_json)
aliases(alias, alias_norm, alias_type, node_id, source)
documents(doc_id primary key, node_id, title, text, doc_type, source, metadata_json)
documents_fts(title, text, content='documents')
edges(edge_id primary key, subject_id, predicate, object_id, relation_label, source, properties_json)
node_neighbors(node_id, edge_id, direction, neighbor_id, predicate, relation_label, source)
```

Recommended indexes:

```sql
CREATE INDEX idx_nodes_source ON nodes(source);
CREATE INDEX idx_aliases_norm ON aliases(alias_norm);
CREATE INDEX idx_aliases_node ON aliases(node_id);
CREATE INDEX idx_edges_subject ON edges(subject_id);
CREATE INDEX idx_edges_object ON edges(object_id);
CREATE INDEX idx_neighbors_node ON node_neighbors(node_id);
CREATE INDEX idx_documents_node ON documents(node_id);
```

Textbook index tables mirror `documents` plus FTS metadata for chapter/section/page.

## Tool API Draft

```text
textbook_search(query, top_k=3, filters=None)
graph_rag_search(query, formula=None, candidate_smiles=None, top_k=5, neighbor_limit=10)
web_search_rag(query, top_k=3)
build_evidence_pack(textbook_hits, graph_hits, web_hits, token_budget=3000)
```

For NMRAgent integration, add one payload field:

```json
{
  "retrieved_evidence_pack": []
}
```

Planner uses it to guide generation strategy. Verifier uses it as advisory context. Neither should treat RAG evidence as proof over NMR rerank and formula consistency.

## Implementation Order

1. Finalize KG artifacts and source manifest.
2. Build `kg_index.sqlite` from the unified KG JSONL.
3. Add `graph_rag_search` using metadata lookup + FTS + one-hop expansion.
4. Ingest textbook into chunks and build `textbook_index.sqlite`.
5. Add lightweight Retrieval Planner that emits query plans and budgets.
6. Add optional Web Search RAG only behind an explicit planner flag.
7. Pass the compact Evidence Pack into `MultiAgentNMRV2` Planner and Verifier payloads.

## Latency Target

Normal mode should aim for:

```text
planner: < 1-2 s
graph_rag_search: < 2-5 s after indexing
textbook_search: < 1-2 s
web_search: disabled by default
evidence pack assembly: < 1 s
```

If retrieval exceeds budget, return partial evidence with a timeout note rather than blocking the NMR workflow.
