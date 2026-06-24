# Unified NS Knowledge Graph Format

This document defines a single intermediate format for the heterogeneous NS knowledge assets configured by `NMR_NS_SOURCE_ROOT`. The goal is to keep graph facts, entity attributes, and retrievable text separate, so a future RAG tool can combine semantic search with graph expansion without mixing incompatible data types.

## Output Layout

A build writes one directory with these files:

```text
artifacts/ns_unified_kg/
├── nodes.jsonl
├── edges.jsonl
├── documents.jsonl
├── aliases.jsonl
└── manifest.json
```

## Core Schemas

### `nodes.jsonl`

Each line is one entity.

```json
{
  "node_id": "mol:inchikey:BZLIDAVUQDTJQF-HWTFSWDCSA-N",
  "labels": ["molecule", "natural_product"],
  "name": "Curvularide C",
  "identifiers": {"inchikey": "...", "source_id": "NPA000001"},
  "properties": {"formula": "C19H37NO5", "smiles": "..."},
  "source": "npatlas",
  "provenance": {"path": "NP_Atlas/NPAtlas_download_2024_09.json", "row": 1}
}
```

### `edges.jsonl`

Each line is one directed graph relation.

```json
{
  "edge_id": "sha1:...",
  "subject_id": "mol:inchikey:...",
  "predicate": "produced_by",
  "object_id": "taxon:npatlas:Curvularia_geniculata",
  "relation_label": "produced by organism",
  "properties": {},
  "source": "npatlas",
  "provenance": {"path": "NP_Atlas/NPAtlas_download_2024_09.json", "row": 1}
}
```

### `documents.jsonl`

Each line is a retrievable text unit. Documents may point back to a node.

```json
{
  "doc_id": "sha1:...",
  "node_id": "mol:inchikey:...",
  "title": "Curvularide C",
  "text": "Curvularide C is a natural product with formula ...",
  "doc_type": "molecule_profile",
  "source": "npatlas",
  "metadata": {"formula": "C19H37NO5"},
  "provenance": {"path": "NP_Atlas/NPAtlas_download_2024_09.json", "row": 1}
}
```

### `aliases.jsonl`

Each line links a lookup string or source identifier to a node.

```json
{
  "alias": "Curvularide C",
  "alias_type": "name",
  "node_id": "mol:inchikey:...",
  "source": "npatlas",
  "provenance": {"path": "NP_Atlas/NPAtlas_download_2024_09.json", "row": 1}
}
```

## Node ID Strategy

Use stable global IDs when a source provides them:

- Molecules with InChIKey: `mol:inchikey:{InChIKey}`.
- ChEBI ontology nodes: `chebi:{CHEBI_ID}`.
- PrimeKG nodes: `primekg:{entity_type}:{source}:{source_id}`.
- DRKG nodes: `drkg:{raw_entity_id}`.
- Taxa or organisms created from source metadata: `taxon:{source}:{normalized_name}`.
- Source-only molecules without InChIKey: `{source}:{source_id}`.
- Synthetic corpus entities: `elementkg:{sha1(entity)}`.

This keeps cross-source molecule merging possible through InChIKey while avoiding unsafe merges for sources that only provide SMILES or local IDs.

## Source Mapping

| Source | Nodes | Edges | Documents | Notes |
| --- | --- | --- | --- | --- |
| COCONUT | molecule, natural_product | optional class/source edges | molecule profiles | Best keyed by InChIKey. |
| NP Atlas | molecule, organism, chemical class | produced_by, has_class | molecule profiles | Rich natural-product metadata. |
| LOTUS `smiles` | molecule, natural_product | none | minimal molecule profiles | SMILES + LOTUS ID only; avoid merging without InChIKey. |
| TTD approved drugs | molecule, approved_drug | none | drug profiles | Block key-value parser. |
| ChEBI | ontology/chemical entities | ontology relations | ontology profiles | Use `lbl`, synonyms, xrefs, ChemROF properties. |
| PrimeKG | biomedical entities | typed edges | none by default | Treat as graph layer, not document text. |
| DRKG | biomedical entities | typed triples | relation glossary docs | Treat as graph layer. |
| ElementKG synthetic descriptions | synthetic knowledge entities | none by default | synthetic knowledge cards | Auxiliary text, not primary factual ground truth. |

## RAG Tool Plan

A dedicated RAG tool can use this intermediate format in three steps:

1. Search `documents.jsonl` with lexical or vector retrieval.
2. Resolve each hit to `node_id`, then fetch aliases and properties from `nodes.jsonl`.
3. Expand local graph neighborhoods from `edges.jsonl`, using source and predicate filters to keep context bounded.

For NMR structure work, the first useful filters are molecule-centric: formula, InChIKey, SMILES/source IDs, natural-product source, chemical class, organism, and nearby ontology or biomedical relations.
