#!/usr/bin/env python3
"""Build a unified JSONL knowledge-graph layer from the NS data pack.

The builder intentionally separates entities, graph relations, retrievable text,
and aliases. It uses only the Python standard library so it can run in a basic
HPC environment and supports small smoke builds through per-source limits.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

DEFAULT_SOURCE_ROOT = Path(os.environ.get("NMR_NS_SOURCE_ROOT", "artifacts/ns_source"))
DEFAULT_OUT_DIR = Path("artifacts/ns_unified_kg")
SCHEMA_VERSION = "ns-unified-kg-v0.1"
csv.field_size_limit(sys.maxsize)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact_id(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)
    return value.strip("_") or "unknown"


def sha1_text(value: str, n: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def mol_node_id(source: str, source_id: str = "", inchikey: str = "") -> str:
    inchikey = clean_text(inchikey)
    if inchikey:
        return f"mol:inchikey:{inchikey}"
    return f"{source}:{compact_id(source_id) or sha1_text(source_id)}"


def chebi_id(raw_id: str) -> str:
    match = re.search(r"CHEBI_(\d+)", raw_id or "")
    return f"chebi:CHEBI_{match.group(1)}" if match else f"chebi:{compact_id(raw_id)}"


def split_pipe(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    return [x.strip() for x in text.split("|") if x.strip()]


def maybe_float(value: Any) -> Any:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return text


def write_jsonl(handle, obj: Dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


class KGWriter:
    def __init__(self, out_dir: Path, *, dedupe_edges: bool = False, max_alias_len: int = 512) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_f = (out_dir / "nodes.jsonl").open("w", encoding="utf-8")
        self.edges_f = (out_dir / "edges.jsonl").open("w", encoding="utf-8")
        self.docs_f = (out_dir / "documents.jsonl").open("w", encoding="utf-8")
        self.aliases_f = (out_dir / "aliases.jsonl").open("w", encoding="utf-8")
        self.seen_nodes = set()
        self.dedupe_edges = dedupe_edges
        self.max_alias_len = int(max_alias_len)
        self.seen_edges = set() if dedupe_edges else None
        self.seen_docs = set()
        self.seen_aliases = set()
        self.counts = {"nodes": 0, "edges": 0, "documents": 0, "aliases": 0}
        self.by_source: Dict[str, Dict[str, int]] = {}

    def _bump(self, source: str, kind: str) -> None:
        self.counts[kind] += 1
        self.by_source.setdefault(source, {"nodes": 0, "edges": 0, "documents": 0, "aliases": 0})[kind] += 1

    def node(self, obj: Dict[str, Any]) -> None:
        node_id = obj.get("node_id")
        if not node_id or node_id in self.seen_nodes:
            return
        self.seen_nodes.add(node_id)
        obj.setdefault("labels", [])
        obj.setdefault("identifiers", {})
        obj.setdefault("properties", {})
        obj.setdefault("provenance", {})
        write_jsonl(self.nodes_f, obj)
        self._bump(obj.get("source", "unknown"), "nodes")

    def edge(self, obj: Dict[str, Any]) -> None:
        seed = "|".join([obj.get("subject_id", ""), obj.get("predicate", ""), obj.get("object_id", ""), obj.get("source", "")])
        edge_id = obj.get("edge_id") or f"sha1:{sha1_text(seed, 24)}"
        if self.dedupe_edges:
            if edge_id in self.seen_edges:
                return
            self.seen_edges.add(edge_id)
        obj["edge_id"] = edge_id
        obj.setdefault("properties", {})
        obj.setdefault("provenance", {})
        write_jsonl(self.edges_f, obj)
        self._bump(obj.get("source", "unknown"), "edges")

    def document(self, obj: Dict[str, Any]) -> None:
        text = clean_text(obj.get("text"))
        if not text:
            return
        seed = "|".join(str(obj.get(k) or "") for k in ("node_id", "title", "source")) + "|" + text[:200]
        doc_id = obj.get("doc_id") or f"sha1:{sha1_text(seed, 24)}"
        if doc_id in self.seen_docs:
            return
        self.seen_docs.add(doc_id)
        obj["doc_id"] = doc_id
        obj["text"] = text
        obj.setdefault("metadata", {})
        obj.setdefault("provenance", {})
        write_jsonl(self.docs_f, obj)
        self._bump(obj.get("source", "unknown"), "documents")

    def alias(self, node_id: str, alias: str, alias_type: str, source: str, provenance: Dict[str, Any]) -> None:
        alias = clean_text(alias)
        if not alias or len(alias) > self.max_alias_len:
            return
        key = (node_id, alias.lower(), alias_type, source)
        if key in self.seen_aliases:
            return
        self.seen_aliases.add(key)
        write_jsonl(self.aliases_f, {
            "alias": alias,
            "alias_type": alias_type,
            "node_id": node_id,
            "source": source,
            "provenance": provenance,
        })
        self._bump(source, "aliases")

    def close(self) -> None:
        self.nodes_f.close()
        self.edges_f.close()
        self.docs_f.close()
        self.aliases_f.close()


def iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buf = ""
        started = False
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            pos = 0
            while True:
                if not started:
                    idx = buf.find("[", pos)
                    if idx < 0:
                        buf = buf[-1:]
                        break
                    pos = idx + 1
                    started = True
                while pos < len(buf) and buf[pos] in " \r\n\t,":
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    buf = buf[pos:]
                    break
                if isinstance(obj, dict):
                    yield obj
                pos = end
                if pos > 1024 * 1024:
                    buf = buf[pos:]
                    pos = 0


def limit_reached(count: int, limit: int) -> bool:
    return limit >= 0 and count >= limit


def log_progress(source: str, count: int, every: int = 100000) -> None:
    if every > 0 and count > 0 and count % every == 0:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {source}: processed {count}", flush=True)


def add_molecule_doc(writer: KGWriter, *, node_id: str, title: str, source: str, provenance: Dict[str, Any], properties: Dict[str, Any], doc_type: str) -> None:
    parts = [title]
    if properties.get("formula"):
        parts.append(f"formula {properties['formula']}")
    if properties.get("molecular_weight"):
        parts.append(f"molecular weight {properties['molecular_weight']}")
    if properties.get("chemical_class"):
        parts.append(f"chemical class {properties['chemical_class']}")
    if properties.get("organisms"):
        parts.append(f"reported organism/source {properties['organisms']}")
    if properties.get("smiles"):
        parts.append(f"SMILES {properties['smiles']}")
    if properties.get("notes"):
        parts.append(str(properties["notes"]))
    writer.document({
        "node_id": node_id,
        "title": title or node_id,
        "text": ". ".join(clean_text(x).rstrip(".") for x in parts if clean_text(x)) + ".",
        "doc_type": doc_type,
        "source": source,
        "metadata": {k: v for k, v in properties.items() if k in {"formula", "smiles", "inchikey", "source_id", "chemical_class"} and v not in ("", None, [])},
        "provenance": provenance,
    })


def build_coconut(root: Path, writer: KGWriter, limit: int) -> None:
    source = "coconut"
    rel = "COCONUT/coconut_csv-04-2026.csv"
    path = root / rel
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            log_progress(source, idx - 1)
            if limit_reached(idx - 1, limit):
                break
            identifier = clean_text(row.get("identifier"))
            inchikey = clean_text(row.get("standard_inchi_key"))
            node_id = mol_node_id(source, identifier, inchikey)
            name = clean_text(row.get("name")) or identifier
            provenance = {"path": rel, "row": idx}
            props = {
                "source_id": identifier,
                "smiles": clean_text(row.get("canonical_smiles")),
                "inchi": clean_text(row.get("standard_inchi")),
                "inchikey": inchikey,
                "formula": clean_text(row.get("molecular_formula")),
                "molecular_weight": maybe_float(row.get("molecular_weight")),
                "chemical_class": clean_text(row.get("chemical_class")),
                "chemical_sub_class": clean_text(row.get("chemical_sub_class")),
                "chemical_super_class": clean_text(row.get("chemical_super_class")),
                "np_classifier_pathway": clean_text(row.get("np_classifier_pathway")),
                "np_classifier_superclass": clean_text(row.get("np_classifier_superclass")),
                "np_classifier_class": clean_text(row.get("np_classifier_class")),
                "organisms": clean_text(row.get("organisms")),
                "dois": split_pipe(row.get("dois")),
                "cas": clean_text(row.get("cas")),
            }
            writer.node({
                "node_id": node_id,
                "labels": ["molecule", "natural_product"],
                "name": name,
                "identifiers": {"source_id": identifier, "inchikey": inchikey},
                "properties": {k: v for k, v in props.items() if v not in ("", None, [])},
                "source": source,
                "provenance": provenance,
            })
            for alias in [identifier, name, row.get("iupac_name"), row.get("cas")]:
                writer.alias(node_id, alias, "identifier" if alias == identifier else "name", source, provenance)
            for syn in split_pipe(row.get("synonyms"))[:20]:
                writer.alias(node_id, syn, "synonym", source, provenance)
            add_molecule_doc(writer, node_id=node_id, title=name, source=source, provenance=provenance, properties=props, doc_type="molecule_profile")


def build_lotus(root: Path, writer: KGWriter, limit: int) -> None:
    source = "lotus"
    rel = "LOTUS/smiles"
    path = root / rel
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            log_progress(source, idx - 1)
            if limit_reached(idx - 1, limit):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            smiles, lotus_id = clean_text(parts[0]), clean_text(parts[1])
            node_id = f"lotus:{compact_id(lotus_id)}"
            provenance = {"path": rel, "row": idx}
            props = {"source_id": lotus_id, "smiles": smiles}
            writer.node({
                "node_id": node_id,
                "labels": ["molecule", "natural_product"],
                "name": lotus_id,
                "identifiers": {"source_id": lotus_id},
                "properties": props,
                "source": source,
                "provenance": provenance,
            })
            writer.alias(node_id, lotus_id, "identifier", source, provenance)
            add_molecule_doc(writer, node_id=node_id, title=lotus_id, source=source, provenance=provenance, properties=props, doc_type="minimal_molecule_profile")


def parse_ttd_blocks(path: Path) -> Iterator[Dict[str, str]]:
    current: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    yield current
                    current = {}
                continue
            if "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            key = key.strip()
            value = value.strip()
            if key in {"DRUG__ID", "DRUGNAME", "DRUGINCH", "DRUGSMIL"}:
                current[key] = value
        if current:
            yield current


def build_ttd(root: Path, writer: KGWriter, limit: int) -> None:
    source = "ttd_approved"
    rel = "P3-07-Approved_smi_inchi.txt"
    path = root / rel
    if not path.exists():
        return
    count = 0
    for idx, row in enumerate(parse_ttd_blocks(path), start=1):
        if not row.get("DRUG__ID"):
            continue
        if limit_reached(count, limit):
            break
        count += 1
        drug_id = clean_text(row.get("DRUG__ID"))
        name = clean_text(row.get("DRUGNAME")) or drug_id
        node_id = f"ttd:{compact_id(drug_id)}"
        provenance = {"path": rel, "record": idx}
        props = {"source_id": drug_id, "smiles": clean_text(row.get("DRUGSMIL")), "inchi": clean_text(row.get("DRUGINCH"))}
        writer.node({
            "node_id": node_id,
            "labels": ["molecule", "approved_drug"],
            "name": name,
            "identifiers": {"source_id": drug_id},
            "properties": props,
            "source": source,
            "provenance": provenance,
        })
        writer.alias(node_id, drug_id, "identifier", source, provenance)
        writer.alias(node_id, name, "name", source, provenance)
        add_molecule_doc(writer, node_id=node_id, title=name, source=source, provenance=provenance, properties=props, doc_type="approved_drug_profile")


def build_npatlas(root: Path, writer: KGWriter, limit: int) -> None:
    source = "npatlas"
    rel = "NP_Atlas/NPAtlas_download_2024_09.json"
    path = root / rel
    if not path.exists():
        return
    for idx, row in enumerate(iter_json_array(path), start=1):
        log_progress(source, idx - 1)
        if limit_reached(idx - 1, limit):
            break
        npaid = clean_text(row.get("npaid") or row.get("id"))
        inchikey = clean_text(row.get("inchikey"))
        name = clean_text(row.get("original_name")) or npaid
        node_id = mol_node_id(source, npaid, inchikey)
        provenance = {"path": rel, "row": idx}
        classyfire = row.get("classyfire") or {}
        npclassifier = row.get("npclassifier") or {}
        org = row.get("origin_organism") or {}
        org_name = " ".join(x for x in [clean_text(org.get("genus")), clean_text(org.get("species"))] if x) or clean_text(org.get("type"))
        props = {
            "source_id": npaid,
            "smiles": clean_text(row.get("smiles")),
            "inchi": clean_text(row.get("inchi")),
            "inchikey": inchikey,
            "formula": clean_text(row.get("mol_formula")),
            "molecular_weight": maybe_float(row.get("mol_weight")),
            "exact_mass": maybe_float(row.get("exact_mass")),
            "chemical_class": clean_text((classyfire.get("class") or {}).get("name")),
            "chemical_super_class": clean_text((classyfire.get("superclass") or {}).get("name")),
            "direct_parent": clean_text((classyfire.get("direct_parent") or {}).get("name")),
            "organisms": org_name,
            "notes": clean_text(classyfire.get("description")),
        }
        writer.node({
            "node_id": node_id,
            "labels": ["molecule", "natural_product"],
            "name": name,
            "identifiers": {"source_id": npaid, "inchikey": inchikey},
            "properties": {k: v for k, v in props.items() if v not in ("", None, [])},
            "source": source,
            "provenance": provenance,
        })
        for alias in [npaid, name]:
            writer.alias(node_id, alias, "identifier" if alias == npaid else "name", source, provenance)
        for syn in row.get("synonyms") or []:
            writer.alias(node_id, syn, "synonym", source, provenance)
        for ext in row.get("external_ids") or []:
            writer.alias(node_id, ext.get("external_db_code"), f"xref:{clean_text(ext.get('external_db_name'))}", source, provenance)
        if org_name:
            org_id = f"taxon:npatlas:{compact_id(org_name)}"
            writer.node({
                "node_id": org_id,
                "labels": ["organism", "taxon"],
                "name": org_name,
                "identifiers": {"source_id": clean_text(org.get("id"))},
                "properties": {"type": clean_text(org.get("type")), "genus": clean_text(org.get("genus")), "species": clean_text(org.get("species"))},
                "source": source,
                "provenance": provenance,
            })
            writer.edge({"subject_id": node_id, "predicate": "produced_by", "object_id": org_id, "relation_label": "produced by organism", "source": source, "provenance": provenance})
        for cls in [props.get("chemical_super_class"), props.get("chemical_class"), props.get("direct_parent"), clean_text(npclassifier.get("pathway")), clean_text(npclassifier.get("superclass")), clean_text(npclassifier.get("class"))]:
            if cls:
                class_id = f"class:npatlas:{compact_id(cls)}"
                writer.node({"node_id": class_id, "labels": ["chemical_class"], "name": cls, "identifiers": {}, "properties": {}, "source": source, "provenance": provenance})
                writer.edge({"subject_id": node_id, "predicate": "has_class", "object_id": class_id, "relation_label": "has chemical class", "source": source, "provenance": provenance})
        add_molecule_doc(writer, node_id=node_id, title=name, source=source, provenance=provenance, properties=props, doc_type="natural_product_profile")



def build_chembl(root: Path, writer: KGWriter, limit: int) -> None:
    source = "chembl"
    rel = "ChemBLdb/chembl_36/chembl_36_sqlite/chembl_36.db"
    path = root / rel
    if not path.exists():
        return
    con = sqlite3.connect(str(path))
    query = """
        SELECT
            md.molregno,
            md.pref_name,
            md.chembl_id,
            md.max_phase,
            md.therapeutic_flag,
            md.dosed_ingredient,
            md.structure_type,
            md.molecule_type,
            md.first_approval,
            md.oral,
            md.parenteral,
            md.topical,
            md.black_box_warning,
            md.natural_product,
            md.first_in_class,
            md.chirality,
            md.prodrug,
            md.inorganic_flag,
            md.withdrawn_flag,
            md.chemical_probe,
            md.orphan,
            md.veterinary,
            cs.standard_inchi,
            cs.standard_inchi_key,
            cs.canonical_smiles
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
    """
    try:
        cur = con.execute(query)
        columns = [d[0] for d in cur.description]
        for idx, values in enumerate(cur, start=1):
            log_progress(source, idx - 1)
            if limit_reached(idx - 1, limit):
                break
            row = dict(zip(columns, values))
            chembl_id = clean_text(row.get("chembl_id")) or f"molregno:{row.get('molregno')}"
            inchikey = clean_text(row.get("standard_inchi_key"))
            node_id = mol_node_id(source, chembl_id, inchikey)
            name = clean_text(row.get("pref_name")) or chembl_id
            provenance = {"path": rel, "row": idx, "molregno": row.get("molregno")}
            props = {
                "source_id": chembl_id,
                "molregno": row.get("molregno"),
                "smiles": clean_text(row.get("canonical_smiles")),
                "inchi": clean_text(row.get("standard_inchi")),
                "inchikey": inchikey,
                "pref_name": clean_text(row.get("pref_name")),
                "max_phase": row.get("max_phase"),
                "therapeutic_flag": row.get("therapeutic_flag"),
                "dosed_ingredient": row.get("dosed_ingredient"),
                "structure_type": clean_text(row.get("structure_type")),
                "molecule_type": clean_text(row.get("molecule_type")),
                "first_approval": row.get("first_approval"),
                "oral": row.get("oral"),
                "parenteral": row.get("parenteral"),
                "topical": row.get("topical"),
                "black_box_warning": row.get("black_box_warning"),
                "natural_product": row.get("natural_product"),
                "first_in_class": row.get("first_in_class"),
                "chirality": row.get("chirality"),
                "prodrug": row.get("prodrug"),
                "inorganic_flag": row.get("inorganic_flag"),
                "withdrawn_flag": row.get("withdrawn_flag"),
                "chemical_probe": row.get("chemical_probe"),
                "orphan": row.get("orphan"),
                "veterinary": row.get("veterinary"),
            }
            labels = ["molecule", "chembl_compound"]
            if row.get("therapeutic_flag") == 1 or row.get("max_phase") == 4:
                labels.append("drug_like_or_approved")
            if row.get("natural_product") == 1:
                labels.append("natural_product")
            writer.node({
                "node_id": node_id,
                "labels": labels,
                "name": name,
                "identifiers": {"source_id": chembl_id, "inchikey": inchikey, "molregno": row.get("molregno")},
                "properties": {k: v for k, v in props.items() if v not in ("", None, [])},
                "source": source,
                "provenance": provenance,
            })
            writer.alias(node_id, chembl_id, "identifier", source, provenance)
            writer.alias(node_id, name, "name", source, provenance)
            doc_props = dict(props)
            doc_props["notes"] = f"ChEMBL molecule type {props.get('molecule_type')}; max phase {props.get('max_phase')}; therapeutic flag {props.get('therapeutic_flag')}; first approval {props.get('first_approval')}."
            add_molecule_doc(writer, node_id=node_id, title=name, source=source, provenance=provenance, properties=doc_props, doc_type="chembl_molecule_profile")
    finally:
        con.close()

def build_primekg(root: Path, writer: KGWriter, edge_limit: int) -> None:
    source = "primekg"
    rel = "PrimeKG/kg.csv"
    path = root / rel
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            log_progress(source, idx - 1)
            if limit_reached(idx - 1, edge_limit):
                break
            x_id = f"primekg:{compact_id(row.get('x_type', 'entity'))}:{compact_id(row.get('x_source', 'source'))}:{compact_id(row.get('x_id', ''))}"
            y_id = f"primekg:{compact_id(row.get('y_type', 'entity'))}:{compact_id(row.get('y_source', 'source'))}:{compact_id(row.get('y_id', ''))}"
            provenance = {"path": rel, "row": idx}
            writer.node({"node_id": x_id, "labels": ["primekg_entity", compact_id(row.get("x_type", "entity"))], "name": clean_text(row.get("x_name")) or x_id, "identifiers": {"source_id": clean_text(row.get("x_id")), "source": clean_text(row.get("x_source"))}, "properties": {"entity_type": clean_text(row.get("x_type"))}, "source": source, "provenance": provenance})
            writer.node({"node_id": y_id, "labels": ["primekg_entity", compact_id(row.get("y_type", "entity"))], "name": clean_text(row.get("y_name")) or y_id, "identifiers": {"source_id": clean_text(row.get("y_id")), "source": clean_text(row.get("y_source"))}, "properties": {"entity_type": clean_text(row.get("y_type"))}, "source": source, "provenance": provenance})
            writer.alias(x_id, row.get("x_name"), "name", source, provenance)
            writer.alias(y_id, row.get("y_name"), "name", source, provenance)
            writer.edge({"subject_id": x_id, "predicate": clean_text(row.get("relation")), "object_id": y_id, "relation_label": clean_text(row.get("display_relation")), "properties": {"x_index": clean_text(row.get("x_index")), "y_index": clean_text(row.get("y_index"))}, "source": source, "provenance": provenance})


def load_drkg_relation_glossary(root: Path) -> Dict[str, Dict[str, str]]:
    rel = "DRKG/relation_glossary.tsv"
    path = root / rel
    out: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = clean_text(row.get("Relation-name"))
            if name:
                out[name] = {clean_text(k): clean_text(v) for k, v in row.items()}
    return out


def build_drkg(root: Path, writer: KGWriter, edge_limit: int) -> None:
    source = "drkg"
    rel = "DRKG/drkg.tsv"
    path = root / rel
    if not path.exists():
        return
    glossary = load_drkg_relation_glossary(root)
    for relation, meta in glossary.items():
        writer.document({
            "node_id": None,
            "title": relation,
            "text": f"DRKG relation {relation}. Data source: {meta.get('Data-source', '')}. Connected entity types: {meta.get('Connected entity-types', '')}. Interaction type: {meta.get('Interaction-type', '')}. Description: {meta.get('Description', '')}.",
            "doc_type": "relation_glossary",
            "source": source,
            "metadata": meta,
            "provenance": {"path": "DRKG/relation_glossary.tsv", "relation": relation},
        })
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            log_progress(source, idx - 1)
            if limit_reached(idx - 1, edge_limit):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            subj, pred, obj = [clean_text(x) for x in parts]
            s_id = f"drkg:{subj}"
            o_id = f"drkg:{obj}"
            provenance = {"path": rel, "row": idx}
            s_type = subj.split("::", 1)[0] if "::" in subj else "entity"
            o_type = obj.split("::", 1)[0] if "::" in obj else "entity"
            writer.node({"node_id": s_id, "labels": ["drkg_entity", compact_id(s_type)], "name": subj, "identifiers": {"source_id": subj}, "properties": {"entity_type": s_type}, "source": source, "provenance": provenance})
            writer.node({"node_id": o_id, "labels": ["drkg_entity", compact_id(o_type)], "name": obj, "identifiers": {"source_id": obj}, "properties": {"entity_type": o_type}, "source": source, "provenance": provenance})
            writer.edge({"subject_id": s_id, "predicate": pred, "object_id": o_id, "relation_label": (glossary.get(pred) or {}).get("Interaction-type", pred), "properties": glossary.get(pred, {}), "source": source, "provenance": provenance})


def chebi_basic_props(node: Dict[str, Any]) -> Dict[str, str]:
    props = {}
    for item in ((node.get("meta") or {}).get("basicPropertyValues") or []):
        pred = clean_text(item.get("pred"))
        val = clean_text(item.get("val"))
        if not pred or not val:
            continue
        key = pred.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        if key in {"generalized_empirical_formula", "inchi_key_string", "inchi_string", "smiles_string", "charge", "mass", "monoisotopic_mass"}:
            props[key] = val
    return props


def build_chebi(root: Path, writer: KGWriter, node_limit: int, edge_limit: int) -> None:
    source = "chebi"
    rel = "chebi/chebi.json"
    path = root / rel
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    graph = (data.get("graphs") or [{}])[0]
    count = 0
    for idx, node in enumerate(graph.get("nodes") or [], start=1):
        log_progress(source + ":nodes", idx - 1)
        if limit_reached(count, node_limit):
            break
        meta = node.get("meta") or {}
        if meta.get("deprecated"):
            continue
        name = clean_text(node.get("lbl"))
        if not name:
            continue
        count += 1
        node_id = chebi_id(clean_text(node.get("id")))
        provenance = {"path": rel, "node_index": idx}
        props = chebi_basic_props(node)
        writer.node({"node_id": node_id, "labels": ["chebi_entity", "ontology_node"], "name": name, "identifiers": {"source_id": clean_text(node.get("id")), "inchikey": props.get("inchi_key_string", "")}, "properties": props, "source": source, "provenance": provenance})
        writer.alias(node_id, name, "name", source, provenance)
        for syn in meta.get("synonyms") or []:
            writer.alias(node_id, syn.get("val"), "synonym", source, provenance)
        for xref in meta.get("xrefs") or []:
            writer.alias(node_id, xref.get("val"), "xref", source, provenance)
        doc_bits = [name]
        if props.get("generalized_empirical_formula"):
            doc_bits.append(f"formula {props['generalized_empirical_formula']}")
        if props.get("smiles_string"):
            doc_bits.append(f"SMILES {props['smiles_string']}")
        writer.document({"node_id": node_id, "title": name, "text": ". ".join(doc_bits) + ".", "doc_type": "ontology_profile", "source": source, "metadata": props, "provenance": provenance})
    edge_count = 0
    for idx, edge in enumerate(graph.get("edges") or [], start=1):
        log_progress(source + ":edges", idx - 1)
        if limit_reached(edge_count, edge_limit):
            break
        sub = chebi_id(clean_text(edge.get("sub")))
        obj = chebi_id(clean_text(edge.get("obj")))
        pred = clean_text(edge.get("pred"))
        if not sub or not obj or not pred:
            continue
        edge_count += 1
        writer.edge({"subject_id": sub, "predicate": pred, "object_id": obj, "relation_label": pred, "source": source, "provenance": {"path": rel, "edge_index": idx}})


def build_elementkg(root: Path, writer: KGWriter, limit: int, docs_per_entity: int) -> None:
    source = "elementkg_synthetic"
    rel = "ElementKG_Synthetic_Corpus_40w/knowlege_description_20w.json"
    path = root / rel
    if not path.exists():
        return
    for idx, row in enumerate(iter_json_array(path), start=1):
        log_progress(source, idx - 1)
        if limit_reached(idx - 1, limit):
            break
        entity = clean_text(row.get("entity"))
        if not entity:
            continue
        node_id = f"elementkg:{sha1_text(entity, 20)}"
        provenance = {"path": rel, "row": idx}
        writer.node({"node_id": node_id, "labels": ["synthetic_knowledge_entity"], "name": entity, "identifiers": {}, "properties": {}, "source": source, "provenance": provenance})
        writer.alias(node_id, entity, "name", source, provenance)
        for j, text in enumerate((row.get("corpus") or [])[:docs_per_entity], start=1):
            writer.document({"node_id": node_id, "title": entity, "text": text, "doc_type": "synthetic_knowledge_card", "source": source, "metadata": {"corpus_index": j}, "provenance": {"path": rel, "row": idx, "corpus_index": j}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified JSONL KG artifacts from the NS knowledge directory.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sources", nargs="+", default=["coconut", "npatlas", "lotus", "ttd", "chembl", "elementkg", "primekg", "drkg", "chebi"])
    parser.add_argument("--limit", type=int, default=100, help="Per-source entity/document limit for table/document sources. Use -1 for no limit.")
    parser.add_argument("--edge-limit", type=int, default=500, help="Per-source edge limit for graph sources. Use -1 for no limit.")
    parser.add_argument("--chebi-node-limit", type=int, default=100)
    parser.add_argument("--elementkg-docs-per-entity", type=int, default=1)
    parser.add_argument("--dedupe-edges", action="store_true", help="Deduplicate edges in memory. Disabled by default for large builds.")
    parser.add_argument("--max-alias-len", type=int, default=512, help="Skip aliases longer than this many characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    writer = KGWriter(args.out_dir, dedupe_edges=args.dedupe_edges, max_alias_len=args.max_alias_len)
    sources = set(args.sources)
    try:
        if "coconut" in sources:
            build_coconut(args.source_root, writer, args.limit)
        if "npatlas" in sources:
            build_npatlas(args.source_root, writer, args.limit)
        if "lotus" in sources:
            build_lotus(args.source_root, writer, args.limit)
        if "ttd" in sources or "ttd_approved" in sources:
            build_ttd(args.source_root, writer, args.limit)
        if "chembl" in sources:
            build_chembl(args.source_root, writer, args.limit)
        if "elementkg" in sources or "elementkg_synthetic" in sources:
            build_elementkg(args.source_root, writer, args.limit, args.elementkg_docs_per_entity)
        if "primekg" in sources:
            build_primekg(args.source_root, writer, args.edge_limit)
        if "drkg" in sources:
            build_drkg(args.source_root, writer, args.edge_limit)
        if "chebi" in sources:
            build_chebi(args.source_root, writer, args.chebi_node_limit, args.edge_limit)
    finally:
        writer.close()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(args.source_root),
        "out_dir": str(args.out_dir),
        "sources": sorted(sources),
        "limits": {"limit": args.limit, "edge_limit": args.edge_limit, "chebi_node_limit": args.chebi_node_limit, "elementkg_docs_per_entity": args.elementkg_docs_per_entity, "dedupe_edges": args.dedupe_edges, "max_alias_len": args.max_alias_len},
        "counts": writer.counts,
        "counts_by_source": writer.by_source,
        "elapsed_sec": round(time.time() - started, 3),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
