"""
NMR Knowledge Graph RAG Tool

Retrieves chemical knowledge from the KG before NMR analysis.
Data files:
  - 10m_elementkg_release.csv : 10M+ triples (Molecule, Reaction, Experiment, etc.)
  - knowlege_description_20w.json : entity descriptions

Usage:
  This tool MUST be called first. It searches by molecular formula to retrieve
  known SMILES candidates and chemical context from PubChem-derived data.
"""

import json
import os
import re
import subprocess

from .decorator import tool
from .path_utils import artifact_path

KG_DATA_DIR = os.environ.get("NMR_KG_DATA_DIR", str(artifact_path("kg")))
CSV_PATH = os.path.join(KG_DATA_DIR, "10m_elementkg_release.csv")
JSON_PATH = os.path.join(KG_DATA_DIR, "knowlege_description_20w.json")

_JSON_CACHE = None


def _load_json():
    global _JSON_CACHE
    if _JSON_CACHE is None:
        try:
            with open(JSON_PATH, "r") as f:
                _JSON_CACHE = json.load(f)
        except Exception:
            _JSON_CACHE = []
    return _JSON_CACHE


def _grep(pattern, flags=None, max_results=50, timeout=20, fixed=False):
    """Run grep on the KG CSV and return stdout string."""
    if not os.path.exists(CSV_PATH):
        return ""
    cmd = ["grep"]
    if fixed:
        cmd.append("-F")
    if flags:
        cmd.extend(flags)
    cmd += ["-m", str(max_results), pattern, CSV_PATH]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


def _search_molecules_by_formula(formula: str, max_molecules: int = 6) -> list:
    """
    Two-stage search:
      1. grep for the formula string to get molecule IDs
      2. grep for those molecule IDs to get SMILES and IUPAC names
    Returns list of dicts: {mol_id, smiles, name}
    """
    # Stage 1: find molecule IDs with matching formula
    raw = _grep(formula, fixed=True, max_results=100)
    mol_ids = []
    seen = set()
    for line in raw.splitlines():
        if "PUBCHEM_MOLECULAR_FORMULA_IS" in line:
            parts = line.split(",")
            if len(parts) >= 5 and parts[0] == "Molecule":
                mol_id = parts[1].strip()
                if mol_id not in seen:
                    mol_ids.append(mol_id)
                    seen.add(mol_id)
                    if len(mol_ids) >= max_molecules:
                        break

    if not mol_ids:
        return []

    # Stage 2: get SMILES + IUPAC name for those molecule IDs in one grep
    mol_pattern = "|".join(re.escape(m) for m in mol_ids)
    # Only pull SMILES and IUPAC_NAME_IS (not MARKUP) lines
    stage2_pattern = (
        f"^Molecule,({mol_pattern}),"
        r"(PUBCHEM_CONNECTIVITY_SMILES_IS|PUBCHEM_IUPAC_NAME_IS),"
    )
    raw2 = _grep(stage2_pattern, flags=["-E"], max_results=60)

    molecules: dict = {}
    for line in raw2.splitlines():
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        mol_id = parts[1].strip()
        rel = parts[2].strip()
        # Tail value: rejoin in case SMILES contains commas (rare)
        val = ",".join(parts[4:]).strip()
        if mol_id not in molecules:
            molecules[mol_id] = {}
        if "SMILES" in rel:
            molecules[mol_id]["smiles"] = val
        elif rel == "PUBCHEM_IUPAC_NAME_IS":
            molecules[mol_id]["name"] = val

    result = []
    for mol_id in mol_ids:
        props = molecules.get(mol_id, {})
        if "smiles" in props:
            result.append(
                {
                    "mol_id": mol_id,
                    "smiles": props["smiles"],
                    "name": props.get("name", "Unknown"),
                }
            )
    return result


def _search_general(keyword: str, max_results: int = 15) -> list:
    """General case-insensitive grep, skip noisy relations."""
    SKIP_RELS = {
        "PUBCHEM_CACTVS_SUBSKEYS_IS",
        "PUBCHEM_BONDANNOTATIONS_IS",
        "PUBCHEM_COORDINATE_TYPE_IS",
    }
    raw = _grep(keyword, flags=["-i"], max_results=max_results)
    triples = []
    for line in raw.splitlines():
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 5:
            rel = parts[2].strip()
            if rel in SKIP_RELS:
                continue
            triples.append(
                {
                    "head_type": parts[0].strip(),
                    "head": parts[1].strip(),
                    "relation": rel,
                    "tail": parts[4].strip(),
                }
            )
    return triples


def _search_json_descriptions(keyword: str) -> list:
    data = _load_json()
    kw_lower = keyword.lower()
    results = []
    for item in data:
        entity = item.get("entity", "")
        if kw_lower in entity.lower() or entity.lower() in kw_lower:
            corpus = item.get("corpus", [])
            desc = corpus[0] if isinstance(corpus, list) and corpus else str(corpus)
            results.append({"entity": entity, "description": desc[:600]})
    return results


def _run_kg_rag_impl(query: str) -> dict:
    """
    Internal KG RAG retrieval. Returns dict with 'observation' and 'valid'.
    Used both by the @tool wrapper and for direct programmatic pre-injection.
    """
    if not query or not query.strip():
        return {
            "observation": "Error: Empty query. Provide formula, compound name, or NMR features.",
            "valid": 0,
        }

    query = query.strip()
    lines = ["=== Knowledge Graph RAG Results ==="]
    total_found = 0

    # --- 1. Formula-based molecule search (most targeted for NMR) ---
    formula_match = re.search(r"C\d+H\d+[A-Z0-9]*", query)
    if formula_match:
        formula = formula_match.group(0)
        lines.append(f"\n[Candidate molecules with formula {formula}]")
        molecules = _search_molecules_by_formula(formula)
        if molecules:
            for m in molecules:
                lines.append(f"  • {m['name']}: {m['smiles']}")
            total_found += len(molecules)
        else:
            lines.append(f"  No molecules found for formula {formula} in KG.")

    # --- 2. JSON entity descriptions ---
    # Try query words (skip short tokens and formula tokens)
    search_words = [
        w for w in query.split()
        if len(w) > 3 and not re.match(r"^[A-Z]\d", w)
    ]
    desc_results = _search_json_descriptions(query)
    if not desc_results and search_words:
        desc_results = _search_json_descriptions(search_words[0])
    for d in desc_results[:2]:
        lines.append(f"\n[Entity: {d['entity']}]\n{d['description']}")
        total_found += 1

    # --- 3. General keyword KG search (fallback) ---
    if total_found < 2 and search_words:
        kw = search_words[0]
        triples = _search_general(kw, max_results=15)
        if triples:
            lines.append(f"\n[KG triples for '{kw}']")
            # Group by relation
            by_rel: dict = {}
            for t in triples[:20]:
                rel = t["relation"]
                by_rel.setdefault(rel, []).append(f"{t['head']} → {t['tail']}")
            for rel, items in list(by_rel.items())[:5]:
                lines.append(f"  [{rel}]: " + " | ".join(items[:3]))
            total_found += len(triples)

    # --- 4. Reaction / Reagent search for the formula ---
    if formula_match and total_found < 3:
        formula = formula_match.group(0)
        reaction_raw = _grep(formula, flags=["-i"], max_results=20)
        reagent_lines = []
        for line in reaction_raw.splitlines():
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 5 and parts[0] in ("Reactant", "Product", "Reagent"):
                reagent_lines.append(
                    f"  {parts[0]}: {parts[1]} --[{parts[2]}]--> {parts[4]}"
                )
        if reagent_lines:
            lines.append(f"\n[Reaction context for {formula}]")
            lines.extend(reagent_lines[:8])
            total_found += len(reagent_lines)

    if total_found == 0:
        lines.append(f"\nNo KG knowledge found for: '{query}'")
        lines.append("Proceeding with NMR spectral analysis directly.")
        valid = 0
    else:
        valid = 1

    lines.append("\n=== End KG Results ===")
    return {"observation": "\n".join(lines), "valid": valid}


@tool(
    name="kg_rag_retrieve",
    description=(
        "[CALL THIS FIRST] Retrieve relevant chemical knowledge from the NMR "
        "Knowledge Graph (RAG). Searches 10M+ chemical triples and entity "
        "descriptions to provide candidate SMILES, compound names, and "
        "reaction context for the given molecular formula or query. "
        "ALWAYS call this before nmr_retrieve, nmr_denovo, or nmr_rerank."
    ),
)
def kg_rag_retrieve(query: str) -> dict:
    """
    Retrieve relevant chemical knowledge from the NMR Knowledge Graph (RAG).

    This tool MUST be called FIRST before any other NMR analysis tool.
    It searches the Knowledge Graph for candidate molecular structures,
    entity descriptions, and reaction context based on the given query.

    Args:
        query (str): Search query. Best results with molecular formula
                     (e.g. "C9H12O2") or compound name (e.g. "benzene ether").
                     Can also include NMR-relevant keywords.

    Returns:
        dict: Contains:
            - observation: Retrieved knowledge summary (candidate SMILES,
                           entity descriptions, KG triples)
            - valid: 1 if results found, 0 if no results
    """
    return _run_kg_rag_impl(query)
