"""
NMRAgent utilities.
"""

import re
from typing import List, Optional


def extract_answer_smiles(text: str) -> str:
    """Extract SMILES from agent answer."""
    for m in re.finditer(r"<answer>\s*([^<\s]+)\s*</answer>", text):
        return m.group(1).strip("\"'")
    m = re.search(r"Answer:\s*([A-Za-z0-9@\[\]()=#\-+/\\%.]+)", text)
    return m.group(1).strip() if m else ""


def extract_smiles_from_trajectory(messages: list) -> list:
    """Extract all SMILES candidates from conversation trajectory."""
    candidates = []
    smiles_re = re.compile(r'^\s*\d+\.\s+[✓✗]?\s*([A-Za-z0-9@\[\]()=#\-+/\\%.]{5,})[\s(]')
    dict_re = re.compile(r"""['"]smiles['"]\s*:\s*['"]([^'"]+)['"]""")
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        for match in smiles_re.finditer(content):
            candidates.append(match.group(1))
        for match in dict_re.finditer(content):
            candidates.append(match.group(1))
    return candidates


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def extract_formula(text: str) -> Optional[str]:
    """Extract molecular formula from text."""
    m = re.search(r"C\d+H\d+[A-Z0-9]*", text)
    return m.group(0) if m else None
