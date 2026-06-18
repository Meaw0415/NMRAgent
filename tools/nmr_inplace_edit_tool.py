"""Late-stage RDKit in-place SMILES edit utilities."""

from __future__ import annotations

from typing import Any, Dict

from rdkit import Chem

from .decorator import tool


def _canonicalize_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _smiles_result(observation: str, mol: Chem.Mol | None, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"observation": observation}
    if mol is not None:
        try:
            result["smiles"] = _canonicalize_mol(mol)
            result["valid"] = True
        except Exception as exc:
            result["smiles"] = ""
            result["valid"] = False
            result["error"] = str(exc)
    else:
        result["smiles"] = ""
        result["valid"] = False
    result.update(extra)
    return result


@tool(name="nmr_canonicalize_smiles")
def nmr_canonicalize_smiles(smiles: str) -> Dict[str, Any]:
    """Canonicalize a SMILES string with RDKit."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return _smiles_result("Invalid SMILES.", None)
    return _smiles_result("Canonicalized SMILES.", mol, original_smiles=smiles)


@tool(name="nmr_replace_atom")
def nmr_replace_atom(smiles: str, atom_idx: int, new_atomic_num: int) -> Dict[str, Any]:
    """Replace one atom in a molecule by atom index."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return _smiles_result("Invalid SMILES.", None, atom_idx=atom_idx, new_atomic_num=new_atomic_num)
    if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
        return _smiles_result("Atom index out of range.", None, atom_idx=atom_idx, new_atomic_num=new_atomic_num)
    if new_atomic_num <= 0:
        return _smiles_result("Invalid atomic number.", None, atom_idx=atom_idx, new_atomic_num=new_atomic_num)

    rw = Chem.RWMol(mol)
    new_atom = Chem.Atom(int(new_atomic_num))
    try:
        rw.ReplaceAtom(int(atom_idx), new_atom, preserveProps=False)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
    except Exception as exc:
        return _smiles_result(
            f"Atom replacement failed: {exc}",
            None,
            atom_idx=atom_idx,
            new_atomic_num=new_atomic_num,
        )
    return _smiles_result(
        "Replaced atom successfully.",
        new_mol,
        atom_idx=atom_idx,
        new_atomic_num=new_atomic_num,
    )


@tool(name="nmr_delete_atom")
def nmr_delete_atom(smiles: str, atom_idx: int) -> Dict[str, Any]:
    """Delete one atom from a molecule by atom index."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return _smiles_result("Invalid SMILES.", None, atom_idx=atom_idx)
    if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
        return _smiles_result("Atom index out of range.", None, atom_idx=atom_idx)

    rw = Chem.RWMol(mol)
    try:
        rw.RemoveAtom(int(atom_idx))
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
    except Exception as exc:
        return _smiles_result(f"Atom deletion failed: {exc}", None, atom_idx=atom_idx)
    return _smiles_result("Deleted atom successfully.", new_mol, atom_idx=atom_idx)
