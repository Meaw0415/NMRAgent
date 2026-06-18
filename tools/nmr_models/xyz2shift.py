"""
Predict NMR shifts directly from atoms + xyz coordinates using the same
UniMol/NMRNet checkpoints used by nmrnet_predict.py.

This module is for fixed-graph, coordinate-aware workflows where we want to
evaluate or optimize a specific conformer instead of regenerating geometry
from SMILES.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Match nmrnet_predict.py path setup
_third_party = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if os.path.isdir(_third_party) and _third_party not in sys.path:
    sys.path.insert(0, _third_party)

from .nmrnet_predict import (  # noqa: E402
    _NMRNET_AVAILABLE,
    _debug,
    _get_default_model_dir,
    _get_predictor,
    _get_equi_class,
    _merge_equi_nmr,
)


def _as_float32_coords(coords: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {arr.shape}")
    return arr


def _normalize_atoms(atoms: Sequence[str]) -> list[str]:
    atom_list = [str(atom).strip() for atom in atoms]
    if not atom_list:
        raise ValueError("atoms must not be empty")
    if any(not atom for atom in atom_list):
        raise ValueError("atoms contains empty symbols")
    return atom_list


def _build_atom_mask_with_hydrogens(atoms: Sequence[str]) -> list[int]:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    atomic_numbers = []
    for symbol in atoms:
        z = int(pt.GetAtomicNumber(str(symbol)))
        if z <= 0:
            raise ValueError(f"Unknown element symbol: {symbol}")
        atomic_numbers.append(z)
    if len(atomic_numbers) > 511:
        raise ValueError(f"Too many atoms for current UniMol padding: {len(atomic_numbers)}")
    return [0] + atomic_numbers + [0] * (512 - 1 - len(atomic_numbers))


def _reference_equi_class(reference_smiles: str, expected_atoms: int) -> np.ndarray:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(reference_smiles)
    if mol is None:
        raise ValueError("Invalid reference_smiles")
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() != expected_atoms:
        raise ValueError(
            f"reference_smiles atom count mismatch: expected {expected_atoms}, got {mol_h.GetNumAtoms()}"
        )
    return _get_equi_class(mol_h)


def _predict_points3d_batch_internal(
    atoms_list: list[list[str]],
    coordinates_list: list[np.ndarray],
    model_dir: str,
    fast: bool,
) -> Dict[str, object]:
    clf = _get_predictor(model_dir, fast=fast)
    infer_data = {
        "atoms": atoms_list,
        "coordinates": [coords.tolist() for coords in coordinates_list],
        "atom_target": [[0.0] * 512 for _ in atoms_list],
        "atom_mask": [_build_atom_mask_with_hydrogens(atoms) for atoms in atoms_list],
    }
    clf.predict(infer_data, datatype="points3d")
    return {
        "cv_pred": clf.cv_pred,
        "cv_label_mask": clf.cv_label_mask,
        "cv_index_mask": np.array(clf.cv_index_mask.tolist()).astype(np.int8),
    }


def predict_shifts_from_xyz(
    atoms: Sequence[str],
    coords: Sequence[Sequence[float]],
    reference_smiles: str | None = None,
    model_dir: str | None = None,
    fast: bool = False,
) -> Dict[str, object]:
    if not _NMRNET_AVAILABLE:
        return {
            "H_shifts": np.array([]),
            "C_shifts": np.array([]),
            "valid": False,
            "error": "NMRNet not available",
        }

    if model_dir is None:
        model_dir = _get_default_model_dir()

    try:
        atom_list = _normalize_atoms(atoms)
        coord_array = _as_float32_coords(coords)
        if len(atom_list) != coord_array.shape[0]:
            raise ValueError(f"len(atoms)={len(atom_list)} != len(coords)={coord_array.shape[0]}")

        packed = _predict_points3d_batch_internal(
            atoms_list=[atom_list],
            coordinates_list=[coord_array],
            model_dir=model_dir,
            fast=fast,
        )

        cv_pred = packed["cv_pred"][0].astype(np.float32)
        cv_label_mask = packed["cv_label_mask"][0]
        index_mask = packed["cv_index_mask"][0]
        nmr_predict = cv_pred[cv_label_mask]
        mol_index = index_mask[cv_label_mask]

        h_shifts = np.sort(nmr_predict[mol_index == 1])
        if reference_smiles:
            equi_class = _reference_equi_class(reference_smiles, len(atom_list))
            c_shifts = np.sort(_merge_equi_nmr(nmr_predict, mol_index, equi_class, element_id=6))
        else:
            c_shifts = np.sort(nmr_predict[mol_index == 6])

        atom_data = []
        mask_positions = np.where(cv_label_mask)[0]
        atom_indices = mask_positions - 1
        for k in range(len(nmr_predict)):
            elem_num = int(mol_index[k])
            if elem_num not in (1, 6):
                continue
            atom_idx = int(atom_indices[k])
            if atom_idx < 0 or atom_idx >= len(atom_list):
                continue
            atom_data.append(
                {
                    "atom_idx": atom_idx,
                    "element": atom_list[atom_idx],
                    "shift": float(nmr_predict[k]),
                }
            )

        return {
            "H_shifts": np.array(h_shifts),
            "C_shifts": np.array(c_shifts),
            "atom_data": atom_data,
            "valid": True,
        }
    except Exception as e:
        _debug(f"predict_shifts_from_xyz error={e}\n{traceback.format_exc()}")
        return {
            "H_shifts": np.array([]),
            "C_shifts": np.array([]),
            "valid": False,
            "error": str(e),
        }


def predict_shifts_from_xyz_batch(
    batch: Sequence[Dict[str, Sequence]],
    model_dir: str | None = None,
    fast: bool = False,
) -> list[Dict[str, object]]:
    if not _NMRNET_AVAILABLE:
        return [
            {
                "H_shifts": np.array([]),
                "C_shifts": np.array([]),
                "valid": False,
                "error": "NMRNet not available",
            }
            for _ in batch
        ]

    if model_dir is None:
        model_dir = _get_default_model_dir()

    atoms_list: list[list[str]] = []
    coordinates_list: list[np.ndarray] = []
    valid_rows: list[int] = []
    results: list[Dict[str, object] | None] = [None] * len(batch)

    for idx, row in enumerate(batch):
        try:
            atom_list = _normalize_atoms(row["atoms"])
            coord_array = _as_float32_coords(row["coords"])
            if len(atom_list) != coord_array.shape[0]:
                raise ValueError(f"len(atoms)={len(atom_list)} != len(coords)={coord_array.shape[0]}")
            atoms_list.append(atom_list)
            coordinates_list.append(coord_array)
            valid_rows.append(idx)
        except Exception as e:
            results[idx] = {
                "H_shifts": np.array([]),
                "C_shifts": np.array([]),
                "valid": False,
                "error": str(e),
            }

    if not atoms_list:
        return [r for r in results if r is not None]

    try:
        packed = _predict_points3d_batch_internal(
            atoms_list=atoms_list,
            coordinates_list=coordinates_list,
            model_dir=model_dir,
            fast=fast,
        )
        cv_preds = packed["cv_pred"]
        cv_label_masks = packed["cv_label_mask"]
        cv_index_masks = packed["cv_index_mask"]

        for local_idx, batch_idx in enumerate(valid_rows):
            cv_pred = cv_preds[local_idx].astype(np.float32)
            cv_label_mask = cv_label_masks[local_idx]
            index_mask = cv_index_masks[local_idx]
            nmr_predict = cv_pred[cv_label_mask]
            mol_index = index_mask[cv_label_mask]
            h_shifts = np.sort(nmr_predict[mol_index == 1])
            reference_smiles = batch[batch_idx].get("reference_smiles")
            if reference_smiles:
                equi_class = _reference_equi_class(reference_smiles, len(atoms_list[local_idx]))
                c_shifts = np.sort(_merge_equi_nmr(nmr_predict, mol_index, equi_class, element_id=6))
            else:
                c_shifts = np.sort(nmr_predict[mol_index == 6])

            atom_list = atoms_list[local_idx]
            atom_data = []
            mask_positions = np.where(cv_label_mask)[0]
            atom_indices = mask_positions - 1
            for k in range(len(nmr_predict)):
                elem_num = int(mol_index[k])
                if elem_num not in (1, 6):
                    continue
                atom_idx = int(atom_indices[k])
                if atom_idx < 0 or atom_idx >= len(atom_list):
                    continue
                atom_data.append(
                    {
                        "atom_idx": atom_idx,
                        "element": atom_list[atom_idx],
                        "shift": float(nmr_predict[k]),
                    }
                )

            results[batch_idx] = {
                "H_shifts": np.array(h_shifts),
                "C_shifts": np.array(c_shifts),
                "atom_data": atom_data,
                "valid": True,
            }
    except Exception as e:
        _debug(f"predict_shifts_from_xyz_batch error={e}\n{traceback.format_exc()}")
        for batch_idx in valid_rows:
            results[batch_idx] = {
                "H_shifts": np.array([]),
                "C_shifts": np.array([]),
                "valid": False,
                "error": str(e),
            }

    return [r if r is not None else {"H_shifts": np.array([]), "C_shifts": np.array([]), "valid": False, "error": "Unknown failure"} for r in results]


def _load_payload_from_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Predict NMR shifts from atoms + xyz coordinates using NMRNet/UniMol")
    ap.add_argument("--input-json", type=str, default="", help='JSON file with {"atoms": [...], "coords": [[x,y,z], ...]}')
    ap.add_argument("--atoms", nargs="*", default=None)
    ap.add_argument("--coords-json", type=str, default="")
    ap.add_argument("--reference-smiles", type=str, default="", help="Optional fixed graph SMILES for equivalent-carbon merging")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    if args.input_json:
        payload = _load_payload_from_json(args.input_json)
        atoms = payload["atoms"]
        coords = payload["coords"]
    else:
        if not args.atoms or not args.coords_json:
            raise SystemExit("Provide either --input-json, or both --atoms and --coords-json.")
        atoms = args.atoms
        coords = json.loads(args.coords_json)

    result = predict_shifts_from_xyz(
        atoms=atoms,
        coords=coords,
        reference_smiles=args.reference_smiles or None,
        fast=args.fast,
    )
    output = {
        "valid": result["valid"],
        "H_shifts": result["H_shifts"].tolist() if isinstance(result["H_shifts"], np.ndarray) else result["H_shifts"],
        "C_shifts": result["C_shifts"].tolist() if isinstance(result["C_shifts"], np.ndarray) else result["C_shifts"],
        "atom_data": result.get("atom_data", []),
        "error": result.get("error", ""),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
