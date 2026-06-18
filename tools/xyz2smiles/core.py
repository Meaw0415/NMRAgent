import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


@dataclass
class XYZToSmilesResult:
    smiles: Optional[str]
    largest_fragment_smiles: Optional[str]
    mol: Optional[Chem.Mol]


def _as_numpy_coords(coords: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {arr.shape}")
    return arr


def _normalize_atoms(atoms: Iterable[str]) -> List[str]:
    atom_list = [str(atom).strip() for atom in atoms]
    if not atom_list:
        raise ValueError("atoms must not be empty")
    if any(not atom for atom in atom_list):
        raise ValueError("atoms contains empty element symbols")
    return atom_list


def build_rdkit_molecule(atoms: Sequence[str], coords: Sequence[Sequence[float]]) -> Chem.Mol:
    atom_list = _normalize_atoms(atoms)
    coord_array = _as_numpy_coords(coords)
    if len(atom_list) != coord_array.shape[0]:
        raise ValueError(f"len(atoms)={len(atom_list)} does not match len(coords)={coord_array.shape[0]}")

    mol = Chem.RWMol()
    for atom_symbol in atom_list:
        mol.AddAtom(Chem.Atom(atom_symbol))

    conf = Chem.Conformer(len(atom_list))
    for idx, pos in enumerate(coord_array):
        conf.SetAtomPosition(idx, pos)
    mol.AddConformer(conf)
    return mol.GetMol()


def get_smiles_from_mol(mol: Chem.Mol, remove_stereo: bool = False) -> Tuple[Optional[str], Optional[str]]:
    try:
        mol_no_h = Chem.RemoveHs(mol)
        if remove_stereo:
            Chem.RemoveStereochemistry(mol_no_h)

        full_smiles = Chem.MolToSmiles(mol_no_h)
        full_smiles = Chem.CanonSmiles(full_smiles)

        fragments = Chem.rdmolops.GetMolFrags(mol_no_h, asMols=True, sanitizeFrags=True)
        if len(fragments) > 1:
            largest_fragment = max(fragments, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(largest_fragment)
            largest_fragment_smiles = Chem.CanonSmiles(Chem.MolToSmiles(largest_fragment))
            return None, largest_fragment_smiles

        return full_smiles, full_smiles
    except Exception:
        return None, None


def _worker_determine_bonds(queue: mp.Queue, mol_block: str, remove_stereo: bool) -> None:
    try:
        mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
        rdDetermineBonds.DetermineBonds(mol)
        smiles, largest = get_smiles_from_mol(mol, remove_stereo=remove_stereo)
        queue.put((smiles, largest))
    except Exception:
        queue.put((None, None))


def determine_bonds_with_timeout(
    mol: Chem.Mol,
    remove_stereo: bool = False,
    timeout_seconds: int = 0,
) -> Tuple[Optional[str], Optional[str]]:
    if timeout_seconds <= 0:
        try:
            mol_copy = Chem.Mol(mol)
            rdDetermineBonds.DetermineBonds(mol_copy)
            return get_smiles_from_mol(mol_copy, remove_stereo=remove_stereo)
        except Exception:
            return None, None

    mol_block = Chem.MolToMolBlock(mol)
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker_determine_bonds, args=(queue, mol_block, remove_stereo))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return None, None

    try:
        return queue.get_nowait()
    except Exception:
        return None, None


def xyz_to_smiles(
    atoms: Sequence[str],
    coords: Sequence[Sequence[float]],
    remove_stereo: bool = False,
    timeout_seconds: int = 0,
) -> XYZToSmilesResult:
    mol = build_rdkit_molecule(atoms=atoms, coords=coords)
    smiles, largest_fragment_smiles = determine_bonds_with_timeout(
        mol=mol,
        remove_stereo=remove_stereo,
        timeout_seconds=timeout_seconds,
    )
    return XYZToSmilesResult(
        smiles=smiles,
        largest_fragment_smiles=largest_fragment_smiles,
        mol=mol,
    )


def parse_xyz_text(xyz_text: str) -> Tuple[List[str], List[List[float]]]:
    lines = [line.rstrip() for line in xyz_text.splitlines()]
    lines = [line for line in lines if line.strip() or len(lines) < 2]
    if not lines:
        raise ValueError("XYZ text is empty")

    try:
        n_atoms = int(lines[0].strip())
        atom_lines = lines[2: 2 + n_atoms]
    except ValueError:
        atom_lines = lines

    atoms: List[str] = []
    coords: List[List[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not atoms:
        raise ValueError("No atoms parsed from XYZ text")
    return atoms, coords

