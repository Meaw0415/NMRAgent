"""
Atom feature generation utilities using RDKit ETKDGv3.

This module generates atom features (atom types and 3D conformations) from SMILES strings
for use with ChefNMR's Diffusion Transformer.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from typing import Dict, List, Optional, Tuple

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Atom type encoding
# Common atom types in organic molecules
ATOM_DECODER = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']
ATOM_ENCODER = {atom: idx for idx, atom in enumerate(ATOM_DECODER)}


def smiles_to_atom_features(
    smiles: str,
    max_n_atoms: int = 300,
    remove_h: bool = False,
    seed: int = 42,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Generate atom features from SMILES using RDKit ETKDGv3.

    Args:
        smiles: SMILES string
        max_n_atoms: Maximum number of atoms (for padding)
        remove_h: Whether to remove hydrogen atoms
        seed: Random seed for conformer generation

    Returns:
        Dictionary containing:
            - atom_coords: (max_n_atoms, 3) 3D coordinates
            - atom_one_hot: (max_n_atoms, num_atom_types) one-hot encoding of atom types
            - atom_mask: (max_n_atoms,) binary mask (1 for real atoms, 0 for padding)
            - n_atoms: int, actual number of atoms
        Returns None if molecule generation or conformer embedding fails.
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens if not removing them
        if not remove_h:
            mol = Chem.AddHs(mol)
            
        # Generate 3D conformer using ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = seed

        # Embed molecule
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id == -1:
            # Embedding failed, try with random coordinates
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            conf_id = 0

        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        except:
            pass  # Continue even if optimization fails

        # Extract atom features
        conf = mol.GetConformer(conf_id)
        n_atoms = mol.GetNumAtoms()

        if n_atoms > max_n_atoms:
            # Molecule too large
            return None

        # Get atom coordinates
        coords = np.zeros((max_n_atoms, 3), dtype=np.float32)
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]

        # Get atom types (one-hot encoding)
        atom_one_hot = np.zeros((max_n_atoms, len(ATOM_DECODER)), dtype=np.float32)
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_symbol = atom.GetSymbol()
            if atom_symbol in ATOM_ENCODER:
                atom_one_hot[i, ATOM_ENCODER[atom_symbol]] = 1.0
            else:
                # Unknown atom type, use last index
                atom_one_hot[i, -1] = 1.0

        # Create mask
        atom_mask = np.zeros(max_n_atoms, dtype=np.float32)
        atom_mask[:n_atoms] = 1.0

        return {
            'atom_coords': coords,
            'atom_one_hot': atom_one_hot,
            'atom_mask': atom_mask,
            'n_atoms': n_atoms,
        }

    except Exception as e:
        # Return None if any error occurs
        return None


def batch_smiles_to_atom_features(
    smiles_list: List[str],
    max_n_atoms: int = 300,
    remove_h: bool = False,
    seed: int = 42,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """
    Generate atom features for a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        max_n_atoms: Maximum number of atoms
        remove_h: Whether to remove hydrogen atoms
        seed: Random seed for conformer generation

    Returns:
        List of feature dictionaries (or None for failed cases)
    """
    results = []
    for i, smiles in enumerate(smiles_list):
        # Use different seed for each molecule for diversity
        features = smiles_to_atom_features(
            smiles,
            max_n_atoms=max_n_atoms,
            remove_h=remove_h,
            seed=seed + i
        )
        results.append(features)
    return results


def collate_atom_features(
    features_list: List[Dict[str, np.ndarray]]
) -> Dict[str, torch.Tensor]:
    """
    Collate atom features into batched tensors.

    Args:
        features_list: List of feature dictionaries

    Returns:
        Dictionary of batched tensors
    """
    batch_size = len(features_list)

    # Stack all features
    atom_coords = torch.from_numpy(
        np.stack([f['atom_coords'] for f in features_list], axis=0)
    )
    atom_one_hot = torch.from_numpy(
        np.stack([f['atom_one_hot'] for f in features_list], axis=0)
    )
    atom_mask = torch.from_numpy(
        np.stack([f['atom_mask'] for f in features_list], axis=0)
    )

    return {
        'atom_coords': atom_coords,
        'atom_one_hot': atom_one_hot,
        'atom_mask': atom_mask,
    }


def get_formula_counts(smiles: str) -> torch.Tensor:
    """
    Get formula counts from SMILES (count of each atom type).

    Args:
        smiles: SMILES string

    Returns:
        Tensor of shape (num_atom_types,) with counts
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(len(ATOM_DECODER), dtype=torch.long)

        # Add hydrogens to get correct count
        mol = Chem.AddHs(mol)

        counts = torch.zeros(len(ATOM_DECODER), dtype=torch.long)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in ATOM_ENCODER:
                counts[ATOM_ENCODER[symbol]] += 1

        return counts

    except:
        return torch.zeros(len(ATOM_DECODER), dtype=torch.long)
