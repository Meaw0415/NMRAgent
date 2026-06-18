"""
Dataset for loading preprocessed NMRGym data with pre-generated atom features.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List


class PreprocessedNMRGymDataset(Dataset):
    """
    Dataset for preprocessed NMRGym data with atom features already generated.

    This is much faster than NMRGymDiffusionDataset because:
    - Atom features are pre-computed during preprocessing
    - No RDKit calls needed during training
    - Molecules with invalid elements already filtered out
    """

    def __init__(
        self,
        preprocessed_jsonl_path: str,
        split: str = 'train',
        max_n_atoms: int = 300,
        h_shift_range: tuple = (-0.01, 0.01),
        c_shift_range: tuple = (-0.1, 0.1),
        shift_aug_p: float = 0.2,
    ):
        """
        Args:
            preprocessed_jsonl_path: Path to preprocessed JSONL file
            split: 'train', 'val', or 'test'
            max_n_atoms: Maximum number of atoms (for padding)
            h_shift_range: H NMR shift augmentation range (only for training)
            c_shift_range: C NMR shift augmentation range (only for training)
            shift_aug_p: Probability of shift augmentation
        """
        self.split = split
        self.max_n_atoms = max_n_atoms
        self.h_shift_range = h_shift_range
        self.c_shift_range = c_shift_range
        self.shift_aug_p = shift_aug_p

        # Load preprocessed data
        print(f"Loading {split} data from {preprocessed_jsonl_path}...")
        self.data = []

        with open(preprocessed_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Only load successful samples
                if item.get('status') == 'success':
                    self.data.append(item)

        print(f"[{split}] Loaded {len(self.data)} preprocessed samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        Returns:
            Dictionary containing:
                - nmr_data: Dict with 'raw_shifts', 'counts', 'types'
                - atom_features: Dict with 'atom_coords', 'atom_one_hot', 'atom_mask'
                - smiles: SMILES string
        """
        item = self.data[idx]
        smiles = item['smiles']

        # Parse NMR data
        h_shifts = item.get('h_shift', [])
        c_shifts = item.get('c_shift', [])

        # Convert to tensors
        h_shifts = torch.tensor(h_shifts, dtype=torch.float32)
        c_shifts = torch.tensor(c_shifts, dtype=torch.float32)

        # NMRGym doesn't have H count info, use 1 as default
        h_counts = torch.ones_like(h_shifts)
        c_counts = torch.zeros_like(c_shifts)

        # Shift augmentation (only for training)
        if self.split == 'train' and random.random() < self.shift_aug_p:
            h_bias = random.uniform(*self.h_shift_range)
            c_bias = random.uniform(*self.c_shift_range)
            h_shifts = h_shifts + h_bias
            c_shifts = c_shifts + c_bias

        # Build input sequence: [H peaks, C peaks]
        all_shifts = torch.cat([h_shifts, c_shifts])
        all_counts = torch.cat([h_counts, c_counts])
        types = torch.cat([
            torch.zeros_like(h_shifts, dtype=torch.long),  # H = 0
            torch.ones_like(c_shifts, dtype=torch.long)    # C = 1
        ])

        nmr_data = {
            'raw_shifts': all_shifts,
            'counts': all_counts,
            'types': types,
        }

        # Load pre-generated atom features and add padding
        atom_features_raw = item['atom_features']
        n_atoms = atom_features_raw['n_atoms']

        # Convert to tensors (without padding)
        atom_coords = torch.tensor(atom_features_raw['atom_coords'], dtype=torch.float32)  # (n_atoms, 3)
        atom_one_hot = torch.tensor(atom_features_raw['atom_one_hot'], dtype=torch.float32)  # (n_atoms, 10)

        # Create padding
        n_pad = self.max_n_atoms - n_atoms
        if n_pad > 0:
            # Pad coords with zeros
            atom_coords = torch.cat([
                atom_coords,
                torch.zeros(n_pad, 3, dtype=torch.float32)
            ], dim=0)

            # Pad one_hot with zeros
            atom_one_hot = torch.cat([
                atom_one_hot,
                torch.zeros(n_pad, atom_one_hot.shape[1], dtype=torch.float32)
            ], dim=0)

        # Create atom mask (1 for real atoms, 0 for padding)
        atom_mask = torch.zeros(self.max_n_atoms, dtype=torch.float32)
        atom_mask[:n_atoms] = 1.0

        atom_features = {
            'atom_coords': atom_coords,       # (max_n_atoms, 3)
            'atom_one_hot': atom_one_hot,     # (max_n_atoms, 10)
            'atom_mask': atom_mask,           # (max_n_atoms,)
        }

        return {
            'nmr_data': nmr_data,
            'atom_features': atom_features,
            'smiles': smiles,
        }


def preprocessed_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for preprocessed NMRGym dataset.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched tensors
    """
    # Extract components
    nmr_raw_shifts = [item['nmr_data']['raw_shifts'] for item in batch]
    nmr_counts = [item['nmr_data']['counts'] for item in batch]
    nmr_types = [item['nmr_data']['types'] for item in batch]

    atom_coords = torch.stack([item['atom_features']['atom_coords'] for item in batch])
    atom_one_hot = torch.stack([item['atom_features']['atom_one_hot'] for item in batch])
    atom_mask = torch.stack([item['atom_features']['atom_mask'] for item in batch])

    smiles_list = [item['smiles'] for item in batch]

    # Pad NMR sequences
    nmr_raw_shifts_padded = pad_sequence(nmr_raw_shifts, batch_first=True, padding_value=0.0)
    nmr_counts_padded = pad_sequence(nmr_counts, batch_first=True, padding_value=0.0)
    nmr_types_padded = pad_sequence(nmr_types, batch_first=True, padding_value=-1)

    # Create padding masks
    nmr_padding_mask = (nmr_types_padded == -1)
    nmr_types_padded[nmr_padding_mask] = 0

    return {
        'nmr_data': {
            'shifts': nmr_raw_shifts_padded,
            'counts': nmr_counts_padded,
            'types': nmr_types_padded,
            'padding_mask': nmr_padding_mask,
        },
        'atom_features': {
            'atom_coords': atom_coords,
            'atom_one_hot': atom_one_hot,
            'atom_mask': atom_mask,
        },
        'smiles_list': smiles_list,
    }
