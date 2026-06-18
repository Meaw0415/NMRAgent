"""
Dataset for NMRGym data with atom feature generation for diffusion training.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.atom_features import smiles_to_atom_features, ATOM_DECODER


class NMRGymDiffusionDataset(Dataset):
    """
    Dataset for NMRGym data that generates both NMR features and atom features.

    This dataset:
    1. Loads NMRGym JSONL data (SMILES + NMR shifts)
    2. Processes NMR data for the encoder
    3. Generates 3D atom features from SMILES using RDKit ETKDGv3
    """

    def __init__(
        self,
        jsonl_path: str,
        max_n_atoms: int = 300,
        remove_h: bool = False,
        split: str = 'train',
        h_shift_range: tuple = (-0.01, 0.01),
        c_shift_range: tuple = (-0.1, 0.1),
        shift_aug_p: float = 0.2,
        seed: int = 42,
        cache_atom_features: bool = True,
    ):
        """
        Args:
            jsonl_path: Path to NMRGym JSONL file
            max_n_atoms: Maximum number of atoms (for padding)
            remove_h: Whether to remove hydrogen atoms
            split: 'train', 'val', or 'test'
            h_shift_range: H NMR shift augmentation range (only for training)
            c_shift_range: C NMR shift augmentation range (only for training)
            shift_aug_p: Probability of shift augmentation
            seed: Random seed for reproducibility
            cache_atom_features: Whether to pre-compute and cache atom features
        """
        self.max_n_atoms = max_n_atoms
        self.remove_h = remove_h
        self.split = split
        self.h_shift_range = h_shift_range
        self.c_shift_range = c_shift_range
        self.shift_aug_p = shift_aug_p
        self.seed = seed

        # Load data from JSONL
        print(f"Loading {split} data from {jsonl_path}...")
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_idx}: {e}")

        print(f"[{split}] Loaded {len(self.data)} samples")

        # Optionally cache atom features
        self.atom_features_cache = None
        if cache_atom_features:
            print(f"Pre-computing atom features for {len(self.data)} molecules...")
            self.atom_features_cache = []
            failed_count = 0

            for i, item in enumerate(self.data):
                if i % 1000 == 0:
                    print(f"  Processed {i}/{len(self.data)} molecules...")

                features = smiles_to_atom_features(
                    item['smiles'],
                    max_n_atoms=max_n_atoms,
                    remove_h=remove_h,
                    seed=seed + i,
                )

                if features is None:
                    failed_count += 1

                self.atom_features_cache.append(features)

            print(f"  Atom feature generation complete. Failed: {failed_count}/{len(self.data)}")

            # Filter out failed cases
            valid_indices = [i for i, f in enumerate(self.atom_features_cache) if f is not None]
            self.data = [self.data[i] for i in valid_indices]
            self.atom_features_cache = [self.atom_features_cache[i] for i in valid_indices]

            print(f"[{split}] Valid samples after filtering: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        Returns:
            Dictionary containing:
                - nmr_data: Dict with 'h_shifts', 'c_shifts', 'h_counts', 'c_counts', 'types'
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

        # Get atom features (from cache or generate on-the-fly)
        if self.atom_features_cache is not None:
            atom_features = self.atom_features_cache[idx]
        else:
            atom_features = smiles_to_atom_features(
                smiles,
                max_n_atoms=self.max_n_atoms,
                remove_h=self.remove_h,
                seed=self.seed + idx,
            )

        # Convert to tensors
        if atom_features is not None:
            atom_features = {
                'atom_coords': torch.from_numpy(atom_features['atom_coords']),
                'atom_one_hot': torch.from_numpy(atom_features['atom_one_hot']),
                'atom_mask': torch.from_numpy(atom_features['atom_mask']),
            }
        else:
            # Return dummy features if generation failed (should not happen if caching)
            atom_features = {
                'atom_coords': torch.zeros((self.max_n_atoms, 3), dtype=torch.float32),
                'atom_one_hot': torch.zeros((self.max_n_atoms, len(ATOM_DECODER)), dtype=torch.float32),
                'atom_mask': torch.zeros(self.max_n_atoms, dtype=torch.float32),
            }

        return {
            'nmr_data': nmr_data,
            'atom_features': atom_features,
            'smiles': smiles,
        }


def nmrgym_diffusion_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for NMRGym diffusion dataset.

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
