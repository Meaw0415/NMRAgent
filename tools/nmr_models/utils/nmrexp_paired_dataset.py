"""
NMR Experimental Paired Dataset - Load paired H/C spectra from pickle
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
from typing import List, Dict
from ..utils.nmr2smiles_dataset import get_formula_counts


class NMRExpPairedDataset(Dataset):
    """
    Dataset for paired H/C NMR spectra from experimental data.
    Each sample contains ONE paired H spectrum and C spectrum from the same source.
    """

    def __init__(
        self,
        pickle_path: str,
        split: str = 'train',
        val_size: int = 10000,
        seed: int = 0,
    ):
        """
        Args:
            pickle_path: Path to paired pickle file
            split: 'train' or 'val'
            val_size: Number of validation samples
            seed: Random seed for train/val split
        """
        print(f"Loading paired data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Loaded {len(self.data)} paired spectra")

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.data))

        if split == 'val':
            self.indices = indices[:val_size].tolist()
        elif split == 'train':
            self.indices = indices[val_size:].tolist()
        else:
            raise ValueError(f"Invalid split: {split}")

        print(f"Split '{split}': {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            {
                'smiles': str,
                'h_shift': List[float],
                'c_shift': List[float],
            }
        """
        data_idx = self.indices[idx]
        item = self.data[data_idx]

        return {
            'smiles': item['smiles'],
            'h_shift': item['h_shift'],
            'c_shift': item['c_shift'],
        }


def nmrexp_paired_collate_fn(batch, tokenizer, max_smiles_len=256, pad_idx=0):
    """
    Collate function for paired NMR data.

    Matches the format of train_nmrgym.py + handles SMILES tokenization.
    Combines H and C shifts into a single spectrum with type indicators.
    H peaks: type = 0
    C peaks: type = 1
    """
    raw_shifts_list = []
    counts_list = []
    types_list = []
    smiles_ids_list = []
    smiles_list = []
    formula_counts_list = []

    # Get special token IDs
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    for item in batch:
        h_shifts = item['h_shift']
        c_shifts = item['c_shift']
        smiles = item['smiles']

        # Concatenate H and C shifts
        all_shifts = torch.tensor(h_shifts + c_shifts, dtype=torch.float32)

        # Type indicators: 0 for H, 1 for C
        types = torch.cat([
            torch.zeros(len(h_shifts), dtype=torch.long),
            torch.ones(len(c_shifts), dtype=torch.long)
        ])

        # Counts: 1 for H (default), 0 for C
        counts = torch.cat([
            torch.ones(len(h_shifts), dtype=torch.float32),
            torch.zeros(len(c_shifts), dtype=torch.float32)
        ])

        raw_shifts_list.append(all_shifts)
        counts_list.append(counts)
        types_list.append(types)
        smiles_list.append(smiles)

        # Tokenize SMILES
        token_ids = tokenizer.encode(smiles, add_special_tokens=False)
        smiles_ids = [bos_token_id] + token_ids + [eos_token_id]
        if len(smiles_ids) > max_smiles_len:
            smiles_ids = smiles_ids[:max_smiles_len - 1] + [eos_token_id]
        smiles_ids_list.append(torch.tensor(smiles_ids, dtype=torch.long))

        # Get formula counts from SMILES
        formula_counts = get_formula_counts(smiles)
        formula_counts_list.append(formula_counts)

    # Pad sequences
    raw_shifts_padded = pad_sequence(raw_shifts_list, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts_list, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types_list, batch_first=True, padding_value=-1)
    smiles_padded = pad_sequence(smiles_ids_list, batch_first=True, padding_value=pad_idx)
    formula_counts_stacked = torch.stack(formula_counts_list, dim=0)

    # Create padding masks
    nmr_padding_mask = (types_padded == -1)
    types_padded[nmr_padding_mask] = 0
    smiles_padding_mask = (smiles_padded == pad_idx)

    return {
        "raw_shifts": raw_shifts_padded,
        "counts": counts_padded,
        "types": types_padded,
        "nmr_padding_mask": nmr_padding_mask,
        "smiles_ids": smiles_padded,
        "smiles_padding_mask": smiles_padding_mask,
        "smiles_list": smiles_list,
        "formula_counts": formula_counts_stacked,
    }
