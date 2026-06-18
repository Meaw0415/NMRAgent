"""
Dataset for NMR2SMILES task

Loads NMR data and SMILES for sequence-to-sequence training.
Supports both BasicSmilesTokenizer and HuggingFace tokenizers.
"""

import lmdb
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Tuple, Union, Any
import random
from rdkit import Chem


# Default atom types for formula embedding (50 types)
DEFAULT_ATOM_TYPES = [
    'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'B', 'Si', 'Se', 'Te', 'As', 'Ge', 'Sb', 'Bi', 'Sn', 'Pb',
    'Na', 'K', 'Li', 'Mg', 'Ca', 'Zn', 'Fe', 'Cu', 'Mn', 'Co',
    'Ni', 'Pd', 'Pt', 'Ag', 'Au', 'Hg', 'Cd', 'Al', 'Ga', 'In',
    'Tl', 'Ti', 'V', 'Cr', 'Mo', 'W', 'Ru', 'Rh', 'Ir', 'Os',
]


def get_formula_counts(smiles: str, atom_types: List[str] = None) -> torch.Tensor:
    """
    Extract atom counts from SMILES string.

    Args:
        smiles: SMILES string
        atom_types: List of atom types to count (default: DEFAULT_ATOM_TYPES)

    Returns:
        formula_counts: (num_atom_types,) tensor of atom counts
    """
    if atom_types is None:
        atom_types = DEFAULT_ATOM_TYPES

    num_atom_types = len(atom_types)
    counts = torch.zeros(num_atom_types, dtype=torch.float32)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Add hydrogens to get accurate H count
            mol = Chem.AddHs(mol)
            atom_type_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}

            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_type_to_idx:
                    counts[atom_type_to_idx[symbol]] += 1
    except:
        pass

    return counts


class NMR2SmilesDataset(Dataset):
    """
    Dataset for NMR to SMILES generation task.

    Returns:
        - NMR features (shifts, counts, types)
        - Target SMILES token indices
    """

    def __init__(
        self,
        lmdb_path: str,
        tokenizer: Any,  # Supports BasicSmilesTokenizer or HuggingFace tokenizers
        max_smiles_len: int = 256,
        split: str = 'train',
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        seed: int = 42,
        h_shift_range: Tuple[float, float] = (-0.01, 0.01),
        c_shift_range: Tuple[float, float] = (-0.1, 0.1),
        shift_aug_p: float = 0.2,
    ):
        """
        Args:
            lmdb_path: Path to LMDB file
            tokenizer: SMILES tokenizer (BasicSmilesTokenizer or HuggingFace)
            max_smiles_len: Maximum SMILES sequence length
            split: 'train', 'val', or 'test'
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            seed: Random seed for split
            h_shift_range: H NMR shift augmentation range
            c_shift_range: C NMR shift augmentation range
            shift_aug_p: Probability of shift augmentation
        """
        self.lmdb_path = lmdb_path
        self.tokenizer = tokenizer
        self.max_smiles_len = max_smiles_len
        self.h_shift_range = h_shift_range
        self.c_shift_range = c_shift_range
        self.shift_aug_p = shift_aug_p
        self.split = split
        # Get special token ids
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        # Open LMDB to get total length
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            self.total_samples = txn.stat()['entries']
        env.close()

        # Create split indices
        np.random.seed(seed)
        all_indices = np.random.permutation(self.total_samples)

        val_num = 10000

        if split == 'val':
            self.indices = all_indices[:val_num].tolist()
        elif split == 'train':
            self.indices = all_indices[val_num:].tolist()

        print(f"[{split}] Samples: {len(self.indices)}")

        # Lazy LMDB initialization
        self.env = None

    def _init_lmdb(self):
        """Initialize LMDB environment (called in worker process)."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                subdir=False,
                readahead=False,
                meminit=False,
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        self._init_lmdb()

        real_idx = self.indices[idx]

        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(real_idx).encode()))

        # Extract NMR data
        c_nmr = data.get('C_nmr', [])
        h_info = data.get('H_nmr', [])
        smiles = data.get('smiles', '')

        # Process H NMR
        h_shifts = torch.tensor([item['shift'] for item in h_info], dtype=torch.float32)
        h_counts = torch.tensor([item['count'] for item in h_info], dtype=torch.float32)
        h_c_shifts = torch.tensor([item.get('c_shift', 0.0) for item in h_info], dtype=torch.float32)

        # Process C NMR
        c_shifts = torch.tensor(c_nmr, dtype=torch.float32)

        # Shift augmentation (only for training)
        if self.split == 'train' and random.random() < self.shift_aug_p:
            h_bias = random.uniform(*self.h_shift_range)
            c_bias = random.uniform(*self.c_shift_range)
            h_shifts = h_shifts + h_bias
            c_shifts = c_shifts + c_bias

        # Build input sequence: [H peaks, C peaks]
        all_shifts = torch.cat([h_shifts, c_shifts])
        c_counts = torch.zeros_like(c_shifts)
        all_counts = torch.cat([h_counts, c_counts])
        types = torch.cat([
            torch.zeros_like(h_shifts, dtype=torch.long),  # H = 0
            torch.ones_like(c_shifts, dtype=torch.long)    # C = 1
        ])

        # Tokenize SMILES using HuggingFace tokenizer
        # Add BOS and EOS manually for decoder training
        token_ids = self.tokenizer.encode(smiles, add_special_tokens=False)
        smiles_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Truncate if too long
        if len(smiles_ids) > self.max_smiles_len:
            smiles_ids = smiles_ids[:self.max_smiles_len - 1] + [self.eos_token_id]

        smiles_ids = torch.tensor(smiles_ids, dtype=torch.long)

        # Get formula counts from SMILES
        formula_counts = get_formula_counts(smiles)
        return {
            "raw_shifts": all_shifts,
            "shifts": all_shifts,
            "counts": all_counts,
            "types": types,
            "smiles_ids": smiles_ids,
            "smiles": smiles,
            "h_c_shifts": h_c_shifts,
            "formula_counts": formula_counts,
        }


def nmr2smiles_collate_fn(batch: List[Dict], pad_idx: int = 0) -> Dict:
    """
    Collate function for NMR2SMILES training.

    Args:
        batch: List of samples
        pad_idx: Padding index for SMILES

    Returns:
        Batched tensors
    """
    raw_shifts = [item['raw_shifts'] for item in batch]
    shifts = [item['shifts'] for item in batch]
    counts = [item['counts'] for item in batch]
    types = [item['types'] for item in batch]
    smiles_ids = [item['smiles_ids'] for item in batch]
    h_c_shifts = [item['h_c_shifts'] for item in batch]
    formula_counts = [item['formula_counts'] for item in batch]

    # Pad NMR sequences
    raw_shifts_padded = pad_sequence(raw_shifts, batch_first=True, padding_value=0.0)
    shifts_padded = pad_sequence(shifts, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types, batch_first=True, padding_value=-1)
    h_c_shifts_padded = pad_sequence(h_c_shifts, batch_first=True, padding_value=0.0)

    # Stack formula counts (no padding needed, all same size)
    formula_counts_stacked = torch.stack(formula_counts, dim=0)

    # Create NMR padding mask
    nmr_padding_mask = (types_padded == -1)
    types_padded[nmr_padding_mask] = 0

    # Pad SMILES sequences
    smiles_padded = pad_sequence(smiles_ids, batch_first=True, padding_value=pad_idx)

    # Create SMILES padding mask
    smiles_padding_mask = (smiles_padded == pad_idx)

    return {
        "raw_shifts": raw_shifts_padded,
        "shifts": shifts_padded,
        "counts": counts_padded,
        "types": types_padded,
        "nmr_padding_mask": nmr_padding_mask,
        "smiles_ids": smiles_padded,
        "smiles_padding_mask": smiles_padding_mask,
        "h_c_shifts": h_c_shifts_padded,
        "formula_counts": formula_counts_stacked,
    }
