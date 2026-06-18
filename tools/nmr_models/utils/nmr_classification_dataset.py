"""
Dataset for NMR superclass classification using NMRGym data.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NMRClassificationDataset(Dataset):
    """
    Dataset for NMR superclass classification.

    Loads NMRGym JSONL data with superclass labels.
    """

    def __init__(
        self,
        jsonl_path,
        class_to_idx,
        split='train',
        h_shift_range=(-0.01, 0.01),
        c_shift_range=(-0.1, 0.1),
        shift_aug_p=0.0,
        filter_unseen_classes=True,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file
            class_to_idx: Dict mapping class names to indices
            split: 'train', 'val', or 'test'
            h_shift_range: Range for H shift augmentation
            c_shift_range: Range for C shift augmentation
            shift_aug_p: Probability of applying shift augmentation
            filter_unseen_classes: Whether to filter out samples with unseen classes
        """
        self.split = split
        self.class_to_idx = class_to_idx
        self.h_shift_range = h_shift_range
        self.c_shift_range = c_shift_range
        self.shift_aug_p = shift_aug_p
        self.filter_unseen_classes = filter_unseen_classes

        # Load data from JSONL
        print(f"Loading {split} data from {jsonl_path}...")
        self.data = []
        skipped_invalid = 0
        skipped_unseen = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    skipped_invalid += 1
                    continue

                # Check if class is in class_to_idx
                superclass = item.get('superclass', 'unknown')
                if superclass not in self.class_to_idx and superclass:
                    if filter_unseen_classes:
                        skipped_unseen += 1
                        continue
                    else:
                        # Assign to unknown class (index -1, will be filtered in collate)
                        print(f"Warning: Unseen class '{superclass}' at line {line_no}")
                        continue

                # Validate data
                h_shifts = item.get('h_shift', [])
                c_shifts = item.get('c_shift', [])

                if not isinstance(h_shifts, list) or not isinstance(c_shifts, list):
                    skipped_invalid += 1
                    continue

                # At least one peak required
                if len(h_shifts) == 0 and len(c_shifts) == 0:
                    skipped_invalid += 1
                    continue

                self.data.append(item)

        print(f"[{split}] Loaded {len(self.data)} samples")
        if skipped_invalid > 0:
            print(f"  Skipped {skipped_invalid} invalid samples")
        if skipped_unseen > 0:
            print(f"  Skipped {skipped_unseen} samples with unseen classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get superclass label
        superclass = item['superclass']
        if superclass:
            label = self.class_to_idx[superclass]
        else:
            label = ''
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

        return {
            "raw_shifts": all_shifts,
            "counts": all_counts,
            "types": types,
            "label": label,
            "superclass": superclass,
            "smiles": smiles
        }


def nmr_classification_collate_fn(batch):
    """Collate function for NMR classification."""
    raw_shifts = [item['raw_shifts'] for item in batch]
    counts = [item['counts'] for item in batch]
    types = [item['types'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    superclasses = [item['superclass'] for item in batch]
    smiles = [item['smiles'] for item in batch]
    # Pad sequences
    raw_shifts_padded = pad_sequence(raw_shifts, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types, batch_first=True, padding_value=-1)

    # Create padding mask
    padding_mask = (types_padded == -1)
    types_padded[padding_mask] = 0

    return {
        "raw_shifts": raw_shifts_padded,
        "counts": counts_padded,
        "types": types_padded,
        "padding_mask": padding_mask,
        "labels": labels,
        "superclasses": superclasses,
        "smiles":smiles
    }


def build_class_mapping(train_jsonl_path, exclude_classes=None):
    """
    Build class-to-index mapping from training data.

    Args:
        train_jsonl_path: Path to training JSONL file
        exclude_classes: List of class names to exclude (e.g., ['Failed'])

    Returns:
        class_to_idx: Dict mapping class names to indices
        idx_to_class: Dict mapping indices to class names
        class_counts: Dict with class counts
    """
    if exclude_classes is None:
        exclude_classes = ['Failed']  # Exclude 'Failed' by default

    print(f"Building class mapping from {train_jsonl_path}...")

    classes = set()
    class_counts = {}

    with open(train_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                superclass = item.get('superclass', 'unknown')

                if superclass not in exclude_classes:
                    classes.add(superclass)
                    class_counts[superclass] = class_counts.get(superclass, 0) + 1
            except Exception:
                continue

    # Sort classes for consistent ordering
    sorted_classes = sorted(classes)

    # Create mappings
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"  Found {len(class_to_idx)} classes (excluding {exclude_classes})")
    print(f"  Total samples: {sum(class_counts.values())}")

    return class_to_idx, idx_to_class, class_counts


def save_class_mapping(class_to_idx, idx_to_class, class_counts, save_path):
    """Save class mapping to JSON file."""
    mapping = {
        'class_to_idx': class_to_idx,
        'idx_to_class': {str(k): v for k, v in idx_to_class.items()},  # JSON keys must be strings
        'class_counts': class_counts,
        'num_classes': len(class_to_idx),
    }

    with open(save_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Class mapping saved to {save_path}")


def load_class_mapping(load_path):
    """Load class mapping from JSON file."""
    with open(load_path, 'r') as f:
        mapping = json.load(f)

    class_to_idx = mapping['class_to_idx']
    idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}  # Convert keys back to int
    class_counts = mapping.get('class_counts', {})
    num_classes = mapping['num_classes']

    print(f"Loaded class mapping from {load_path}")
    print(f"  Num classes: {num_classes}")

    return class_to_idx, idx_to_class, class_counts
