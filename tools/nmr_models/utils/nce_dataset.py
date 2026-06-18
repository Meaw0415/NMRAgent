"""
NCE Dataset for Isomer Training

从isomer_nce.lmdb加载数据，支持NCE Loss训练
- 每个样本返回anchor的NMR特征、正样本fingerprint、以及同formula的负样本fingerprints
- 不预加载fingerprints，从LMDB实时读取
"""

import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
import random


class IsomerNCEDataset(Dataset):
    """
    NCE训练数据集

    每个样本返回:
    - NMR特征 (shifts, counts, types等)
    - 正样本fingerprint
    - 负样本fingerprints (同formula的其他isomers)
    """

    def __init__(
        self,
        lmdb_path: str,
        index_path: str,
        num_negatives: int = 255,
        h_shift_range: Tuple[float, float] = (-0.01, 0.01),
        c_shift_range: Tuple[float, float] = (-0.1, 0.1),
        shift_aug_p: float = 0.2,
    ):
        """
        Args:
            lmdb_path: isomer_nce.lmdb路径
            index_path: formula_index.pkl路径
            num_negatives: 每个anchor的负样本数量
            h_shift_range: H NMR漂移范围
            c_shift_range: C NMR漂移范围
            shift_aug_p: 漂移增强概率
        """
        self.lmdb_path = lmdb_path
        self.num_negatives = num_negatives
        self.h_shift_range = h_shift_range
        self.c_shift_range = c_shift_range
        self.shift_aug_p = shift_aug_p

        # 加载索引
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)

        self.num_formulas = self.index['num_formulas']
        self.isomers_per_formula = self.index['isomers_per_formula']

        # 使用所有数据训练，不划分
        self.formula_indices = list(range(self.num_formulas))

        # 构建样本索引: (formula_idx, isomer_idx)
        self.samples = []
        for formula_idx in self.formula_indices:
            for isomer_idx in range(self.isomers_per_formula):
                self.samples.append((formula_idx, isomer_idx))

        print(f"[train] Formulas: {len(self.formula_indices)}, Samples: {len(self.samples)}")

        # 懒加载LMDB
        self.env = None

    def _init_lmdb(self):
        """初始化LMDB（在worker进程中）"""
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
        return len(self.samples)

    def _get_sample(self, formula_idx: int, isomer_idx: int) -> Dict:
        """从LMDB获取单个样本"""
        key = f"{formula_idx}_{isomer_idx}".encode()
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))
        return data

    def _get_fingerprint(self, formula_idx: int, isomer_idx: int) -> np.ndarray:
        """从LMDB获取fingerprint"""
        key = f"{formula_idx}_{isomer_idx}".encode()
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))
        return np.unpackbits(data['fp']).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        self._init_lmdb()

        formula_idx, isomer_idx = self.samples[idx]

        # 获取anchor样本
        data = self._get_sample(formula_idx, isomer_idx)

        # 处理NMR特征
        c_nmr = torch.tensor(data['c_nmr'], dtype=torch.float32)
        h_shifts = torch.tensor(data['h_shifts'], dtype=torch.float32)
        h_counts = torch.tensor(data['h_counts'], dtype=torch.float32)
        h_c_shifts = torch.tensor(data['h_c_shifts'], dtype=torch.float32)

        # 漂移增强
        if random.random() < self.shift_aug_p:
            h_bias = random.uniform(*self.h_shift_range)
            c_bias = random.uniform(*self.c_shift_range)
            h_shifts = h_shifts + h_bias
            c_nmr = c_nmr + c_bias

        # 构建输入序列：[H peaks, C peaks]
        all_shifts = torch.cat([h_shifts, c_nmr])
        c_counts = torch.zeros_like(c_nmr)
        all_counts = torch.cat([h_counts, c_counts])
        types = torch.cat([
            torch.zeros_like(h_shifts, dtype=torch.long),
            torch.ones_like(c_nmr, dtype=torch.long)
        ])

        # 正样本fingerprint (从data中直接获取，不需要再次读取)
        positive_fp = torch.from_numpy(np.unpackbits(data['fp']).astype(np.float32))

        # 负样本fingerprints (从LMDB实时读取)
        # 随机选择num_negatives个其他isomers
        all_isomer_indices = list(range(self.isomers_per_formula))
        all_isomer_indices.remove(isomer_idx)  # 排除自己

        if self.num_negatives >= len(all_isomer_indices):
            neg_indices = all_isomer_indices
        else:
            neg_indices = random.sample(all_isomer_indices, self.num_negatives)

        # 从LMDB读取负样本fingerprints
        negative_fps = []
        for neg_idx in neg_indices:
            neg_fp = self._get_fingerprint(formula_idx, neg_idx)
            negative_fps.append(neg_fp)

        negative_fps = torch.from_numpy(np.stack(negative_fps))  # (num_neg, 2048)

        return {
            "raw_shifts": all_shifts,
            "shifts": all_shifts,
            "counts": all_counts,
            "types": types,
            "fingerprint": positive_fp,  # 正样本fingerprint (2048,)
            "negative_fps": negative_fps,  # 负样本fingerprints (num_neg, 2048)
            "h_c_shifts": h_c_shifts,
            "c_shifts": c_nmr,
            "smiles": data['smiles'],
            "formula_idx": formula_idx,
        }


def nce_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for NCE training
    """
    raw_shifts = [item['raw_shifts'] for item in batch]
    shifts = [item['shifts'] for item in batch]
    counts = [item['counts'] for item in batch]
    types = [item['types'] for item in batch]
    fingerprints = [item['fingerprint'] for item in batch]
    negative_fps = [item['negative_fps'] for item in batch]
    h_c_shifts = [item['h_c_shifts'] for item in batch]
    c_shifts_list = [item['c_shifts'] for item in batch]

    # Pad sequences
    raw_shifts_padded = pad_sequence(raw_shifts, batch_first=True, padding_value=0.0)
    shifts_padded = pad_sequence(shifts, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types, batch_first=True, padding_value=-1)
    h_c_shifts_padded = pad_sequence(h_c_shifts, batch_first=True, padding_value=0.0)

    padding_mask = (types_padded == -1)
    types_padded[padding_mask] = 0

    # Stack fingerprints
    fingerprints_stacked = torch.stack(fingerprints, dim=0)  # (B, 2048)
    negative_fps_stacked = torch.stack(negative_fps, dim=0)  # (B, num_neg, 2048)

    return {
        "raw_shifts": raw_shifts_padded,
        "shifts": shifts_padded,
        "counts": counts_padded,
        "types": types_padded,
        "padding_mask": padding_mask,
        "fingerprints": fingerprints_stacked,
        "negative_fps": negative_fps_stacked,
        "h_c_shifts": h_c_shifts_padded,
        "c_shifts_list": c_shifts_list,
    }
