import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import lmdb
import pickle
import random  # 用于生成漂移值

class NMRDataset(Dataset):
    def __init__(self, data_source,
                 h_x_min=0.01, h_x_max=16.0,
                 c_x_min=0.01, c_x_max=230.0,
                 lmdb_max_readers=256,
                 h_shift_range=(-0.01, 0.01),  # 氢谱漂移范围
                 c_shift_range=(-0.1, 0.1),  # 碳谱漂移范围
                 shift_aug_p=0.2):  # 漂移增强的概率
        """
        data_source: LMDB文件路径
        h_x_min/x_max, c_x_min/x_max: H/C 范围参数(保留用于兼容性，但不再用于归一化)
        lmdb_max_readers: LMDB的最大读取器数量
        h_shift_range: 氢谱漂移范围
        c_shift_range: 碳谱漂移范围
        shift_aug_p: 漂移增强的概率
        """
        self.h_x_min = h_x_min
        self.h_x_max = h_x_max
        self.c_x_min = c_x_min
        self.c_x_max = c_x_max
        self.h_shift_range = h_shift_range  # 氢谱漂移范围
        self.c_shift_range = c_shift_range  # 碳谱漂移范围
        self.shift_aug_p = shift_aug_p  # 漂移增强的概率
        
        self.env = lmdb.open(
            data_source,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=lmdb_max_readers,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # with self.env.begin(write=False) as txn:
        #     self.length = min(txn.stat()['entries'], 10000)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            value = txn.get(str(idx).encode())
            if value is None:
                raise IndexError(f"Index {idx} not found in LMDB")
            sample = pickle.loads(value)

        # -------------------- 提取原始谱数据 --------------------
        c_shifts = torch.tensor(sample.get('C_nmr', []), dtype=torch.float32)
        h_info = sample.get('H_nmr', [])
        h_shifts = torch.tensor([item['shift'] for item in h_info], dtype=torch.float32)
        h_counts = torch.tensor([item['count'] for item in h_info], dtype=torch.float32)

        # 新增: 提取每个H peak对应的C shift值 (c_shift字段)
        h_c_shifts = torch.tensor([item.get('c_shift', 0.0) for item in h_info], dtype=torch.float32)

        # -------------------- 漂移操作 --------------------
        # 随机决定是否应用漂移
        if random.random() < self.shift_aug_p:
            # 对氢谱（H NMR）漂移
            h_shift_bias = random.uniform(self.h_shift_range[0], self.h_shift_range[1])
            h_shifts += h_shift_bias

            # 对碳谱（C NMR）漂移
            c_shift_bias = random.uniform(self.c_shift_range[0], self.c_shift_range[1])
            c_shifts += c_shift_bias

        all_shifts = torch.cat([h_shifts, c_shifts])  # 原始值（ppm），直接使用不归一化
        c_counts = torch.zeros_like(c_shifts)
        all_counts = torch.cat([h_counts, c_counts])
        types = torch.cat([
            torch.zeros_like(h_shifts, dtype=torch.long),
            torch.ones_like(c_shifts, dtype=torch.long)
        ])

        # -------------------- 提取SMILES并计算Morgan fingerprint --------------------
        smiles = sample.get('smiles', '')

        # 新增: 如果fp已经存储在LMDB中（二进制形式），直接使用
        if 'fp' in sample and sample['fp'] is not None:
            fp = np.frombuffer(sample['fp'], dtype=np.uint8)
            fp = np.unpackbits(fp) #解码为2进制
            fingerprint = torch.tensor(fp, dtype=torch.float32)
        else:
            # 否则从SMILES计算Morgan fingerprint
            mol = Chem.MolFromSmiles(smiles)
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)
                fingerprint = torch.tensor(np.array(fp), dtype=torch.float32)
            else:
                # 如果SMILES无效，使用零向量
                fingerprint = torch.zeros(2048, dtype=torch.float32)

        # 不再进行归一化，raw_shifts 和 shifts 使用相同的原始 ppm 值
        return {
            "raw_shifts": all_shifts,  # 用于 FourierFeatures (原始 ppm)
            "shifts": all_shifts,      # 用于 loss / regression (原始 ppm)
            "counts": all_counts,
            "types": types,
            "fingerprint": fingerprint,  # 新增：Morgan fingerprint
            "smiles": smiles,  # 新增：保留SMILES用于调试
            "h_c_shifts": h_c_shifts,  # 新增：每个H peak对应的C shift值
            "c_shifts": c_shifts,  # 新增：所有C shifts（用于InfoNCE负样本）
        }


def collate_fn(batch):
    # 对应字段分别 pad
    raw_shifts = [item['raw_shifts'] for item in batch]
    shifts = [item['shifts'] for item in batch]
    counts = [item['counts'] for item in batch]
    types = [item['types'] for item in batch]
    fingerprints = [item['fingerprint'] for item in batch]  # 新增
    h_c_shifts = [item['h_c_shifts'] for item in batch]  # 新增：每个H对应的C shift
    c_shifts_list = [item['c_shifts'] for item in batch]  # 新增：每个样本的所有C shifts

    raw_shifts_padded = pad_sequence(raw_shifts, batch_first=True, padding_value=0.0)
    shifts_padded = pad_sequence(shifts, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types, batch_first=True, padding_value=-1)

    # 新增：pad h_c_shifts（每个H peak对应的C shift）
    h_c_shifts_padded = pad_sequence(h_c_shifts, batch_first=True, padding_value=0.0)

    padding_mask = (types_padded == -1)
    types_padded[padding_mask] = 0  # 修正 padding 占位符

    # Stack fingerprints (all have same size)
    fingerprints_stacked = torch.stack(fingerprints, dim=0)  # [B, 2048]

    return {
        "raw_shifts": raw_shifts_padded,  # 给 FourierFeatures
        "shifts": shifts_padded,          # 给 loss
        "counts": counts_padded,
        "types": types_padded,
        "padding_mask": padding_mask,
        "fingerprints": fingerprints_stacked,  # 新增：[B, 2048]
        "h_c_shifts": h_c_shifts_padded,  # 新增：[B, num_H] 每个H对应的C shift
        "c_shifts_list": c_shifts_list,  # 新增：List[Tensor] 每个样本的所有C shifts（不pad，用于InfoNCE）
    }
