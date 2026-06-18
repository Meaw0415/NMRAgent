#!/usr/bin/env python3
"""
NMR谱图检索：从测试数据中搜索最相似的库谱图
使用LMDB存储 + Faiss索引进行快速检索

特性：
1. 使用cosine similarity进行相似度计算（而非L2距离）
2. 支持formula filter模式：先筛选相同分子式的谱图再进行匹配
"""

import logging
import os
import lmdb
import pickle
import numpy as np
import faiss
import torch
import json

logger = logging.getLogger(__name__)
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from ..shared_lmdb import open_shared_lmdb

# Use relative import - models are in the same directory
from .models.nmr_bert import NMR_BERT

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def compute_cosine_similarity(vec1, vec2):
    """
    计算两个向量的cosine similarity

    Args:
        vec1: numpy array (n,) or (1, n)
        vec2: numpy array (m, n)

    Returns:
        similarities: numpy array (m,)
    """
    # 确保是2D数组
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)

    # 归一化
    vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)

    # 计算cosine similarity (内积)
    similarities = np.dot(vec1_norm, vec2_norm.T).flatten()

    return similarities


def smiles_to_formula(smiles):
    """
    从SMILES字符串计算分子式

    Args:
        smiles: SMILES字符串

    Returns:
        formula: 分子式字符串，如 "C6H12O6"，如果解析失败返回None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return formula
    except Exception:
        return None


def get_canonical_smiles(smiles):
    """
    将SMILES转换为canonical形式

    Args:
        smiles: SMILES字符串

    Returns:
        canonical SMILES，如果解析失败返回None
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        #return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


class NMRSearcher:
    """
    NMR谱图检索器：编码查询谱图 → 搜索 → 返回最相似结果

    使用cosine similarity作为相似度指标
    支持两种模式：
    - 不启用formula filter: 使用Faiss进行近似最近邻搜索
    - 启用formula filter: 先筛选相同分子式的谱图，再计算相似度
    """

    def __init__(self, model_checkpoint, retrieval_dir, embedding_dim=768,
                 enable_formula_filter=False):
        """
        Args:
            model_checkpoint: 预训练模型路径
            retrieval_dir: 检索系统目录（包含LMDB和Faiss索引）
            embedding_dim: Embedding维度
            enable_formula_filter: 是否启用formula filter功能
        """
        self.model_checkpoint = model_checkpoint
        self.retrieval_dir = Path(retrieval_dir)
        self.embedding_dim = embedding_dim
        self.enable_formula_filter = enable_formula_filter

        # 检索系统文件路径
        self.lmdb_path = os.environ.get(
            "NMR_RETRIEVAL_LMDB",
            str(self.retrieval_dir / "embeddings_with_smiles.lmdb"),
        )
        if not Path(self.lmdb_path).exists():
            alt_lmdb = self.retrieval_dir / "embeddings_with_smiles_gaussian.lmdb"
            if alt_lmdb.exists():
                self.lmdb_path = str(alt_lmdb)

        self.faiss_index_path = os.environ.get(
            "NMR_FAISS_INDEX_PATH",
            str(self.retrieval_dir / "faiss.index"),
        )
        self.id_map_path = os.environ.get(
            "NMR_ID_MAP_PATH",
            str(self.retrieval_dir / "id_mapping.pkl"),
        )
        self.formula_map_path = os.environ.get(
            "NMR_FORMULA_MAP_PATH",
            str(self.retrieval_dir / "formula_to_ids_new.pkl"),
        )

        parts_dir = self.retrieval_dir / "embeddings_with_smiles_gaussian.lmdb.parts"
        if not Path(self.lmdb_path).exists() and parts_dir.exists():
            raise FileNotFoundError(
                "LMDB not found. Reconstruct embeddings_with_smiles_gaussian.lmdb "
                f"from parts in {parts_dir} first."
            )

        # 检查文件是否存在
        for path, name in [
            (self.lmdb_path, "LMDB"),
            (self.id_map_path, "ID mapping"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        self.has_faiss_index = Path(self.faiss_index_path).exists()
        if not self.has_faiss_index and not enable_formula_filter:
            raise FileNotFoundError(f"Faiss index not found: {self.faiss_index_path}")

        # 检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires a GPU.")

        # 初始化CUDA
        print("Initializing CUDA...")
        torch.cuda.init()
        torch.cuda.set_device(0)
        self.device = 'cuda:0'  # 设置 device 属性
        _ = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # 加载模型
        print("Loading model...")
        self.model = self._load_model()

        # 加载Faiss索引（formula filter 模式下可选）
        self.index = None
        if self.has_faiss_index:
            print("Loading Faiss index...")
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"  Index contains {self.index.ntotal:,} vectors")
        else:
            print("Faiss index not found, formula-filter-only retrieval is enabled")

        # 加载ID映射
        print("Loading ID mapping...")
        with open(self.id_map_path, 'rb') as f:
            self.id_to_key = pickle.load(f)
        print(f"  Loaded {len(self.id_to_key):,} ID mappings")

        # 打开LMDB（只读）
        print("Opening LMDB...")
        self.lmdb_env = open_shared_lmdb(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            subdir=False,
        )

        # 如果启用formula filter，加载formula映射
        self.formula_to_ids = None
        if enable_formula_filter:
            if not Path(self.formula_map_path).exists():
                raise FileNotFoundError(f"Formula mapping not found: {self.formula_map_path}")
            print("Loading formula index...")
            with open(self.formula_map_path, 'rb') as f:
                self.formula_to_ids = pickle.load(f)
            print(f"  Loaded {len(self.formula_to_ids):,} formulas")

        print("✓ Searcher initialized successfully")

    def _load_model(self):
        """加载预训练模型"""
        checkpoint = torch.load(self.model_checkpoint, map_location='cpu')

        # 从checkpoint提取配置
        model_config = checkpoint.get('config', {})
        model = NMR_BERT(
            d_model=model_config.get('d_model', 768),
            nhead=model_config.get('nhead', 12),
            num_encoder_layers=model_config.get('num_encoder_layers', 16),
            dim_feedforward=model_config.get('dim_feedforward', 3072),
            dropout=model_config.get('dropout', 0.1),
            h_bin_size=model_config.get('h_bin_size', 0.01),
            h_max=model_config.get('h_max', 16.0),
            c_bin_size=model_config.get('c_bin_size', 0.1),
            c_max=model_config.get('c_max', 230.0),
            fp_sim_bin_size=model_config.get('fp_sim_bin_size', 0.005),
            fp_sim_mlp_hidden=model_config.get('fp_sim_mlp_hidden', 512),
            use_identity_embedding=model_config.get('use_identity_embedding', True),
            use_count_embedding=model_config.get('use_count_embedding', False),
            lr=1e-4,
            weight_decay=0.0,
            n_warmup_steps=None
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(torch.device(self.device))

        return model

    @torch.no_grad()
    def encode_spectrum(self, spectrum):
        """
        编码单个谱图为embedding

        Args:
            spectrum: Dict with 'H_nmr' and 'C_nmr' (or 'h_shift' and 'c_shift')

        Returns:
            embedding: (embedding_dim,)
        """
        device = next(self.model.parameters()).device

        # 准备输入（兼容两种格式）
        batch_data = self._collate_spectra([spectrum])

        # 转移到GPU
        shifts = batch_data['shifts'].to(device, non_blocking=True)
        counts = batch_data['counts'].to(device, non_blocking=True)
        types = batch_data['types'].to(device, non_blocking=True)
        padding_mask = batch_data['padding_mask'].to(device, non_blocking=True)

        # 前向传播
        _, _, encoder_output = self.model(shifts, counts, types, padding_mask)

        # 平均池化（mask掉padding）
        mask = ~padding_mask.unsqueeze(-1)
        pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1)

        # 转到CPU
        embedding = pooled.cpu().numpy()[0]

        return embedding

    def _collate_spectra(self, spectra_batch):
        """
        准备输入数据（支持多种格式）

        Args:
            spectra_batch: List of dicts with either:
                - 'H_nmr' and 'C_nmr' (库格式)
                - 'h_shift' and 'c_shift' (测试格式)
                - 'H_shifts' and 'C_shifts' (LMDB格式，numpy数组)

        Returns:
            dict with padded tensors
        """
        from torch.nn.utils.rnn import pad_sequence

        raw_shifts_list = []
        shifts_list = []
        counts_list = []
        types_list = []

        for sample in spectra_batch:
            # 提取H NMR数据（兼容三种格式）
            if 'H_nmr' in sample:
                h_info = sample['H_nmr']
                if isinstance(h_info, list) and len(h_info) > 0 and isinstance(h_info[0], dict):
                    h_shifts = torch.tensor([item['shift'] for item in h_info], dtype=torch.float32)
                    h_counts = torch.tensor([item['count'] for item in h_info], dtype=torch.float32)
                else:
                    h_shifts = torch.tensor(h_info if isinstance(h_info, list) else [], dtype=torch.float32)
                    h_counts = torch.zeros_like(h_shifts)
            elif 'h_shift' in sample:
                h_shifts = torch.tensor(sample['h_shift'], dtype=torch.float32)
                h_counts = torch.zeros_like(h_shifts)  # 测试数据通常没有count信息
            elif 'H_shifts' in sample:
                # 支持numpy数组格式
                h_shifts = torch.tensor(sample['H_shifts'], dtype=torch.float32)
                h_counts = torch.zeros_like(h_shifts)
            else:
                h_shifts = torch.tensor([], dtype=torch.float32)
                h_counts = torch.tensor([], dtype=torch.float32)

            # 提取C NMR数据
            if 'C_nmr' in sample:
                c_shifts = torch.tensor(sample['C_nmr'], dtype=torch.float32)
            elif 'c_shift' in sample:
                c_shifts = torch.tensor(sample['c_shift'], dtype=torch.float32)
            elif 'C_shifts' in sample:
                # 支持numpy数组格式
                c_shifts = torch.tensor(sample['C_shifts'], dtype=torch.float32)
            else:
                c_shifts = torch.tensor([], dtype=torch.float32)

            c_counts = torch.zeros_like(c_shifts)

            # 拼接H和C数据
            all_shifts = torch.cat([h_shifts, c_shifts])
            all_counts = torch.cat([h_counts, c_counts])
            types = torch.cat([
                torch.zeros_like(h_shifts, dtype=torch.long),
                torch.ones_like(c_shifts, dtype=torch.long)
            ])

            raw_shifts_list.append(all_shifts)
            shifts_list.append(all_shifts)
            counts_list.append(all_counts)
            types_list.append(types)

        # Padding
        raw_shifts_padded = pad_sequence(raw_shifts_list, batch_first=True, padding_value=0.0)
        shifts_padded = pad_sequence(shifts_list, batch_first=True, padding_value=0.0)
        counts_padded = pad_sequence(counts_list, batch_first=True, padding_value=0.0)
        types_padded = pad_sequence(types_list, batch_first=True, padding_value=-1)

        # 创建padding mask
        padding_mask = (types_padded == -1)
        types_padded[padding_mask] = 0

        return {
            "raw_shifts": raw_shifts_padded,
            "shifts": shifts_padded,
            "counts": counts_padded,
            "types": types_padded,
            "padding_mask": padding_mask,
        }

    def search(self, query_spectrum, k=10, nprobe=128, query_smiles=None, disable_formula_filter=False):
        """
        搜索最相似的k个谱图（使用cosine similarity）

        Args:
            query_spectrum: Dict with NMR data
            k: 返回top-k结果
            nprobe: Faiss搜索参数（越大越准确但越慢，仅在不使用formula filter时有效）
            query_smiles: 查询分子的SMILES（如果启用formula filter则需要）

        Returns:
            tuple: (results, target_in_library)
                results: List of dicts with:
                    - 'cosine_similarity': Cosine similarity (越大越相似)
                    - 'H_nmr': H NMR数据
                    - 'C_nmr': C NMR数据
                    - 'smiles': SMILES字符串
                target_in_library: bool, 仅在formula filter模式且提供query_smiles时有效，
                                  表示target smiles是否存在于库中
        """
        # 编码查询谱图
        query_embedding = self.encode_spectrum(query_spectrum)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # 如果启用formula filter且未禁用
        if self.enable_formula_filter and not disable_formula_filter:
            if query_smiles is None:
                raise ValueError("query_smiles is required when formula filter is enabled")

            # 计算查询分子的formula
            query_formula = smiles_to_formula(query_smiles)
            if query_formula is None:
                raise ValueError(f"Failed to parse SMILES: {query_smiles}")

            # 获取相同formula的所有候选ID
            candidate_ids = self.formula_to_ids.get(query_formula, [])
            if len(candidate_ids) == 0:
                logger.debug(f"Warning: No spectra found with formula {query_formula}")
                return [], False  # 库中不存在该formula，target不在库中

            logger.debug(f"Found {len(candidate_ids)} candidates with formula {query_formula}")

            # 将query_smiles转换为canonical形式用于比较
            query_canonical = get_canonical_smiles(query_smiles)
            if query_canonical is None:
                print(f"Warning: Failed to canonicalize query SMILES: {query_smiles}")
                return [], False

            # 从LMDB读取候选embeddings，并检查target是否在库中
            candidate_embeddings = []
            candidate_data = []
            target_in_library = False
            skipped_corrupted = 0
            with self.lmdb_env.begin() as txn:
                for cand_id in candidate_ids:
                    try:
                        # 获取LMDB key
                        lmdb_key = self.id_to_key.get(cand_id)
                        if lmdb_key is None:
                            continue

                        # 从LMDB读取数据
                        value = txn.get(lmdb_key)
                        if value is None:
                            continue

                        data = pickle.loads(value)
                        embedding = data.get('embedding')
                        if embedding is None:
                            continue

                        candidate_embeddings.append(embedding)
                        candidate_data.append({
                            "data": data,
                            "db_id": int(cand_id),
                            "db_key": lmdb_key.decode() if isinstance(lmdb_key, bytes) else str(lmdb_key),
                        })

                        # 检查是否为target smiles（使用canonical SMILES比较）
                        candidate_smiles = data.get('smiles')
                        if candidate_smiles:
                            candidate_canonical = get_canonical_smiles(candidate_smiles)
                            if candidate_canonical and candidate_canonical == query_canonical:
                                target_in_library = True
                    except Exception as e:
                        # LMDB损坏或读取错误，跳过该条目
                        skipped_corrupted += 1
                        continue

            if skipped_corrupted > 0:
                print(f"  Warning: Skipped {skipped_corrupted} corrupted entries for formula {query_formula}")

            if len(candidate_embeddings) == 0:
                return [], target_in_library  # 没有有效candidates

            # 计算cosine similarity
            candidate_embeddings = np.array(candidate_embeddings, dtype=np.float32)
            similarities = compute_cosine_similarity(query_embedding, candidate_embeddings)

            # 排序并返回top-k
            top_indices = np.argsort(similarities)[::-1][:k]
            results = []
            for idx in top_indices:
                row = candidate_data[idx]
                data = row["data"]
                results.append({
                    'cosine_similarity': float(similarities[idx]),
                    'H_nmr': data.get('H_nmr', []),
                    'C_nmr': data.get('C_nmr', []),
                    'smiles': data.get('smiles', None),
                    'canonical_smiles': data.get('canonical_smiles', get_canonical_smiles(data.get('smiles'))),
                    'formula': data.get('formula', ''),
                    'db_id': row['db_id'],
                    'db_key': row['db_key'],
                })

            return results, target_in_library

        else:
            # 不使用formula filter：使用Faiss搜索大约的k个邻居
            if self.index is None:
                raise RuntimeError(
                    "Faiss index is not available. Provide a formula to use formula-filter "
                    "retrieval, or add faiss.index for global ANN search."
                )

            # 设置搜索参数
            if hasattr(self.index, 'nprobe'):
                # IVFPQ索引：nprobe控制搜索的聚类数
                self.index.nprobe = nprobe
            elif hasattr(self.index, 'hnsw'):
                # HNSW索引：efSearch控制搜索精度
                # 参考NMR-Solver: efSearch = 2*k (server.py:106)
                self.index.hnsw.efSearch = max(nprobe, 2 * k)
                # HNSW索引使用归一化向量（参考NMR-Solver search.py:34）
                # 归一化查询向量以匹配索引中的归一化向量
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm

            # 搜索更多的候选（因为我们需要用cosine similarity重新排序）
            search_k = min(k * 10, self.index.ntotal)  # 搜索10倍的k
            distances, indices = self.index.search(query_embedding, search_k)

            # 从LMDB中获取embeddings并计算cosine similarity
            candidate_embeddings = []
            candidate_data = []
            skipped_corrupted = 0
            with self.lmdb_env.begin() as txn:
                for idx in indices[0]:
                    if idx == -1:
                        continue

                    try:
                        # 获取LMDB key
                        lmdb_key = self.id_to_key[int(idx)]

                        # 从LMDB读取数据
                        value = txn.get(lmdb_key)
                        if value is None:
                            continue

                        data = pickle.loads(value)
                        embedding = data.get('embedding')
                        if embedding is None:
                            continue

                        candidate_embeddings.append(embedding)
                        candidate_data.append({
                            "data": data,
                            "db_id": int(idx),
                            "db_key": lmdb_key.decode() if isinstance(lmdb_key, bytes) else str(lmdb_key),
                        })
                    except Exception as e:
                        # LMDB损坏或读取错误，跳过该条目
                        skipped_corrupted += 1
                        continue

            if skipped_corrupted > 0:
                print(f"  Warning: Skipped {skipped_corrupted} corrupted entries during Faiss search")

            if len(candidate_embeddings) == 0:
                return [], None  # 非formula filter模式不检查target_in_library

            # 计算cosine similarity
            candidate_embeddings = np.array(candidate_embeddings, dtype=np.float32)
            similarities = compute_cosine_similarity(query_embedding, candidate_embeddings)

            # 排序并返回top-k
            top_indices = np.argsort(similarities)[::-1][:k]
            results = []
            for idx in top_indices:
                row = candidate_data[idx]
                data = row["data"]
                results.append({
                    'cosine_similarity': float(similarities[idx]),
                    'H_nmr': data.get('H_nmr', []),
                    'C_nmr': data.get('C_nmr', []),
                    'smiles': data.get('smiles', None),
                    'canonical_smiles': data.get('canonical_smiles', get_canonical_smiles(data.get('smiles'))),
                    'formula': data.get('formula', ''),
                    'db_id': row['db_id'],
                    'db_key': row['db_key'],
                })

            return results, None  # 非formula filter模式不检查target_in_library

    def search_batch(self, query_file, output_file, k=10, nprobe=128,
                     max_queries=None):
        """
        批量搜索：从JSONL文件读取查询，输出结果到JSONL

        Args:
            query_file: 查询谱图JSONL文件路径
            output_file: 输出结果JSONL文件路径
            k: 每个查询返回top-k结果
            nprobe: Faiss搜索参数（仅在不使用formula filter时有效）
            max_queries: 最大查询数量（用于测试）
        """
        print(f"\n=== Batch Search ===")
        print(f"Query file: {query_file}")
        print(f"Output file: {output_file}")
        print(f"Top-k: {k}, nprobe: {nprobe}")
        print(f"Formula filter: {'Enabled' if self.enable_formula_filter else 'Disabled'}")

        # 统计查询数量
        with open(query_file, 'r') as f:
            total_queries = sum(1 for _ in f)

        if max_queries is not None:
            total_queries = min(total_queries, max_queries)

        print(f"Total queries: {total_queries:,}")

        # 批量搜索
        processed_queries = 0
        failed_queries = 0
        skipped_queries = 0

        with open(query_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for i, line in enumerate(tqdm(f_in, total=total_queries, desc="Searching")):
                if max_queries is not None and i >= max_queries:
                    break

                query = json.loads(line)

                # 提取SMILES（如果启用formula filter）
                query_smiles = None
                if self.enable_formula_filter:
                    query_smiles = query.get('smiles')
                    if query_smiles is None:
                        print(f"Warning: Query {i} missing 'smiles' field, skipping")
                        skipped_queries += 1
                        continue

                # 搜索
                try:
                    results, target_in_library = self.search(query, k=k, nprobe=nprobe,
                                        query_smiles=query_smiles)
                    processed_queries += 1
                except Exception as e:
                    print(f"\nError processing query {i}: {e}")
                    print(f"  Query SMILES: {query.get('smiles', 'N/A')}")
                    failed_queries += 1
                    results = []
                    target_in_library = None

                # 写入结果（即使失败也写入空结果）
                output = {
                    'query': query,
                    'results': results
                }
                # 如果启用formula filter，记录target_in_library状态
                if self.enable_formula_filter and target_in_library is not None:
                    output['target_in_library'] = target_in_library

                f_out.write(json.dumps(output) + '\n')

        print(f"\n✓ Search completed:")
        print(f"  Processed: {processed_queries:,}")
        print(f"  Failed: {failed_queries:,}")
        print(f"  Skipped: {skipped_queries:,}")
        print(f"  Results saved to: {output_file}")

    def close(self):
        """关闭LMDB环境"""
        self.lmdb_env.close()


def main():
    import argparse
    import sys
    import os

    # 添加当前目录到Python路径，以便导入analyze_search_results
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Search NMR library using cosine similarity")
    parser.add_argument("--checkpoint", type=str, default=os.environ.get("NMR_SEARCH_CKPT", "artifacts/checkpoints/search.pth"),
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--retrieval_dir", type=str, default=os.environ.get("NMR_CKPT_DIR", "artifacts/checkpoints"),
                        help="Directory containing retrieval system (LMDB + Faiss)")
    parser.add_argument("--query_file", type=str, default=os.environ.get("NMR_QUERY_FILE", "artifacts/benchmarks/nmrgym_test.jsonl"),
                        help="JSONL file with query spectra (must include 'smiles' field if using formula filter)")
    parser.add_argument("--output_file", type=str, default=os.environ.get("NMR_OUTPUT_FILE", "results/results_formula_k_30.jsonl"),
                        help="Output JSONL file for search results")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of top results to return (default: 10)")
    parser.add_argument("--nprobe", type=int, default=128,
                        help="Faiss nprobe parameter (default: 128, higher=more accurate but slower, only used when formula filter is disabled)")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Maximum number of queries to process (for testing)")
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Embedding dimension (default: 768)")
    parser.add_argument("--enable_formula_filter", action='store_true',
                        help="Enable formula filter: only search spectra with the same molecular formula (requires SMILES in query)")
    parser.add_argument("--analyze_results", action='store_true',
                        help="Automatically analyze search results and print statistics")
    parser.add_argument("--top_k_analysis", type=str, default='1,5,10',
                        help="Comma-separated list of k values for top-k accuracy analysis (default: 1,5,10)")
    parser.add_argument("--save_samples", type=int, default=10,
                        help="Number of sample queries to save for detailed analysis (default: 10)")

    args = parser.parse_args()

    # 创建搜索器
    searcher = NMRSearcher(
        model_checkpoint=args.checkpoint,
        retrieval_dir=args.retrieval_dir,
        embedding_dim=args.embedding_dim,
        enable_formula_filter=args.enable_formula_filter
    )

    # 批量搜索
    searcher.search_batch(
        query_file=args.query_file,
        output_file=args.output_file,
        k=args.k,
        nprobe=args.nprobe,
        max_queries=args.max_queries
    )

    # 关闭
    searcher.close()
    print("\n✓ Search completed successfully")

    # 如果启用结果分析，自动分析结果
    if args.analyze_results:
        print("\n" + "="*80)
        print("Analyzing Search Results")
        print("="*80)

        try:
            from analyze_search_results import analyze_search_results, compare_query_and_top1

            # 解析top-k值
            k_values = [int(k.strip()) for k in args.top_k_analysis.split(',')]

            # 运行分析
            stats = analyze_search_results(args.output_file, top_k_list=k_values)

            # 保存完整的统计结果到JSON文件
            stats_output = args.output_file.replace('.jsonl', '_stats.json')
            print(f"\nSaving statistics to {stats_output}")
            with open(stats_output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"✓ Statistics saved")

            # 保存抽样的详细结果
            if args.save_samples > 0:
                samples_output = args.output_file.replace('.jsonl', '_samples.json')
                print(f"\nSaving sample queries to {samples_output}")

                # 收集抽样结果
                sample_results = []
                with open(args.output_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= args.save_samples:
                            break
                        data = json.loads(line)

                        # 提取查询信息
                        query = data['query']
                        results = data['results']

                        if not results:
                            continue

                        # 查找正确答案的排名
                        query_smiles = query.get('smiles', '')
                        correct_rank = None
                        if query_smiles:
                            for rank, r in enumerate(results):
                                if r.get('smiles') == query_smiles:
                                    correct_rank = rank + 1
                                    break

                        # 构建抽样数据
                        sample_info = {
                            'query_index': i,
                            'query_smiles': query_smiles,
                            'query_h_nmr': query.get('h_shift', []),
                            'query_c_nmr': query.get('c_shift', []),
                            'correct_rank': correct_rank,
                            'top_k_results': []
                        }

                        # 添加top-k结果
                        for rank, r in enumerate(results[:max(k_values)]):
                            result_info = {
                                'rank': rank + 1,
                                'smiles': r.get('smiles', ''),
                                'cosine_similarity': r.get('cosine_similarity'),
                                'is_correct': r.get('smiles') == query_smiles if query_smiles else None,
                                'h_nmr': r.get('H_nmr', []),
                                'c_nmr': r.get('C_nmr', [])
                            }

                            # 计算Tanimoto similarity
                            if query_smiles and r.get('smiles'):
                                try:
                                    from rdkit import Chem
                                    from rdkit.Chem import rdFingerprintGenerator, DataStructs
                                    mol1 = Chem.MolFromSmiles(query_smiles)
                                    mol2 = Chem.MolFromSmiles(r['smiles'])
                                    if mol1 and mol2:
                                        mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                                        fp1 = mgen.GetFingerprint(mol1)
                                        fp2 = mgen.GetFingerprint(mol2)
                                        result_info['tanimoto_similarity'] = DataStructs.TanimotoSimilarity(fp1, fp2)
                                except:
                                    pass

                            sample_info['top_k_results'].append(result_info)

                        sample_results.append(sample_info)

                # 保存到文件
                with open(samples_output, 'w') as f:
                    json.dump(sample_results, f, indent=2)
                print(f"✓ Saved {len(sample_results)} sample queries")

            # 显示示例比较（打印到控制台）
            if args.save_samples > 0:
                print("\n" + "="*80)
                print(f"Sample Query Comparisons (showing first {min(args.save_samples, 10)} queries)")
                print("="*80)
                compare_query_and_top1(args.output_file, max_examples=min(args.save_samples, 10))

            print("\n✓ Result analysis completed")

        except ImportError as e:
            print(f"\n✗ Could not import analyze_search_results: {e}")
            print("  Skipping result analysis")
        except Exception as e:
            print(f"\n✗ Error during result analysis: {e}")
            print("  Results were saved but analysis failed")


if __name__ == "__main__":
    main()
