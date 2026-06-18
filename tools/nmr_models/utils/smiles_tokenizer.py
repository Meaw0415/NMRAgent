"""
SMILES Tokenizer for NMR2SMILES task

Provides character-level and regex-based tokenization for SMILES strings.
Compatible with HuggingFace tokenizer interface.
"""

import os
from pathlib import Path
import re
from typing import List, Dict, Optional, Union
import json


class BasicSmilesTokenizer:
    """
    SMILES Tokenizer with support for:
    - Character-level tokenization
    - Regex-based tokenization (handles multi-char tokens like Cl, Br, @@)
    """

    # Regex pattern for SMILES tokens
    SMILES_REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocab_file: Optional[str] = None):
        """
        Args:
            vocab_file: Path to vocabulary JSON file. If None, uses default vocab.
        """
        self.pattern = re.compile(self.SMILES_REGEX)

        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Build default vocabulary from common SMILES tokens."""
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        # Common SMILES tokens
        common_tokens = [
            # Atoms
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'Cl', 'Br', 'I',
            'B', 'b', 'Si', 'Se', 'se', 'Te', 'te',
            # Bonds
            '-', '=', '#', ':', '/', '\\',
            # Rings
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '%10', '%11', '%12', '%13', '%14', '%15',
            # Branches
            '(', ')',
            # Stereochemistry
            '@', '@@',
            # Charges and others
            '+', '.', '*',
            # Common bracket atoms
            '[C]', '[c]', '[N]', '[n]', '[O]', '[o]', '[S]', '[s]',
            '[C@H]', '[C@@H]', '[C@]', '[C@@]',
            '[N+]', '[N-]', '[O-]', '[S-]',
            '[nH]', '[NH]', '[NH2]', '[NH3+]',
            '[O+]', '[OH]', '[OH-]', '[OH2+]',
            '[S+]', '[SH]',
            '[H]', '[2H]', '[3H]',
            '[Na]', '[Na+]', '[K]', '[K+]',
            '[Ca]', '[Ca+2]', '[Mg]', '[Mg+2]',
            '[Fe]', '[Fe+2]', '[Fe+3]',
            '[Zn]', '[Zn+2]', '[Cu]', '[Cu+2]',
            '[Pt]', '[Pd]', '[Ag]', '[Au]',
        ]

        all_tokens = special_tokens + common_tokens
        self.token2idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.pad_idx = self.token2idx[self.PAD_TOKEN]
        self.bos_idx = self.token2idx[self.BOS_TOKEN]
        self.eos_idx = self.token2idx[self.EOS_TOKEN]
        self.unk_idx = self.token2idx[self.UNK_TOKEN]

        self.vocab_size = len(self.token2idx)

    # HuggingFace compatible properties
    @property
    def pad_token_id(self) -> int:
        return self.pad_idx

    @property
    def bos_token_id(self) -> int:
        return self.bos_idx

    @property
    def eos_token_id(self) -> int:
        return self.eos_idx

    @property
    def unk_token_id(self) -> int:
        return self.unk_idx

    def __len__(self) -> int:
        return self.vocab_size

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into tokens."""
        tokens = self.pattern.findall(smiles)
        return tokens

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        add_bos: bool = None,
        add_eos: bool = None,
    ) -> List[int]:
        """
        Encode SMILES string to token indices.

        Args:
            smiles: SMILES string
            add_special_tokens: Add BOS and EOS tokens (HuggingFace compatible)
            add_bos: Add BOS token at start (legacy, overrides add_special_tokens)
            add_eos: Add EOS token at end (legacy, overrides add_special_tokens)

        Returns:
            List of token indices
        """
        # Handle legacy parameters
        if add_bos is None:
            add_bos = add_special_tokens
        if add_eos is None:
            add_eos = add_special_tokens

        tokens = self.tokenize(smiles)
        indices = []

        if add_bos:
            indices.append(self.bos_idx)

        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                # Try to add unknown token to vocab dynamically
                indices.append(self.unk_idx)

        if add_eos:
            indices.append(self.eos_idx)

        return indices

    def decode(
        self,
        ids: Union[List[int], 'torch.Tensor'],
        skip_special_tokens: bool = True,
        skip_special: bool = None,  # Legacy parameter
    ) -> str:
        """
        Decode token indices back to SMILES string.

        Args:
            ids: List of token indices or tensor
            skip_special_tokens: Skip special tokens (HuggingFace compatible)
            skip_special: Legacy parameter name

        Returns:
            SMILES string
        """
        # Handle legacy parameter
        if skip_special is not None:
            skip_special_tokens = skip_special

        # Convert tensor to list if needed
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()

        special_indices = {self.pad_idx, self.bos_idx, self.eos_idx}

        tokens = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if skip_special_tokens and idx in special_indices:
                continue
            if idx in self.idx2token:
                tokens.append(self.idx2token[idx])
            else:
                tokens.append(self.UNK_TOKEN)

        return ''.join(tokens)

    def add_token(self, token: str) -> int:
        """Add a new token to vocabulary."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            self.vocab_size = len(self.token2idx)
        return self.token2idx[token]

    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'token2idx': self.token2idx,
                'special_tokens': {
                    'pad': self.PAD_TOKEN,
                    'bos': self.BOS_TOKEN,
                    'eos': self.EOS_TOKEN,
                    'unk': self.UNK_TOKEN,
                }
            }, f, indent=2)

    def load_vocab(self, path: str):
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.token2idx = data['token2idx']
        # Convert string keys to int for idx2token
        self.idx2token = {int(idx) if isinstance(idx, str) else idx: token
                         for token, idx in self.token2idx.items()}

        self.pad_idx = self.token2idx[self.PAD_TOKEN]
        self.bos_idx = self.token2idx[self.BOS_TOKEN]
        self.eos_idx = self.token2idx[self.EOS_TOKEN]
        self.unk_idx = self.token2idx[self.UNK_TOKEN]
        self.vocab_size = len(self.token2idx)

    def build_vocab_from_data(self, smiles_list: List[str]):
        """Build vocabulary from a list of SMILES strings."""
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            for token in tokens:
                self.add_token(token)

    @classmethod
    def from_pretrained(cls, vocab_file: str) -> 'BasicSmilesTokenizer':
        """
        Load tokenizer from vocabulary file (HuggingFace-style API).

        Args:
            vocab_file: Path to vocabulary JSON file

        Returns:
            BasicSmilesTokenizer instance
        """
        return cls(vocab_file=vocab_file)


# Backward compatibility alias
SmilesTokenizer = BasicSmilesTokenizer


class ChemBertaSmilesTokenizer:
    """
    SMILES Tokenizer that uses ChemBERTa's vocabulary but with correct SMILES tokenization.

    This tokenizer:
    1. Uses ChemBERTa's vocabulary (593 tokens including stereochemistry)
    2. Uses regex-based tokenization that properly handles bracket atoms and stereochemistry
    3. Is compatible with HuggingFace tokenizer interface
    """

    # SMILES tokenization regex - handles bracket atoms, multi-char atoms, stereochemistry
    # Order matters: longer patterns first
    SMILES_REGEX = re.compile(
        r"(\[[^\]]+\]|Br|Cl|>>|%\d{2}|\d|b|c|n|o|p|s|B|C|N|O|P|S|F|I|@@|@|\.|\(|\)|=|#|-|\+|\\|\/|:|~|\*)"
    )

    def __init__(self, chemberta_path: Optional[str] = None):
        """
        Initialize tokenizer by loading ChemBERTa's vocabulary.

        Args:
            chemberta_path: Path to ChemBERTa checkpoint directory
        """
        from transformers import AutoTokenizer

        if chemberta_path is None:
            chemberta_path = os.environ.get("NMR_CHEMBERTA_PATH", str(Path(__file__).resolve().parents[3] / "artifacts" / "chemberta"))

        hf_tokenizer = AutoTokenizer.from_pretrained(chemberta_path)
        self.token2idx = hf_tokenizer.get_vocab()
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # Set special token indices (ChemBERTa specific)
        self.pad_idx = self.token2idx.get('[PAD]', 0)
        self.unk_idx = self.token2idx.get('[UNK]', 11)
        self.cls_idx = self.token2idx.get('[CLS]', 12)
        self.sep_idx = self.token2idx.get('[SEP]', 13)
        self.mask_idx = self.token2idx.get('[MASK]', 14)
        self.bos_idx = self.token2idx.get('<s>', 591)
        self.eos_idx = self.token2idx.get('</s>', 592)

        self.vocab_size = len(self.token2idx)

        # Special tokens
        self.PAD_TOKEN = '[PAD]'
        self.UNK_TOKEN = '[UNK]'
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'

    # HuggingFace compatible properties
    @property
    def pad_token_id(self) -> int:
        return self.pad_idx

    @property
    def bos_token_id(self) -> int:
        return self.bos_idx

    @property
    def eos_token_id(self) -> int:
        return self.eos_idx

    @property
    def unk_token_id(self) -> int:
        return self.unk_idx

    @property
    def cls_token_id(self) -> int:
        return self.cls_idx

    @property
    def sep_token_id(self) -> int:
        return self.sep_idx

    @property
    def mask_token_id(self) -> int:
        return self.mask_idx

    def __len__(self) -> int:
        return self.vocab_size

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into tokens using regex."""
        tokens = self.SMILES_REGEX.findall(smiles)
        return tokens

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        add_bos: bool = None,
        add_eos: bool = None,
    ) -> List[int]:
        """
        Encode SMILES string to token indices.

        Args:
            smiles: SMILES string
            add_special_tokens: Add BOS and EOS tokens (HuggingFace compatible)
            add_bos: Add BOS token at start (legacy, overrides add_special_tokens)
            add_eos: Add EOS token at end (legacy, overrides add_special_tokens)

        Returns:
            List of token indices
        """
        if add_bos is None:
            add_bos = add_special_tokens
        if add_eos is None:
            add_eos = add_special_tokens

        tokens = self.tokenize(smiles)
        indices = []

        if add_bos:
            indices.append(self.bos_idx)

        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.unk_idx)

        if add_eos:
            indices.append(self.eos_idx)

        return indices

    def decode(
        self,
        ids: Union[List[int], 'torch.Tensor'],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token indices back to SMILES string.

        Args:
            ids: List of token indices or tensor
            skip_special_tokens: Skip special tokens

        Returns:
            SMILES string
        """
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()

        special_indices = {self.pad_idx, self.bos_idx, self.eos_idx, self.cls_idx, self.sep_idx}

        tokens = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if skip_special_tokens and idx in special_indices:
                continue
            if idx in self.idx2token:
                tokens.append(self.idx2token[idx])
            else:
                tokens.append(self.UNK_TOKEN)

        return ''.join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self.token2idx.copy()

    @classmethod
    def from_pretrained(cls, chemberta_path: str) -> 'ChemBertaSmilesTokenizer':
        """Load tokenizer from ChemBERTa checkpoint (HuggingFace-style API)."""
        return cls(chemberta_path=chemberta_path)


def build_vocab_from_lmdb(lmdb_path: str, output_path: str, max_samples: int = 1000000):
    """
    Build vocabulary from LMDB dataset.

    Args:
        lmdb_path: Path to LMDB file
        output_path: Path to save vocabulary JSON
        max_samples: Maximum samples to scan
    """
    import lmdb
    import pickle
    from tqdm import tqdm

    tokenizer = BasicSmilesTokenizer()

    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        total = min(txn.stat()['entries'], max_samples)

        for i in tqdm(range(total), desc="Building vocab"):
            data = pickle.loads(txn.get(str(i).encode()))
            smiles = data.get('smiles', '')
            if smiles:
                tokens = tokenizer.tokenize(smiles)
                for token in tokens:
                    tokenizer.add_token(token)

    env.close()

    tokenizer.save_vocab(output_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, required=True, help='LMDB path')
    parser.add_argument('--output', type=str, default='smiles_vocab.json', help='Output vocab file')
    parser.add_argument('--max-samples', type=int, default=1000000)
    args = parser.parse_args()

    build_vocab_from_lmdb(args.lmdb, args.output, args.max_samples)
