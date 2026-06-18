"""
NMR De Novo Generation Tool

This tool uses the UltraNMR model to generate candidate SMILES directly from NMR spectra.
"""

import os
import sys
import torch
import numpy as np
import re
import json
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from .decorator import tool
from .path_utils import artifact_path
from .pool_store import save_pool

# Global model cache
_DENOVO_MODEL = None
_DENOVO_TOKENIZER = None
_DENOVO_DEVICE = None
_DENOVO_LOAD_ERROR = None  # Stores last load error message


def _get_chemberta_path() -> str:
    """Resolve ChemBERTa tokenizer path from env or known local cache."""
    env_path = os.environ.get("NMR_CHEMBERTA_PATH")
    if env_path:
        return env_path

    candidates = [
        str(artifact_path("chemberta")),
        str(Path.home() / ".cache/huggingface/hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots"),
    ]

    for cand in candidates:
        if not cand:
            continue
        path = Path(cand)
        if path.is_dir():
            if path.name == "snapshots":
                snapshots = sorted([p for p in path.iterdir() if p.is_dir()])
                if snapshots:
                    return str(snapshots[-1])
            return str(path)

    return str(artifact_path("chemberta"))


def _get_denovo_checkpoint_path() -> str:
    """Resolve denovo checkpoint path from env or checkpoint root."""
    if os.environ.get("NMR_DENOVO_CKPT"):
        return os.environ["NMR_DENOVO_CKPT"]
    ckpt_dir = os.environ.get("NMR_CKPT_DIR", str(artifact_path("checkpoints")))
    return str(Path(ckpt_dir) / "denovo_v3.pth")


# ============== ChemBerta SMILES Tokenizer ==============

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

    def __init__(self, chemberta_path: str = None):
        """
        Initialize tokenizer by loading ChemBERTa's vocabulary.

        Args:
            chemberta_path: Path to ChemBERTa checkpoint directory
        """
        chemberta_path = chemberta_path or _get_chemberta_path()
        self.token2idx = self._load_vocab(chemberta_path)
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

    @staticmethod
    def _load_vocab(chemberta_path: str) -> Dict[str, int]:
        """Load token->id mapping from HF tokenizer dir or a plain vocab.json."""
        path = Path(chemberta_path)

        # Support passing a bare vocab.json path directly.
        if path.is_file() and path.suffix == ".json":
            with open(path, "r") as f:
                vocab = json.load(f)
            if all(isinstance(k, str) and isinstance(v, int) for k, v in vocab.items()):
                return vocab
            raise ValueError(f"Invalid vocab JSON format: {chemberta_path}")

        # Fallback to Hugging Face tokenizer loading.
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(
            chemberta_path,
            local_files_only=path.is_dir(),
        )
        return hf_tokenizer.get_vocab()

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
        ids: Union[List[int], torch.Tensor],
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


# ============== Formula Counts Helper ==============

def get_formula_counts(formula: str, num_atom_types: int = 50) -> torch.Tensor:
    """
    Convert molecular formula string to atom counts tensor.

    Args:
        formula: Molecular formula string (e.g., "C7H8")
        num_atom_types: Number of atom types (default: 50)

    Returns:
        torch.Tensor: (num_atom_types,) tensor with atom counts

    Atom type mapping (matches UltraNMR training):
        Index 0-9: C, H, O, N, S, P, F, Cl, Br, I
        Index 10-19: B, Si, Se, Te, As, Ge, Sb, Bi, Sn, Pb
        Index 20-29: Na, K, Li, Mg, Ca, Zn, Fe, Cu, Mn, Co
        Index 30-39: Ni, Pd, Pt, Ag, Au, Hg, Cd, Al, Ga, In
        Index 40-49: Tl, Ti, V, Cr, Mo, W, Ru, Rh, Ir, Os
    """
    import re

    # Atom type mapping (matches DEFAULT_ATOM_TYPES from nmr2smiles_dataset.py)
    DEFAULT_ATOM_TYPES = [
        'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I',
        'B', 'Si', 'Se', 'Te', 'As', 'Ge', 'Sb', 'Bi', 'Sn', 'Pb',
        'Na', 'K', 'Li', 'Mg', 'Ca', 'Zn', 'Fe', 'Cu', 'Mn', 'Co',
        'Ni', 'Pd', 'Pt', 'Ag', 'Au', 'Hg', 'Cd', 'Al', 'Ga', 'In',
        'Tl', 'Ti', 'V', 'Cr', 'Mo', 'W', 'Ru', 'Rh', 'Ir', 'Os',
    ]

    atom_type_to_idx = {atom: idx for idx, atom in enumerate(DEFAULT_ATOM_TYPES)}
    counts = torch.zeros(num_atom_types, dtype=torch.long)

    # Parse formula using regex
    # Match patterns like C7, H8, Cl2, etc.
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    for element, count_str in matches:
        if element in atom_type_to_idx:
            count = int(count_str) if count_str else 1
            counts[atom_type_to_idx[element]] += count

    return counts


def _filter_by_formula(candidates: List[Dict], target_formula: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter candidates by molecular formula.

    Args:
        candidates: List of candidate dictionaries with 'smiles' key
        target_formula: Target molecular formula (e.g., "C7H8")

    Returns:
        Tuple of (matched_candidates, unmatched_candidates)
    """
    from rdkit import Chem

    matched = []
    unmatched = []

    for cand in candidates:
        try:
            mol = Chem.MolFromSmiles(cand['smiles'])
            if mol:
                mol_with_h = Chem.AddHs(mol)
                formula = Chem.rdMolDescriptors.CalcMolFormula(mol_with_h)

                if formula == target_formula:
                    matched.append(cand)
                else:
                    unmatched.append(cand)
            else:
                unmatched.append(cand)
        except:
            unmatched.append(cand)

    return matched, unmatched


# ============== Model Loading ==============


def _load_denovo_model():
    """Load the de novo generation model (lazy loading)."""
    global _DENOVO_MODEL, _DENOVO_TOKENIZER, _DENOVO_DEVICE, _DENOVO_LOAD_ERROR

    if _DENOVO_MODEL is not None:
        return _DENOVO_MODEL, _DENOVO_TOKENIZER, _DENOVO_DEVICE

    try:
        from .nmr_models.models.nmr2smiles import SmilesDecoder, NMR2SmilesModel
        from .nmr_models.models.nmr_bert import NMR_BERT

        print("[NMR De Novo] Loading UltraNMR model...")

        # Configuration
        config = {
            "tokenizer_type": "chemberta",
            "tokenizer_path": _get_chemberta_path(),
            "d_model": 768,
            "nhead": 12,
            "num_encoder_layers": 16,
            "dim_feedforward": 3072,
            "dropout": 0.1,
            "num_decoder_layers": 6,
            "decoder_nhead": 12,
            "decoder_dim_feedforward": 3072,
            "h_bin_size": 0.01,
            "h_max": 16.0,
            "c_bin_size": 0.1,
            "c_max": 230.0,
            "use_identity_embedding": True,
            "use_count_embedding": False,
            "max_smiles_len": 256,
            "use_count_fusion": False,
            "use_formula_embedding": True,
            "num_atom_types": 50,  # For formula_embedding input size
        }

        # Initialize tokenizer
        tokenizer = ChemBertaSmilesTokenizer(
            chemberta_path=config["tokenizer_path"]
        )

        # Initialize encoder
        encoder = NMR_BERT(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            h_bin_size=config['h_bin_size'],
            h_max=config['h_max'],
            c_bin_size=config['c_bin_size'],
            c_max=config['c_max'],
            fp_sim_bin_size=0.005,  # Must match checkpoint: 201 bins = int(1.0/0.005) + 1
            use_identity_embedding=config['use_identity_embedding'],
            use_count_embedding=config['use_count_embedding'],
        )

        # Initialize decoder
        decoder = SmilesDecoder(
            vocab_size=len(tokenizer),
            d_model=config["d_model"],
            nhead=config["decoder_nhead"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config["decoder_dim_feedforward"],
            dropout=config["dropout"],
            max_seq_len=config["max_smiles_len"],
            pad_idx=tokenizer.pad_token_id,
            pretrained_embedding=None,
            freeze_embedding=False,
        )

        # Create model
        model = NMR2SmilesModel(
            encoder=encoder,
            decoder=decoder,
            freeze_encoder=False,
            use_count_fusion=config['use_count_fusion'],
            use_formula_embedding=config['use_formula_embedding'],
            num_atom_types=config['num_atom_types'],
        )

        # Load checkpoint and auto-detect formula embedding
        checkpoint_path = _get_denovo_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Auto-detect formula embedding from checkpoint
        has_formula_embedding = any('formula_embedding' in k for k in checkpoint['model_state_dict'].keys())
        print(f"[NMR De Novo] Formula embedding detected: {has_formula_embedding}")

        # Recreate model with correct config
        if not has_formula_embedding:
            model = NMR2SmilesModel(
                encoder=encoder,
                decoder=decoder,
                freeze_encoder=False,
                use_count_fusion=config['use_count_fusion'],
                use_formula_embedding=False,  # Disable formula embedding
                num_atom_types=config['num_atom_types'],
            )

        model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        _DENOVO_MODEL = model
        _DENOVO_TOKENIZER = tokenizer
        _DENOVO_DEVICE = device

        print(f"[NMR De Novo] Model loaded successfully on {device}")
        return model, tokenizer, device

    except Exception as e:
        _DENOVO_LOAD_ERROR = str(e)
        print(f"[NMR De Novo] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return None


def _decode_smiles(token_ids, tokenizer, eos_idx):
    """Decode token IDs to SMILES string."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # Find EOS token
    if eos_idx in token_ids:
        eos_pos = token_ids.index(eos_idx)
        token_ids = token_ids[:eos_pos]

    # Decode
    smiles = tokenizer.decode(token_ids, skip_special_tokens=True)
    smiles = smiles.strip()

    # Try to find a valid SMILES substring if full string is invalid
    # Sometimes the model generates valid SMILES followed by garbage
    from rdkit import Chem

    # First try the full string
    if Chem.MolFromSmiles(smiles) is not None:
        return smiles

    # Try progressively shorter substrings
    for length in range(len(smiles), max(0, len(smiles) - 50), -1):
        substring = smiles[:length]
        if Chem.MolFromSmiles(substring) is not None:
            return substring

    return smiles  # Return original even if invalid


@torch.no_grad()
def generate_candidates_beam_search(
    h_shifts: List[float],
    c_shifts: List[float],
    formula: str,
    beam_size: int = 10,
    max_len: int = 256,
    length_penalty: float = 1.0
) -> List[Dict]:
    """
    Generate candidate SMILES using beam search.

    Args:
        h_shifts: List of H-NMR chemical shifts
        c_shifts: List of C-NMR chemical shifts
        formula: Molecular formula (e.g., "C7H8")
        beam_size: Number of beams for beam search
        max_len: Maximum sequence length
        length_penalty: Length penalty for beam search

    Returns:
        List of candidates with scores and canonical forms
    """
    model, tokenizer, device = _load_denovo_model()

    if model is None:
        return []

    # Prepare input
    h_shifts_tensor = torch.tensor(h_shifts, dtype=torch.float32).unsqueeze(0).to(device)
    c_shifts_tensor = torch.tensor(c_shifts, dtype=torch.float32).unsqueeze(0).to(device)

    # H counts (default to 1 for each peak)
    h_counts = torch.ones_like(h_shifts_tensor)
    c_counts = torch.zeros_like(c_shifts_tensor)

    # Combine shifts and types
    all_shifts = torch.cat([h_shifts_tensor, c_shifts_tensor], dim=1)
    all_counts = torch.cat([h_counts, c_counts], dim=1)
    types = torch.cat([
        torch.zeros_like(h_shifts_tensor, dtype=torch.long),  # H = 0
        torch.ones_like(c_shifts_tensor, dtype=torch.long)    # C = 1
    ], dim=1)

    # Create padding mask
    nmr_padding_mask = torch.zeros_like(types, dtype=torch.bool)

    # Get formula counts
    formula_counts = get_formula_counts(formula).unsqueeze(0).to(device)

    # Beam search
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id

    try:
        sequences, scores = model.beam_search_generate(
            shifts=all_shifts,
            counts=all_counts,
            types=types,
            nmr_padding_mask=nmr_padding_mask,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            formula_counts=formula_counts,  # Use formula counts
            beam_size=beam_size,
            max_len=max_len,
            length_penalty=length_penalty,
        )

        # Decode candidates
        candidates = []
        seen_canonical = set()  # Track unique canonical SMILES

        for beam_idx in range(beam_size):
            # Decode token IDs to SMILES
            token_ids = sequences[0, beam_idx, 1:].tolist()

            # Find EOS token
            if eos_idx in token_ids:
                eos_pos = token_ids.index(eos_idx)
                token_ids = token_ids[:eos_pos]

            # Decode
            gen_smiles = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

            # Canonicalize
            gen_canonical = _canonicalize_smiles(gen_smiles)

            # Only add valid and unique candidates
            if gen_canonical is not None and gen_canonical not in seen_canonical:
                seen_canonical.add(gen_canonical)
                candidates.append({
                    'smiles': gen_canonical,
                    'score': float(scores[0, beam_idx].item()),
                    'source': 'denovo',
                    'rank': len(candidates) + 1
                })

        return candidates

    except Exception as e:
        print(f"[NMR De Novo] Beam search failed: {e}")
        import traceback
        traceback.print_exc()
        return []


@tool(
    name="nmr_denovo",
    description="Generate candidate SMILES directly from NMR spectra and molecular formula using deep learning model (de novo generation)."
)
def nmr_denovo(
    h_shifts: str,
    c_shifts: str,
    formula: str,
    top_k: int = 10,
    save_pool_file: bool = False,
    pool_path: str = ""
) -> dict:
    """
    Generate candidate molecular structures from NMR spectra using de novo generation.

    This tool uses a transformer-based model to directly generate SMILES from NMR peaks,
    without relying on database search. It's complementary to nmr_retrieve.

    Args:
        h_shifts (str): Comma-separated H-NMR chemical shifts (ppm), e.g., "7.3,7.2,2.3"
        c_shifts (str): Comma-separated C-NMR chemical shifts (ppm), e.g., "138.0,129.0,21.0"
        formula (str): Molecular formula (e.g., "C7H8")
        top_k (int): Number of top candidates to return (default: 10, max: 20)

    Returns:
        dict: Dictionary containing:
            - observation: Human-readable summary
            - candidates: List of candidate SMILES with scores
            - count: Number of candidates returned

    Example:
        >>> nmr_denovo("7.3,7.2,2.3", "138.0,129.0,21.0", "C7H8", top_k=5)
        {
            "observation": "Generated 5 candidate structures from NMR spectra...",
            "candidates": [
                {"smiles": "Cc1ccccc1", "score": 0.95, "rank": 1},
                ...
            ],
            "count": 5
        }
    """
    try:
        # Parse input
        h_list = h_shifts if isinstance(h_shifts, list) else [float(x.strip()) for x in h_shifts.split(',') if x.strip()]
        c_list = c_shifts if isinstance(c_shifts, list) else [float(x.strip()) for x in c_shifts.split(',') if x.strip()]
        h_list = [float(x) for x in h_list]
        c_list = [float(x) for x in c_list]

        if not h_list and not c_list:
            return {
                "observation": "Error: At least one of H-NMR or C-NMR shifts must be provided.",
                "candidates": [],
                "count": 0
            }

        if not formula:
            return {
                "observation": "Error: Molecular formula is required for de novo generation.",
                "candidates": [],
                "count": 0
            }

        # Limit top_k
        top_k = min(max(1, top_k), 20)
        # Use larger beam_size to increase chance of formula matches
        # But cap at reasonable limit to avoid being too slow
        beam_size = min(max(top_k * 2, 15), 30)  # Between 15-30

        # Generate candidates
        candidates = generate_candidates_beam_search(
            h_shifts=h_list,
            c_shifts=c_list,
            formula=formula,
            beam_size=beam_size,
            max_len=256,
            length_penalty=1.0
        )

        if not candidates:
            load_err = _DENOVO_LOAD_ERROR or "unknown reason"
            return {
                "observation": f"Error: Failed to generate valid candidates. Model load error: {load_err}",
                "candidates": [],
                "count": 0
            }

        # Filter by formula
        matched, unmatched = _filter_by_formula(candidates, formula)

        # Debug: print filtering results
        print(f"[NMR De Novo] Generated {len(candidates)} total, {len(matched)} exact matches, {len(unmatched)} approximate")

        # Return top_k candidates, prioritizing matched ones
        # Don't wait for all top_k to be exact matches - return what we have
        final_candidates = matched[:top_k]
        remaining_slots = top_k - len(final_candidates)

        if remaining_slots > 0:
            # Add best unmatched candidates to fill up to top_k
            final_candidates.extend(unmatched[:remaining_slots])

        # Format observation
        obs_lines = [
            f"Generated {len(candidates)} candidate structures from NMR spectra (de novo):",
            f"  Input: {len(h_list)} H-NMR peaks, {len(c_list)} C-NMR peaks",
            f"  Formula: {formula}",
            f"  Exact matches: {len(matched)}/{len(candidates)}",
            "",
            "Top candidates:"
        ]

        for i, cand in enumerate(final_candidates[:5], 1):
            # Check if this candidate matches formula
            is_match = cand in matched
            match_marker = "✓" if is_match else "~"
            obs_lines.append(f"  {i}. {cand['smiles']} (score: {cand['score']:.4f}) {match_marker}")

        if len(final_candidates) > 5:
            obs_lines.append(f"  ... and {len(final_candidates) - 5} more")

        obs_lines.append("")
        obs_lines.append("✓ = exact formula match, ~ = approximate match")

        response = {
            "observation": "\n".join(obs_lines),
            "candidates": final_candidates,
            "count": len(final_candidates)
        }
        if save_pool_file:
            response["pool_path"] = save_pool(
                final_candidates,
                prefix="denovo_pool",
                query={"formula": formula, "h_shifts": h_list, "c_shifts": c_list},
                metadata={"tool": "nmr_denovo"},
                path=pool_path,
            )
        return response

    except Exception as e:
        return {
            "observation": f"Error in de novo generation: {str(e)}",
            "candidates": [],
            "count": 0
        }
