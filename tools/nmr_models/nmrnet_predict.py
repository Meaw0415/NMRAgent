"""
NMRNet prediction module using UniMol NMRNet ensemble models.

This module provides NMR spectrum prediction using NMRNet ensemble models.
All dependencies are installed in the conda nmr_agent environment.
"""

import sys
import os
import time
import traceback

# Add third_party/ to path for unicore / unimol_tools (copied from NMR-Solver)
_third_party = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party'))
if os.path.isdir(_third_party) and _third_party not in sys.path:
    sys.path.insert(0, _third_party)
# Fallback: old external_packages location
_external_pkg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'external_packages')
if os.path.exists(_external_pkg_path) and _external_pkg_path not in sys.path:
    sys.path.insert(0, _external_pkg_path)

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..path_utils import artifact_path


def _debug_enabled() -> bool:
    return os.environ.get("NMR_RERANK_DEBUG", "0").lower() not in {"", "0", "false", "no"}


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[NMRNet DEBUG] {msg}", flush=True)

# ============== NMRNet Backend ==============

_NMRNET_AVAILABLE = False
_NMRNET_PREDICTOR = None
_NMRNET_PREDICTOR_FAST = None  # Fast predictor for reranking (n_splits=1)


def _get_default_model_dir() -> str:
    """Resolve NMRNet checkpoint directory from env or known layouts."""
    env_dir = os.environ.get("NMR_NMRNET_DIR")
    if env_dir:
        return env_dir

    ckpt_root = os.environ.get("NMR_CKPT_DIR")
    if ckpt_root:
        nested = os.path.join(ckpt_root, "NMRNet_ckpt")
        if os.path.isdir(nested):
            return nested

    local_ssd = str(artifact_path("checkpoints", "NMRNet_ckpt"))
    if os.path.isdir(local_ssd):
        return local_ssd

    local_solver = os.path.expanduser("~/NMR-Solver/model/model/nmrnet")
    if os.path.isdir(local_solver):
        return local_solver

    return local_ssd

try:
    from unimol_tools import MolPredict
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles
    _NMRNET_AVAILABLE = True
    print("[NMRNet] NMRNet backend available")
except ImportError as e:
    print(f"[NMRNet] NMRNet not available: {e}")


def _get_predictor(model_dir: str, fast: bool = False):
    """Get or create NMRNet predictor.

    Args:
        model_dir: Path to model directory
        fast: If True, use n_splits=1 for faster prediction (for reranking)
    """
    global _NMRNET_PREDICTOR, _NMRNET_PREDICTOR_FAST

    if fast:
        if _NMRNET_PREDICTOR_FAST is None:
            t0 = time.time()
            _NMRNET_PREDICTOR_FAST = MolPredict(load_model=model_dir, n_splits=1)
            _debug(f"FAST predictor initialized in {time.time() - t0:.2f}s")
        return _NMRNET_PREDICTOR_FAST
    else:
        if _NMRNET_PREDICTOR is None:
            t0 = time.time()
            _NMRNET_PREDICTOR = MolPredict(load_model=model_dir, n_splits=5)
            _debug(f"ENSEMBLE predictor initialized in {time.time() - t0:.2f}s")
        return _NMRNET_PREDICTOR


def _get_atomic_numbers(mol):
    """Get atomic numbers from molecule."""
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def _get_equi_class(mol):
    """Get equivalence classes for atoms."""
    return np.array(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)).astype(np.int16)


def _merge_equi_nmr(nmr, atom_index, equi_class, element_id=6):
    """Merge NMR shifts for equivalent atoms."""
    mask = atom_index == element_id
    equi_class = equi_class[mask]
    nmr = nmr[mask]
    unique_equi_class = np.unique(equi_class)
    return np.array([np.mean(nmr[equi_class == _class]) for _class in unique_equi_class])


def _predict_nmr_from_mol_internal(mol, model_dir: str, fast: bool = False):
    """Internal function to predict NMR from molecule.

    Args:
        mol: RDKit molecule
        model_dir: Path to model directory
        fast: If True, use n_splits=1 for faster prediction
    """
    # Get predictor
    clf = _get_predictor(model_dir, fast=fast)

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Prepare data
    infer_data = {
        'mol': [mol],
        'atom_target': [[0.0]*512],
        'atom_mask': [],
    }

    atom_num = _get_atomic_numbers(mol)
    infer_data['atom_mask'].append([0] + atom_num + [0]*(512-1-len(atom_num)))

    # Predict
    test_pred = clf.predict(infer_data, datatype='mol')

    # Extract predictions
    cv_pred = clf.cv_pred[0].astype(np.float32)
    cv_label_mask = clf.cv_label_mask[0]
    index_mask = np.array(clf.cv_index_mask.tolist()).astype(np.int8)[0]

    nmr_predict = cv_pred[cv_label_mask]
    mol_index = index_mask[cv_label_mask]

    # Get equivalence classes
    equi_class = _get_equi_class(mol)

    # Extract H and C NMR
    H_nmr = np.sort(nmr_predict[mol_index == 1])
    C_nmr = np.sort(_merge_equi_nmr(nmr_predict, mol_index, equi_class, element_id=6))

    return H_nmr, C_nmr


def predict_nmr_nmrnet(smiles: str, model_dir: str = None, fast: bool = False) -> Dict:
    """
    Predict NMR spectrum using NMRNet ensemble.

    Args:
        smiles: SMILES string
        model_dir: Path to NMRNet models (default: $NMR_CKPT_DIR/NMRNet_ckpt)
        fast: If True, use n_splits=1 for faster prediction (recommended for reranking)

    Returns:
        dict: {
            'H_shifts': np.ndarray,
            'C_shifts': np.ndarray,
            'valid': bool
        }
    """
    if not _NMRNET_AVAILABLE:
        return {
            'H_shifts': np.array([]),
            'C_shifts': np.array([]),
            'valid': False,
            'error': 'NMRNet not available'
        }

    if model_dir is None:
        model_dir = _get_default_model_dir()

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'H_shifts': np.array([]),
                'C_shifts': np.array([]),
                'valid': False,
                'error': 'Invalid SMILES'
            }

        # Predict using NMRNet
        h_shifts, c_shifts = _predict_nmr_from_mol_internal(mol, model_dir, fast=fast)

        return {
            'H_shifts': np.array(h_shifts),
            'C_shifts': np.array(c_shifts),
            'valid': True
        }

    except Exception as e:
        return {
            'H_shifts': np.array([]),
            'C_shifts': np.array([]),
            'valid': False,
            'error': str(e)
        }


def predict_nmr_batch(smiles_list: List[str], model_dir: str = None,
                      fast: bool = False, include_atom_data: bool = False) -> List[Dict]:
    """
    Batch predict NMR spectra for multiple SMILES.

    Args:
        smiles_list: List of SMILES strings
        model_dir: Path to NMRNet models
        fast: If True, use n_splits=1 for faster prediction
        include_atom_data: If True, also return per-atom predictions as
            'atom_data': list of {'heavy_idx': int, 'parent_idx': int,
                                   'element': str, 'shift': float}
            heavy_idx: atom index in original (no-H) mol; -1 for added H
            parent_idx: for H atoms, the heavy atom they are bonded to;
                        for C atoms, same as heavy_idx

    Returns:
        List of prediction results
    """
    if not _NMRNET_AVAILABLE:
        return [{'H_shifts': np.array([]), 'C_shifts': np.array([]),
                 'valid': False, 'error': 'NMRNet not available'}
                for _ in smiles_list]

    if model_dir is None:
        model_dir = _get_default_model_dir()

    _debug(
        f"predict_nmr_batch start total={len(smiles_list)} fast={fast} "
        f"include_atom_data={include_atom_data} model_dir={model_dir}"
    )
    results = []

    # Process in batches of 10 to avoid memory issues
    batch_size = 10
    for i in range(0, len(smiles_list), batch_size):
        batch_t0 = time.time()
        batch = smiles_list[i:i+batch_size]
        _debug(f"batch_start idx={i // batch_size} batch_len={len(batch)}")

        # Parse all SMILES in batch
        mols = []
        valid_indices = []
        for idx, smiles in enumerate(batch):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(idx)
        _debug(
            f"batch_parsed idx={i // batch_size} valid={len(valid_indices)} "
            f"invalid={len(batch) - len(valid_indices)}"
        )

        # Predict for valid molecules
        if mols:
            try:
                prep_t0 = time.time()
                clf = _get_predictor(model_dir, fast=fast)

                # Prepare batch data
                mols_h = [Chem.AddHs(mol) for mol in mols]
                infer_data = {
                    'mol': mols_h,
                    'atom_target': [[0.0]*512 for _ in mols],
                    'atom_mask': [],
                }

                for mol in infer_data['mol']:
                    atom_num = _get_atomic_numbers(mol)
                    infer_data['atom_mask'].append([0] + atom_num + [0]*(512-1-len(atom_num)))
                _debug(
                    f"batch_prepare idx={i // batch_size} "
                    f"prepare_time={time.time() - prep_t0:.2f}s"
                )

                # Batch predict
                pred_t0 = time.time()
                clf.predict(infer_data, datatype='mol')
                _debug(
                    f"batch_predict idx={i // batch_size} "
                    f"predict_time={time.time() - pred_t0:.2f}s"
                )

                # Extract results for each molecule
                post_t0 = time.time()
                batch_results = [None] * len(batch)
                for j, mol_idx in enumerate(valid_indices):
                    try:
                        cv_pred = clf.cv_pred[j].astype(np.float32)
                        cv_label_mask = clf.cv_label_mask[j]
                        index_mask = np.array(clf.cv_index_mask.tolist()).astype(np.int8)[j]

                        nmr_predict = cv_pred[cv_label_mask]
                        mol_index = index_mask[cv_label_mask]

                        equi_class = _get_equi_class(infer_data['mol'][j])

                        H_nmr = np.sort(nmr_predict[mol_index == 1])
                        C_nmr = np.sort(_merge_equi_nmr(nmr_predict, mol_index, equi_class, element_id=6))

                        result = {
                            'H_shifts': H_nmr,
                            'C_shifts': C_nmr,
                            'valid': True,
                        }

                        if include_atom_data:
                            # cv_label_mask selects positions in the 512-length atom_mask.
                            # atom_mask = [0(CLS)] + atom_nums + [0(pad)...]
                            # So position k in atom_mask → atom (k-1) in mol_h (0-indexed).
                            mask_positions = np.where(cv_label_mask)[0]  # positions in 512-array
                            atom_h_indices  = mask_positions - 1         # atom indices in mol_h
                            n_heavy = mols[j].GetNumAtoms()
                            mol_h  = mols_h[j]

                            atom_data = []
                            for k in range(len(nmr_predict)):
                                elem_num = int(mol_index[k])
                                if elem_num not in (1, 6):   # only H and C
                                    continue
                                h_idx   = int(atom_h_indices[k])
                                shift   = float(nmr_predict[k])
                                element = 'H' if elem_num == 1 else 'C'

                                if element == 'C':
                                    heavy_idx  = h_idx  # C atoms keep original index
                                    parent_idx = h_idx
                                else:
                                    # H atom: heavy_idx = -1 (added H not in original mol)
                                    heavy_idx = -1
                                    nbrs = mol_h.GetAtomWithIdx(h_idx).GetNeighbors()
                                    parent_idx = nbrs[0].GetIdx() if nbrs else -1

                                atom_data.append({
                                    'heavy_idx':  heavy_idx,
                                    'parent_idx': parent_idx,
                                    'element':    element,
                                    'shift':      shift,
                                })
                            result['atom_data'] = atom_data

                        batch_results[mol_idx] = result
                    except:
                        batch_results[mol_idx] = {
                            'H_shifts': np.array([]),
                            'C_shifts': np.array([]),
                            'valid': False,
                            'error': 'Prediction failed'
                        }

                # Fill in invalid molecules
                for idx in range(len(batch)):
                    if batch_results[idx] is None:
                        batch_results[idx] = {
                            'H_shifts': np.array([]),
                            'C_shifts': np.array([]),
                            'valid': False,
                            'error': 'Invalid SMILES'
                        }

                results.extend(batch_results)
                _debug(
                    f"batch_end idx={i // batch_size} post_time={time.time() - post_t0:.2f}s "
                    f"total_time={time.time() - batch_t0:.2f}s"
                )

            except Exception as e:
                _debug(
                    f"batch_error idx={i // batch_size} error={e}\n"
                    f"{traceback.format_exc()}"
                )
                # If batch fails, fall back to individual predictions
                for smiles in batch:
                    single_t0 = time.time()
                    results.append(predict_nmr_nmrnet(smiles, model_dir))
                    _debug(
                        f"fallback_single idx={i // batch_size} "
                        f"time={time.time() - single_t0:.2f}s"
                    )
        else:
            # All invalid
            for _ in batch:
                results.append({
                    'H_shifts': np.array([]),
                    'C_shifts': np.array([]),
                    'valid': False,
                    'error': 'Invalid SMILES'
                })
            _debug(f"batch_all_invalid idx={i // batch_size}")

    _debug(f"predict_nmr_batch end total_results={len(results)}")
    return results
    """
    Batch predict NMR spectra for multiple SMILES.

    Args:
        smiles_list: List of SMILES strings
        model_dir: Path to NMRNet models
        fast: If True, use n_splits=1 for faster prediction

    Returns:
        List of prediction results
    """
    if not _NMRNET_AVAILABLE:
        return [{'H_shifts': np.array([]), 'C_shifts': np.array([]),
                 'valid': False, 'error': 'NMRNet not available'}
                for _ in smiles_list]

    if model_dir is None:
        model_dir = _get_default_model_dir()

    results = []

    # Process in batches of 10 to avoid memory issues
    batch_size = 10
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]

        # Parse all SMILES in batch
        mols = []
        valid_indices = []
        for idx, smiles in enumerate(batch):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(idx)

        # Predict for valid molecules
        if mols:
            try:
                clf = _get_predictor(model_dir, fast=fast)

                # Prepare batch data
                infer_data = {
                    'mol': [Chem.AddHs(mol) for mol in mols],
                    'atom_target': [[0.0]*512 for _ in mols],
                    'atom_mask': [],
                }

                for mol in infer_data['mol']:
                    atom_num = _get_atomic_numbers(mol)
                    infer_data['atom_mask'].append([0] + atom_num + [0]*(512-1-len(atom_num)))

                # Batch predict
                clf.predict(infer_data, datatype='mol')

                # Extract results for each molecule
                batch_results = [None] * len(batch)
                for j, mol_idx in enumerate(valid_indices):
                    try:
                        cv_pred = clf.cv_pred[j].astype(np.float32)
                        cv_label_mask = clf.cv_label_mask[j]
                        index_mask = np.array(clf.cv_index_mask.tolist()).astype(np.int8)[j]

                        nmr_predict = cv_pred[cv_label_mask]
                        mol_index = index_mask[cv_label_mask]

                        equi_class = _get_equi_class(infer_data['mol'][j])

                        H_nmr = np.sort(nmr_predict[mol_index == 1])
                        C_nmr = np.sort(_merge_equi_nmr(nmr_predict, mol_index, equi_class, element_id=6))

                        batch_results[mol_idx] = {
                            'H_shifts': H_nmr,
                            'C_shifts': C_nmr,
                            'valid': True
                        }
                    except:
                        batch_results[mol_idx] = {
                            'H_shifts': np.array([]),
                            'C_shifts': np.array([]),
                            'valid': False,
                            'error': 'Prediction failed'
                        }

                # Fill in invalid molecules
                for idx in range(len(batch)):
                    if batch_results[idx] is None:
                        batch_results[idx] = {
                            'H_shifts': np.array([]),
                            'C_shifts': np.array([]),
                            'valid': False,
                            'error': 'Invalid SMILES'
                        }

                results.extend(batch_results)

            except Exception as e:
                # If batch fails, fall back to individual predictions
                for smiles in batch:
                    results.append(predict_nmr_nmrnet(smiles, model_dir))
        else:
            # All invalid
            for _ in batch:
                results.append({
                    'H_shifts': np.array([]),
                    'C_shifts': np.array([]),
                    'valid': False,
                    'error': 'Invalid SMILES'
                })

    return results


def calc_nmr_similarity_nmrnet(pred_h: np.ndarray, pred_c: np.ndarray,
                                target_h: np.ndarray, target_c: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate NMR similarity using weighted set matching algorithm.

    Args:
        pred_h: Predicted H-NMR shifts
        pred_c: Predicted C-NMR shifts
        target_h: Target H-NMR shifts
        target_c: Target C-NMR shifts

    Returns:
        Tuple of (h_score, c_score, combined_score)
    """
    from scipy.optimize import linear_sum_assignment

    def set_match_score_weighted(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
        """Compute weighted matching score between two sets of NMR data."""
        if len(x) == 0 or len(y) == 0:
            return 1.0 if (len(x) == 0 and len(y) == 0) else 0.0

        dist_matrix = np.abs(x[:, np.newaxis] - y[np.newaxis, :])
        score_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

        cost_matrix = -score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_score = score_matrix[row_ind, col_ind].sum()

        score = total_score / np.sqrt(len(x) * len(y))
        return score

    try:
        # Use weighted matching with appropriate thresholds
        h_score = set_match_score_weighted(pred_h, target_h, sigma=0.5)
        c_score = set_match_score_weighted(pred_c, target_c, sigma=2.0)

        # Combined score (weighted average)
        combined = 0.5 * h_score + 0.5 * c_score

        return h_score, c_score, combined

    except Exception as e:
        print(f"[NMRNet] Similarity calculation failed: {e}")
        return 0.0, 0.0, 0.0


# ============== Test Function ==============

if __name__ == "__main__":
    print("Testing NMRNet prediction...")

    # Test on toluene
    smiles = "Cc1ccccc1"
    result = predict_nmr_nmrnet(smiles)

    print(f"\nSMILES: {smiles}")
    print(f"Valid: {result['valid']}")

    if result['valid']:
        print(f"H-NMR peaks: {len(result['H_shifts'])}")
        print(f"C-NMR peaks: {len(result['C_shifts'])}")
        print(f"H-NMR: {result['H_shifts']}")
        print(f"C-NMR: {result['C_shifts']}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
