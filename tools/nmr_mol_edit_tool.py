"""
NMR-guided molecular editing tools.

nmr_mol_edit       — manual add/remove/replace of functional groups
nmr_fragment_edit  — NMR-guided fragment swap using dynamic pool from
                     rerank candidates (cut logic from NMR-Solver)
nmr_refine         — iterative rerank + fragment_edit loop (n rounds)
"""
import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from .decorator import tool
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors
rdBase.DisableLog('rdApp.error')


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _interpret_c_shift(ppm: float) -> str:
    """Interpret C-NMR shift to suggest functional group."""
    if 200 <= ppm <= 220:
        return "ketone/aldehyde C=O"
    elif 160 <= ppm < 200:
        return "carboxylic acid/ester/amide C=O or aromatic C"
    elif 120 <= ppm < 160:
        return "aromatic C or C=C"
    elif 90 <= ppm < 120:
        return "acetal/ketal C"
    elif 50 <= ppm < 90:
        return "C-O (ether/alcohol) or C-N"
    elif 20 <= ppm < 50:
        return "aliphatic C-C"
    elif 0 <= ppm < 20:
        return "methyl C"
    else:
        return "unusual shift"

def _formula_diff(formula1: str, formula2: str) -> dict:
    """Calculate atom difference between two formulas."""
    from collections import Counter
    import re
    def parse(f):
        return Counter({m[0]: int(m[1] or 1) for m in re.findall(r'([A-Z][a-z]?)(\d*)', f)})
    c1, c2 = parse(formula1), parse(formula2)
    diff = {}
    for elem in set(c1.keys()) | set(c2.keys()):
        d = c2.get(elem, 0) - c1.get(elem, 0)
        if d != 0:
            diff[elem] = d
    return diff

def _sanitize(mol, desc):
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return {"observation": f"{desc} → {smi}", "smiles": smi, "valid": 1}
    except Exception as e:
        return {"observation": f"Invalid result after edit: {e}", "valid": 0}

def _predict_nmr_score(smiles: str, query_c: List[float], query_h: List[float],
                       nmr_predictor) -> float:
    """Predict NMR and calculate match score with query."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 0.0

        # Predict NMR (use nmr_predictor from rerank tool)
        pred_c, pred_h = nmr_predictor.predict(smiles)

        # Calculate set similarity score
        from .nmr_rerank_tool import _set_match_score
        c_score = _set_match_score(query_c, pred_c, sigma=10.0) if query_c and pred_c else 0
        h_score = _set_match_score(query_h, pred_h, sigma=1.0) if query_h and pred_h else 0

        return (c_score + h_score) / 2
    except:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# NMR-Solver fragment cut/stitch logic (adapted from src/core/operation.py)
# ──────────────────────────────────────────────────────────────────────────────

def _get_complement_cut_type(cut_type):
    if len(cut_type[1]) == 1:
        return ((cut_type[0][1], cut_type[0][0]), cut_type[1])
    else:
        return ((cut_type[0][1], cut_type[0][0],
                 cut_type[0][3], cut_type[0][2]), cut_type[1])


def _cut_non_ring(mol) -> List[Tuple]:
    """Cut all non-ring bonds; return (fragment_mol, cut_type) pairs."""
    fragments_mol = []
    bond_map = {Chem.BondType.SINGLE: '-',
                Chem.BondType.DOUBLE: '=',
                Chem.BondType.TRIPLE: '#'}

    patt = Chem.MolFromSmarts('[*]~;!@[*]')
    for bis in mol.GetSubstructMatches(patt):
        bond = mol.GetBondBetweenAtoms(bis[0], bis[1])
        bs = [bond.GetIdx()]
        frags_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True,
                                          dummyLabels=[(1, 1)])
        atom_sym = [mol.GetAtomWithIdx(bis[i]).GetSymbol() for i in range(2)]
        btype = bond_map.get(bond.GetBondType(), '-')
        frags = list(Chem.GetMolFrags(frags_mol, asMols=True,
                                       sanitizeFrags=False))
        if len(frags) != 2:
            continue
        if (all(f.GetNumAtoms() > 2 for f in frags) or
                all(s not in ('H', 'F', 'Cl', 'Br', 'I') for s in atom_sym)):
            ct = ((atom_sym[0], atom_sym[1]), (btype,))
            fragments_mol += [(frags[0], ct),
                               (frags[1], _get_complement_cut_type(ct))]
        elif (frags[1].GetNumAtoms() == 2 and btype == '-'
              and atom_sym[1] in ('H', 'F', 'Cl', 'Br', 'I')):
            ct = ((atom_sym[0], 'X'), (btype,))
            fragments_mol += [(frags[0], ct)]
        elif (frags[0].GetNumAtoms() == 2 and btype == '-'
              and atom_sym[0] in ('H', 'F', 'Cl', 'Br', 'I')):
            ct = ((atom_sym[1], 'X'), (btype,))
            fragments_mol += [(frags[1], ct)]
    return fragments_mol


def _cut_ring(mol) -> List[Tuple]:
    """Cut pairs of non-aromatic ring bonds; return (fragment_mol, cut_type) pairs."""
    fragments_mol = []
    bond_map = {Chem.BondType.SINGLE: '-',
                Chem.BondType.DOUBLE: '=',
                Chem.BondType.TRIPLE: '#'}
    structure_bond = {
        '[R]!:@[R]!:@[R]!:@[R]': (0, 1, 2, 3),
        '[R]!:@[R]!:@[R]':        (0, 1, 1, 2),
    }
    for smarts, bond in structure_bond.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for bis in mol.GetSubstructMatches(patt):
            pairs = ((bis[bond[0]], bis[bond[1]]),
                     (bis[bond[2]], bis[bond[3]]))
            bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in pairs]
            frags_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True,
                                              dummyLabels=[(1, 1), (2, 2)])
            atom_sym = [mol.GetAtomWithIdx(pairs[i][j]).GetSymbol()
                        for i in range(2) for j in range(2)]
            btype = [bond_map.get(
                         mol.GetBondBetweenAtoms(x, y).GetBondType(), '-')
                     for x, y in pairs]
            try:
                frags = Chem.GetMolFrags(frags_mol, asMols=True,
                                          sanitizeFrags=False)
                if len(frags) != 2:
                    continue
                ct = ((atom_sym[0], atom_sym[1],
                       atom_sym[2], atom_sym[3]),
                      (btype[0], btype[1]))
                fragments_mol += [(frags[0], ct),
                                   (frags[1], _get_complement_cut_type(ct))]
            except ValueError:
                continue
    return fragments_mol


def _stitch(frag1, frag2, bond_type):
    """Reconnect two fragments using SMARTS reactions (NMR-Solver style)."""
    bt = bond_type if isinstance(bond_type, (list, tuple)) else [bond_type]
    try:
        rxn1 = AllChem.ReactionFromSmarts(
            f'[*:1]~[1*].[1*]~[*:2]>>[*:1]{bt[0]}[*:2]')
        prod = rxn1.RunReactants((frag1, frag2))
        if not prod:
            return None
        mol = prod[0][0]
        if len(bt) == 2:
            mol.UpdatePropertyCache(strict=False)
            rxn2 = AllChem.ReactionFromSmarts(
                f'([*:1]~[2*].[2*]~[*:2])>>[*:1]{bt[1]}[*:2]')
            prod2 = rxn2.RunReactants((mol,))
            if not prod2:
                return None
            mol = prod2[0][0]
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Fragment Pool
# ──────────────────────────────────────────────────────────────────────────────

def _build_fragment_pool(candidates: List[Dict]) -> List[Dict]:
    """
    Cut every candidate into fragments and annotate each fragment with:
      - frag_smiles:  canonical SMILES of fragment (with dummy atoms)
      - cut_type:     reconnection key
      - c_shifts:     predicted C shifts of C atoms inside this fragment
      - source_smiles: which candidate this came from
      - heavy_idxs:  set of heavy atom indices (in source mol) in this fragment
    """
    pool = []
    for cand in candidates:
        smi = cand.get('smiles', '')
        atom_data = cand.get('atom_data', [])
        if not smi or not atom_data:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Build heavy_idx → predicted_C_shift map from atom_data
        c_shift_map = {a['heavy_idx']: a['shift']
                       for a in atom_data
                       if a['element'] == 'C' and a['heavy_idx'] >= 0}

        all_cuts = _cut_non_ring(mol) + _cut_ring(mol)
        seen_frag_smiles = set()
        for frag_mol, cut_type in all_cuts:
            try:
                # Identify which original heavy atoms are in this fragment.
                # Dummy atoms ([1*]/[2*]) have atomic number 0.
                frag_heavy_idxs = set()
                for atom in frag_mol.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        continue  # dummy
                    # Map atom index in frag_mol back to original mol.
                    # FragmentOnBonds preserves original atom indices when
                    # addDummies=True (dummies are appended at the end).
                    orig_idx = atom.GetAtomMapNum()
                    if orig_idx > 0:
                        frag_heavy_idxs.add(orig_idx - 1)  # 1-indexed → 0-indexed
                    else:
                        # Fallback: use atom index directly (valid when mol
                        # hasn't been reordered — holds for FragmentOnBonds)
                        idx = atom.GetIdx()
                        if idx < mol.GetNumAtoms():
                            frag_heavy_idxs.add(idx)

                frag_c_shifts = [c_shift_map[i]
                                 for i in frag_heavy_idxs
                                 if i in c_shift_map]

                frag_smi = Chem.MolToSmiles(frag_mol)
                if frag_smi in seen_frag_smiles:
                    continue
                seen_frag_smiles.add(frag_smi)

                pool.append({
                    'frag_mol':      frag_mol,
                    'frag_smiles':   frag_smi,
                    'cut_type':      cut_type,
                    'c_shifts':      sorted(frag_c_shifts),
                    'source_smiles': smi,
                    'heavy_idxs':    frag_heavy_idxs,
                })
            except Exception:
                continue
    return pool


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@tool(
    name="nmr_mol_edit",
    description=(
        "Edit a molecule by adding, removing, or replacing a functional group/substructure. "
        "Use after rerank to fix unmatched NMR peaks. "
        "action='add': attach group_smiles to the molecule (optionally at position_smarts). "
        "action='remove': remove substructure matching group_smarts. "
        "action='replace': replace OLD_SMARTS>>NEW_SMILES (e.g. '[Cl]>>F')."
    ),
)
def nmr_mol_edit(smiles: str, action: str, group: str, position: str = "") -> dict:
    """
    Edit molecular structure.

    Args:
        smiles:   Input SMILES string
        action:   'add', 'remove', or 'replace'
        group:    For 'add': SMILES of group to attach.
                  For 'remove': SMARTS pattern to delete.
                  For 'replace': 'OLD_SMARTS>>NEW_SMILES' (e.g. '[Cl]>>F').
        position: Optional SMARTS specifying attachment atom (for 'add' only).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"observation": f"Invalid input SMILES: {smiles}", "valid": 0}

    if action == "add":
        grp_mol = Chem.MolFromSmiles(group)
        if grp_mol is None:
            return {"observation": f"Invalid group SMILES: {group}", "valid": 0}
        target_idx = None
        if position:
            patt = Chem.MolFromSmarts(position)
            if patt and mol.HasSubstructMatch(patt):
                target_idx = mol.GetSubstructMatch(patt)[0]
        if target_idx is None:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0:
                    target_idx = atom.GetIdx(); break
        if target_idx is None:
            return {"observation": "No suitable attachment point found.", "valid": 0}
        combined = Chem.CombineMols(mol, grp_mol)
        em = Chem.EditableMol(combined)
        em.AddBond(target_idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
        return _sanitize(em.GetMol(), f"Added {group}")

    elif action == "remove":
        patt = Chem.MolFromSmarts(group)
        if patt is None:
            return {"observation": f"Invalid SMARTS: {group}", "valid": 0}
        if not mol.HasSubstructMatch(patt):
            return {"observation": f"Pattern '{group}' not found.", "valid": 0}
        match = mol.GetSubstructMatch(patt)
        em = Chem.EditableMol(mol)
        for idx in sorted(match, reverse=True):
            em.RemoveAtom(int(idx))
        return _sanitize(em.GetMol(), f"Removed {group}")

    elif action == "replace":
        if ">>" not in group:
            return {"observation": "Use format 'OLD_SMARTS>>NEW_SMILES'", "valid": 0}
        old_smarts, new_smi = group.split(">>", 1)
        patt = Chem.MolFromSmarts(old_smarts.strip())
        repl = Chem.MolFromSmiles(new_smi.strip())
        if patt is None or repl is None:
            return {"observation": "Invalid SMARTS or replacement SMILES.", "valid": 0}
        if not mol.HasSubstructMatch(patt):
            return {"observation": f"Pattern '{old_smarts}' not found.", "valid": 0}
        res = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=False)
        if not res:
            return {"observation": "Replacement failed.", "valid": 0}
        return _sanitize(res[0], f"Replaced {old_smarts} → {new_smi}")

    else:
        return {"observation": f"Unknown action '{action}'.", "valid": 0}


@tool(
    name="nmr_fragment_edit",
    description=(
        "NMR-guided fragment swap using a dynamic pool built from rerank candidates. "
        "Cuts each candidate with NMR-Solver bond-breaking rules, annotates fragments "
        "with per-atom NMR predictions, then swaps poorly-matched fragments in the top "
        "candidate with better-matching fragments from the pool. "
        "Use this after nmr_rerank when the top candidate has unmatched C peaks."
    ),
)
def nmr_fragment_edit(
    candidates: str,
    c_shifts: str,
    h_shifts: str = "",
    top_k: int = 5,
    c_tol: float = 10.0,
    max_new: int = 10,
) -> dict:
    """
    Fragment-level NMR-guided editing.

    Args:
        candidates: JSON list of reranked candidates (output from nmr_rerank).
                    Each entry must have 'smiles', 'atom_data', 'c_assignment',
                    'c_unmatched' fields (populated automatically by nmr_rerank).
        c_shifts:   Comma-separated query C-NMR shifts (ppm).
        h_shifts:   Comma-separated query H-NMR shifts (ppm, optional).
        top_k:      Number of top candidates to use for fragment pool (default 5).
        c_tol:      C-NMR tolerance for fragment matching in ppm (default 10.0).
        max_new:    Maximum number of new candidate SMILES to return (default 10).

    Returns:
        dict with 'observation' and 'new_candidates' list of new SMILES to rerank.
    """
    try:
        # ── Parse inputs ──────────────────────────────────────────────────────
        if isinstance(candidates, str):
            try:
                cand_list = json.loads(candidates)
            except Exception:
                import ast
                cand_list = ast.literal_eval(candidates)
        else:
            cand_list = candidates

        q_c = np.array([float(x.strip()) for x in c_shifts.split(',') if x.strip()])
        q_h = (np.array([float(x.strip()) for x in h_shifts.split(',') if x.strip()])
               if h_shifts.strip() else np.array([]))

        if not cand_list:
            return {"observation": "No candidates provided.", "new_candidates": []}

        # Sort by nmr_similarity, take top_k for pool
        cand_list = sorted(cand_list,
                           key=lambda x: x.get('nmr_similarity', 0), reverse=True)
        top_cand   = cand_list[0]
        pool_cands = cand_list[:top_k]

        # ── Identify unmatched / poorly-matched C peaks in top candidate ──────
        c_assignment = top_cand.get('c_assignment', [])
        c_unmatched  = list(top_cand.get('c_unmatched', []))

        # Also flag matched pairs with large residual (> c_tol)
        bad_peaks = list(c_unmatched)
        for entry in c_assignment:
            q_ppm, p_ppm, residual = entry[0], entry[1], entry[2]
            if residual > c_tol:
                bad_peaks.append(q_ppm)

        if not bad_peaks:
            return {
                "observation": (
                    f"Top candidate '{top_cand['smiles']}' has no poorly-matched "
                    f"C peaks (tol={c_tol} ppm). No fragment edits needed."
                ),
                "new_candidates": [],
            }

        # ── Build fragment pool from top-k candidates ─────────────────────────
        pool = _build_fragment_pool(pool_cands)
        if not pool:
            return {
                "observation": "Fragment pool is empty (atom_data missing from candidates). "
                               "Ensure nmr_rerank was called with include_atom_data=True.",
                "new_candidates": [],
            }

        # ── Cut top candidate into fragments ──────────────────────────────────
        top_mol = Chem.MolFromSmiles(top_cand['smiles'])
        if top_mol is None:
            return {"observation": "Invalid top candidate SMILES.", "new_candidates": []}

        top_cuts = _cut_non_ring(top_mol) + _cut_ring(top_mol)

        # ── For each cut of the top mol, find its C shifts from atom_data ─────
        top_atom_data = top_cand.get('atom_data', [])
        top_c_map = {a['heavy_idx']: a['shift']
                     for a in top_atom_data
                     if a['element'] == 'C' and a['heavy_idx'] >= 0}

        def _frag_c_shifts(frag_mol):
            shifts = []
            for atom in frag_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    continue
                idx = atom.GetIdx()
                if idx < top_mol.GetNumAtoms() and idx in top_c_map:
                    shifts.append(top_c_map[idx])
            return shifts

        # ── Try swapping each top-mol fragment with pool fragments ─────────────
        # Focus on fragments whose C shifts cover a bad_peak.
        new_smiles_set = set()
        obs_lines = [
            f"Fragment edit: top candidate = {top_cand['smiles']}",
            f"Bad C peaks (residual > {c_tol} or unmatched): {[round(p,1) for p in bad_peaks]}",
        ]

        # Add chemical shift interpretation for unmatched peaks
        if c_unmatched:
            obs_lines.append("Unmatched C peaks analysis:")
            for ppm in c_unmatched[:5]:  # Show top 5
                obs_lines.append(f"  {ppm:.1f} ppm → likely {_interpret_c_shift(ppm)}")

        obs_lines.append(f"Fragment pool size: {len(pool)}")
        obs_lines.append("")

        for top_frag, cut_type in top_cuts:
            frag_c = _frag_c_shifts(top_frag)

            # Check if this fragment "owns" any of the bad peaks
            # (within c_tol of a bad_peak)
            covers_bad = any(
                any(abs(s - bp) < c_tol for s in frag_c)
                for bp in bad_peaks
            )
            if not frag_c or not covers_bad:
                continue

            # Find pool fragments with compatible cut_type and better shift coverage
            complement_ct = _get_complement_cut_type(cut_type)
            for pool_entry in pool:
                if pool_entry['cut_type'] != complement_ct:
                    continue
                if pool_entry['source_smiles'] == top_cand['smiles']:
                    continue  # don't swap with itself

                # Check if pool fragment covers the bad peaks better
                pool_c = pool_entry['c_shifts']
                improvement = sum(
                    1 for bp in bad_peaks
                    if any(abs(s - bp) < c_tol for s in pool_c)
                )
                if improvement == 0:
                    continue

                # Stitch
                new_mol = _stitch(top_frag, pool_entry['frag_mol'],
                                   cut_type[1])
                if new_mol is None:
                    continue
                try:
                    Chem.SanitizeMol(new_mol)
                    new_smi = Chem.MolToSmiles(new_mol)
                    if new_smi not in new_smiles_set:
                        new_smiles_set.add(new_smi)
                        obs_lines.append(
                            f"  Swap: {Chem.MolToSmiles(top_frag)} "
                            f"(C:{[round(s,1) for s in frag_c]}) "
                            f"→ {pool_entry['frag_smiles']} "
                            f"(C:{[round(s,1) for s in pool_c]}) "
                            f"covers {improvement} bad peaks → {new_smi}"
                        )
                except Exception:
                    continue

            if len(new_smiles_set) >= max_new:
                break

        new_candidates = list(new_smiles_set)[:max_new]

        if not new_candidates:
            obs_lines.append("No compatible fragment swaps found. "
                             "Try nmr_mol_edit for manual edits.")
        else:
            obs_lines.append(f"\n{len(new_candidates)} new candidates generated. "
                             "Pass them to nmr_rerank for scoring.")

        return {
            "observation": "\n".join(obs_lines),
            "new_candidates": new_candidates,
        }

    except Exception as e:
        import traceback
        return {
            "observation": f"Error: {e}\n{traceback.format_exc()}",
            "new_candidates": [],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Iterative Refine Tool  (rerank → fragment_edit) × n
# ──────────────────────────────────────────────────────────────────────────────

def _rerank_pool(smiles_pool: List[str],
                 h_list: List[float], c_list: List[float],
                 formula: str,
                 backend, c_sigma: float = 10.0, h_sigma: float = 1.0,
                 fast: bool = True) -> List[Dict]:
    """
    Internal rerank: predict NMR for each SMILES in pool, score, sort.
    Returns list of candidate dicts (same schema as nmr_rerank output).
    """
    from scipy.optimize import linear_sum_assignment

    def _set_sim(x, y, sigma):
        if len(x) == 0 or len(y) == 0:
            return 0.0
        m = np.exp(-(x[:, None] - y[None, :]) ** 2 / (2 * sigma ** 2))
        r, c = linear_sum_assignment(-m)
        return float(m[r, c].sum() / np.sqrt(len(x) * len(y)))

    def _peak_assign_local(query, pred, atom_heavy_idxs=None):
        if len(query) == 0 or len(pred) == 0:
            return [], [float(q) for q in query]
        residual_matrix = np.abs(query[:, None] - pred[None, :])
        r, c = linear_sum_assignment(residual_matrix)
        assigned = set(r)
        if atom_heavy_idxs is not None:
            matched = [(float(query[ri]), float(pred[ci]),
                        float(residual_matrix[ri, ci]),
                        int(atom_heavy_idxs[ci]))
                       for ri, ci in zip(r, c)]
        else:
            matched = [(float(query[ri]), float(pred[ci]),
                        float(residual_matrix[ri, ci]))
                       for ri, ci in zip(r, c)]
        unmatched = [float(query[i]) for i in range(len(query)) if i not in assigned]
        return matched, unmatched

    # Filter by formula if needed
    if formula:
        filtered = []
        for smi in smiles_pool:
            mol = Chem.MolFromSmiles(smi)
            if mol and rdMolDescriptors.CalcMolFormula(mol) == formula.strip():
                filtered.append(smi)
        smiles_pool = filtered
    if not smiles_pool:
        return []

    pred_results = backend.predict_nmr_batch(smiles_pool, fast=fast,
                                              include_atom_data=True)
    q_h = np.array(h_list, dtype=float)
    q_c = np.array(c_list, dtype=float)
    reranked = []
    for smi, pred in zip(smiles_pool, pred_results):
        if not pred.get('valid', False):
            continue
        pred_h = np.array(pred['H_shifts'], dtype=float)
        pred_c = np.array(pred['C_shifts'], dtype=float)
        h_sim = _set_sim(q_h, pred_h, h_sigma) if len(q_h) and len(pred_h) else 0.0
        c_sim = _set_sim(q_c, pred_c, c_sigma) if len(q_c) and len(pred_c) else 0.0
        nmr_sim = (h_sim + c_sim) / 2.0 if (len(q_h) > 0 and len(q_c) > 0) \
                  else (h_sim if len(q_h) > 0 else c_sim)

        atom_data = pred.get('atom_data', [])
        if atom_data:
            h_atoms = [(a['shift'], a['parent_idx']) for a in atom_data if a['element'] == 'H']
            c_atoms = [(a['shift'], a['heavy_idx'])  for a in atom_data if a['element'] == 'C']
            pa_h = np.array([s for s, _ in h_atoms], dtype=float)
            pa_c = np.array([s for s, _ in c_atoms], dtype=float)
            idx_h = [i for _, i in h_atoms]
            idx_c = [i for _, i in c_atoms]
            h_matched, h_unmatched = _peak_assign_local(q_h, pa_h, idx_h)
            c_matched, c_unmatched = _peak_assign_local(q_c, pa_c, idx_c)
        else:
            h_matched, h_unmatched = _peak_assign_local(q_h, pred_h)
            c_matched, c_unmatched = _peak_assign_local(q_c, pred_c)

        reranked.append({
            'smiles':         smi,
            'nmr_similarity': nmr_sim,
            'original_score': 0.0,
            'source':         'pool',
            'original_rank':  0,
            'h_assignment':   h_matched,
            'h_unmatched':    h_unmatched,
            'c_assignment':   c_matched,
            'c_unmatched':    c_unmatched,
            'atom_data':      atom_data,
        })

    reranked.sort(key=lambda x: x['nmr_similarity'], reverse=True)
    for i, cand in enumerate(reranked, 1):
        cand['rank'] = i
    return reranked


@tool(
    name="nmr_refine",
    description=(
        "Iteratively refine candidate structures via (rerank → NMR-guided fragment edit) × n rounds. "
        "Each round: (1) rerank all candidates by NMR similarity with per-atom assignment, "
        "(2) fragment-edit the top candidates to generate new structurally diverse alternatives, "
        "(3) merge new candidates into the pool (deduplicated). "
        "Returns the final reranked top-k with full peak assignment. "
        "Use this ONCE after combining nmr_retrieve + nmr_denovo candidates."
    ),
)
def nmr_refine(
    candidates: str,
    h_shifts: str,
    c_shifts: str,
    n_iter: int = 2,
    top_k: int = 10,
    c_tol: float = 10.0,
    max_new: int = 10,
    formula: str = "",
) -> dict:
    """
    Iterative NMR-guided refinement: (rerank → fragment_edit) × n_iter.

    Args:
        candidates: JSON list of candidate dicts or SMILES strings.
        h_shifts:   Comma-separated H-NMR shifts (ppm).
        c_shifts:   Comma-separated C-NMR shifts (ppm).
        n_iter:     Number of refinement rounds (default 2).
        top_k:      Top-k candidates to return in final result (default 10).
        c_tol:      C-NMR tolerance for fragment matching (ppm, default 10.0).
        max_new:    Max new candidates from fragment edit per round (default 10).
        formula:    Optional molecular formula filter (e.g. "C15H16O3").

    Returns:
        dict with 'observation', 'candidates' (reranked list), 'count'.
    """
    import traceback as _tb
    try:
        # ── Load NMRNet ───────────────────────────────────────────────────────
        from .nmr_rerank_tool import _load_nmrnet_backend
        backend = _load_nmrnet_backend()
        if backend is None:
            return {"observation": "Error: NMRNet backend unavailable.",
                    "candidates": [], "count": 0}

        # ── Parse inputs ──────────────────────────────────────────────────────
        if isinstance(candidates, str):
            try:
                cand_list = json.loads(candidates)
            except Exception:
                import ast
                cand_list = ast.literal_eval(candidates)
        else:
            cand_list = list(candidates)

        h_list = [float(x.strip()) for x in h_shifts.split(',') if x.strip()]
        c_list = [float(x.strip()) for x in c_shifts.split(',') if x.strip()]

        if not cand_list:
            return {"observation": "No candidates provided.", "candidates": [], "count": 0}

        # Normalise to list-of-dicts and canonicalise SMILES
        if cand_list and isinstance(cand_list[0], str):
            cand_list = [{"smiles": s} for s in cand_list]

        seen: set = set()
        pool: List[str] = []
        for c in cand_list:
            smi = c.get('smiles', '') if isinstance(c, dict) else str(c)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen.add(canon)
                pool.append(canon)

        obs_lines = [
            f"nmr_refine: {len(pool)} initial candidates, {n_iter} rounds",
            f"  H peaks: {len(h_list)},  C peaks: {len(c_list)}",
            "",
        ]

        current_reranked: List[Dict] = []

        # ── Main refinement loop ──────────────────────────────────────────────
        for round_idx in range(1, n_iter + 1):
            obs_lines.append(f"── Round {round_idx}/{n_iter} (pool size: {len(pool)}) ──")

            # Step A: rerank the whole pool
            current_reranked = _rerank_pool(
                pool, h_list, c_list, formula, backend,
                c_sigma=10.0, h_sigma=1.0, fast=True)

            if not current_reranked:
                obs_lines.append("  Rerank returned no results. Stopping.")
                break

            top1 = current_reranked[0]
            obs_lines.append(
                f"  Top-1: {top1['smiles']}  (NMR sim={top1['nmr_similarity']:.4f})")

            # Step B: fragment edit top-k candidates
            top_for_edit = current_reranked[:max(top_k, 5)]
            bad_c = [e[0] for e in top1.get('c_assignment', []) if e[2] > c_tol]
            bad_c += list(top1.get('c_unmatched', []))

            if not bad_c:
                obs_lines.append("  No poorly-matched C peaks — skipping fragment edit.")
            else:
                obs_lines.append(
                    f"  Bad C peaks: {[round(p, 1) for p in bad_c]}")
                frag_pool = _build_fragment_pool(top_for_edit)
                top_mol = Chem.MolFromSmiles(top1['smiles'])
                top_cuts = (_cut_non_ring(top_mol) + _cut_ring(top_mol)
                            if top_mol is not None else [])
                top_c_map = {a['heavy_idx']: a['shift']
                             for a in top1.get('atom_data', [])
                             if a['element'] == 'C' and a['heavy_idx'] >= 0}

                new_smiles_set: set = set()
                for top_frag, cut_type in top_cuts:
                    frag_c = [top_c_map[atom.GetIdx()]
                              for atom in top_frag.GetAtoms()
                              if atom.GetAtomicNum() != 0
                              and atom.GetIdx() < (top_mol.GetNumAtoms() if top_mol else 0)
                              and atom.GetIdx() in top_c_map]
                    covers_bad = any(
                        any(abs(s - bp) < c_tol for s in frag_c)
                        for bp in bad_c)
                    if not frag_c or not covers_bad:
                        continue
                    complement_ct = _get_complement_cut_type(cut_type)
                    for pe in frag_pool:
                        if pe['cut_type'] != complement_ct:
                            continue
                        if pe['source_smiles'] == top1['smiles']:
                            continue
                        improvement = sum(
                            1 for bp in bad_c
                            if any(abs(s - bp) < c_tol for s in pe['c_shifts']))
                        if improvement == 0:
                            continue
                        new_mol = _stitch(top_frag, pe['frag_mol'], cut_type[1])
                        if new_mol is None:
                            continue
                        try:
                            Chem.SanitizeMol(new_mol)
                            ns = Chem.MolToSmiles(new_mol)
                            if ns not in seen:
                                new_smiles_set.add(ns)
                        except Exception:
                            continue
                        if len(new_smiles_set) >= max_new:
                            break
                    if len(new_smiles_set) >= max_new:
                        break

                new_smiles = list(new_smiles_set)[:max_new]
                obs_lines.append(f"  Fragment edit generated {len(new_smiles)} new candidates.")
                for ns in new_smiles:
                    seen.add(ns)
                    pool.append(ns)

        # ── Final rerank ──────────────────────────────────────────────────────
        if not current_reranked:
            current_reranked = _rerank_pool(
                pool, h_list, c_list, formula, backend,
                c_sigma=10.0, h_sigma=1.0, fast=True)

        final_top = current_reranked[:top_k]

        obs_lines += [
            "",
            f"Final top-{len(final_top)} candidates (pool={len(pool)}):",
        ]
        for i, cand in enumerate(final_top[:5], 1):
            obs_lines.append(
                f"  {i}. {cand['smiles']}  (NMR sim={cand['nmr_similarity']:.4f})")
            c_parts = [f"{e[0]:.1f}->{e[1]:.1f}(Δ{e[2]:.1f})"
                       for e in cand['c_assignment'][:6]]
            if cand['c_unmatched']:
                c_parts.append(f"unmatched:{[round(x,1) for x in cand['c_unmatched']]}")
            if c_parts:
                obs_lines.append(f"     C: {', '.join(c_parts)}")
        if len(final_top) > 5:
            obs_lines.append(f"  ... and {len(final_top) - 5} more")

        return {
            "observation": "\n".join(obs_lines),
            "candidates":  final_top,
            "count":       len(final_top),
        }

    except Exception as e:
        return {
            "observation": f"Error in nmr_refine: {e}\n{_tb.format_exc()}",
            "candidates":  [],
            "count":       0,
        }
