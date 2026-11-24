"""Verify explicit spectral hypotheses against existing NMRNet predictions."""
from collections import Counter
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.optimize import linear_sum_assignment

from .decorator import tool


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol)) if mol else None


def _classes(mol):
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    return Counter(ranks[a.GetIdx()] for a in mol.GetAtoms() if a.GetAtomicNum() == 6)


def _assign(query, predicted, allowed):
    if not query or not predicted:
        return [], list(query), list(predicted)
    cost = np.array([[abs(q - p['shift']) if allowed(q, p) else 1e6 for p in predicted] for q in query])
    rows, cols = linear_sum_assignment(cost)
    matched, used_q, used_p = [], set(), set()
    for i, j in zip(rows, cols):
        if cost[i, j] >= 1e6:
            continue
        used_q.add(int(i)); used_p.add(int(j))
        matched.append({'query_shift': query[i], 'pred_shift': predicted[j]['shift'],
                        'residual': float(cost[i, j]), 'parent_idx': predicted[j]['parent_idx'],
                        'heavy_idx': predicted[j]['heavy_idx']})
    return matched, [q for i, q in enumerate(query) if i not in used_q], [p for i, p in enumerate(predicted) if i not in used_p]


def verify_candidate(candidate, hypothesis):
    mol = _mol(candidate['smiles'])
    atoms = candidate.get('atom_data') or []
    if not atoms:
        return {'smiles': candidate['smiles'], 'error': 'Per-atom predictions unavailable'}
    h = [p for p in atoms if p['element'] == 'H']
    c = [p for p in atoms if p['element'] == 'C']
    exchangeable = [p for p in h if mol.GetAtomWithIdx(p['parent_idx']).GetAtomicNum() in {7, 8, 16}]
    if hypothesis.get('allow_missing_exchangeable_h'):
        h = [p for p in h if p not in exchangeable]
    carbonyl = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetAtomicNum() == 8 for b in a.GetBonds())}
    methoxy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetTotalNumHs() == 3 and any(n.GetAtomicNum() == 8 for n in a.GetNeighbors())}
    sp2 = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP2}
    close = lambda q, values: any(abs(q - float(v)) < 1e-6 for v in values)
    def allow_h(q, p):
        if close(q, hypothesis.get('methoxy_h_shifts', [])):
            return p['parent_idx'] in methoxy
        if close(q, hypothesis.get('sp2_h_shifts', [])):
            return p['parent_idx'] in sp2
        return True
    def allow_c(q, p):
        return not close(q, hypothesis.get('carbonyl_shifts', [])) or p['heavy_idx'] in carbonyl
    hm, hq, hp = _assign(hypothesis['h_shifts'], h, allow_h)
    cm, cq, cp = _assign(hypothesis['c_shifts'], c, allow_c)
    groups = _classes(mol)
    return {'smiles': candidate['smiles'], 'original_rank': candidate.get('rank'),
            'source': candidate.get('source'), 'carbon_symmetry_group_count': len(groups),
            'carbon_symmetry_multiplicities': sorted(groups.values()),
            'observed_unique_c_shift_count': len(set(hypothesis['c_shifts'])),
            'carbonyl_atom_indices': sorted(carbonyl), 'methoxy_parent_indices': sorted(methoxy),
            'h_matched': hm, 'c_matched': cm, 'unmatched_query_h': hq, 'unmatched_query_c': cq,
            'unused_predicted_h': hp,
            'unused_nonexchangeable_h': [p for p in hp if p not in exchangeable],
            'unused_exchangeable_h': [p for p in hp if p in exchangeable], 'unused_c': cp,
            'excluded_exchangeable_h_count': len(exchangeable) if hypothesis.get('allow_missing_exchangeable_h') else 0,
            'methoxy_h_coverage': {str(i): sum(p['parent_idx'] == i for p in hm) for i in sorted(methoxy)},
            'c_mean_absolute_residual': float(np.mean([p['residual'] for p in cm])) if cm else None,
            'c_max_absolute_residual': max((p['residual'] for p in cm), default=None)}


@tool(name='nmr_verify_hypothesis', description='Audit candidate exclusions and test a labeled integration/chemical-assignment hypothesis using per-atom NMRNet predictions.')
def nmr_verify_hypothesis(candidates, pool_candidates, formula, hypothesis):
    candidates = json.loads(candidates) if isinstance(candidates, str) else candidates
    pool = json.loads(pool_candidates) if isinstance(pool_candidates, str) else pool_candidates
    hypothesis = json.loads(hypothesis) if isinstance(hypothesis, str) else hypothesis
    valid, exclusions, seen = [], [], set()
    for row in pool:
        mol = _mol(row.get('smiles', ''))
        reason = None
        if mol is None:
            reason = 'invalid_smiles'
        else:
            canon = Chem.MolToSmiles(mol)
            actual_formula = rdMolDescriptors.CalcMolFormula(mol)
            if canon in seen:
                reason = 'duplicate'
            elif actual_formula != formula:
                reason = 'formula_mismatch:' + actual_formula
            seen.add(canon)
        if reason:
            exclusions.append({'smiles': row.get('smiles'), 'reason': reason})
        else:
            valid.append(row)
    covered = {Chem.MolToSmiles(_mol(row['smiles'])) for row in candidates}
    repeated_c = len(set(hypothesis['c_shifts'])) < len(hypothesis['c_shifts'])
    missing = [row for row in valid if repeated_c and len(_classes(_mol(row['smiles']))) == len(set(hypothesis['c_shifts']))
               and Chem.MolToSmiles(_mol(row['smiles'])) not in covered]
    if missing:
        from .nmr_models.nmrnet_predict import predict_nmr_batch
        predictions = predict_nmr_batch([Chem.MolToSmiles(_mol(r['smiles'])) for r in missing], fast=True, include_atom_data=True)
        candidates = list(candidates) + [{**row, 'atom_data': pred.get('atom_data', [])} for row, pred in zip(missing, predictions)]
    return {'label': hypothesis.get('label', 'hypothesis'), 'hypothesis': hypothesis,
            'pool_count': len(pool), 'formula_valid_unique_count': len(valid), 'exclusions': exclusions,
            'additional_symmetric_candidates_predicted': len(missing),
            'candidates': [verify_candidate(row, hypothesis) for row in candidates],
            'observation': 'Original shifts and predictions are retained. These are explicit integration and assignment hypotheses; residuals are reported without changing NMRNet predictions.'}
