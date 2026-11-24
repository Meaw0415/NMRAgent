import unittest

from rdkit import Chem
from tools.nmr_verification_tool import nmr_verify_hypothesis, verify_candidate


class VerificationTests(unittest.TestCase):
    def test_explicit_integration_preserves_individual_methyl_hydrogens(self):
        data = [{'element': 'H', 'shift': 3.3, 'heavy_idx': -1, 'parent_idx': 0} for _ in range(3)]
        data += [{'element': 'H', 'shift': 5.5, 'heavy_idx': -1, 'parent_idx': 1},
                 {'element': 'C', 'shift': 50., 'heavy_idx': 0, 'parent_idx': 0}]
        result = verify_candidate({'smiles': 'CO', 'atom_data': data},
            {'h_shifts': [3.3] * 3, 'c_shifts': [50.], 'methoxy_h_shifts': [3.3], 'allow_missing_exchangeable_h': True})
        self.assertEqual(len(result['h_matched']), 3)
        self.assertEqual(result['methoxy_h_coverage'], {'0': 3})
        self.assertEqual(result['excluded_exchangeable_h_count'], 1)
        self.assertEqual(result['unused_nonexchangeable_h'], [])
        baseline = verify_candidate({'smiles': 'CO', 'atom_data': data},
            {'h_shifts': [3.3] * 3, 'c_shifts': [50.]})
        self.assertEqual(baseline['unused_nonexchangeable_h'], [])
        self.assertEqual(len(baseline['unused_exchangeable_h']), 1)

    def test_carbonyl_constraint_does_not_force_assignment_to_other_carbons(self):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles('COC(C)=O'))
        mol = Chem.MolFromSmiles(smiles)
        data = [{'element': 'C', 'shift': 164., 'heavy_idx': a.GetIdx(), 'parent_idx': a.GetIdx()}
                for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        result = verify_candidate({'smiles': smiles, 'atom_data': data},
                                  {'h_shifts': [], 'c_shifts': [164., 164.], 'carbonyl_shifts': [164.]})
        self.assertEqual(len(result['c_matched']), 1)
        self.assertEqual(result['unmatched_query_c'], [164.])

    def test_formula_and_duplicate_exclusions_are_explicit(self):
        pool = [{'smiles': 'CO'}, {'smiles': 'C'}, {'smiles': 'CO'}]
        data = [{'element': 'C', 'shift': 50., 'heavy_idx': 0, 'parent_idx': 0}]
        result = nmr_verify_hypothesis([{'smiles': 'CO', 'atom_data': data}], pool, 'CH4O',
                                      {'h_shifts': [], 'c_shifts': [50.]})
        self.assertEqual(result['formula_valid_unique_count'], 1)
        self.assertEqual([x['reason'] for x in result['exclusions']], ['formula_mismatch:CH4', 'duplicate'])


if __name__ == '__main__':
    unittest.main()
