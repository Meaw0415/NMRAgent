# xyz2smiles

Standalone utility extracted from the `chefnmr` bond-reconstruction path.

Input:
- atom symbols
- 3D coordinates

Output:
- canonical SMILES from RDKit `DetermineBonds`

## Python

```python
from tools.xyz2smiles.core import xyz_to_smiles

atoms = ["O", "H", "H"]
coords = [
    [0.000, 0.000, 0.000],
    [0.757, 0.586, 0.000],
    [-0.757, 0.586, 0.000],
]

result = xyz_to_smiles(atoms, coords)
print(result.smiles)
```

## CLI with XYZ

```bash
python -m tools.xyz2smiles.cli --xyz-file water.xyz
```

## CLI with JSON

```bash
python -m tools.xyz2smiles.cli \
  --atoms O H H \
  --coords-json '[[0,0,0],[0.757,0.586,0],[-0.757,0.586,0]]'
```

## Notes

- This tool does not depend on the original `chefnmr` package.
- It relies on RDKit `MolFromMolBlock` + `rdDetermineBonds.DetermineBonds`.
- For problematic geometries, set `--timeout-seconds 2`.
