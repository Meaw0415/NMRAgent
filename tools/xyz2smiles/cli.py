import argparse
import json
from pathlib import Path

from .core import parse_xyz_text, xyz_to_smiles


def _load_from_json(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    atoms = payload["atoms"]
    coords = payload["coords"]
    return atoms, coords


def _load_from_xyz(path: str):
    xyz_text = Path(path).read_text(encoding="utf-8")
    return parse_xyz_text(xyz_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert atom symbols + 3D coordinates into SMILES.")
    parser.add_argument("--input-json", type=str, default="", help='JSON file with {"atoms": [...], "coords": [[x,y,z], ...]}')
    parser.add_argument("--xyz-file", type=str, default="", help="Path to XYZ file.")
    parser.add_argument("--atoms", nargs="*", default=None, help="Atom symbols, e.g. C H H H H")
    parser.add_argument("--coords-json", type=str, default="", help='JSON string like [[0,0,0],[0,0,1],...]')
    parser.add_argument("--remove-stereo", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_json:
        atoms, coords = _load_from_json(args.input_json)
    elif args.xyz_file:
        atoms, coords = _load_from_xyz(args.xyz_file)
    else:
        if not args.atoms or not args.coords_json:
            raise SystemExit("Provide either --input-json, --xyz-file, or both --atoms and --coords-json.")
        atoms = args.atoms
        coords = json.loads(args.coords_json)

    result = xyz_to_smiles(
        atoms=atoms,
        coords=coords,
        remove_stereo=args.remove_stereo,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "smiles": result.smiles,
                "largest_fragment_smiles": result.largest_fragment_smiles,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

