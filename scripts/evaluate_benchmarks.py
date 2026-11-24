#!/usr/bin/env python3
"""Evaluate predictions exported by run_benchmark.py without model/API calls."""
import argparse
import json
from pathlib import Path

from run_benchmark import load_records, score_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = score_records(load_records(args.predictions))
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
