# Case-study inputs

| Compound | Formula | Input | Execution trace |
| --- | --- | --- | --- |
| 1 | C20H30O3 | [input.json](compound_1/input.json) | [trace.log](compound_1/trace.log) |
| 2 | C20H14O8 | [input.json](compound_2/input.json) | [trace.log](compound_2/trace.log) |

Configure the environment, model/data assets, and LLM as described in the root [README](../../README.md). Run commands from the repository root.

Validate either input before inference:

```bash
python scripts/run_benchmark.py --benchmark case_study \
  --input examples/case_studies/compound_1/input.json \
  --validate-only --output-dir results/validate_compound_1
```

Run compound 1 or compound 2:

```bash
python scripts/run_benchmark.py --benchmark case_study \
  --input examples/case_studies/compound_1/input.json \
  --output-dir results/compound_1

python scripts/run_benchmark.py --benchmark case_study \
  --input examples/case_studies/compound_2/input.json \
  --output-dir results/compound_2
```

Both commands use the configured LLM; `--model` overrides it. `bash examples/run_demo.sh` is a shortcut for compound 1.

Inputs contain `formula`, `h_shifts`, `c_shifts`, and an `idx`. To evaluate structure accuracy, add an optional `reference_smiles` to your input before inference. The runner retains it for evaluation and passes only the formula and shifts to the agent.

Each run writes `predictions.jsonl`, `summary.json`, and per-case `trace.jsonl` / `state.json` under the selected output directory. Evaluate a labeled case with:

```bash
python scripts/evaluate_benchmarks.py \
  --predictions results/compound_1/predictions.jsonl \
  --output results/compound_1/evaluation.json
```

Reference-based metrics are `null` for an unlabeled case. See [metric definitions](../../README.md#6-outputs-and-evaluation) for exact-match normalization, structural similarity, and verifier candidate Top-k.
