# Inputs, inference, and evaluation

See the root [README](../README.md) for dependency installation, model/data paths, and LLM configuration.

| Input type | Examples | Runner |
| --- | --- | --- |
| NMRGym | [nmrgym.json](benchmark_inputs/nmrgym.json) | `scripts/infer_nmrgym.sh` |
| nmrshiftdb | [nmrshiftdb.json](benchmark_inputs/nmrshiftdb.json) | `scripts/infer_nmrshiftdb.sh` |
| Exp450 | [exp450.json](benchmark_inputs/exp450.json) | `scripts/infer_exp450.sh` |
| Case studies | [Compound 1 and 2](case_studies/README.md) | `scripts/run_benchmark.py --benchmark case_study` |

Run commands from the repository root.

## Input format

Inputs are JSON objects/arrays or JSONL. Each record supplies a molecular `formula` and at least one of `h_shifts` / `c_shifts`, in ppm. Use a unique `idx` per selected input. Repeated shifts are preserved. An optional `reference_smiles` is used only for evaluation.

```json
{
  "idx": "example_1",
  "formula": "C2H6O",
  "h_shifts": [1.2, 1.2, 1.2, 3.6, 3.6],
  "c_shifts": [18.0, 58.0],
  "reference_smiles": "CCO"
}
```

The loader also accepts NMRGym's `answer` and `extra_info` fields, flat `gt_smiles` / `smiles` references, and `h_shift` / `c_shift` aliases. The formula and shifts form the inference input; reference structures and dataset prompts are excluded from agent calls.

## Validate examples

Validation requires RDKit and does not call the API or load neural models:

```bash
for benchmark in nmrgym nmrshiftdb exp450; do
  bash "scripts/infer_${benchmark}.sh" \
    --input "examples/benchmark_inputs/${benchmark}.json" \
    --validate-only --output-dir "results/validate_${benchmark}"
done

bash examples/run_demo.sh --validate-only --output-dir results/validate_compound_1
```

## Run inference

After configuring the assets and LLM, run a single benchmark example:

```bash
bash scripts/infer_nmrgym.sh \
  --input examples/benchmark_inputs/nmrgym.json \
  --limit 1 --output-dir results/nmrgym_example
```

Use the other wrappers in the table for nmrshiftdb or Exp450. Full-data defaults are `artifacts/benchmarks/nmrgym_test.json`, `artifacts/benchmarks/nmrshiftdb_test.json`, and `artifacts/benchmarks/exp450.json`; pass `--input` to select another file. Remove `--limit` for a full run.

Run a case:

```bash
python scripts/run_benchmark.py --benchmark case_study \
  --input examples/case_studies/compound_2/input.json \
  --output-dir results/compound_2
```

The LLM model comes from the API configuration and can be overridden with `--model`. All runs save actual request/response and tool traces under their output directory. Use a new output directory for each attempt; `--resume` skips existing records, including errors.

## Evaluate predictions

```bash
python scripts/evaluate_benchmarks.py \
  --predictions results/nmrgym_example/predictions.jsonl \
  --output results/nmrgym_example/evaluation.json
```

The same command evaluates a case output by changing the prediction path. References must be supplied in the input to obtain reference-based metrics; the bundled case inputs contain only formula and shifts. Without references those metrics are `null`.

Final-answer accuracy uses canonical SMILES with stereochemistry removed. Structural similarity is Morgan–Tanimoto (radius 2, 2,048 bits, no chirality). Invalid, missing, failed, and unaccepted predictions count as failures for labeled samples. Verifier candidate Top-1/5/10 is reported as a separate diagnostic. See [output layout and metric definitions](../README.md#6-outputs-and-evaluation).
