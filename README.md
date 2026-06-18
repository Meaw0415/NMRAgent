# NMRAgent

NMRAgent is a standalone agent framework for NMR-based molecular structure elucidation. This repository keeps the reusable agent core and tool abstractions separate from local experiments, private checkpoints, and machine-specific infrastructure.

## Scope

This open-source repository includes:
- `agents/`: the LangGraph-based agent runtime
- `tools/`: retrieval, de novo, reranking, editing, KG, and solver-style tool modules
- `configs/`: environment-driven configuration
- `Service/`: lightweight retrieval service wrapper
- `scripts/run_nmr.py`: minimal single-sample entrypoint

This repository intentionally does not include large checkpoints, LMDB caches, or benchmark dumps, but it now includes `third_party/` dependencies required by the current rerank / UniMol-related code path.

## Repository Layout

```text
NMRAgent/
├── agents/
├── configs/
├── tools/
├── Service/
├── scripts/
└── artifacts/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Optional dependencies such as `torch`, `vllm`, `transformers`, or private model runtimes should be installed based on which tools you plan to use.

## Configuration

Runtime assets can be configured from a single file. The loader checks these files in order:
- `configs/runtime_assets.local.json`
- `configs/runtime_assets.json`
- `configs/runtime_assets.example.json`

`runtime_assets.local.json` is intended for your internal machine paths. `runtime_assets.example.json` is the publishable template for GitHub.

The loader exports environment variables before tools are initialized, so existing code can keep using `os.environ`.

Common variables:
- `NMR_CKPT_DIR`: retrieval / denovo / rerank checkpoint root
- `NMR_KG_DATA_DIR`: KG CSV and JSON directory
- `NMR_CHEMBERTA_PATH`: ChemBERTa tokenizer directory
- `NMR_OFFLINE_LMDB`: offline cache path
- `NMRBENCH_ROOT`: benchmark cache root
- `NMR_FRAGMENT_LIB_DIR`: fragment LMDB + index directory
- `OPENAI_API_KEY`: API key for `--backend openai`
- `OPENAI_BASE_URL`: optional OpenAI-compatible endpoint

By default, the publishable example points under `artifacts/`. For your current machine, you can keep `configs/runtime_assets.local.json` with absolute paths and avoid setting variables manually every time.

## Quick Start

```bash
python scripts/run_nmr.py   --formula C8H10N4O2   --h-shifts 3.4 3.6   --c-shifts 28.1 149.5   --backend openai   --model gpt-4o-mini
```

To start the retrieval service:

```bash
bash Service/start_retrieval_service.sh
```

## Open-Source Notes

Several tools still expect external checkpoints or data products that are not bundled here. The code is now repository-relative and environment-configurable, but you will still need to decide which artifacts to publish, document, or replace before a public GitHub release.

## Runtime Config Files

- `configs/runtime_assets.example.json`: repository-relative example for open source release
- `configs/runtime_assets.local.json`: local absolute-path config for your machine
- `third_party/`: copied dependency tree required by the current NMRNet / UniMol stack
