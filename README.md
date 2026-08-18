# NMRAgent

Official repository for **Towards Generalizable and Evidential Nuclear Magnetic Resonance-Based Molecular Structure Elucidation via Large Language Model Agent**.

**Paper:** [arXiv:2606.29776](https://arxiv.org/abs/2606.29776)

![NMRAgent overview](docs/assets/NMRAgent_BC1.png)

[PDF version of the overview figure](docs/assets/NMRAgent_BC1.pdf)

NMRAgent is a LangGraph-based agent framework for NMR-driven molecular structure elucidation. The repository contains the agent runtime, tool wrappers, service entrypoints, and environment templates. Large model/data artifacts and machine-local configs are not part of the public repository.

## Table of Contents

- [1. Install Environment](#1-install-environment)
- [2. Configure Data and Model Paths](#2-configure-data-and-model-paths)
- [3. Configure LLM API](#3-configure-llm-api)
- [4. One-Shot Multi-Agent Run Mode](#4-one-shot-multi-agent-run-mode)
- [5. Terminal Multi-Turn Chat Mode](#5-terminal-multi-turn-chat-mode)
- [6. Web Chat Frontend](#6-web-chat-frontend)
- [7. Persistent Retrieval and De Novo Services](#7-persistent-retrieval-and-de-novo-services)
- [8. Full Local Chat Stack](#8-full-local-chat-stack)
- [9. Optional Web RAG Keys](#9-optional-web-rag-keys)
- [10. Data Availability](#10-data-availability)
- [11. Acknowledgements](#11-acknowledgements)
- [12. Citation](#12-citation)

## 1. Install Environment

Use the clean conda environment file first:

```bash
conda env create -f environment.yml
conda activate nmragent
pip install -e .
```

`environment.yml` is the public, reproducible environment. Do not publish machine snapshots such as `environment.full.lock.yml` or `environment.nmragent.yml`.

## 2. Configure Data and Model Paths

All checkpoint/data paths are configured through runtime asset config files. The loader checks these files in order:

```text
configs/runtime_assets.local.json
configs/runtime_assets.json
configs/runtime_assets.example.json
```

For a new machine, copy the example and edit the local file:

```bash
cp configs/runtime_assets.example.json configs/runtime_assets.local.json
```

Put private absolute paths only in `configs/runtime_assets.local.json`. This file is ignored by git.

Important fields:

```json
{
  "env": {
    "NMR_CKPT_DIR": "artifacts/checkpoints",
    "NMR_SEARCH_CKPT": "artifacts/checkpoints/search.pth",
    "NMR_DENOVO_CKPT": "artifacts/checkpoints/denovo_v3.pth",
    "NMR_NMRNET_DIR": "artifacts/checkpoints/NMRNet_ckpt",
    "NMR_RETRIEVAL_LMDB": "artifacts/checkpoints/embeddings_with_smiles.lmdb",
    "NMR_FAISS_INDEX_PATH": "artifacts/checkpoints/faiss.index",
    "NMR_ID_MAP_PATH": "artifacts/checkpoints/id_mapping.pkl",
    "NMR_FORMULA_MAP_PATH": "artifacts/checkpoints/formula_to_ids_new.pkl",
    "NMR_KG_DATA_DIR": "artifacts/kg",
    "NMR_CHEMBERTA_PATH": "artifacts/chemberta"
  }
}
```

Meaning:

- `NMR_CKPT_DIR`: root directory for model/data artifacts
- `NMR_SEARCH_CKPT`: retrieval model checkpoint
- `NMR_DENOVO_CKPT`: de novo generation checkpoint
- `NMR_NMRNET_DIR`: NMRNet/rerank checkpoint directory
- `NMR_RETRIEVAL_LMDB`: retrieval LMDB database
- `NMR_FAISS_INDEX_PATH`: retrieval FAISS index
- `NMR_ID_MAP_PATH`: retrieval ID map
- `NMR_FORMULA_MAP_PATH`: formula-to-ID map
- `NMR_KG_DATA_DIR`: KG/RAG files
- `NMR_CHEMBERTA_PATH`: ChemBERTa tokenizer/model path

The public repository should only contain `configs/runtime_assets.example.json`. Real paths should stay in `configs/runtime_assets.local.json`.

## 3. Configure LLM API

LLM backend config is loaded from:

```text
configs/multi_agent_api.local.yaml
configs/multi_agent_api.yaml
configs/multi_agent_api.example.yaml
```

For local use:

```bash
cp configs/multi_agent_api.example.yaml configs/multi_agent_api.local.yaml
```

Example:

```yaml
openai_compatible:
  backend: openai
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  max_tokens: 4096
  temperature: 0.0
```

Then set:

```bash
export OPENAI_API_KEY=...
```

Do not commit `configs/multi_agent_api.local.yaml` if it contains keys or private endpoints.

## 4. One-Shot Multi-Agent Run Mode

Use this mode for one NMR case and one final JSON result:

```bash
python scripts/run_multi_agent_nmr.py \
  --formula C20H30O3 \
  --h-shifts 5.68 3.19 2.98 2.68 2.59 1.86 1.77 1.69 1.57 1.56 1.44 1.32 1.23 1.20 1.11 1.02 0.94 \
  --c-shifts 210.38 182.46 172.23 111.19 86.84 56.39 46.03 42.31 41.72 40.55 40.20 37.25 34.24 33.51 21.71 19.45 18.34 18.07 17.90 17.51 \
  --model gpt-4o-mini \
  --textbook-rag \
  --kg-rag
```

Dry-run wiring test without loading retrieval/de novo/rerank tools:

```bash
python scripts/run_multi_agent_nmr.py \
  --formula C8H10N4O2 \
  --h-shifts 3.4 3.6 \
  --c-shifts 28.1 149.5 \
  --dry-run-tools \
  --print-trace-tail 20
```

Useful flags:

```bash
--textbook-rag --textbook-rag-top-k 5
--kg-rag --kg-rag-top-k 5 --kg-rag-neighbor-limit 0
--web-rag --web-rag-top-k 5
--use-service-tools
```

`--use-service-tools` makes retrieval/de novo calls through HTTP services instead of initializing those models in the agent process.

## 5. Terminal Multi-Turn Chat Mode

Use terminal chat when you want to paste data over multiple turns:

```bash
python scripts/chat_agent.py --textbook-rag --kg-rag
```

Commands inside terminal chat:

```text
/show   show accumulated formula/shifts/note
/rag    preview RAG evidence
/solve  run the multi-agent solver
/reset  clear current case
/quit   exit
```

## 6. Web Chat Frontend

Start the web chat UI:

```bash
python scripts/chat_agent_web.py --host 0.0.0.0 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Web chat behavior:

- ordinary messages go to the configured OpenAI-compatible chat API
- pasted Formula / 1H / 13C text is accumulated into the current LangGraph thread state
- `rag` previews evidence
- `solve` runs the multi-agent NMR solver
- `demo` loads the built-in demo case
- `show` shows current accumulated state
- `reset` clears the current session

If you have persistent model services running, enable `Use retrieval/denovo services` in the UI before sending `solve`.

## 7. Persistent Retrieval and De Novo Services

Persistent services avoid repeatedly loading retrieval/de novo models inside the chat or CLI process.

Start both model services:

```bash
bash Service/start_nmr_services_tmux.sh
```

Default endpoints:

```text
Retrieval service: http://127.0.0.1:8011/retrieve
De novo service:   http://127.0.0.1:8012/denovo
```

Health check:

```bash
bash Service/check_nmr_services.sh
```

If a port is occupied, override it:

```bash
NMR_DENOVO_SERVICE_PORT=8014 bash Service/start_denovo_service.sh
```

Then point the agent/chat process to the custom URL:

```bash
export NMR_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8011
export NMR_DENOVO_SERVICE_URL=http://127.0.0.1:8014
```

## 8. Full Local Chat Stack

For local development, this helper starts retrieval, de novo, and web chat in tmux sessions:

```bash
bash Service/start_chat_stack.sh
```

Defaults:

```text
Retrieval: http://127.0.0.1:8011
De novo:   http://127.0.0.1:8012
Chat UI:   http://127.0.0.1:7860
```

Override ports when needed:

```bash
NMR_DENOVO_SERVICE_PORT=8014 bash Service/start_chat_stack.sh
```

Logs:

```text
logs/services/retrieval_8011.log
logs/services/denovo_8012.log
logs/chat_agent_web_7860.log
```

## 9. Optional Web RAG Keys

Web RAG is optional. Configure one provider:

```bash
export TAVILY_API_KEY=...
# or
export SERPER_API_KEY=...
# or
export BRAVE_SEARCH_API_KEY=...
```

## 10. Data Availability

Benchmark and database datasets used in this study are publicly available. The source data for the PubChem-NMRNet database are available from Hugging Face at [SimNMR-PubChem](https://huggingface.co/datasets/yqj01/SimNMR-PubChem). The NMRGym dataset is available at [Hugging Face](https://huggingface.co/datasets/meaw0415/NMRGym). The processed nmrshiftdb dataset is available at [Zenodo](https://zenodo.org/records/19142375), and the Exp450 dataset is available at [Zenodo](https://doi.org/10.5281/zenodo.16952024). Detailed information on dataset construction, preprocessing, and benchmark splits is provided in the Supplementary Information.

For construction of the retrieval database, a subset of NMR spectra was simulated from molecular structures using the commercially available Gaussian quantum-chemistry software.

The downstream structural elucidation and structural revision cases were obtained from publicly available literature, including [Altechromone A](https://pubs.acs.org/doi/10.1021/np1005604), [Samoquasine A](https://pubs.acs.org/doi/10.1021/acs.jnatprod.8b00319), the [natural tetrahydroquinoxaline-6-carboxylic acid isolated from *Caulis Sinomenii*](https://pubs.acs.org/doi/10.1021/acs.jnatprod.5c01215), [C5-hydroxy-cyclo(L-Pro-L-Leu)](https://chemrxiv.org/doi/10.26434/chemrxiv-2026-gb0hz), and the [coumarin dimers isolated from *Hydrangea davidii*](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/mrc.70105). The only non-publicly sourced downstream case was the *Vitex trifolia* case, for which experimental NMR data were provided by Peking Union Medical College/Chinese Academy of Medical Sciences (PUMC/CAMS).

## 11. Acknowledgements

We thank our collaborators for helpful discussions on NMR spectroscopy, natural-product structure elucidation, and experimental validation. We gratefully acknowledge funding and institutional support from **The Hong Kong University of Science and Technology (Guangzhou) (HKUST(GZ))**. We also thank Peking Union Medical College/Chinese Academy of Medical Sciences (PUMC/CAMS) for providing experimental NMR data for the *Vitex trifolia* case. We acknowledge the maintainers and contributors of the open-source software and public scientific resources that make NMRAgent possible, including LangGraph, RDKit, FAISS, and the NMR data resources used in this project.

## 12. Citation

If you find NMRAgent useful in your research, please cite our paper:

**Paper:** [Towards Generalizable and Evidential Nuclear Magnetic Resonance-Based Molecular Structure Elucidation via Large Language Model Agent](https://arxiv.org/abs/2606.29776)

```bibtex
@article{fang2026towards,
  title={Towards Generalizable and Evidential Nuclear Magnetic Resonance-Based Molecular Structure Elucidation via Large Language Model Agent},
  author={Fang, Zheng and Yang, Chen and Tan, Yusen and Zhao, Yunpeng and Xu, Fanjie and Xiang, Hongxin and Sun, Hanyu and Gao, Hanyu and Wang, Xiaojian and Du, Wenjie and Li, Yuqiang and Xia, Jun},
  journal={arXiv preprint arXiv:2606.29776},
  year={2026},
  doi={10.48550/arXiv.2606.29776}
}
```
