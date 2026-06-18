"""
NMRAgent configuration.

Defaults are repository-oriented and safe for public release. Users are expected
to point environment variables at their own checkpoints, caches, and datasets.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"


def _env_path(name: str, default: Path | str) -> str:
    return os.environ.get(name, str(default))


@dataclass
class PathConfig:
    """All data/model paths. Override via env vars or constructor."""

    nmr_ckpt_dir: str = _env_path("NMR_CKPT_DIR", _ARTIFACTS_DIR / "checkpoints")
    kg_data_dir: str = _env_path("NMR_KG_DATA_DIR", _ARTIFACTS_DIR / "kg")
    base_llm: str = os.environ.get("NMR_BASE_LLM", "gpt-4o-mini")
    chemberta_path: str = _env_path("NMR_CHEMBERTA_PATH", _ARTIFACTS_DIR / "chemberta")
    offline_lmdb: str = _env_path("NMR_OFFLINE_LMDB", _ARTIFACTS_DIR / "offline" / "offline_cache.lmdb")
    benchmark_data: str = _env_path("NMR_BENCH_DATA", _ARTIFACTS_DIR / "benchmarks" / "nmrgym_test.json")
    fragment_library_dir: str = _env_path("NMR_FRAGMENT_LIB_DIR", _ARTIFACTS_DIR / "fragments")

    @property
    def kg_csv_path(self) -> str:
        return os.path.join(self.kg_data_dir, "10m_elementkg_release.csv")

    @property
    def kg_json_path(self) -> str:
        return os.path.join(self.kg_data_dir, "knowlege_description_20w.json")

    @property
    def search_model_path(self) -> str:
        return os.path.join(self.nmr_ckpt_dir, "search.pth")

    @property
    def denovo_model_path(self) -> str:
        return os.path.join(self.nmr_ckpt_dir, "denovo_v3.pth")

    @property
    def fragment_lmdb_path(self) -> str:
        return os.path.join(self.fragment_library_dir, "fragment_library.lmdb")

    @property
    def fragment_index_path(self) -> str:
        return os.path.join(self.fragment_library_dir, "fragment_shift_index.pkl")


@dataclass
class BackendConfig:
    """LLM backend configuration."""

    backend: str = os.environ.get("NMR_BACKEND", "vllm")
    model: str = os.environ.get("NMR_MODEL", os.environ.get("NMR_BASE_LLM", "gpt-4o-mini"))
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "")
    tensor_parallel_size: int = int(os.environ.get("NMR_TP_SIZE", "4"))
    gpu_memory_utilization: float = float(os.environ.get("NMR_GPU_MEM", "0.85"))
    max_model_len: int = int(os.environ.get("NMR_MAX_MODEL_LEN", "32768"))
    temperature: float = float(os.environ.get("NMR_TEMPERATURE", "0.7"))
    top_p: float = float(os.environ.get("NMR_TOP_P", "0.95"))
    max_tokens: int = int(os.environ.get("NMR_MAX_TOKENS", "4096"))


@dataclass
class AgentConfig:
    """Agent behavior settings."""

    force_kg_rag: bool = True
    max_iterations: int = 6
    batch_size: int = 16
    h_sigma: float = 1.0
    c_sigma: float = 10.0


@dataclass
class NMRConfig:
    """Full configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_env(cls) -> "NMRConfig":
        return cls()

    @classmethod
    def for_openai(cls, model: str = "gpt-4o-mini", api_key: str = None) -> "NMRConfig":
        cfg = cls()
        cfg.backend.backend = "openai"
        cfg.backend.model = model
        if api_key:
            cfg.backend.openai_api_key = api_key
        return cfg

    @classmethod
    def for_vllm(cls, model_path: str = None, gpus: str = "0,1,2,3") -> "NMRConfig":
        cfg = cls()
        cfg.backend.backend = "vllm"
        if model_path:
            cfg.backend.model = model_path
        cfg.backend.tensor_parallel_size = len(gpus.split(","))
        return cfg

    @classmethod
    def for_offline(cls, lmdb_path: str = None) -> "NMRConfig":
        cfg = cls()
        cfg.agent.force_kg_rag = False
        if lmdb_path:
            cfg.paths.offline_lmdb = lmdb_path
        return cfg
