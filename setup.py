from setuptools import find_packages, setup

setup(
    name="nmragent",
    version="0.1.0",
    description="Standalone NMR agent framework for molecular structure elucidation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain-core>=0.3.0",
        "openai>=1.0.0",
        "rdkit",
        "scipy",
        "numpy",
        "faiss-cpu",
        "lmdb",
        "msgpack",
        "tqdm",
        "requests",
        "flask",
    ],
    extras_require={
        "vllm": ["vllm"],
        "transformers": ["transformers", "accelerate"],
        "openai": ["langchain-openai>=0.2.0"],
        "all": ["vllm", "transformers", "accelerate", "langchain-openai>=0.2.0"],
    },
)
