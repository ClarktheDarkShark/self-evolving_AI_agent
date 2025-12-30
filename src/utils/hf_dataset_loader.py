from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset


@dataclass
class HfEnvLoadSpec:
    dataset_name: str = "csyq/LifelongAgentBench"
    env: str = "os_interaction"
    split: str = "train"
    cache_dir: Optional[str] = None


def load_hf_env_parquet(spec: HfEnvLoadSpec) -> Dataset:
    api = HfApi()
    files = api.list_repo_files(repo_id=spec.dataset_name, repo_type="dataset")

    env_lc = spec.env.lower()
    split_lc = spec.split.lower()

    parquet_candidates = [
        f
        for f in files
        if f.lower().endswith(".parquet")
        and env_lc in f.lower()
        and split_lc in f.lower()
    ]

    if not parquet_candidates:
        parquet_candidates = [
            f
            for f in files
            if f.lower().endswith(".parquet") and env_lc in f.lower()
        ]

    if not parquet_candidates:
        raise RuntimeError(
            f"No parquet files found for env={spec.env} in dataset={spec.dataset_name}. "
            f"Repo files (sample): {files[:50]}"
        )

    local_paths: List[str] = []
    for filename in parquet_candidates:
        local_paths.append(
            hf_hub_download(
                repo_id=spec.dataset_name,
                repo_type="dataset",
                filename=filename,
                cache_dir=spec.cache_dir,
            )
        )

    dataset = load_dataset("parquet", data_files=local_paths, split=spec.split)
    return dataset
