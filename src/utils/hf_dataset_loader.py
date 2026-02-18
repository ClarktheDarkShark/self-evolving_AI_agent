from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import os


@dataclass
class HfEnvLoadSpec:
    dataset_name: str = "csyq/LifelongAgentBench"
    env: str = "os_interaction"
    split: str = "train"
    cache_dir: Optional[str] = None



def _scan_cached_parquets(dataset_name: str, cache_dir: str | None) -> List[str]:
    """
    Find .parquet files already present in the HF hub cache for this dataset repo.
    Works offline. Returns absolute file paths.
    """
    # Default hub cache root
    hub_root = None
    if cache_dir:
        # huggingface_hub uses cache_dir as root, but many people pass HF_HOME-ish paths.
        # We'll try both the exact cache_dir and cache_dir/hub.
        candidates = [cache_dir, os.path.join(cache_dir, "hub")]
        for c in candidates:
            if os.path.isdir(c):
                hub_root = c
                break
    if hub_root is None:
        hub_root = os.path.expanduser("~/.cache/huggingface/hub")

    owner_repo = dataset_name.replace("/", "--")
    repo_dir = os.path.join(hub_root, f"datasets--{owner_repo}")

    parquet_paths: List[str] = []
    # Look under snapshots/*/**.parquet
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return parquet_paths

    for snap in sorted(os.listdir(snapshots_dir)):
        snap_dir = os.path.join(snapshots_dir, snap)
        if not os.path.isdir(snap_dir):
            continue
        for root, _, files in os.walk(snap_dir):
            for fn in files:
                if fn.lower().endswith(".parquet"):
                    parquet_paths.append(os.path.join(root, fn))
    return parquet_paths

def load_hf_env_parquet(spec) -> Dataset:
    """
    Offline-first loader:
    - never calls HfApi.list_repo_files()
    - tries to use cached parquet files in hub cache
    - only if you want online behavior, you can add an optional online discovery path
    """
    env_lc = (spec.env or "").lower()
    split_lc = (spec.split or "").lower()

    # 1) Try scanning local hub cache for parquet files
    cached = _scan_cached_parquets(spec.dataset_name, spec.cache_dir)

    # Filter by env/split in path/name
    parquet_candidates = [
        p for p in cached
        if env_lc in p.lower() and split_lc in p.lower()
    ]
    if not parquet_candidates:
        parquet_candidates = [p for p in cached if env_lc in p.lower()]

    if parquet_candidates:
        # load_dataset("parquet") ignores split semantics; split here is just a label.
        # If you have multiple files for "train"/"test" etc, your filename filters should separate them.
        return load_dataset("parquet", data_files=parquet_candidates, split="train")

    # 2) If nothing cached, try hf_hub_download in LOCAL-ONLY mode *if you already know filenames*
    # (You currently don't, because you removed list_repo_files.)
    # If you want, you can raise a clear error here:
    raise RuntimeError(
        f"Offline load failed: no cached parquet files found for "
        f"dataset={spec.dataset_name} env={spec.env} split={spec.split}. "
        f"Cache_dir={spec.cache_dir}. "
        f"Tip: run once online (or mirror locally) so the hub cache contains the parquet files."
    )

