from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolgenPipelineConfig:
    pipeline: str
    agg_n: int
    registry_root: str
    registry_dir: str
    name_prefix: str
    registry_root_from_env: bool


def _parse_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def get_toolgen_pipeline_config(default_registry_root: str) -> ToolgenPipelineConfig:
    pipeline_raw = os.environ.get("TOOLGEN_PIPELINE", "baseline").strip().lower()
    pipeline = pipeline_raw if pipeline_raw in {"baseline", "aggregate3"} else "baseline"
    if pipeline == "aggregate3":
        agg_n = 10
    else:
        agg_n = _parse_int(os.environ.get("TOOLGEN_AGG_N", "3"), 3)
    registry_root_env = os.environ.get("TOOL_REGISTRY_ROOT")
    registry_root_from_env = registry_root_env is not None
    registry_root = (registry_root_env or default_registry_root).strip()
    if pipeline == "baseline" and not registry_root_from_env:
        registry_dir = registry_root
    else:
        registry_dir = os.path.join(registry_root, pipeline)
    name_prefix = "" if pipeline == "baseline" else "agg3__"
    return ToolgenPipelineConfig(
        pipeline=pipeline,
        agg_n=agg_n,
        registry_root=registry_root,
        registry_dir=registry_dir,
        name_prefix=name_prefix,
        registry_root_from_env=registry_root_from_env,
    )
