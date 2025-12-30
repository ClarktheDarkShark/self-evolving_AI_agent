from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CONFIG_PATHS = [
    "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml",
    "configs/assignments/experiments/llama_31_8b_instruct/instance/os_interaction/instance/standard.yaml",
    "configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    for config_path in CONFIG_PATHS:
        full_path = repo_root / config_path
        if not full_path.exists():
            print(f"[run_all_tasks] Missing config: {full_path}")
            return 1
        print(f"[run_all_tasks] Running: {config_path}")
        result = subprocess.run(
            [sys.executable, "src/run_experiment.py", "--config_path", config_path],
            cwd=repo_root,
            check=False,
        )
        if result.returncode != 0:
            print(f"[run_all_tasks] Failed for {config_path} (exit={result.returncode})")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
