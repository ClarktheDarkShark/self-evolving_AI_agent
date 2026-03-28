from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_CONFIG_DIR = (
    PROJECT_ROOT
    / "configs"
    / "assignments"
    / "experiments"
    / "llama_31_8b_instruct"
    / "instance"
    / "knowledge_graph"
    / "instance"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

CONFIG_TEMPLATE = """\
import:
- ../task.yaml
- ../../../agent.yaml
- ../../../../../../definition.yaml

assignment_config:
  callback_dict:
    callback_0:
      name: current_session_saving_callback
    callback_1:
      name: consecutive_abnormal_agent_inference_process_handling_callback
  output_dir: outputs/{{TIMESTAMP}}
  sample_order:
    - "{sample_index}"

environment_config:
  use_task_client_flag: true

task_dict:
  knowledge_graph:
    parameters:
      ontology_dir_path: null
      data_source: huggingface
      hf_dataset_name: csyq/LifelongAgentBench
      hf_dataset_config: default
      hf_data_dir: knowledge_graph
      hf_split: train
"""


def _write_single_sample_config(*, sample_index: str, stem: str) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = TMP_CONFIG_DIR / f"{stem}.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(sample_index=sample_index),
        encoding="utf-8",
    )
    return config_path


def _find_latest_run_dir(config_stem: str) -> Path | None:
    candidates = sorted(
        OUTPUT_ROOT.glob(f"run_all_*/knowledge_graph/{config_stem}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_run_dir(run_dir: Path) -> dict[str, Any]:
    task_outcomes_path = run_dir / "task_outcomes.json"
    current_session_path = run_dir / "current_session.json"
    exception_path = run_dir / "exception.txt"
    generated_tools_path = run_dir / "generated_tools.log"

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task_outcomes_path": str(task_outcomes_path) if task_outcomes_path.exists() else None,
        "generated_tools_path": str(generated_tools_path) if generated_tools_path.exists() else None,
        "exception_path": str(exception_path) if exception_path.exists() else None,
    }

    if task_outcomes_path.exists():
        outcomes = _load_json(task_outcomes_path)
        outcome = None
        if isinstance(outcomes, list) and outcomes:
            outcome = outcomes[0]
        elif isinstance(outcomes, dict):
            results = outcomes.get("results")
            if isinstance(results, list) and results:
                outcome = results[0]
        if isinstance(outcome, dict):
            summary.update(
                {
                    "sample_index": outcome.get("sample_index"),
                    "sample_status": outcome.get("sample_status")
                    or ("completed" if outcome.get("completed") else "not_completed"),
                    "finish_reason": outcome.get("finish_reason")
                    or outcome.get("not_completed_reason"),
                    "task_output": outcome.get("task_output"),
                    "evaluation_outcome": outcome.get("outcome")
                    or (
                        outcome.get("evaluation_record", {}) or {}
                    ).get("outcome"),
                }
            )
    elif current_session_path.exists():
        current_session = _load_json(current_session_path)
        summary.update(
            {
                "sample_index": current_session.get("sample_index"),
                "sample_status": current_session.get("sample_status"),
                "finish_reason": current_session.get("finish_reason"),
            }
        )
    return summary


def run_sample(*, sample_index: str, label: str) -> dict[str, Any]:
    config_stem = f"pal_batch_{label}_{sample_index}"
    config_path = _write_single_sample_config(
        sample_index=sample_index,
        stem=config_stem,
    )
    command = [
        sys.executable,
        "-c",
        (
            "import scripts.run_all_with_servers as m; "
            f'm.CONFIG_PATHS=["{config_path.relative_to(PROJECT_ROOT).as_posix()}"]; '
            "raise SystemExit(m.main())"
        ),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ".:scripts"
    env["ENABLE_PAL_AGENT"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    run_dir = _find_latest_run_dir(config_stem)
    summary = {
        "sample_index": sample_index,
        "config_path": str(config_path),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if run_dir is not None:
        summary.update(_summarize_run_dir(run_dir))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    summaries = [
        run_sample(sample_index=str(sample_index), label=args.label)
        for sample_index in args.samples
    ]

    summary_text = json.dumps(summaries, indent=2, sort_keys=True)
    if args.summary_path:
        summary_path = Path(args.summary_path)
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
