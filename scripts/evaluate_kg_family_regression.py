from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "configs" / "evaluation" / "knowledge_graph_family_regression.json"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_sample_index(raw_value: Any) -> str:
    return str(raw_value or "").strip()


def evaluate_family_regression(*, runs_path: Path, manifest_path: Path) -> dict[str, Any]:
    runs = _load_json(runs_path)
    manifest = _load_json(manifest_path)
    if not isinstance(runs, list):
        raise ValueError("runs.json must contain a list of session records")
    families = manifest.get("families") or {}
    by_sample: dict[str, dict[str, Any]] = {}
    for item in runs:
        if not isinstance(item, dict):
            continue
        sample_index = _normalize_sample_index(item.get("sample_index"))
        if sample_index:
            by_sample[sample_index] = item

    summary: dict[str, Any] = {
        "runs_path": str(runs_path),
        "manifest_path": str(manifest_path),
        "families": {},
    }
    totals = defaultdict(int)

    for family_name, split_dict in families.items():
        family_summary: dict[str, Any] = {}
        if not isinstance(split_dict, dict):
            continue
        for split_name in ("tuning", "held_out"):
            sample_ids = [
                _normalize_sample_index(sample_id)
                for sample_id in (split_dict.get(split_name) or [])
                if _normalize_sample_index(sample_id)
            ]
            matched_records = [
                by_sample[sample_id]
                for sample_id in sample_ids
                if sample_id in by_sample
            ]
            completed = sum(
                1
                for record in matched_records
                if str(record.get("sample_status") or "").strip() == "completed"
            )
            correct = sum(
                1
                for record in matched_records
                if str(
                    ((record.get("evaluation_record") or {}).get("outcome") or "")
                ).strip()
                == "correct"
            )
            wrong_completed = sum(
                1
                for record in matched_records
                if str(record.get("sample_status") or "").strip() == "completed"
                and str(
                    ((record.get("evaluation_record") or {}).get("outcome") or "")
                ).strip()
                != "correct"
            )
            split_summary = {
                "expected_samples": sample_ids,
                "matched_samples": [record.get("sample_index") for record in matched_records],
                "matched_count": len(matched_records),
                "completed_count": completed,
                "correct_count": correct,
                "wrong_completed_count": wrong_completed,
            }
            family_summary[split_name] = split_summary
            totals[f"{split_name}_matched"] += len(matched_records)
            totals[f"{split_name}_correct"] += correct
            totals[f"{split_name}_wrong_completed"] += wrong_completed
        summary["families"][family_name] = family_summary

    summary["totals"] = dict(totals)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Path to runs.json")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to family regression manifest",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs)
    if not runs_path.is_absolute():
        runs_path = PROJECT_ROOT / runs_path
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    summary = evaluate_family_regression(
        runs_path=runs_path,
        manifest_path=manifest_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
