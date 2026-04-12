from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_kg_family_policy_evolution as family_policy_harness
from src.sage.candidate_compare import (
    aggregate_stage_metrics,
    baseline_satisfies_locked_family,
    compare_aggregate_metrics,
    load_archived_baseline_record,
    stage_definitively_lost,
    summarize_sample_semantic_metrics,
    write_archived_baseline_record,
)
from src.sage.family_policy_evolution import ENV_COMPARE_LOCK_FAMILY
from src.sage.family_policy_evolution import (
    build_candidate_signature,
    build_family_policy_store,
)
from src.sage.reusable_tool_families import get_baseline_reusable_family_policy_bundles


COMPARE_MODE_FULL = "family_locked_candidate_compare_full"
COMPARE_MODE_SAGE_ONLY = "family_locked_candidate_compare_sage_only"
SAGE_ONLY_BYPASS_ENV = "SAGE_TOOL_EVOLUTION_SKIP_MANUAL_FALLBACK"


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_sample_order_from_config(config_path: Path) -> list[str]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assignment_config = payload.get("assignment_config") or {}
    sample_order = assignment_config.get("sample_order") or []
    return [str(item or "").strip() for item in sample_order if str(item or "").strip()]


def _comparison_mode(*, sage_only_screen: bool) -> str:
    return COMPARE_MODE_SAGE_ONLY if sage_only_screen else COMPARE_MODE_FULL


def _comparison_env(*, family_name: str, sage_only_screen: bool) -> dict[str, str]:
    env = {
        ENV_COMPARE_LOCK_FAMILY: str(family_name or "").strip(),
    }
    if sage_only_screen:
        env[SAGE_ONLY_BYPASS_ENV] = "1"
    return env


def _parse_extra_env(raw_values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_value in raw_values:
        cleaned = str(raw_value or "").strip()
        if not cleaned:
            continue
        key, separator, value = cleaned.partition("=")
        cleaned_key = key.strip()
        if not separator or not cleaned_key:
            raise SystemExit(f"invalid_extra_env_assignment:{cleaned}")
        parsed[cleaned_key] = value.strip()
    return parsed


def _baseline_archive_path(
    *,
    baseline_root: Path,
    family_name: str,
    evaluation_mode: str,
    baseline_version: str,
    sample_index: str,
) -> Path:
    return (
        baseline_root
        / str(family_name or "").strip()
        / str(evaluation_mode or "").strip()
        / str(baseline_version or "").strip()
        / f"{str(sample_index or '').strip()}.json"
    )


def _run_and_summarize(
    *,
    sample_index: str,
    label: str,
    family_name: str,
    store_path: Path,
    bundle_version: str,
    evaluation_mode: str,
    parent_output_dir: Path | None,
    extra_env: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_summary = family_policy_harness._run_sample_with_policy(
        sample_index=sample_index,
        label=label,
        family_name=family_name,
        store_path=store_path,
        promotion_enabled=False,
        override_version=bundle_version,
        evaluation_mode=evaluation_mode,
        parent_output_dir=parent_output_dir,
        extra_env=extra_env,
    )
    semantic_metrics = summarize_sample_semantic_metrics(
        run_summary,
        locked_family=family_name,
    )
    return dict(run_summary), semantic_metrics


def compare_family_candidate(
    *,
    family_name: str,
    sample_indices: Sequence[str],
    store_path: Path,
    candidate_version: str,
    baseline_version: str,
    summary_path: Path | None = None,
    baseline_archive_root: Path | None = None,
    parent_output_dir: Path | None = None,
    sage_only_screen: bool = False,
    screen_unsatisfied_baseline: bool = False,
    stop_on_definitive_loss: bool = True,
    label: str = "",
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    cleaned_family = str(family_name or "").strip()
    cleaned_samples = [
        str(item or "").strip() for item in sample_indices if str(item or "").strip()
    ]
    evaluation_mode = _comparison_mode(sage_only_screen=sage_only_screen)
    comparison_env = _comparison_env(
        family_name=cleaned_family,
        sage_only_screen=sage_only_screen,
    )
    merged_extra_env = {
        **comparison_env,
        **{
            str(key or "").strip(): str(value or "").strip()
            for key, value in (extra_env or {}).items()
            if str(key or "").strip()
        },
    }
    baseline_root = (
        baseline_archive_root.resolve()
        if baseline_archive_root is not None
        else (store_path / "_baseline_compare_archive").resolve()
    )
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate_bundle = store.get_bundle(cleaned_family, version=candidate_version)
    candidate_signature = (
        build_candidate_signature(bundle=candidate_bundle, fields_changed=())
        if candidate_bundle is not None
        else {}
    )
    baseline_bundle = store.get_bundle(cleaned_family, version=baseline_version)
    baseline_signature = (
        build_candidate_signature(bundle=baseline_bundle, fields_changed=())
        if baseline_bundle is not None
        else {}
    )

    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    baseline_reuse_count = 0
    screened_out_samples: list[dict[str, Any]] = []
    stopped_early = False

    cleaned_label = str(label or "").strip() or f"{cleaned_family}_{evaluation_mode}"
    for sample_index in cleaned_samples:
        baseline_context = family_policy_harness._build_evaluation_cache_context(
            family_name=cleaned_family,
            bundle_version=baseline_version,
            sample_index=sample_index,
            evaluation_mode=evaluation_mode,
            extra_env=merged_extra_env,
        )
        archive_path = _baseline_archive_path(
            baseline_root=baseline_root,
            family_name=cleaned_family,
            evaluation_mode=evaluation_mode,
            baseline_version=baseline_version,
            sample_index=sample_index,
        )
        archived_baseline = load_archived_baseline_record(
            archive_path,
            expected_context=baseline_context,
        )
        if archived_baseline is not None:
            baseline_run_summary = dict(archived_baseline["run_summary"])
            baseline_semantic_metrics = dict(archived_baseline["semantic_metrics"])
            baseline_semantic_metrics["archived_baseline_reused"] = True
            baseline_reuse_count += 1
        else:
            baseline_run_summary, baseline_semantic_metrics = _run_and_summarize(
                sample_index=sample_index,
                label=f"{cleaned_label}_baseline_{sample_index}",
                family_name=cleaned_family,
                store_path=store_path,
                bundle_version=baseline_version,
                evaluation_mode=evaluation_mode,
                parent_output_dir=parent_output_dir,
                extra_env=merged_extra_env,
            )
            write_archived_baseline_record(
                archive_path,
                context=baseline_context,
                run_summary=baseline_run_summary,
                semantic_metrics=baseline_semantic_metrics,
            )

        if screen_unsatisfied_baseline and not baseline_satisfies_locked_family(
            baseline_semantic_metrics
        ) or (
            screen_unsatisfied_baseline
            and str(baseline_run_summary.get("finish_reason") or "").strip().startswith(
                "sage_query_plan_invalid:family_compare_locked_query_shape:"
            )
        ):
            screened_out_samples.append(
                {
                    "sample_index": sample_index,
                    "reason": "baseline_locked_family_unsatisfied",
                    "baseline_run_summary": dict(baseline_run_summary),
                    "baseline_semantic_metrics": dict(baseline_semantic_metrics),
                }
            )
            per_sample_rows.append(
                {
                    "sample_index": sample_index,
                    "screened_out": True,
                    "screen_reason": "baseline_locked_family_unsatisfied",
                    "baseline_run_summary": baseline_run_summary,
                    "baseline_semantic_metrics": baseline_semantic_metrics,
                }
            )
            continue

        candidate_run_summary, candidate_semantic_metrics = _run_and_summarize(
            sample_index=sample_index,
            label=f"{cleaned_label}_candidate_{sample_index}",
            family_name=cleaned_family,
            store_path=store_path,
            bundle_version=candidate_version,
            evaluation_mode=evaluation_mode,
            parent_output_dir=parent_output_dir,
            extra_env=merged_extra_env,
        )
        baseline_rows.append(dict(baseline_semantic_metrics))
        candidate_rows.append(dict(candidate_semantic_metrics))
        per_sample_rows.append(
            {
                "sample_index": sample_index,
                "baseline_run_summary": baseline_run_summary,
                "baseline_semantic_metrics": baseline_semantic_metrics,
                "candidate_run_summary": candidate_run_summary,
                "candidate_semantic_metrics": candidate_semantic_metrics,
            }
        )
        if stop_on_definitive_loss:
            baseline_aggregate = aggregate_stage_metrics(baseline_rows)
            candidate_aggregate = aggregate_stage_metrics(candidate_rows)
            if stage_definitively_lost(
                baseline=baseline_aggregate,
                candidate=candidate_aggregate,
            ):
                stopped_early = True
                break

    baseline_aggregate = aggregate_stage_metrics(baseline_rows)
    candidate_aggregate = aggregate_stage_metrics(candidate_rows)
    comparison = compare_aggregate_metrics(
        baseline=baseline_aggregate,
        candidate=candidate_aggregate,
    )
    summary = {
        "family_name": cleaned_family,
        "candidate_version": str(candidate_version or "").strip(),
        "candidate_signature": candidate_signature,
        "baseline_version": str(baseline_version or "").strip(),
        "baseline_signature": baseline_signature,
        "sample_indices": cleaned_samples,
        "evaluated_sample_count": len(per_sample_rows),
        "evaluation_mode": evaluation_mode,
        "sage_only_screen": bool(sage_only_screen),
        "locked_family": cleaned_family,
        "extra_env": merged_extra_env,
        "store_path": str(store_path),
        "baseline_archive_root": str(baseline_root),
        "baseline_reuse_count": baseline_reuse_count,
        "screen_unsatisfied_baseline": bool(screen_unsatisfied_baseline),
        "screened_out_sample_count": len(screened_out_samples),
        "screened_out_samples": screened_out_samples,
        "stopped_early": stopped_early,
        "family_lock_pure": bool(
            int(baseline_aggregate.get("family_lock_violation_count") or 0) == 0
            and int(candidate_aggregate.get("family_lock_violation_count") or 0) == 0
        ),
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "comparison": comparison,
        "per_sample": per_sample_rows,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--store-path", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--baseline-version", default="2026-03-31")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--baseline-archive-root", default="")
    parser.add_argument("--parent-output-dir", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--sage-only-screen", action="store_true")
    parser.add_argument("--screen-unsatisfied-baseline", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args()

    config_path = _resolve_path(args.config_path) if args.config_path else None
    sample_indices = list(args.samples or [])
    if config_path is not None:
        sample_indices.extend(_load_sample_order_from_config(config_path))
    cleaned_samples = [str(item or "").strip() for item in sample_indices if str(item or "").strip()]
    if not cleaned_samples:
        raise SystemExit("offline_compare_requires_samples_or_config_path")

    summary = compare_family_candidate(
        family_name=args.family,
        sample_indices=cleaned_samples,
        store_path=_resolve_path(args.store_path),
        candidate_version=args.candidate_version,
        baseline_version=args.baseline_version,
        summary_path=_resolve_path(args.summary_path) if args.summary_path else None,
        baseline_archive_root=(
            _resolve_path(args.baseline_archive_root)
            if args.baseline_archive_root
            else None
        ),
        parent_output_dir=(
            _resolve_path(args.parent_output_dir) if args.parent_output_dir else None
        ),
        sage_only_screen=bool(args.sage_only_screen),
        screen_unsatisfied_baseline=bool(args.screen_unsatisfied_baseline),
        stop_on_definitive_loss=not bool(args.no_early_stop),
        label=args.label,
        extra_env=_parse_extra_env(args.extra_env),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
