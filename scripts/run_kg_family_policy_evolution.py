from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pal_kg_batch_runner import PROJECT_ROOT, run_sample
from src.pal.family_policy_evolution import (
    ENV_ENABLE_EVOLUTION,
    ENV_ENABLE_PROMOTION,
    ENV_ENABLED_FAMILIES,
    ENV_OVERRIDE_VERSIONS,
    ENV_STORE_PATH,
    build_family_policy_store,
    classify_family_failure,
    summarize_sample_metrics,
    version_reuse_compatible,
)
from src.pal.reusable_tool_families import get_baseline_reusable_family_policy_bundles


PROMOTION_EVALUATION_MODE = "family_inline_promotion_gate"
PROMOTION_EVALUATION_CONTRACT_VERSION = "2026-03-31__family_inline_promotion_gate_v1"
MAX_PRIOR_SUCCESS_SAMPLES = 1
ENV_PROMOTION_GATE_MODE = "PAL_FAMILY_POLICY_PROMOTION_GATE_MODE"
POLICY_FINGERPRINT_FILES = (
    PROJECT_ROOT / "src" / "agents" / "instance" / "pal_agent_controller.py",
    PROJECT_ROOT / "src" / "pal" / "family_policy_evolution.py",
    PROJECT_ROOT / "src" / "pal" / "kg_benchmark_adapter.py",
    PROJECT_ROOT / "src" / "pal" / "plausibility_validator.py",
    PROJECT_ROOT / "src" / "pal" / "policy_contracts.py",
    PROJECT_ROOT / "src" / "pal" / "reusable_tool_families.py",
    PROJECT_ROOT / "scripts" / "run_kg_family_policy_evolution.py",
)
EXECUTION_FINGERPRINT_FILES = (
    PROJECT_ROOT / "scripts" / "pal_kg_batch_runner.py",
    PROJECT_ROOT / "scripts" / "run_all_with_servers.py",
)
EXECUTION_ENV_KEYS = (
    "ENABLE_PAL_AGENT",
    "LIFELONG_FUSEKI_IMAGE",
    "LIFELONG_FUSEKI_PORT",
    "LIFELONG_KG_CONTAINER_NAME",
    "LIFELONG_KG_DATA_DIR",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    ENV_PROMOTION_GATE_MODE,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _policy_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in POLICY_FINGERPRINT_FILES:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _execution_environment_fingerprint() -> str:
    env_snapshot = {
        key: str(os.environ.get(key) or "").strip()
        for key in EXECUTION_ENV_KEYS
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "env": env_snapshot,
                "python_version": sys.version,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for path in EXECUTION_FINGERPRINT_FILES:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _build_evaluation_cache_context(
    *,
    family_name: str,
    bundle_version: str,
    sample_index: str,
    evaluation_mode: str = PROMOTION_EVALUATION_MODE,
) -> dict[str, Any]:
    return {
        "family_name": str(family_name or "").strip(),
        "bundle_version": str(bundle_version or "").strip(),
        "sample_index": str(sample_index or "").strip(),
        "evaluation_mode": str(evaluation_mode or "").strip(),
        "promotion_evaluation_contract_version": PROMOTION_EVALUATION_CONTRACT_VERSION,
        "policy_fingerprint": _policy_fingerprint(),
        "execution_environment_fingerprint": _execution_environment_fingerprint(),
    }


def _build_success_bank_context(
    *,
    family_name: str,
    active_version: str,
) -> dict[str, Any]:
    return {
        "family_name": str(family_name or "").strip(),
        "source_version": str(active_version or "").strip(),
        "evaluation_mode": PROMOTION_EVALUATION_MODE,
        "promotion_evaluation_contract_version": PROMOTION_EVALUATION_CONTRACT_VERSION,
        "policy_fingerprint": _policy_fingerprint(),
        "execution_environment_fingerprint": _execution_environment_fingerprint(),
    }


def _promotion_gate_mode() -> str:
    mode = str(os.environ.get(ENV_PROMOTION_GATE_MODE) or "").strip().lower()
    if mode in {"soft_improvement", "non_regression"}:
        return mode
    return "trigger_first"


def _contexts_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def _context_contains(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    observed_dict = dict(observed)
    return all(observed_dict.get(key) == value for key, value in dict(expected).items())


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _extract_run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    dangerous_overreach_count = int(summary.get("dangerous_overreach_count") or 0)
    run_dir_text = str(summary.get("run_dir") or "").strip()
    if run_dir_text:
        run_dir = Path(run_dir_text)
        generated_tools_path = run_dir / "generated_tools.log"
        if generated_tools_path.exists():
            dangerous_overreach_count = 0
            for payload in _load_json_lines(generated_tools_path):
                if payload.get("event") == "pal_attempt_decision_finalized" and bool(
                    payload.get("dangerous_overreach")
                ):
                    dangerous_overreach_count += 1
    sample_status = str(summary.get("sample_status") or "").strip()
    evaluation_outcome = str(summary.get("evaluation_outcome") or "").strip()
    return {
        **summary,
        "dangerous_overreach_count": dangerous_overreach_count,
        "sample_status": sample_status,
        "evaluation_outcome": evaluation_outcome,
    }


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _latest_attempt_decision(run_dir: Path, *, sample_index: str) -> dict[str, Any]:
    generated_tools_path = run_dir / "generated_tools.log"
    latest: dict[str, Any] = {}
    target_sample = str(sample_index or "").strip()
    for payload in _load_json_lines(generated_tools_path):
        if str(payload.get("sample_index") or "").strip() != target_sample:
            continue
        if payload.get("event") == "pal_attempt_decision_finalized":
            latest = payload
    return latest


def _load_query_plan_for_tool(run_dir: Path, tool_name: str) -> dict[str, Any]:
    cleaned_tool_name = str(tool_name or "").strip()
    if not cleaned_tool_name:
        return {}
    plan_path = run_dir / "pal_query_artifacts" / f"{cleaned_tool_name}.plan.json"
    if not plan_path.exists():
        return {}
    try:
        payload = _load_json(plan_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _relation_names_from_query_plan(query_plan: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for relation_path in query_plan.get("relation_paths") or []:
        if not isinstance(relation_path, Mapping):
            continue
        relation_name = str(
            relation_path.get("relation")
            or relation_path.get("normalized_relation")
            or relation_path.get("relation_name")
            or ""
        ).strip()
        if relation_name and relation_name not in names:
            names.append(relation_name)
    return names


def _build_scaffold_signature(query_plan: Mapping[str, Any]) -> str:
    query_shape = str(query_plan.get("query_shape") or "n/a").strip() or "n/a"
    answer_role = str(
        query_plan.get("projection_role")
        or query_plan.get("projection_var_role")
        or query_plan.get("answer_role")
        or "n/a"
    ).strip() or "n/a"
    relation_names = _relation_names_from_query_plan(query_plan)
    relation_part = "|".join(relation_names[:3]) if relation_names else "n/a"
    return f"{query_shape}|{answer_role}|{relation_part}"


def _has_pending_candidate_for_sample(
    pending_candidates: Sequence[Mapping[str, Any]], sample_index: str
) -> bool:
    target = str(sample_index or "").strip()
    return any(
        str((item.get("trigger_context") or {}).get("sample_index") or "").strip() == target
        for item in pending_candidates
    )


def _derive_failure_reasons(
    *,
    sample_status: str,
    evaluation_outcome: str,
    decision_payload: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    cleaned_status = str(sample_status or "").strip()
    cleaned_outcome = str(evaluation_outcome or "").strip()
    if cleaned_status == "completed" and cleaned_outcome != "correct":
        reasons.append("trusted_incorrect_completion")
    elif cleaned_status:
        reasons.append(f"sample_status:{cleaned_status}")
    if cleaned_outcome and cleaned_outcome != "correct":
        reasons.append(f"evaluation_outcome:{cleaned_outcome}")
    if bool(decision_payload.get("dangerous_overreach")):
        raw_reasons = decision_payload.get("dangerous_overreach_reasons") or []
        if raw_reasons:
            reasons.extend(
                f"dangerous_overreach:{str(item or '').strip()}"
                for item in raw_reasons
                if str(item or "").strip()
            )
        else:
            reasons.append("dangerous_overreach:unspecified")
    for denial_reason in decision_payload.get("materialization_denial_reasons") or []:
        cleaned = str(denial_reason or "").strip()
        if cleaned:
            reasons.append(cleaned)
    return reasons


def _maybe_create_candidate_from_run_summary(
    *,
    family_name: str,
    run_summary: Mapping[str, Any],
    store_path: Path,
) -> dict[str, Any] | None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    sample_index = str(run_summary.get("sample_index") or "").strip()
    if not sample_index:
        return None
    if _has_pending_candidate_for_sample(store.get_pending_candidates(family_name), sample_index):
        return None

    sample_status = str(run_summary.get("sample_status") or "").strip()
    evaluation_outcome = str(run_summary.get("evaluation_outcome") or "").strip()
    if sample_status == "completed" and evaluation_outcome == "correct":
        return None

    run_dir = Path(str(run_summary.get("run_dir") or "")).resolve()
    decision_payload = _latest_attempt_decision(run_dir, sample_index=sample_index)
    if str(decision_payload.get("selected_family") or "").strip() != str(family_name or "").strip():
        return None

    query_plan = _load_query_plan_for_tool(
        run_dir, str(decision_payload.get("tool_name") or "").strip()
    )
    if not query_plan:
        return None
    scaffold_signature = _build_scaffold_signature(query_plan)
    relation_names = _relation_names_from_query_plan(query_plan)
    if not scaffold_signature or scaffold_signature.endswith("|n/a"):
        return None

    failure_reasons = _derive_failure_reasons(
        sample_status=sample_status,
        evaluation_outcome=evaluation_outcome,
        decision_payload=decision_payload,
    )
    failure_class = classify_family_failure(
        family_name=family_name,
        sample_status=sample_status,
        evaluation_outcome=evaluation_outcome,
        relation_names=relation_names,
        failure_reasons=failure_reasons,
        dangerous_overreach=bool(decision_payload.get("dangerous_overreach")),
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature=scaffold_signature,
        relation_names=relation_names,
        failure_reasons=failure_reasons,
        failure_class=failure_class,
        trigger_context={
            "sample_index": sample_index,
            "sample_status": sample_status,
            "evaluation_outcome": evaluation_outcome,
            "selected_family": family_name,
            "family_bundle_version": str(
                decision_payload.get("family_bundle_version") or ""
            ).strip(),
            "tool_name": str(decision_payload.get("tool_name") or "").strip(),
            "scaffold_signature": scaffold_signature,
            "relation_names": relation_names,
            "run_dir": str(run_dir),
        },
    )
    if candidate is None:
        return None
    return {
        "event": "family_policy_candidate_synthesized_from_run",
        "family_name": family_name,
        "candidate_version": candidate.candidate_version,
        "failure_class": failure_class,
        "trigger_sample": sample_index,
        "sample_status": sample_status,
        "evaluation_outcome": evaluation_outcome,
        "scaffold_signature": scaffold_signature,
        "relation_names": relation_names,
        "failure_reasons": failure_reasons,
        "store_path": str(store_path),
    }


def _run_sample_with_policy(
    *,
    sample_index: str,
    label: str,
    family_name: str,
    store_path: Path,
    promotion_enabled: bool,
    override_version: str | None = None,
    evaluation_mode: str = PROMOTION_EVALUATION_MODE,
    parent_output_dir: Path | None = None,
) -> dict[str, Any]:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    bundle_version = str(
        override_version or store.get_active_version(family_name) or ""
    ).strip()
    cache_context = _build_evaluation_cache_context(
        family_name=family_name,
        bundle_version=bundle_version,
        sample_index=sample_index,
        evaluation_mode=evaluation_mode,
    )
    cache_key = _hash_payload(cache_context)
    cached = store.get_cached_evaluation(family_name, cache_key=cache_key)
    if isinstance(cached, Mapping) and _contexts_match(
        cached.get("context") or {},
        cache_context,
    ):
        cached_result = dict(cached.get("result") or {})
        cached_result.update(
            {
                "evaluation_cache_hit": True,
                "evaluation_cache_key": cache_key,
                "evaluation_context": cache_context,
                "evaluation_bundle_version": bundle_version,
            }
        )
        print(
            "[family_policy_cache] hit "
            f"family={family_name} version={bundle_version} sample={sample_index} "
            f"mode={evaluation_mode}"
        )
        return cached_result

    env_updates = {
        ENV_ENABLE_EVOLUTION: "1",
        ENV_ENABLE_PROMOTION: "1" if promotion_enabled else "0",
        ENV_ENABLED_FAMILIES: family_name,
        ENV_STORE_PATH: str(store_path),
    }
    if override_version:
        env_updates[ENV_OVERRIDE_VERSIONS] = json.dumps({family_name: override_version})
    else:
        env_updates.pop(ENV_OVERRIDE_VERSIONS, None)
    print(
        "[family_policy_cache] miss "
        f"family={family_name} version={bundle_version} sample={sample_index} "
        f"mode={evaluation_mode}"
    )
    with _temporary_env(env_updates):
        summary = run_sample(
            sample_index=sample_index,
            label=label,
            parent_output_dir=parent_output_dir,
        )
    result = _extract_run_metrics(summary)
    result.update(
        {
            "evaluation_cache_hit": False,
            "evaluation_cache_key": cache_key,
            "evaluation_context": cache_context,
            "evaluation_bundle_version": bundle_version,
        }
    )
    store.set_cached_evaluation(
        family_name,
        cache_key=cache_key,
        context=cache_context,
        result=result,
    )
    return result


def _evaluate_inline_promotion_gate(
    *,
    trigger_baseline: Mapping[str, Any],
    trigger_candidate: Mapping[str, Any],
    prior_success_baseline: Sequence[Mapping[str, Any]],
    prior_success_candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def _evaluation_is_valid(sample: Mapping[str, Any]) -> bool:
        sample_status = str(sample.get("sample_status") or "").strip()
        if not sample_status:
            return False
        returncode = sample.get("returncode")
        if returncode is None:
            return True
        try:
            return int(returncode) == 0
        except Exception:
            return False

    trigger_baseline_summary = summarize_sample_metrics(trigger_baseline)
    trigger_candidate_summary = summarize_sample_metrics(trigger_candidate)
    trigger_baseline_valid = _evaluation_is_valid(trigger_baseline)
    trigger_candidate_valid = _evaluation_is_valid(trigger_candidate)
    trigger_improved = (
        trigger_baseline_valid
        and trigger_candidate_valid
        and (
            int(trigger_candidate_summary["material_improvement_score"])
            > int(trigger_baseline_summary["material_improvement_score"])
        )
    )
    trigger_not_regressed = (
        trigger_baseline_valid
        and trigger_candidate_valid
        and int(trigger_candidate_summary["correct_completed"])
        >= int(trigger_baseline_summary["correct_completed"])
        and int(trigger_candidate_summary["wrong_completed"])
        <= int(trigger_baseline_summary["wrong_completed"])
        and int(trigger_candidate_summary["dangerous_overreach_count"])
        <= int(trigger_baseline_summary["dangerous_overreach_count"])
        and int(trigger_candidate_summary["material_improvement_score"])
        >= int(trigger_baseline_summary["material_improvement_score"])
    )
    trigger_soft_improved = (
        trigger_baseline_valid
        and trigger_candidate_valid
        and (
            int(trigger_candidate_summary["correct_completed"])
            > int(trigger_baseline_summary["correct_completed"])
            or int(trigger_candidate_summary["wrong_completed"])
            < int(trigger_baseline_summary["wrong_completed"])
            or int(trigger_candidate_summary["dangerous_overreach_count"])
            < int(trigger_baseline_summary["dangerous_overreach_count"])
            or int(trigger_candidate_summary["material_improvement_score"])
            > int(trigger_baseline_summary["material_improvement_score"])
        )
    )
    prior_success_guard_present = bool(prior_success_baseline and prior_success_candidate)
    prior_success_evaluation_valid = (
        prior_success_guard_present
        and _evaluation_is_valid(prior_success_baseline[0])
        and _evaluation_is_valid(prior_success_candidate[0])
    )
    regression_guard_available = prior_success_guard_present and prior_success_evaluation_valid
    prior_success_not_regressed = True
    if regression_guard_available:
        baseline_summary = summarize_sample_metrics(prior_success_baseline[0])
        candidate_summary = summarize_sample_metrics(prior_success_candidate[0])
        prior_success_not_regressed = (
            int(candidate_summary["correct_completed"]) >= int(baseline_summary["correct_completed"])
            and int(candidate_summary["wrong_completed"]) <= int(baseline_summary["wrong_completed"])
            and int(candidate_summary["dangerous_overreach_count"])
            <= int(baseline_summary["dangerous_overreach_count"])
        )
    gate_mode = _promotion_gate_mode()
    if gate_mode == "soft_improvement":
        trigger_gate_passed = trigger_soft_improved
    elif gate_mode == "non_regression":
        trigger_gate_passed = trigger_not_regressed
    else:
        trigger_gate_passed = trigger_improved
    promote = trigger_gate_passed and regression_guard_available and prior_success_not_regressed
    reasons: list[str] = []
    if not trigger_baseline_valid:
        reasons.append("trigger_baseline_invalid")
    if not trigger_candidate_valid:
        reasons.append("trigger_evaluation_invalid")
    if not trigger_gate_passed:
        reasons.append("trigger_failure_not_improved")
    if not prior_success_guard_present:
        reasons.append("awaiting_prior_trusted_success")
    elif not prior_success_evaluation_valid:
        reasons.append("prior_success_evaluation_invalid")
    elif not prior_success_not_regressed:
        reasons.append("prior_success_regressed")
    return {
        "promote": promote,
        "gate_checks": {
            "trigger_baseline_valid": trigger_baseline_valid,
            "trigger_candidate_valid": trigger_candidate_valid,
            "trigger_improved": trigger_improved,
            "trigger_not_regressed": trigger_not_regressed,
            "trigger_soft_improved": trigger_soft_improved,
            "trigger_gate_passed": trigger_gate_passed,
            "regression_guard_present": prior_success_guard_present,
            "regression_guard_available": regression_guard_available,
            "prior_success_evaluation_valid": prior_success_evaluation_valid,
            "prior_success_not_regressed": prior_success_not_regressed,
        },
        "gate_mode": gate_mode,
        "reasons": reasons,
        "baseline_summary": {
            "trigger": trigger_baseline_summary,
            "prior_success": summarize_sample_metrics(prior_success_baseline[0])
            if regression_guard_available
            else {},
        },
        "candidate_summary": {
            "trigger": trigger_candidate_summary,
            "prior_success": summarize_sample_metrics(prior_success_candidate[0])
            if regression_guard_available
            else {},
        },
    }


def _evaluate_candidate(
    *,
    family_name: str,
    candidate_version: str,
    promote: bool,
    store_path: Path,
    label_prefix: str,
    trigger_baseline_summary: Mapping[str, Any],
    parent_output_dir: Path | None = None,
) -> dict[str, Any]:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    pending_candidates = {
        item["candidate_version"]: item for item in store.get_pending_candidates(family_name)
    }
    candidate_payload = pending_candidates.get(candidate_version)
    if candidate_payload is None:
        raise KeyError(f"pending_candidate_not_found:{family_name}:{candidate_version}")
    active_version = store.get_active_version(family_name)
    trigger_sample = str(
        (candidate_payload.get("trigger_context") or {}).get("sample_index") or ""
    ).strip()
    trigger_baseline = _extract_run_metrics(dict(trigger_baseline_summary))
    prior_success_baseline: list[dict[str, Any]] = []
    prior_success_candidate: list[dict[str, Any]] = []
    prior_success_samples: list[str] = []
    prior_success_baseline_reused = False
    success_bank = store.get_trusted_success_bank(family_name)
    bank_metadata = store.get_trusted_success_bank_metadata(family_name)
    expected_bank_context = _build_success_bank_context(
        family_name=family_name,
        active_version=active_version,
    )
    bank_context = bank_metadata.get("evaluation_context") or {}
    if isinstance(bank_context, dict) and version_reuse_compatible(
        str(bank_metadata.get("source_version") or "").strip(),
        active_version,
    ):
        if str(bank_context.get("source_version") or "").strip() and (
            str(bank_context.get("source_version") or "").strip() != active_version
        ):
            bank_context = {
                **dict(bank_context),
                "source_version": active_version,
            }
        if _context_contains(
            expected_bank_context,
            bank_context,
        ):
            prior_success_samples = [
                sample_id
                for sample_id in success_bank
                if sample_id and sample_id != trigger_sample
            ][:MAX_PRIOR_SUCCESS_SAMPLES]
    if not prior_success_samples:
        evaluation_payload = {
            "family_name": family_name,
            "active_version": active_version,
            "candidate_version": candidate_version,
            "evaluation_mode": PROMOTION_EVALUATION_MODE,
            "promotion_evaluation_contract_version": PROMOTION_EVALUATION_CONTRACT_VERSION,
            "trigger": {
                "baseline": trigger_baseline,
                "candidate": {},
            },
            "prior_success": {
                "sample_ids": [],
                "baseline": [],
                "candidate": [],
                "trusted_success_bank_reused": False,
            },
            "gate_stage": "inline",
            "evaluation_stats": {
                "cache_hits": 0,
                "cache_misses": 0,
                "executed_runs": 0,
                "total_evaluation_requests": 0,
                "potential_max_requests": 3,
                "prior_success_evaluated": 0,
                "prior_success_skipped": MAX_PRIOR_SUCCESS_SAMPLES,
                "regression_guard_available": False,
                "stage_reached": "awaiting_prior_success",
            },
            "promotion_gate": {
                "promote": False,
                "gate_checks": {
                    "trigger_baseline_valid": True,
                    "trigger_candidate_valid": False,
                    "trigger_improved": False,
                    "regression_guard_present": False,
                    "regression_guard_available": False,
                    "prior_success_evaluation_valid": False,
                    "prior_success_not_regressed": False,
                },
                "reasons": ["awaiting_prior_trusted_success"],
                "baseline_summary": {
                    "trigger": summarize_sample_metrics(trigger_baseline),
                    "prior_success": {},
                },
                "candidate_summary": {
                    "trigger": {},
                    "prior_success": {},
                },
            },
        }
        store.update_candidate_evaluation(
            family_name,
            candidate_version=candidate_version,
            evaluation_results=evaluation_payload,
        )
        return evaluation_payload

    trigger_candidate = _run_sample_with_policy(
        sample_index=trigger_sample,
        label=f"{label_prefix}_candidate_trigger",
        family_name=family_name,
        store_path=store_path,
        promotion_enabled=False,
        override_version=candidate_version,
        parent_output_dir=parent_output_dir,
    )
    gate_result = _evaluate_inline_promotion_gate(
        trigger_baseline=trigger_baseline,
        trigger_candidate=trigger_candidate,
        prior_success_baseline=[],
        prior_success_candidate=[],
    )
    all_results = [trigger_candidate]
    if not gate_result["gate_checks"]["trigger_gate_passed"]:
        reasons = [
            reason
            for reason in (gate_result.get("reasons") or [])
            if reason != "awaiting_prior_trusted_success"
        ]
        if not reasons:
            reasons = ["trigger_failure_not_improved"]
        gate_result = {
            **gate_result,
            "reasons": reasons,
            "gate_checks": {
                **dict(gate_result.get("gate_checks") or {}),
                "regression_guard_present": bool(prior_success_samples),
                "regression_guard_available": bool(prior_success_samples),
            },
        }
    elif gate_result["gate_checks"]["trigger_gate_passed"]:
        sample_id = prior_success_samples[0]
        cached_success_summary = (
            (bank_metadata.get("evaluation_results") or {}).get("run_summary")
            if isinstance(bank_metadata, Mapping)
            else None
        )
        if (
            isinstance(cached_success_summary, Mapping)
            and str(cached_success_summary.get("sample_index") or "").strip() == sample_id
        ):
            prior_success_baseline.append(
                _extract_run_metrics(dict(cached_success_summary))
            )
            prior_success_baseline_reused = True
        else:
            prior_success_baseline.append(
                _run_sample_with_policy(
                    sample_index=sample_id,
                    label=f"{label_prefix}_baseline_success",
                    family_name=family_name,
                    store_path=store_path,
                    promotion_enabled=False,
                    override_version=active_version,
                    parent_output_dir=parent_output_dir,
                )
            )
        prior_success_candidate.append(
            _run_sample_with_policy(
                sample_index=sample_id,
                label=f"{label_prefix}_candidate_success",
                family_name=family_name,
                store_path=store_path,
                promotion_enabled=False,
                override_version=candidate_version,
                parent_output_dir=parent_output_dir,
            )
        )
        gate_result = _evaluate_inline_promotion_gate(
            trigger_baseline=trigger_baseline,
            trigger_candidate=trigger_candidate,
            prior_success_baseline=prior_success_baseline,
            prior_success_candidate=prior_success_candidate,
        )
        all_results.extend(prior_success_baseline)
        all_results.extend(prior_success_candidate)
    evaluation_results = {
        "family_name": family_name,
        "active_version": active_version,
        "candidate_version": candidate_version,
        "evaluation_mode": PROMOTION_EVALUATION_MODE,
        "promotion_evaluation_contract_version": PROMOTION_EVALUATION_CONTRACT_VERSION,
        "trigger": {
            "baseline": trigger_baseline,
            "candidate": trigger_candidate,
        },
            "prior_success": {
                "sample_ids": prior_success_samples if prior_success_candidate else [],
                "baseline": prior_success_baseline,
                "candidate": prior_success_candidate,
                "trusted_success_bank_reused": bool(prior_success_candidate),
                "baseline_reused_from_bank_metadata": prior_success_baseline_reused,
            },
        }
    cache_hits = sum(bool(item.get("evaluation_cache_hit")) for item in all_results)
    cache_misses = sum(not bool(item.get("evaluation_cache_hit")) for item in all_results)
    total_evaluation_requests = len(all_results)
    evaluation_payload = {
        **evaluation_results,
        "gate_stage": "inline",
        "evaluation_stats": {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "executed_runs": cache_misses,
            "total_evaluation_requests": total_evaluation_requests,
            "potential_max_requests": 3,
            "prior_success_evaluated": len(prior_success_candidate),
            "prior_success_skipped": max(0, MAX_PRIOR_SUCCESS_SAMPLES - len(prior_success_candidate)),
            "regression_guard_available": bool(prior_success_candidate),
            "prior_success_baseline_reused": prior_success_baseline_reused,
            "stage_reached": (
                "trigger_only_rejected"
                if not prior_success_candidate and not gate_result["promote"]
                else "inline"
            ),
        },
        "promotion_gate": gate_result,
    }
    store.update_candidate_evaluation(
        family_name,
        candidate_version=candidate_version,
        evaluation_results=evaluation_payload,
    )
    if promote and gate_result["promote"]:
        store.promote_candidate(
            family_name,
            candidate_version=candidate_version,
            evaluation_results=evaluation_payload,
            promotion_reason="inline_trigger_and_regression_checks_passed",
        )
    elif promote and "awaiting_prior_trusted_success" not in set(gate_result.get("reasons") or []):
        store.reject_candidate(
            family_name,
            candidate_version=candidate_version,
            evaluation_results=evaluation_payload,
            rejection_reason=",".join(gate_result.get("reasons") or ["inline_promotion_gate_failed"]),
        )
    return evaluation_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--store-path", default="")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--trigger-baseline-summary-path", default="")
    parser.add_argument("--parent-output-dir", default="")
    args = parser.parse_args()

    store_path = (
        Path(args.store_path).resolve()
        if args.store_path
        else (PROJECT_ROOT / "outputs" / f"{args.label}_family_policy_store").resolve()
    )
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )

    trigger_baseline_by_sample: dict[str, dict[str, Any]] = {}
    if args.trigger_baseline_summary_path:
        summary_path = Path(args.trigger_baseline_summary_path)
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        if summary_path.exists():
            loaded = _load_json(summary_path)
            if isinstance(loaded, Mapping):
                sample_key = str(loaded.get("sample_index") or "").strip()
                if sample_key:
                    trigger_baseline_by_sample[sample_key] = dict(loaded)
    parent_output_dir = (
        Path(args.parent_output_dir).resolve() if args.parent_output_dir else None
    )

    run_summaries: list[dict[str, Any]] = []
    for sample_index in args.samples:
        run_summary = trigger_baseline_by_sample.get(str(sample_index).strip())
        if run_summary is None:
            run_summary = _run_sample_with_policy(
                sample_index=str(sample_index),
                label=args.label,
                family_name=args.family,
                store_path=store_path,
                promotion_enabled=args.promote,
                override_version=None,
                parent_output_dir=parent_output_dir,
            )
        else:
            run_summary = dict(run_summary)
            run_summary.setdefault("evaluation_bundle_version", store.get_active_version(args.family))
            run_summary.setdefault("evaluation_cache_hit", True)
            run_summary.setdefault("evaluation_mode", PROMOTION_EVALUATION_MODE)
        run_summaries.append(run_summary)
        synthesized_candidate_event = _maybe_create_candidate_from_run_summary(
            family_name=args.family,
            run_summary=run_summary,
            store_path=store_path,
        )
        if synthesized_candidate_event is not None:
            run_summaries.append(synthesized_candidate_event)
            store = build_family_policy_store(
                baseline_bundles=get_baseline_reusable_family_policy_bundles(),
                store_path=store_path,
            )
        for candidate_payload in store.get_pending_candidates(args.family):
            if (
                str((candidate_payload.get("trigger_context") or {}).get("sample_index") or "").strip()
                != str(sample_index).strip()
            ):
                continue
            evaluation_payload = _evaluate_candidate(
                family_name=args.family,
                candidate_version=str(candidate_payload["candidate_version"]),
                promote=args.promote,
                store_path=store_path,
                label_prefix=f"{args.label}_{sample_index}",
                trigger_baseline_summary=run_summary,
                parent_output_dir=parent_output_dir,
            )
            run_summaries.append(
                {
                    "event": "family_policy_candidate_evaluated",
                    "family_name": args.family,
                    "candidate_version": candidate_payload["candidate_version"],
                    "store_path": str(store_path),
                    "evaluation": evaluation_payload,
                }
            )
            store = build_family_policy_store(
                baseline_bundles=get_baseline_reusable_family_policy_bundles(),
                store_path=store_path,
            )

    summary_text = json.dumps(run_summaries, indent=2, sort_keys=True)
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
