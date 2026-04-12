# scripts/run_all_with_servers.py
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import kg_sparql_server
import yaml
from src.sage.family_policy_evolution import (
    build_family_policy_store,
    build_success_plan_archetype,
    merge_success_plan_archetypes,
    version_reuse_compatible,
)
from src.sage.reusable_tool_families import get_baseline_reusable_family_policy_bundles

CONFIG_PATHS = [
    # "configs/assignments/experiments/llama_31_8b_instruct/instance/os_interaction/instance/standard.yaml",
    "configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml",
    # "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml",
    
]
CONFIG_PATHS_ENV = "LIFELONG_CONFIG_PATHS"

SPARQL_ENDPOINT = "http://127.0.0.1:3001/kb/sparql"

# Fuseki server (serve-only; no dump loading)
FUSEKI_CONTAINER = os.getenv("LIFELONG_KG_CONTAINER_NAME", "lifelong_fuseki")
FUSEKI_IMAGE = os.getenv("LIFELONG_FUSEKI_IMAGE", "stain/jena-fuseki:latest")
FUSEKI_PLATFORM = os.getenv("LIFELONG_KG_DOCKER_PLATFORM", "linux/amd64")  # ARM host typically needs amd64

FUSEKI_HOST_PORT = int(os.getenv("LIFELONG_FUSEKI_PORT", "3001"))
FUSEKI_DATASET = os.getenv("LIFELONG_FUSEKI_DATASET", "kb")  # path /kb and databases/kb
KG_DATA_DIR_ENV = "LIFELONG_KG_DATA_DIR"
CLIENT_WALL_TIMEOUT_S = int(os.getenv("LIFELONG_CLIENT_TIMEOUT_S", "1800"))
CLIENT_IDLE_TIMEOUT_S = int(os.getenv("LIFELONG_CLIENT_IDLE_TIMEOUT_S", "600"))
CLIENT_WATCHDOG_POLL_S = float(os.getenv("LIFELONG_CLIENT_WATCHDOG_POLL_S", "5"))
ENABLE_SAGE_AGENT = os.getenv("ENABLE_SAGE_AGENT") == "1"
ENABLE_STANDARD_FAMILY_EVOLUTION = (
    os.getenv("SAGE_ENABLE_STANDARD_FAMILY_EVOLUTION") == "1"
)
FAMILY_EVOLUTION_ENV = "SAGE_ENABLE_FAMILY_POLICY_EVOLUTION"
FAMILY_PROMOTION_ENV = "SAGE_ENABLE_FAMILY_POLICY_PROMOTION"
FAMILY_ENABLED_FAMILIES_ENV = "SAGE_FAMILY_POLICY_EVOLUTION_FAMILIES"
FAMILY_STORE_PATH_ENV = "SAGE_FAMILY_POLICY_STORE_PATH"
PERSISTENT_FAMILY_STORE_ENV = "SAGE_PERSISTENT_FAMILY_POLICY_STORE"
RESET_FAMILY_STORE_ENV = "SAGE_RESET_FAMILY_POLICY_STORE"
TOOL_EVOLUTION_SIGNAL_ENV = "SAGE_RECORD_TOOL_EVOLUTION_SIGNAL"
INLINE_FAMILY_EVOLUTION_FAMILIES_ENV = "SAGE_INLINE_FAMILY_EVOLUTION_FAMILIES"
INLINE_FAMILY_EVOLUTION_BUDGET_S = int(
    os.getenv("SAGE_STANDARD_FAMILY_EVOLUTION_BUDGET_S", "360")
)
INLINE_TRUSTED_SUCCESS_BANK_SIZE = 1
DEFAULT_FAMILY_REGRESSION_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "evaluation"
    / "knowledge_graph_family_regression.json"
)
_RESETTED_FAMILY_STORE_PATHS: set[str] = set()


@dataclass(frozen=True)
class ClientWatchdogResult:
    exit_code: int
    timed_out_reason: str | None = None
    current_session: dict[str, object] | None = None


def _family_evolution_enabled_for(family_name: str) -> bool:
    configured = {
        str(item or "").strip()
        for item in os.getenv(FAMILY_ENABLED_FAMILIES_ENV, "").split(",")
        if str(item or "").strip()
    }
    if not configured:
        return True
    return str(family_name or "").strip() in configured


def _inline_family_evolution_allowed_for(family_name: str) -> bool:
    configured = {
        str(item or "").strip()
        for item in os.getenv(INLINE_FAMILY_EVOLUTION_FAMILIES_ENV, "").split(",")
        if str(item or "").strip()
    }
    if not configured:
        return True
    return str(family_name or "").strip() in configured


def _persistent_family_store_enabled() -> bool:
    configured = str(os.getenv(PERSISTENT_FAMILY_STORE_ENV) or "").strip()
    if configured:
        return configured == "1"
    return ENABLE_STANDARD_FAMILY_EVOLUTION or (
        os.getenv("SAGE_ENABLE_STANDARD_FAMILY_EVOLUTION") == "1"
    )


def _default_persistent_family_store_path(repo_root: Path, task_name: str) -> Path:
    return (
        repo_root
        / "outputs"
        / "persistent_family_policy_store"
        / str(task_name or "default").strip()
    ).resolve()


def _resolve_family_policy_store_path(
    *,
    repo_root: Path,
    aggregate_output_dir: Path,
    task_name: str,
) -> Path:
    configured_store_path = str(os.getenv(FAMILY_STORE_PATH_ENV) or "").strip()
    if configured_store_path:
        store_path = Path(configured_store_path)
        if not store_path.is_absolute():
            store_path = repo_root / store_path
        return store_path.resolve()
    if _persistent_family_store_enabled():
        return _default_persistent_family_store_path(repo_root, task_name)
    return (aggregate_output_dir / "family_policy_store").resolve()


def _maybe_reset_family_policy_store(store_path: Path) -> None:
    if os.getenv(RESET_FAMILY_STORE_ENV) != "1":
        return
    store_key = str(store_path.resolve())
    if store_key in _RESETTED_FAMILY_STORE_PATHS:
        return
    shutil.rmtree(store_path, ignore_errors=True)
    _RESETTED_FAMILY_STORE_PATHS.add(store_key)


def _load_sample_order_from_config(config_path: Path) -> list[str] | None:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    sample_order = ((payload.get("assignment_config") or {}).get("sample_order"))
    if not isinstance(sample_order, list):
        return None
    cleaned = [str(item or "").strip() for item in sample_order if str(item or "").strip()]
    return cleaned or None


def _write_single_sample_config(
    *,
    source_config_path: Path,
    sample_index: str,
    stem: str,
) -> Path:
    payload = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_config:{source_config_path}")
    assignment_config = payload.setdefault("assignment_config", {})
    assignment_config["sample_order"] = [str(sample_index)]
    temp_config_path = source_config_path.parent / f"{stem}.yaml"
    temp_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return temp_config_path


def _delete_file_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _load_session_for_sample(output_dir: Path, sample_index: str) -> dict[str, object] | None:
    runs_path = output_dir / "runs.json"
    try:
        if not runs_path.exists():
            return None
        payload = json.loads(runs_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    for item in reversed(payload):
        if not isinstance(item, dict):
            continue
        if str(item.get("sample_index") or "").strip() == str(sample_index or "").strip():
            return item
    return None


def _latest_selected_family_for_sample(output_dir: Path, sample_index: str) -> str | None:
    generated_tools_path = output_dir / "generated_tools.log"
    if not generated_tools_path.exists():
        return None
    selected_family: str | None = None
    fallback_family: str | None = None
    for line in generated_tools_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("sample_index") or "").strip() != str(sample_index or "").strip():
            continue
        cleaned_family = str(payload.get("selected_family") or "").strip()
        if not cleaned_family:
            continue
        if payload.get("event") == "sage_attempt_decision_finalized":
            selected_family = cleaned_family
        elif payload.get("event") == "sage_attempt_decision" and not selected_family:
            fallback_family = cleaned_family
    return selected_family or fallback_family


def _cleanup_family_evolution_artifacts(repo_root: Path, label: str) -> None:
    config_dir = (
        repo_root
        / "configs"
        / "assignments"
        / "experiments"
        / "llama_31_8b_instruct"
        / "instance"
        / "knowledge_graph"
        / "instance"
    )
    for path in config_dir.glob(f"sage_batch_{label}*.yaml"):
        _delete_file_if_exists(path)
    outputs_root = repo_root / "outputs"
    for run_root in outputs_root.glob("run_all_*"):
        if not run_root.is_dir():
            continue
        kg_dir = run_root / "knowledge_graph"
        if not kg_dir.is_dir():
            continue
        matched = False
        for child in kg_dir.iterdir():
            if child.is_dir() and child.name.startswith(f"sage_batch_{label}"):
                shutil.rmtree(child, ignore_errors=True)
                matched = True
        if matched:
            try:
                next(kg_dir.iterdir())
            except StopIteration:
                shutil.rmtree(run_root, ignore_errors=True)


def _append_generated_tool_event(log_path: Path | None, payload: dict[str, object]) -> None:
    if not log_path:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def _load_family_policy_store_summary(store_path: Path, family_name: str) -> dict[str, object]:
    family_path = store_path / f"{family_name}.json"
    if not family_path.exists():
        return {
            "family_name": family_name,
            "active_version": "",
            "pending_candidates": 0,
            "candidate_versions": [],
        }
    try:
        payload = json.loads(family_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "family_name": family_name,
            "active_version": "",
            "pending_candidates": 0,
            "candidate_versions": [],
        }
    versions = payload.get("versions") or {}
    if not isinstance(versions, dict):
        versions = {}
    candidate_versions = [
        str(version_name)
        for version_name, version_payload in versions.items()
        if isinstance(version_payload, dict)
        and str(version_payload.get("status") or "").strip() == "candidate"
    ]
    return {
        "family_name": family_name,
        "active_version": str(payload.get("active_version") or "").strip(),
        "pending_candidates": len(candidate_versions),
        "candidate_versions": candidate_versions,
    }


def _load_family_policy_store_payload(store_path: Path, family_name: str) -> dict[str, object]:
    family_path = store_path / f"{family_name}.json"
    if not family_path.exists():
        return {}
    try:
        payload = json.loads(family_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_family_regression_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    families = payload.get("families") or {}
    return families if isinstance(families, dict) else {}


def _latest_attempt_decision_for_sample(output_dir: Path, sample_index: str) -> dict[str, object]:
    generated_tools_path = output_dir / "generated_tools.log"
    if not generated_tools_path.exists():
        return {}
    latest: dict[str, object] = {}
    fallback: dict[str, object] = {}
    for line in generated_tools_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("sample_index") or "").strip() != str(sample_index or "").strip():
            continue
        if payload.get("event") == "sage_attempt_decision":
            fallback = payload
        if payload.get("event") == "sage_attempt_decision_finalized":
            latest = payload
    return latest or fallback


def _latest_macro_solver_review_for_sample(
    output_dir: Path,
    sample_index: str,
) -> dict[str, object]:
    generated_tools_path = output_dir / "generated_tools.log"
    if not generated_tools_path.exists():
        return {}
    latest: dict[str, object] = {}
    for line in generated_tools_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("sample_index") or "").strip() != str(sample_index or "").strip():
            continue
        if payload.get("event") == "sage_macro_solver_review":
            latest = payload
    return latest


def _latest_query_plan_for_sample(output_dir: Path, sample_index: str) -> dict[str, object] | None:
    generated_tools_path = output_dir / "generated_tools.log"
    if not generated_tools_path.exists():
        return None
    latest_plan_path = ""
    for line in generated_tools_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("sample_index") or "").strip() != str(sample_index or "").strip():
            continue
        if payload.get("event") not in {"sage_query_artifact_saved", "sage_query_plan_generated"}:
            continue
        plan_path = str(payload.get("plan_artifact_path") or "").strip()
        if plan_path:
            latest_plan_path = plan_path
    if not latest_plan_path:
        return None
    try:
        loaded = json.loads(Path(latest_plan_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(loaded) if isinstance(loaded, dict) else None


def _build_inline_trigger_baseline_summary(
    *,
    output_dir: Path,
    sample_index: str,
    session_record: dict[str, object],
) -> dict[str, object]:
    evaluation_record = session_record.get("evaluation_record")
    evaluation_outcome = ""
    if isinstance(evaluation_record, dict):
        evaluation_outcome = str(evaluation_record.get("outcome") or "").strip()
    decision_payload = _latest_attempt_decision_for_sample(output_dir, sample_index)
    dangerous_overreach_count = int(bool(decision_payload.get("dangerous_overreach")))
    return {
        "sample_index": str(sample_index or "").strip(),
        "sample_status": str(session_record.get("sample_status") or "").strip(),
        "evaluation_outcome": evaluation_outcome,
        "dangerous_overreach_count": dangerous_overreach_count,
        "run_dir": str(output_dir),
    }


def _write_trigger_baseline_summary(
    *,
    output_dir: Path,
    sample_index: str,
    session_record: dict[str, object],
) -> Path:
    summary = _build_inline_trigger_baseline_summary(
        output_dir=output_dir,
        sample_index=sample_index,
        session_record=session_record,
    )
    path = output_dir / f"family_policy_trigger_baseline_{sample_index}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_trusted_success_bank_context(
    *,
    family_name: str,
    active_version: str,
) -> dict[str, object]:
    # Keep the boundary runner's harvested successes compatible with the inline
    # promotion harness gate. Otherwise the harness treats a warm family as cold.
    from scripts.run_kg_family_policy_evolution import _build_success_bank_context

    return _build_success_bank_context(
        family_name=family_name,
        active_version=active_version,
    )


def _record_trusted_family_success(
    *,
    output_dir: Path,
    store_path: Path,
    family_name: str,
    sample_index: str,
    active_version: str,
    session_record: dict[str, object] | None = None,
    progress_log_path: Path | None = None,
) -> list[str]:
    decision_payload = _latest_attempt_decision_for_sample(output_dir, sample_index)
    if str(decision_payload.get("selected_family") or "").strip() != str(family_name or "").strip():
        return []
    if str(decision_payload.get("family_bundle_version") or "").strip() != str(active_version or "").strip():
        return []
    trust_contract = decision_payload.get("trust_contract") or {}
    materialization_allowed = bool(
        (trust_contract.get("materialization_allowed") if isinstance(trust_contract, dict) else False)
        or decision_payload.get("materialization_allowed")
    )
    dangerous_overreach = bool(
        (trust_contract.get("dangerous_overreach") if isinstance(trust_contract, dict) else False)
        or decision_payload.get("dangerous_overreach")
    )
    tool_result_status = str(decision_payload.get("tool_result_status") or "").strip().lower()
    tool_result_solves_task = bool(decision_payload.get("tool_result_solves_task"))
    tool_result_trusted = bool(
        decision_payload.get("tool_result_trusted_for_materialization")
    )
    macro_solver_review = _latest_macro_solver_review_for_sample(
        output_dir,
        sample_index,
    )
    solver_review_present = "accepted_final" in macro_solver_review
    solver_accepted_final = bool(macro_solver_review.get("accepted_final"))
    if (
        not materialization_allowed
        or dangerous_overreach
        or tool_result_status != "success"
        or not tool_result_solves_task
        or not tool_result_trusted
        or (solver_review_present and not solver_accepted_final)
    ):
        return []
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    metadata = store.get_trusted_success_bank_metadata(family_name)
    existing_ids: list[str] = []
    existing_archetypes: list[dict[str, object]] = []
    if version_reuse_compatible(
        str(metadata.get("source_version") or "").strip(),
        str(active_version or "").strip(),
    ):
        existing_ids = store.get_trusted_success_bank(family_name)
        evaluation_context = metadata.get("evaluation_context") or {}
        if isinstance(evaluation_context, dict):
            existing_archetypes = [
                dict(item)
                for item in (evaluation_context.get("success_plan_archetypes") or [])
                if isinstance(item, dict)
            ]
    updated_ids = [
        sample_id
        for sample_id in [*existing_ids, str(sample_index or "").strip()]
        if sample_id
    ]
    deduped: list[str] = []
    for sample_id in updated_ids:
        if sample_id not in deduped:
            deduped.append(sample_id)
    deduped = deduped[-INLINE_TRUSTED_SUCCESS_BANK_SIZE:]
    query_plan = _latest_query_plan_for_sample(output_dir, sample_index)
    success_archetypes = existing_archetypes
    if query_plan:
        archetype = build_success_plan_archetype(query_plan)
        if archetype:
            success_archetypes = merge_success_plan_archetypes(
                existing_archetypes,
                archetype,
            )
    evaluation_context = _build_trusted_success_bank_context(
        family_name=family_name,
        active_version=active_version,
    )
    if success_archetypes:
        evaluation_context["success_plan_archetypes"] = success_archetypes
    evaluation_results: dict[str, object] = {
        "source": "inline_standard_run_success_harvest",
        "sample_index": str(sample_index or "").strip(),
    }
    if isinstance(session_record, dict):
        evaluation_results["run_summary"] = _build_inline_trigger_baseline_summary(
            output_dir=output_dir,
            sample_index=sample_index,
            session_record=session_record,
        )
    store.set_trusted_success_bank(
        family_name,
        sample_ids=deduped,
        source_version=active_version,
        evaluation_results=evaluation_results,
        evaluation_context=evaluation_context,
    )
    _append_generated_tool_event(
        progress_log_path,
        {
            "event": "sage_family_policy_trusted_success_recorded",
            "family_name": family_name,
            "sample_index": str(sample_index or "").strip(),
            "active_version": active_version,
            "trusted_success_bank": deduped,
            "trusted_success_bank_ready": len(deduped) >= INLINE_TRUSTED_SUCCESS_BANK_SIZE,
            "success_plan_archetype_count": len(success_archetypes),
        },
    )
    return deduped


def _tool_evolution_phase_enabled() -> bool:
    configured = str(os.getenv(TOOL_EVOLUTION_SIGNAL_ENV) or "").strip()
    if configured:
        return configured == "1"
    if os.getenv(FAMILY_EVOLUTION_ENV) == "1":
        return True
    return os.getenv("SAGE_ENABLE_STANDARD_FAMILY_EVOLUTION") == "1"


def _is_trusted_tool_success(
    *,
    decision_payload: dict[str, object],
    macro_solver_review: dict[str, object],
) -> bool:
    trust_contract = decision_payload.get("trust_contract") or {}
    materialization_allowed = bool(
        (trust_contract.get("materialization_allowed") if isinstance(trust_contract, dict) else False)
        or decision_payload.get("materialization_allowed")
    )
    dangerous_overreach = bool(
        (trust_contract.get("dangerous_overreach") if isinstance(trust_contract, dict) else False)
        or decision_payload.get("dangerous_overreach")
    )
    tool_result_status = str(decision_payload.get("tool_result_status") or "").strip().lower()
    tool_result_solves_task = bool(decision_payload.get("tool_result_solves_task"))
    tool_result_trusted = bool(
        decision_payload.get("tool_result_trusted_for_materialization")
    )
    solver_review_present = "accepted_final" in macro_solver_review
    solver_accepted_final = bool(macro_solver_review.get("accepted_final"))
    return bool(
        materialization_allowed
        and not dangerous_overreach
        and tool_result_status == "success"
        and tool_result_solves_task
        and tool_result_trusted
        and (not solver_review_present or solver_accepted_final)
    )


def _normalize_tool_evolution_failure_labels(raw_reason: str) -> list[str]:
    cleaned = str(raw_reason or "").strip()
    if not cleaned:
        return []
    labels = [cleaned]
    leaf = cleaned.split(":")[-1].strip()
    if leaf and leaf not in labels:
        labels.append(leaf)
    return labels


def _record_tool_evolution_signal(
    *,
    output_dir: Path,
    store_path: Path,
    family_name: str,
    sample_index: str,
    active_version: str,
    session_record: dict[str, object],
    progress_log_path: Path | None = None,
) -> None:
    if not _tool_evolution_phase_enabled():
        return
    decision_payload = _latest_attempt_decision_for_sample(output_dir, sample_index)
    if str(decision_payload.get("selected_family") or "").strip() != str(family_name or "").strip():
        return
    if str(decision_payload.get("family_bundle_version") or "").strip() != str(active_version or "").strip():
        return
    query_plan = _latest_query_plan_for_sample(output_dir, sample_index)
    if not query_plan:
        return
    macro_solver_review = _latest_macro_solver_review_for_sample(
        output_dir,
        sample_index,
    )
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    context = store.get_tool_evolution_context(family_name)
    if not version_reuse_compatible(
        str(context.get("source_version") or "").strip(),
        str(active_version or "").strip(),
    ):
        preferred_patterns: list[dict[str, object]] = []
        avoid_patterns: list[dict[str, object]] = []
    else:
        preferred_patterns = [
            dict(item)
            for item in (context.get("preferred_patterns") or [])
            if isinstance(item, dict)
        ]
        avoid_patterns = [
            dict(item)
            for item in (context.get("avoid_patterns") or [])
            if isinstance(item, dict)
        ]

    sample_status = str(session_record.get("sample_status") or "").strip()
    evaluation_record = session_record.get("evaluation_record")
    evaluation_outcome = ""
    if isinstance(evaluation_record, dict):
        evaluation_outcome = str(evaluation_record.get("outcome") or "").strip()

    last_signal: dict[str, object] = {
        "sample_index": str(sample_index or "").strip(),
        "sample_status": sample_status,
        "evaluation_outcome": evaluation_outcome,
    }
    if sample_status == "completed" and evaluation_outcome == "correct":
        if not _is_trusted_tool_success(
            decision_payload=decision_payload,
            macro_solver_review=macro_solver_review,
        ):
            return
        preferred_patterns = merge_success_plan_archetypes(
            preferred_patterns,
            build_success_plan_archetype(query_plan),
        )
        last_signal["signal_type"] = "trusted_success"
    else:
        finish_reason = str(session_record.get("finish_reason") or "").strip()
        failure_labels: list[str] = []
        bypass_prefix = "sage_tool_failure_bypassed:"
        if bypass_prefix in finish_reason:
            failure_labels.extend(
                _normalize_tool_evolution_failure_labels(
                    finish_reason.split(bypass_prefix, 1)[1]
                )
            )
        elif sample_status == "completed" and evaluation_outcome and evaluation_outcome != "correct":
            if _is_trusted_tool_success(
                decision_payload=decision_payload,
                macro_solver_review=macro_solver_review,
            ):
                failure_labels.append("trusted_incorrect_completion")
            else:
                failure_labels.append("completed_incorrect")
        failure_labels = list(dict.fromkeys(label for label in failure_labels if label))
        if not failure_labels:
            return
        failure_pattern = build_success_plan_archetype(query_plan)
        if failure_pattern:
            failure_pattern = {
                **failure_pattern,
                "failure_labels": failure_labels,
            }
            existing_index = {
                str(item.get("pattern_signature") or "").strip(): idx
                for idx, item in enumerate(avoid_patterns)
                if str(item.get("pattern_signature") or "").strip()
            }
            signature = str(failure_pattern.get("pattern_signature") or "").strip()
            if signature and signature in existing_index:
                prior = dict(avoid_patterns[existing_index[signature]])
                prior_labels = [
                    str(label or "").strip()
                    for label in (prior.get("failure_labels") or [])
                    if str(label or "").strip()
                ]
                prior["failure_labels"] = list(
                    dict.fromkeys([*prior_labels, *failure_labels])
                )
                avoid_patterns[existing_index[signature]] = prior
            elif signature:
                avoid_patterns.append(failure_pattern)
                avoid_patterns = avoid_patterns[-4:]
        last_signal["signal_type"] = "clean_failure"
        last_signal["failure_labels"] = failure_labels

    store.set_tool_evolution_context(
        family_name,
        source_version=active_version,
        preferred_patterns=preferred_patterns,
        avoid_patterns=avoid_patterns,
        last_signal=last_signal,
    )
    _append_generated_tool_event(
        progress_log_path,
        {
            "event": "sage_tool_evolution_signal_recorded",
            "family_name": family_name,
            "sample_index": str(sample_index or "").strip(),
            "active_version": active_version,
            "preferred_pattern_count": len(preferred_patterns),
            "avoid_pattern_count": len(avoid_patterns),
            **last_signal,
        },
    )


def _oldest_pending_candidate_for_family(
    *,
    store_path: Path,
    family_name: str,
) -> dict[str, object] | None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    pending = store.get_pending_candidates(family_name)
    if not pending:
        return None
    return dict(pending[0])


def _run_between_sample_family_evolution(
    *,
    repo_root: Path,
    family_name: str,
    sample_index: str,
    store_path: Path,
    label: str,
    progress_log_path: Path | None = None,
    trigger_baseline_summary_path: Path | None = None,
    inline_budget_s: int = INLINE_FAMILY_EVOLUTION_BUDGET_S,
    parent_output_dir: Path | None = None,
    trigger_source: str = "sample_failure_or_wrong_completion",
    boundary_sample_index: str | None = None,
) -> int:
    command = [
        sys.executable,
        "scripts/run_kg_family_policy_evolution.py",
        "--family",
        family_name,
        "--samples",
        str(sample_index),
        "--label",
        label,
        "--store-path",
        str(store_path),
        "--promote",
    ]
    if trigger_baseline_summary_path is not None:
        command.extend(
            [
                "--trigger-baseline-summary-path",
                str(trigger_baseline_summary_path),
            ]
        )
    if parent_output_dir is not None:
        command.extend(
            [
                "--parent-output-dir",
                str(parent_output_dir),
            ]
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env["ENABLE_SAGE_AGENT"] = "1"
    env[FAMILY_EVOLUTION_ENV] = "1"
    env[FAMILY_PROMOTION_ENV] = "1"
    env[FAMILY_STORE_PATH_ENV] = str(store_path)
    env[FAMILY_ENABLED_FAMILIES_ENV] = family_name
    env["SAGE_ENABLE_STANDARD_FAMILY_EVOLUTION"] = "0"
    before_summary = _load_family_policy_store_summary(store_path, family_name)
    start_ts = time.monotonic()
    print(
        "[run_all_with_servers] Family evolution start "
        f"family={family_name} sample={sample_index} "
        f"trigger_source={trigger_source} "
        f"boundary_sample={boundary_sample_index or sample_index} "
        f"active_version={before_summary.get('active_version') or 'n/a'} "
        f"pending_candidates={before_summary.get('pending_candidates')}"
    )
    _append_generated_tool_event(
        progress_log_path,
        {
            "event": "sage_family_policy_evaluation_started",
            "family_name": family_name,
            "sample_index": str(sample_index),
            "label": label,
            "store_path": str(store_path),
            "inline_budget_s": int(inline_budget_s),
            "trigger_baseline_reused": bool(trigger_baseline_summary_path),
            "trigger_source": str(trigger_source or "").strip(),
            "boundary_sample_index": str(boundary_sample_index or sample_index),
            "parent_output_dir": str(parent_output_dir) if parent_output_dir is not None else "",
            **before_summary,
        },
    )
    fd, temp_log_name = tempfile.mkstemp(
        prefix=f"family_policy_evolution_{family_name}_{sample_index}_",
        suffix=".log",
    )
    os.close(fd)
    temp_log_path = Path(temp_log_name)
    return_code = -1
    try:
        with temp_log_path.open("w", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                text=True,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_heartbeat_s = 0.0
            while True:
                return_code = proc.poll()
                if return_code is not None:
                    break
                time.sleep(5)
                elapsed_s = time.monotonic() - start_ts
                if inline_budget_s > 0 and elapsed_s > float(inline_budget_s):
                    print(
                        "[run_all_with_servers] Family evolution budget exceeded "
                        f"family={family_name} sample={sample_index} "
                        f"budget_s={int(inline_budget_s)} elapsed_s={int(elapsed_s)}"
                    )
                    _append_generated_tool_event(
                        progress_log_path,
                        {
                            "event": "sage_family_policy_evaluation_budget_exceeded",
                            "family_name": family_name,
                            "sample_index": str(sample_index),
                            "label": label,
                            "budget_s": int(inline_budget_s),
                            "elapsed_s": int(elapsed_s),
                        },
                    )
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.time() + 5.0
                    while time.time() < deadline and proc.poll() is None:
                        time.sleep(0.2)
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    return_code = 124
                    break
                if elapsed_s - last_heartbeat_s < 15.0:
                    continue
                last_heartbeat_s = elapsed_s
                print(
                    "[run_all_with_servers] Family evolution running "
                    f"family={family_name} sample={sample_index} "
                    f"trigger_source={trigger_source} elapsed_s={int(elapsed_s)}"
                )
                _append_generated_tool_event(
                    progress_log_path,
                    {
                        "event": "sage_family_policy_evaluation_heartbeat",
                        "family_name": family_name,
                        "sample_index": str(sample_index),
                        "label": label,
                        "elapsed_s": int(elapsed_s),
                        "trigger_source": str(trigger_source or "").strip(),
                        "boundary_sample_index": str(boundary_sample_index or sample_index),
                    },
                )
    finally:
        log_text = ""
        try:
            log_text = temp_log_path.read_text(encoding="utf-8")
        except Exception:
            log_text = ""
        try:
            temp_log_path.unlink(missing_ok=True)
        except Exception:
            pass
    after_summary = _load_family_policy_store_summary(store_path, family_name)
    elapsed_s = int(time.monotonic() - start_ts)
    print(
        "[run_all_with_servers] Family evolution end "
        f"family={family_name} sample={sample_index} exit={return_code} "
        f"trigger_source={trigger_source} "
        f"boundary_sample={boundary_sample_index or sample_index} "
        f"active_version={after_summary.get('active_version') or 'n/a'} "
        f"pending_candidates={after_summary.get('pending_candidates')} "
        f"elapsed_s={elapsed_s}"
    )
    _append_generated_tool_event(
        progress_log_path,
        {
            "event": "sage_family_policy_evaluation_finished",
            "family_name": family_name,
            "sample_index": str(sample_index),
            "label": label,
            "exit_code": return_code,
            "elapsed_s": elapsed_s,
            "before": before_summary,
            "after": after_summary,
            "trigger_source": str(trigger_source or "").strip(),
            "boundary_sample_index": str(boundary_sample_index or sample_index),
            "parent_output_dir": str(parent_output_dir) if parent_output_dir is not None else "",
        },
    )
    if return_code != 0:
        print(
            f"[run_all_with_servers] Family evolution harness failed for family={family_name} "
            f"sample={sample_index} exit={return_code}"
        )
        if log_text:
            print(log_text[-2000:])
    _cleanup_family_evolution_artifacts(repo_root, label)
    return int(return_code)


def _append_log(log_path: Path | None, text: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _log_json_preview(
    *,
    source: str,
    status_code: str,
    content_type: str,
    body: str,
) -> None:
    preview = (body or "").replace("\n", "\\n")[:200]
    print(
        f"[json-parse] source={source} status={status_code} "
        f"content_type={content_type} body_head={preview}"
    )


def _run(cmd: list[str], *, log_path: Path | None = None) -> subprocess.CompletedProcess:
    _append_log(log_path, f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        _append_log(log_path, result.stdout)
    if result.stderr:
        _append_log(log_path, result.stderr)
    return result


def sparql_probe(endpoint: str, timeout_s: int = 5) -> tuple[bool, bool, str | None]:
    query = "ASK WHERE { ?s ?p ?o }"
    body = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/sparql-results+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            _log_json_preview(
                source=endpoint,
                status_code=str(resp.status),
                content_type=str(resp.headers.get("Content-Type", "")),
                body=raw,
            )
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "boolean" in payload:
                    return True, bool(payload["boolean"]), None
            except Exception:
                pass
            return True, False, f"Unexpected response (first 200 chars): {raw[:200]}"
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return True, False, f"HTTPError {e.code}: {detail[:200]}"
    except Exception as e:
        return False, False, str(e)


def _sparql_health(timeout_s: float = 5.0) -> kg_sparql_server.SparqlHealth:
    return kg_sparql_server.sparql_health(SPARQL_ENDPOINT, timeout_s=timeout_s)


def _log_sparql_health(
    *,
    context: str,
    health: kg_sparql_server.SparqlHealth,
    fuseki_log_path: Path | None = None,
) -> None:
    base_message = (
        f"[KG endpoint] context={context} endpoint={SPARQL_ENDPOINT} "
        f"reachable={health.reachable} has_data={health.has_data}"
    )
    print(base_message)
    _append_log(fuseki_log_path, base_message)
    if health.error:
        error_message = f"[KG endpoint] context={context} error={health.error}"
        print(error_message)
        _append_log(fuseki_log_path, error_message)


def _wait_for_server(url: str, timeout_s: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 405:
                try:
                    with urllib.request.urlopen(url, timeout=2) as resp:
                        if 200 <= resp.status < 300:
                            return True
                except (urllib.error.URLError, TimeoutError):
                    pass
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    return False


def _tail_file(path: Path, n_lines: int = 200) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception as exc:
        return f"[run_all_with_servers] Unable to read log file: {exc}\n"


def _sanitize_log_name(config_path: str) -> str:
    return config_path.replace("/", "_").replace(".yaml", "") + ".log"


def _extract_task_name(config_path: str) -> str:
    parts = config_path.split("/")
    if "instance" in parts:
        idx = parts.index("instance")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(config_path).stem


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_docker_containers_on_port(port: int) -> None:
    ps = subprocess.run(
        ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}"],
        capture_output=True, text=True, check=False,
    )
    ids = [x.strip() for x in (ps.stdout or "").splitlines() if x.strip()]
    for cid in ids:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, text=True, check=False)



def _kill_port(port: int) -> None:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    pids = [int(pid) for pid in result.stdout.split() if pid.strip().isdigit()]
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 3
    while time.time() < deadline:
        if not any(_pid_exists(pid) for pid in pids):
            return
        time.sleep(0.2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _preflight_kill_ports(ports: list[int]) -> None:
    for port in ports:
        _kill_port(port)
    time.sleep(1)


def _list_output_dirs(outputs_root: Path) -> list[Path]:
    if not outputs_root.exists():
        return []
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
    return [p for p in outputs_root.iterdir() if p.is_dir() and pattern.match(p.name)]


def _pick_output_dir(before: list[Path], after: list[Path]) -> Path | None:
    before_set = {p.resolve() for p in before}
    new_dirs = [p for p in after if p.resolve() not in before_set]
    if new_dirs:
        new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return new_dirs[0]
    if after:
        after.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return after[0]
    return None


def _latest_client_progress_mtime(output_dir: Path) -> float | None:
    candidates = [
        output_dir / "generated_tools.log",
        output_dir / "current_session.json",
        output_dir / "exception.txt",
        output_dir / "task_outcomes.json",
        output_dir / "runs.json",
        output_dir / "metric.json",
    ]
    latest: float | None = None
    for path in candidates:
        try:
            if path.exists():
                mtime = path.stat().st_mtime
                latest = mtime if latest is None else max(latest, mtime)
        except Exception:
            continue
    return latest


def _read_current_session(output_dir: Path) -> dict[str, object] | None:
    current_session_path = output_dir / "current_session.json"
    try:
        if not current_session_path.exists():
            return None
        payload = json.loads(current_session_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _extract_current_sample_index(session_payload: dict[str, object] | None) -> str | None:
    if not isinstance(session_payload, dict):
        return None
    sample_index = session_payload.get("sample_index")
    if sample_index is None:
        return None
    sample_index_text = str(sample_index).strip()
    return sample_index_text or None


def _record_timed_out_current_session(output_dir: Path, reason: str) -> str | None:
    current_session = _read_current_session(output_dir)
    sample_index = _extract_current_sample_index(current_session)
    if not sample_index or current_session is None:
        return None

    current_session["sample_status"] = "agent_unknown_error"
    current_session["finish_reason"] = f"[run_all_with_servers] {reason}"
    current_session["task_output"] = {"answer": None}
    current_session["evaluation_record"] = {
        "outcome": "incorrect",
        "detail_dict": {
            "f1_score": 0.0,
            "executable_flag": False,
        },
    }
    current_session.setdefault("tool_invoked", [])
    current_session["tool_invoked_any"] = bool(current_session.get("tool_invoked_any"))

    runs_path = output_dir / "runs.json"
    runs_payload: list[dict[str, object]] = []
    try:
        if runs_path.exists():
            raw_runs = json.loads(runs_path.read_text(encoding="utf-8"))
            if isinstance(raw_runs, list):
                runs_payload = [item for item in raw_runs if isinstance(item, dict)]
    except Exception:
        runs_payload = []

    if any(str(item.get("sample_index") or "").strip() == sample_index for item in runs_payload):
        try:
            (output_dir / "current_session.json").write_text(
                json.dumps(current_session, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return sample_index

    runs_payload.append(current_session)
    runs_path.write_text(json.dumps(runs_payload, indent=2), encoding="utf-8")
    (output_dir / "current_session.json").write_text(
        json.dumps(current_session, indent=2),
        encoding="utf-8",
    )
    return sample_index


def _append_client_timeout_exception(output_dir: Path, reason: str) -> None:
    exception_path = output_dir / "exception.txt"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_log(exception_path, f"Time: {stamp}")
    _append_log(exception_path, f"Exception: [run_all_with_servers] {reason}")
    _append_log(exception_path, "NoneType: None")
    _append_log(exception_path, "")


def _terminate_process_group(proc: subprocess.Popen[str], *, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _run_client_with_watchdog(
    client_cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    log_path: Path,
    wall_timeout_s: int,
    idle_timeout_s: int,
    poll_s: float,
) -> ClientWatchdogResult:
    proc = subprocess.Popen(
        client_cmd,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    start_ts = time.time()
    last_progress_mtime = _latest_client_progress_mtime(output_dir)
    current_sample_index: str | None = None
    current_sample_start_ts = start_ts
    current_sample_progress_ts = start_ts
    try:
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                return ClientWatchdogResult(exit_code=exit_code)

            now = time.time()
            current_session = _read_current_session(output_dir)
            observed_sample_index = _extract_current_sample_index(current_session)
            if observed_sample_index != current_sample_index:
                current_sample_index = observed_sample_index
                current_sample_start_ts = now
                current_sample_progress_ts = now

            latest_mtime = _latest_client_progress_mtime(output_dir)
            if (
                latest_mtime is not None
                and (last_progress_mtime is None or latest_mtime > last_progress_mtime)
            ):
                last_progress_mtime = latest_mtime
                current_sample_progress_ts = now

            timed_out_reason = None
            if wall_timeout_s > 0 and now - current_sample_start_ts > wall_timeout_s:
                timed_out_reason = f"sample_wall_timeout:{wall_timeout_s}s"
            elif idle_timeout_s > 0 and now - current_sample_progress_ts > idle_timeout_s:
                timed_out_reason = f"sample_idle_timeout:{idle_timeout_s}s"

            if timed_out_reason:
                message = (
                    f"[run_all_with_servers] Sample watchdog triggered for "
                    f"{output_dir} sample={current_sample_index or 'unknown'}: "
                    f"{timed_out_reason}"
                )
                print(message)
                _append_log(log_path, message)
                _append_client_timeout_exception(output_dir, timed_out_reason)
                _terminate_process_group(proc)
                return ClientWatchdogResult(
                    exit_code=124,
                    timed_out_reason=timed_out_reason,
                    current_session=current_session,
                )

            time.sleep(max(poll_s, 0.5))
    finally:
        _terminate_process_group(proc, grace_s=1.0)


def _merge_runs(output_dir: Path, combined_dir: Path) -> None:
    runs_path = output_dir / "runs.json"
    if not runs_path.exists():
        return
    combined_path = combined_dir / "runs.json"
    try:
        raw = runs_path.read_text(encoding="utf-8")
        _log_json_preview(
            source=str(runs_path),
            status_code="file",
            content_type="application/json",
            body=raw,
        )
        new_runs = json.loads(raw)
    except Exception:
        return
    if not isinstance(new_runs, list):
        return
    existing: list[dict[str, object]] = []
    if combined_path.exists():
        try:
            raw_existing = combined_path.read_text(encoding="utf-8")
            _log_json_preview(
                source=str(combined_path),
                status_code="file",
                content_type="application/json",
                body=raw_existing,
            )
            existing = json.loads(raw_existing)
        except Exception:
            existing = []
    if not isinstance(existing, list):
        existing = []
    combined_path.write_text(
        json.dumps(existing + new_runs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _merge_metrics(output_dir: Path, combined_dir: Path) -> None:
    metric_path = output_dir / "metric.json"
    runs_path = output_dir / "runs.json"
    if not metric_path.exists() or not runs_path.exists():
        return
    try:
        raw_metric = metric_path.read_text(encoding="utf-8")
        _log_json_preview(
            source=str(metric_path),
            status_code="file",
            content_type="application/json",
            body=raw_metric,
        )
        metric_data = json.loads(raw_metric)
        raw_runs = runs_path.read_text(encoding="utf-8")
        _log_json_preview(
            source=str(runs_path),
            status_code="file",
            content_type="application/json",
            body=raw_runs,
        )
        runs_data = json.loads(raw_runs)
    except Exception:
        return
    task_name = None
    if isinstance(runs_data, list) and runs_data:
        task_name = runs_data[0].get("task_name")
    if not task_name:
        task_name = output_dir.name
    combined_path = combined_dir / "metric.json"
    combined_metric = {}
    if combined_path.exists():
        try:
            raw_combined = combined_path.read_text(encoding="utf-8")
            _log_json_preview(
                source=str(combined_path),
                status_code="file",
                content_type="application/json",
                body=raw_combined,
            )
            combined_metric = json.loads(raw_combined)
        except Exception:
            combined_metric = {}
    if not isinstance(combined_metric, dict):
        combined_metric = {}
    combined_metric.setdefault(task_name, []).append(metric_data)
    combined_path.write_text(
        json.dumps(combined_metric, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _copy_aux_files(output_dir: Path, combined_dir: Path, tag: str) -> None:
    for filename in ("config.yaml", "exception.txt", "singleton_logger_client.log", "singleton_logger_server.log"):
        src = output_dir / filename
        if not src.exists():
            continue
        dest = combined_dir / f"{tag}_{filename}"
        try:
            shutil.copy2(src, dest)
        except Exception:
            continue


def _resolve_fuseki_data_root(repo_root: Path) -> Path:
    env_dir = os.getenv(KG_DATA_DIR_ENV, "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (repo_root / "data" / "knowledge_graph" / "fuseki_db").resolve()


def _docker_container_running(name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode == 0 and name in (r.stdout or "")


def _start_fuseki_serve_only(repo_root: Path, fuseki_log_path: Path) -> bool:
    """
    Start Fuseki from existing TDB2 directory only (no loading).
    """
    data_root = _resolve_fuseki_data_root(repo_root)
    dataset_dir = data_root / "databases" / FUSEKI_DATASET

    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print("[run_all_with_servers] KG dataset directory missing/empty.")
        print(f"[run_all_with_servers] Expected non-empty TDB2 dir: {dataset_dir}")
        return False

    _append_log(fuseki_log_path, "Starting Fuseki (serve-only from existing TDB2).")
    _run(["docker", "rm", "-f", FUSEKI_CONTAINER], log_path=fuseki_log_path)

    # IMPORTANT on macOS:
    # lsof won't reliably reveal Docker's port forwarder. If any container publishes this port,
    # docker run will fail with "port is already allocated". Kill those containers first.
    _stop_docker_containers_on_port(FUSEKI_HOST_PORT)

    # Also try to kill any normal local processes using the port (non-Docker)
    _kill_port(FUSEKI_HOST_PORT)

    run_cmd = [
        "docker", "run", "-d",
        "--name", FUSEKI_CONTAINER,
        "--platform", FUSEKI_PLATFORM,
        "-p", f"{FUSEKI_HOST_PORT}:{FUSEKI_HOST_PORT}",
        "-e", "ADMIN_PASSWORD=admin",
        "-v", f"{data_root}:/fuseki",
        FUSEKI_IMAGE,
        "./fuseki-server",
        "--port", str(FUSEKI_HOST_PORT),
        "--tdb2", "--loc", f"/fuseki/databases/{FUSEKI_DATASET}",
        f"/{FUSEKI_DATASET}",
    ]

    res = _run(run_cmd, log_path=fuseki_log_path)
    if res.returncode != 0:
        print("[run_all_with_servers] docker run failed starting Fuseki.")

        # # Print the REAL docker error immediately
        # if res.stderr:
        print("[run_all_with_servers] docker stderr:")
        print(res.stderr.strip())

        # if res.stdout:
        print("[run_all_with_servers] docker stdout:")
        print(res.stdout.strip())

        print(f"[run_all_with_servers] Fuseki log: {fuseki_log_path}")
        print(_tail_file(fuseki_log_path))
        return False

    # Show log tail right after start (helps immediately)
    # print("Log: ", _tail_file(fuseki_log_path))

    # Print last container logs on start to make failures obvious
    logs = subprocess.run(
        ["docker", "logs", "--tail", "80", FUSEKI_CONTAINER],
        capture_output=True,
        text=True,
        check=False,
    )
    if logs.stdout:
        _append_log(fuseki_log_path, "--- docker logs (tail 80) ---\n" + logs.stdout)

    return True



def _ensure_fuseki_ready(
    repo_root: Path,
    fuseki_log_path: Path,
    *,
    context: str,
    runtime_state: dict[str, bool] | None = None,
) -> tuple[bool, bool]:
    """
    Returns: (started_by_script, ok)
    """
    initial_health = _sparql_health()
    _log_sparql_health(
        context=f"{context}:preflight",
        health=initial_health,
        fuseki_log_path=fuseki_log_path,
    )
    if initial_health.reachable and initial_health.has_data:
        return False, True

    restart_message = (
        f"[run_all_with_servers] KG endpoint unhealthy during {context}; "
        "starting or restarting Fuseki."
    )
    print(restart_message)
    _append_log(fuseki_log_path, restart_message)

    started = True
    ok = _start_fuseki_serve_only(repo_root, fuseki_log_path)
    if not ok:
        return started, False
    if runtime_state is not None:
        runtime_state["managed_by_script"] = True

    health = kg_sparql_server.wait_for_sparql_ready(SPARQL_ENDPOINT, timeout_s=120)
    _log_sparql_health(
        context=f"{context}:post_start",
        health=health,
        fuseki_log_path=fuseki_log_path,
    )
    if not (health.reachable and health.has_data):
        print("[run_all_with_servers] SPARQL readiness failed after starting Fuseki.")
        print(f"[run_all_with_servers] Expected endpoint: {SPARQL_ENDPOINT}")
        print(f"[run_all_with_servers] Reachable: {health.reachable}")
        print(f"[run_all_with_servers] KB loaded: {health.has_data}")
        if health.error:
            print(f"[run_all_with_servers] Error: {health.error}")
            pass

        # surface container status/logs to console
        ps = subprocess.run(["docker", "ps", "-a", "--filter", f"name={FUSEKI_CONTAINER}"],
                            capture_output=True, text=True, check=False)
        if ps.stdout:
            print(ps.stdout.strip())
            pass
        logs = subprocess.run(["docker", "logs", "--tail", "80", FUSEKI_CONTAINER],
                              capture_output=True, text=True, check=False)
        if logs.stdout:
            print(logs.stdout.strip())

        print(f"[run_all_with_servers] Fuseki log: {fuseki_log_path}")
        return started, False

    return started, True


def _ensure_kg_endpoint_available(
    repo_root: Path,
    fuseki_log_path: Path,
    *,
    context: str,
    runtime_state: dict[str, bool],
) -> bool:
    _, ok = _ensure_fuseki_ready(
        repo_root,
        fuseki_log_path,
        context=context,
        runtime_state=runtime_state,
    )
    return ok


def _fuseki_watchdog_loop(
    repo_root: Path,
    fuseki_log_path: Path,
    runtime_state: dict[str, bool],
    stop_event: threading.Event,
    *,
    interval_s: float = 5.0,
    recovery_cooldown_s: float = 15.0,
) -> None:
    last_recovery_attempt = 0.0
    while not stop_event.wait(interval_s):
        health = _sparql_health(timeout_s=3.0)
        if health.reachable and health.has_data:
            continue

        _log_sparql_health(
            context="runtime_watchdog:unhealthy",
            health=health,
            fuseki_log_path=fuseki_log_path,
        )
        now = time.time()
        if now - last_recovery_attempt < recovery_cooldown_s:
            continue
        last_recovery_attempt = now
        try:
            _ensure_kg_endpoint_available(
                repo_root,
                fuseki_log_path,
                context="runtime_watchdog",
                runtime_state=runtime_state,
            )
        except Exception as exc:
            message = f"[run_all_with_servers] Fuseki watchdog recovery failed: {exc}"
            print(message)
            _append_log(fuseki_log_path, message)


def _run_one(
    config_path: str,
    combined_dir: Path,
    *,
    output_dir_override: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    is_kg = "knowledge_graph" in config_path

    repo_root = Path(__file__).resolve().parents[1]
    full_path = repo_root / config_path
    if not full_path.exists():
        print(f"[run_all_with_servers] Missing config: {full_path}")
        return 1

    logs_dir = repo_root / "logs" / "run_all_with_servers"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / _sanitize_log_name(config_path)
    fuseki_log_path = log_path.with_suffix(".fuseki.log")
    task_name = _extract_task_name(config_path)
    config_name = Path(config_path).stem

    # SELF-CONTAINED: ensure Fuseki is up BEFORE anything else for KG
    fuseki_runtime_state = {"managed_by_script": False}
    fuseki_watchdog_stop: threading.Event | None = None
    fuseki_watchdog_thread: threading.Thread | None = None
    if is_kg:
        _, ok = _ensure_fuseki_ready(
            repo_root,
            fuseki_log_path,
            context="startup",
            runtime_state=fuseki_runtime_state,
        )
        if not ok:
            return 1

    server_cmd = [
        sys.executable,
        "src/distributed_deployment_utils/start_server.py",
        "--config_path",
        config_path,
    ]
    client_cmd = [
        sys.executable,
        "src/run_experiment.py",
        "--config_path",
        config_path,
    ]

    env = os.environ.copy()
    env_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env_py_path}" if env_py_path else str(repo_root)
    env["LIFELONG_OUTPUT_DIR"] = str(
        output_dir_override or (combined_dir / task_name / config_name)
    )
    env["LIFELONG_OUTPUT_TAG"] = ""
    for key, value in (extra_env or {}).items():
        env[key] = value
    if is_kg:
        ontology_dir = str(repo_root / "data" / "knowledge_graph" / "ontology")
        if not env.get("KG_ONTOLOGY_DIR"):
            env["KG_ONTOLOGY_DIR"] = ontology_dir
        if not env.get("LIFELONG_KG_ONTOLOGY_DIR"):
            env["LIFELONG_KG_ONTOLOGY_DIR"] = ontology_dir
        env["SAGE_SPARQL_ENDPOINT_URL"] = SPARQL_ENDPOINT

    recovered_timed_out_samples: set[str] = set()
    should_retry_client = True
    final_result_code = 1
    try:
        while should_retry_client:
            should_retry_client = False
            fuseki_watchdog_stop = None
            fuseki_watchdog_thread = None
            print(f"[run_all_with_servers] Starting server: {config_path}")
            _preflight_kill_ports([8000, 8001])

            log_fp = log_path.open("a", encoding="utf-8")
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=repo_root,
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            try:
                if not _wait_for_server("http://127.0.0.1:8000/api/ping"):
                    print("[run_all_with_servers] Task server did not become ready on 8000.")
                    print(f"[run_all_with_servers] Server log: {log_path}")
                    print(_tail_file(log_path))
                    return 1

                if not _wait_for_server("http://127.0.0.1:8001/api/ping"):
                    print("[run_all_with_servers] ChatHistoryItemFactory server did not become ready on 8001.")
                    print(f"[run_all_with_servers] Server log: {log_path}")
                    print(_tail_file(log_path))
                    return 1

                if is_kg:
                    ok = _ensure_kg_endpoint_available(
                        repo_root,
                        fuseki_log_path,
                        context="pre_client",
                        runtime_state=fuseki_runtime_state,
                    )
                    if not ok:
                        print("[run_all_with_servers] KG endpoint unavailable after server startup.")
                        print(f"[run_all_with_servers] Fuseki log: {fuseki_log_path}")
                        print(_tail_file(fuseki_log_path))
                        return 1
                    fuseki_watchdog_stop = threading.Event()
                    fuseki_watchdog_thread = threading.Thread(
                        target=_fuseki_watchdog_loop,
                        args=(
                            repo_root,
                            fuseki_log_path,
                            fuseki_runtime_state,
                            fuseki_watchdog_stop,
                        ),
                        daemon=True,
                    )
                    fuseki_watchdog_thread.start()

                print(f"[run_all_with_servers] Running client: {config_path}")
                result = _run_client_with_watchdog(
                    client_cmd,
                    cwd=repo_root,
                    env=env,
                    output_dir=Path(env["LIFELONG_OUTPUT_DIR"]),
                    log_path=log_path,
                    wall_timeout_s=CLIENT_WALL_TIMEOUT_S,
                    idle_timeout_s=CLIENT_IDLE_TIMEOUT_S,
                    poll_s=CLIENT_WATCHDOG_POLL_S,
                )
                final_result_code = result.exit_code
                if result.exit_code == 0:
                    final_result_code = 0
                    continue

                if result.timed_out_reason:
                    recovered_sample_index = _record_timed_out_current_session(
                        Path(env["LIFELONG_OUTPUT_DIR"]),
                        result.timed_out_reason,
                    )
                    if recovered_sample_index and recovered_sample_index not in recovered_timed_out_samples:
                        recovered_timed_out_samples.add(recovered_sample_index)
                        message = (
                            "[run_all_with_servers] Recorded timed-out sample and will restart client: "
                            f"sample={recovered_sample_index} reason={result.timed_out_reason}"
                        )
                        print(message)
                        _append_log(log_path, message)
                        should_retry_client = True
                        continue

                print(f"[run_all_with_servers] Client failed for {config_path} (exit={result.exit_code})")
                print(f"[run_all_with_servers] Server log: {log_path}")
                print(_tail_file(log_path))
                return result.exit_code

            finally:
                if fuseki_watchdog_stop is not None:
                    fuseki_watchdog_stop.set()
                if fuseki_watchdog_thread is not None:
                    fuseki_watchdog_thread.join(timeout=5)
                if server_proc.poll() is None:
                    try:
                        os.killpg(server_proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    server_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(server_proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(2)
                try:
                    log_fp.close()
                except Exception:
                    pass
    finally:
        if is_kg and fuseki_runtime_state["managed_by_script"]:
            try:
                _run(["docker", "rm", "-f", FUSEKI_CONTAINER], log_path=fuseki_log_path)
            except Exception:
                pass

    return final_result_code


def _run_one_with_sample_boundary_family_evolution(
    config_path: str,
    combined_dir: Path,
) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    full_path = repo_root / config_path
    sample_order = _load_sample_order_from_config(full_path)
    if not sample_order:
        return _run_one(config_path, combined_dir)

    task_name = _extract_task_name(config_path)
    config_name = Path(config_path).stem
    aggregate_output_dir = combined_dir / task_name / config_name
    aggregate_output_dir.mkdir(parents=True, exist_ok=True)
    store_path = _resolve_family_policy_store_path(
        repo_root=repo_root,
        aggregate_output_dir=aggregate_output_dir,
        task_name=task_name,
    )
    _maybe_reset_family_policy_store(store_path)

    extra_env = {
        FAMILY_EVOLUTION_ENV: "1",
        FAMILY_PROMOTION_ENV: "0",
        FAMILY_STORE_PATH_ENV: str(store_path),
    }
    configured_families = str(os.getenv(FAMILY_ENABLED_FAMILIES_ENV) or "").strip()
    if configured_families:
        extra_env[FAMILY_ENABLED_FAMILIES_ENV] = configured_families

    temp_config_paths: dict[str, Path] = {}
    created_temp_config_paths: list[Path] = []
    for sample_index in sample_order:
        if len(sample_order) == 1:
            temp_config_paths[str(sample_index)] = full_path
            continue
        temp_stem = f"{config_name}__sample_{sample_index}"
        temp_config_path = _write_single_sample_config(
            source_config_path=full_path,
            sample_index=str(sample_index),
            stem=temp_stem,
        )
        temp_config_paths[str(sample_index)] = temp_config_path
        created_temp_config_paths.append(temp_config_path)

    try:
        for sample_index in sample_order:
            temp_config_path = temp_config_paths[str(sample_index)]
            _delete_file_if_exists(aggregate_output_dir / "config.yaml")
            code = _run_one(
                temp_config_path.relative_to(repo_root).as_posix(),
                combined_dir,
                output_dir_override=aggregate_output_dir,
                extra_env=extra_env,
            )
            session_record = _load_session_for_sample(aggregate_output_dir, str(sample_index))
            if code != 0 and not isinstance(session_record, dict):
                return code
            if code != 0:
                print(
                    "[run_all_with_servers] Continuing after per-sample task failure: "
                    f"sample={sample_index} config={config_name} exit={code}"
                )

            if not isinstance(session_record, dict):
                continue
            sample_status = str(session_record.get("sample_status") or "").strip()
            evaluation_record = session_record.get("evaluation_record")
            evaluation_outcome = ""
            if isinstance(evaluation_record, dict):
                evaluation_outcome = str(evaluation_record.get("outcome") or "").strip()
            selected_family = _latest_selected_family_for_sample(
                aggregate_output_dir,
                str(sample_index),
            )
            if not selected_family or not _family_evolution_enabled_for(selected_family):
                continue
            active_summary = _load_family_policy_store_summary(store_path, selected_family)
            active_version = str(active_summary.get("active_version") or "").strip()
            if (
                sample_status == "completed"
                and evaluation_outcome == "correct"
            ):
                _record_trusted_family_success(
                    output_dir=aggregate_output_dir,
                    store_path=store_path,
                    family_name=selected_family,
                    sample_index=str(sample_index),
                    active_version=active_version,
                    session_record=session_record,
                    progress_log_path=aggregate_output_dir / "generated_tools.log",
                )
            _record_tool_evolution_signal(
                output_dir=aggregate_output_dir,
                store_path=store_path,
                family_name=selected_family,
                sample_index=str(sample_index),
                active_version=active_version,
                session_record=session_record,
                progress_log_path=aggregate_output_dir / "generated_tools.log",
            )
            current_sample_is_correct = (
                sample_status == "completed" and evaluation_outcome == "correct"
            )
            pending_candidate = _oldest_pending_candidate_for_family(
                store_path=store_path,
                family_name=selected_family,
            )
            if current_sample_is_correct and pending_candidate is None:
                continue
            evaluation_target_sample = str(sample_index)
            trigger_source = "sample_failure_or_wrong_completion"
            if current_sample_is_correct and pending_candidate is not None:
                evaluation_target_sample = str(
                    ((pending_candidate or {}).get("trigger_context") or {}).get(
                        "sample_index"
                    )
                    or sample_index
                ).strip() or str(sample_index)
                trigger_source = "pending_candidate_after_trusted_success"
                print(
                    "[run_all_with_servers] Family evolution reactivated pending candidate "
                    f"family={selected_family} boundary_sample={sample_index} "
                    f"trigger_sample={evaluation_target_sample}"
                )
                _append_generated_tool_event(
                    aggregate_output_dir / "generated_tools.log",
                    {
                        "event": "sage_family_policy_pending_candidate_reactivated",
                        "family_name": selected_family,
                        "boundary_sample_index": str(sample_index),
                        "trigger_sample_index": evaluation_target_sample,
                        "active_version": active_version,
                        "candidate_version": str(
                            (pending_candidate or {}).get("candidate_version") or ""
                        ).strip(),
                    },
                )
            if not _inline_family_evolution_allowed_for(selected_family):
                print(
                    "[run_all_with_servers] Family evolution skipped "
                    f"family={selected_family} sample={evaluation_target_sample} reason=family_not_inline_eligible"
                )
                _append_generated_tool_event(
                    aggregate_output_dir / "generated_tools.log",
                    {
                        "event": "sage_family_policy_evaluation_skipped",
                        "family_name": selected_family,
                        "sample_index": evaluation_target_sample,
                        "reason": "family_not_inline_eligible",
                        "active_version": active_version,
                    },
                )
                continue
            trigger_session_record = _load_session_for_sample(
                aggregate_output_dir,
                evaluation_target_sample,
            )
            if not isinstance(trigger_session_record, dict):
                _append_generated_tool_event(
                    aggregate_output_dir / "generated_tools.log",
                    {
                        "event": "sage_family_policy_evaluation_skipped",
                        "family_name": selected_family,
                        "sample_index": evaluation_target_sample,
                        "reason": "trigger_session_missing",
                        "active_version": active_version,
                    },
                )
                continue
            trigger_baseline_summary_path = _write_trigger_baseline_summary(
                output_dir=aggregate_output_dir,
                sample_index=evaluation_target_sample,
                session_record=trigger_session_record,
            )
            label = f"inlineevo_{config_name}_{evaluation_target_sample}_{selected_family}"
            try:
                harness_code = _run_between_sample_family_evolution(
                    repo_root=repo_root,
                    family_name=selected_family,
                    sample_index=evaluation_target_sample,
                    store_path=store_path,
                    label=label,
                    progress_log_path=aggregate_output_dir / "generated_tools.log",
                    trigger_baseline_summary_path=trigger_baseline_summary_path,
                    parent_output_dir=aggregate_output_dir / "family_policy_inline_runs",
                    trigger_source=trigger_source,
                    boundary_sample_index=str(sample_index),
                )
            finally:
                _delete_file_if_exists(trigger_baseline_summary_path)
            if harness_code == 124:
                store = build_family_policy_store(
                    baseline_bundles=get_baseline_reusable_family_policy_bundles(),
                    store_path=store_path,
                )
                for candidate_payload in store.get_pending_candidates(selected_family):
                    trigger_sample = str(
                        (candidate_payload.get("trigger_context") or {}).get("sample_index") or ""
                    ).strip()
                    if trigger_sample != evaluation_target_sample:
                        continue
                    store.reject_candidate(
                        selected_family,
                        candidate_version=str(candidate_payload.get("candidate_version") or "").strip(),
                        evaluation_results={
                            "event": "family_policy_inline_timeout",
                            "family_name": selected_family,
                            "sample_index": evaluation_target_sample,
                            "active_version": active_version,
                            "timeout_s": INLINE_FAMILY_EVOLUTION_BUDGET_S,
                        },
                        rejection_reason="inline_timeout",
                    )
                _append_generated_tool_event(
                    aggregate_output_dir / "generated_tools.log",
                    {
                        "event": "sage_family_policy_evaluation_timeout",
                        "family_name": selected_family,
                        "sample_index": evaluation_target_sample,
                        "active_version": active_version,
                        "timeout_s": INLINE_FAMILY_EVOLUTION_BUDGET_S,
                    },
                )
                print(
                    "[run_all_with_servers] Continuing after family evolution timeout rejection: "
                    f"family={selected_family} sample={evaluation_target_sample}"
                )
            elif harness_code != 0:
                print(
                    "[run_all_with_servers] Continuing after family evolution harness failure: "
                    f"family={selected_family} sample={evaluation_target_sample}"
                )
    finally:
        for temp_config_path in created_temp_config_paths:
            _delete_file_if_exists(temp_config_path)
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    combined_dir = repo_root / "outputs" / f"run_all_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    combined_dir.mkdir(parents=True, exist_ok=True)

    config_paths_override = str(os.getenv(CONFIG_PATHS_ENV) or "").strip()
    if config_paths_override:
        configured_paths = [
            item.strip()
            for item in re.split(r"[\n,]+", config_paths_override)
            if item.strip()
        ]
    else:
        configured_paths = list(CONFIG_PATHS)

    for config_path in configured_paths:
        use_sample_boundary_evolution = (
            ENABLE_STANDARD_FAMILY_EVOLUTION
            and ENABLE_SAGE_AGENT
            and "knowledge_graph" in config_path
        )
        if use_sample_boundary_evolution:
            code = _run_one_with_sample_boundary_family_evolution(
                config_path,
                combined_dir,
            )
        else:
            code = _run_one(config_path, combined_dir)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
