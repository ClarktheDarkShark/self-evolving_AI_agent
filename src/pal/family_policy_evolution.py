from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.pal.policy_contracts import FamilyPolicyBundle


ENV_ENABLE_EVOLUTION = "PAL_ENABLE_FAMILY_POLICY_EVOLUTION"
ENV_ENABLE_PROMOTION = "PAL_ENABLE_FAMILY_POLICY_PROMOTION"
ENV_ENABLED_FAMILIES = "PAL_FAMILY_POLICY_EVOLUTION_FAMILIES"
ENV_STORE_PATH = "PAL_FAMILY_POLICY_STORE_PATH"
ENV_OVERRIDE_VERSIONS = "PAL_FAMILY_POLICY_OVERRIDE_VERSIONS"


@dataclass(frozen=True)
class FamilyPolicyCandidateUpdate:
    family_name: str
    base_version: str
    candidate_version: str
    bundle: FamilyPolicyBundle
    fields_changed: tuple[str, ...]
    reason_for_change: str
    trigger_context: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "base_version": self.base_version,
            "candidate_version": self.candidate_version,
            "bundle": self.bundle.as_dict(),
            "fields_changed": list(self.fields_changed),
            "reason_for_change": self.reason_for_change,
            "trigger_context": dict(self.trigger_context),
        }


def classify_family_failure(
    *,
    family_name: str,
    sample_status: str,
    evaluation_outcome: str,
    relation_names: Sequence[str],
    failure_reasons: Sequence[str],
    dangerous_overreach: bool = False,
) -> str:
    cleaned_family = str(family_name or "").strip()
    cleaned_status = str(sample_status or "").strip()
    cleaned_outcome = str(evaluation_outcome or "").strip()
    cleaned_relations = [
        str(item or "").strip()
        for item in relation_names
        if str(item or "").strip()
    ]
    cleaned_reasons = [
        str(item or "").strip()
        for item in failure_reasons
        if str(item or "").strip()
    ]

    if dangerous_overreach or any(
        reason.startswith("dangerous_overreach:") for reason in cleaned_reasons
    ):
        return "validator_miss"
    if any("query_uses_unplanned_predicate:" in reason for reason in cleaned_reasons):
        return "forbidden_relation_family_miss"
    if any(reason.startswith("sample_status:agent_unknown_error") for reason in cleaned_reasons):
        if any("anchor" in reason for reason in cleaned_reasons):
            return "bad_routing"
        return "bad_fallback_ordering"
    if cleaned_status == "completed" and cleaned_outcome == "incorrect":
        if any(relation.startswith("type.") for relation in cleaned_relations):
            return "forbidden_relation_family_miss"
        if cleaned_family in {
            "count_over_direct_relation",
            "count_over_joined_set",
            "multi_anchor_intersection",
            "shared_type_intersection",
        }:
            return "weak_applicability_boundary"
        return "validator_miss"
    return "weak_applicability_boundary"


def family_policy_evolution_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_EVOLUTION) == "1"


def family_policy_promotion_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_PROMOTION) == "1"


def family_policy_enabled_for(family_name: str) -> bool:
    if not family_policy_evolution_enabled():
        return False
    configured = {
        str(item or "").strip()
        for item in os.environ.get(ENV_ENABLED_FAMILIES, "").split(",")
        if str(item or "").strip()
    }
    if not configured:
        return True
    return str(family_name or "").strip() in configured


def resolve_family_policy_store_path() -> Path:
    configured = str(os.environ.get(ENV_STORE_PATH) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    output_dir = Path(os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/pal_runtime"))
    return (output_dir / "family_policy_store").resolve()


def parse_override_versions() -> dict[str, str]:
    raw = str(os.environ.get(ENV_OVERRIDE_VERSIONS) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    return {
        str(key or "").strip(): str(value or "").strip()
        for key, value in parsed.items()
        if str(key or "").strip() and str(value or "").strip()
    }


def bundle_from_dict(payload: Mapping[str, Any]) -> FamilyPolicyBundle:
    return FamilyPolicyBundle(
        family_name=str(payload.get("family_name") or "").strip(),
        version=str(payload.get("version") or "").strip(),
        renderer_name=str(payload.get("renderer_name") or "").strip(),
        applicability_conditions=tuple(payload.get("applicability_conditions") or ()),
        validator_expectations=tuple(payload.get("validator_expectations") or ()),
        repair_policy=tuple(payload.get("repair_policy") or ()),
        allowed_materialization_modes=tuple(payload.get("allowed_materialization_modes") or ()),
        blocked_scaffold_signatures=tuple(payload.get("blocked_scaffold_signatures") or ()),
        forbidden_overreach_patterns=tuple(payload.get("forbidden_overreach_patterns") or ()),
        forbidden_relation_families=tuple(payload.get("forbidden_relation_families") or ()),
    )


def _dedupe_strings(items: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


class FamilyPolicyStore:
    def __init__(
        self,
        *,
        store_path: Path,
        baseline_bundles: Mapping[str, FamilyPolicyBundle],
    ) -> None:
        self.store_path = store_path
        self.baseline_bundles = {
            str(name or "").strip(): bundle
            for name, bundle in baseline_bundles.items()
            if str(name or "").strip()
        }
        self.store_path.mkdir(parents=True, exist_ok=True)

    def _family_path(self, family_name: str) -> Path:
        return self.store_path / f"{family_name}.json"

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _bootstrap_family_payload(self, family_name: str) -> dict[str, Any]:
        baseline = self.baseline_bundles.get(family_name)
        if baseline is None:
            raise KeyError(f"unknown_family_policy_bundle:{family_name}")
        now = datetime.now(UTC).isoformat()
        return {
            "family_name": family_name,
            "active_version": baseline.version,
            "trusted_success_bank": [],
            "versions": {
                baseline.version: {
                    "status": "active",
                    "parent_version": None,
                    "bundle": baseline.as_dict(),
                    "fields_changed": [],
                    "reason_for_change": "baseline_from_code",
                    "evaluation_results": {},
                    "promotion_decision": {
                        "decision": "baseline",
                        "reason": "baseline_from_code",
                    },
                    "created_at": now,
                    "updated_at": now,
                }
            },
        }

    def _load_family_payload(self, family_name: str) -> dict[str, Any]:
        family_name = str(family_name or "").strip()
        if not family_name:
            raise KeyError("empty_family_name")
        family_path = self._family_path(family_name)
        if not family_path.exists():
            payload = self._bootstrap_family_payload(family_name)
            self._write_json(family_path, payload)
            return payload
        loaded = json.loads(family_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            payload = self._bootstrap_family_payload(family_name)
            self._write_json(family_path, payload)
            return payload
        if str(loaded.get("family_name") or "").strip() != family_name:
            loaded["family_name"] = family_name
        if "versions" not in loaded or not isinstance(loaded.get("versions"), dict):
            loaded = self._bootstrap_family_payload(family_name)
            self._write_json(family_path, loaded)
        return loaded

    def _save_family_payload(self, family_name: str, payload: Mapping[str, Any]) -> None:
        self._write_json(self._family_path(family_name), payload)

    def get_active_version(self, family_name: str) -> str:
        payload = self._load_family_payload(family_name)
        active = str(payload.get("active_version") or "").strip()
        if active:
            return active
        baseline = self.baseline_bundles[family_name]
        return baseline.version

    def get_bundle(
        self,
        family_name: str,
        *,
        version: str | None = None,
    ) -> Optional[FamilyPolicyBundle]:
        payload = self._load_family_payload(family_name)
        selected_version = str(version or payload.get("active_version") or "").strip()
        if not selected_version:
            return self.baseline_bundles.get(family_name)
        version_payload = (payload.get("versions") or {}).get(selected_version)
        if not isinstance(version_payload, Mapping):
            return self.baseline_bundles.get(family_name)
        bundle_payload = version_payload.get("bundle")
        if not isinstance(bundle_payload, Mapping):
            return self.baseline_bundles.get(family_name)
        return bundle_from_dict(bundle_payload)

    def get_active_bundle(self, family_name: str) -> Optional[FamilyPolicyBundle]:
        override_version = parse_override_versions().get(str(family_name or "").strip())
        return self.get_bundle(family_name, version=override_version)

    def get_trusted_success_bank(self, family_name: str) -> list[str]:
        payload = self._load_family_payload(family_name)
        return [
            str(item or "").strip()
            for item in (payload.get("trusted_success_bank") or [])
            if str(item or "").strip()
        ]

    def get_trusted_success_bank_metadata(self, family_name: str) -> dict[str, Any]:
        payload = self._load_family_payload(family_name)
        metadata = payload.get("trusted_success_bank_metadata") or {}
        return dict(metadata) if isinstance(metadata, Mapping) else {}

    def set_trusted_success_bank(
        self,
        family_name: str,
        *,
        sample_ids: Sequence[str],
        source_version: str,
        evaluation_results: Mapping[str, Any] | None = None,
        evaluation_context: Mapping[str, Any] | None = None,
    ) -> None:
        payload = self._load_family_payload(family_name)
        cleaned_ids = [
            str(item or "").strip()
            for item in sample_ids
            if str(item or "").strip()
        ]
        payload["trusted_success_bank"] = cleaned_ids
        metadata = {
            "sample_ids": cleaned_ids,
            "source_version": str(source_version or "").strip(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if evaluation_results:
            metadata["evaluation_results"] = dict(evaluation_results)
        if evaluation_context:
            metadata["evaluation_context"] = dict(evaluation_context)
        payload["trusted_success_bank_metadata"] = metadata
        self._save_family_payload(family_name, payload)

    def get_cached_evaluation(
        self,
        family_name: str,
        *,
        cache_key: str,
    ) -> dict[str, Any] | None:
        payload = self._load_family_payload(family_name)
        cache_payload = (payload.get("evaluation_cache") or {}).get(str(cache_key or "").strip())
        if not isinstance(cache_payload, Mapping):
            return None
        return dict(cache_payload)

    def set_cached_evaluation(
        self,
        family_name: str,
        *,
        cache_key: str,
        context: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        payload = self._load_family_payload(family_name)
        evaluation_cache = payload.get("evaluation_cache")
        if not isinstance(evaluation_cache, dict):
            evaluation_cache = {}
            payload["evaluation_cache"] = evaluation_cache
        evaluation_cache[str(cache_key or "").strip()] = {
            "context": dict(context),
            "result": dict(result),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._save_family_payload(family_name, payload)

    def create_candidate_update(
        self,
        *,
        family_name: str,
        scaffold_signature: str,
        relation_names: Sequence[str],
        failure_reasons: Sequence[str],
        trigger_context: Mapping[str, Any],
        failure_class: str | None = None,
    ) -> FamilyPolicyCandidateUpdate | None:
        payload = self._load_family_payload(family_name)
        active_version = str(payload.get("active_version") or "").strip()
        active_bundle = self.get_bundle(family_name, version=active_version)
        if active_bundle is None:
            return None

        cleaned_failure_class = str(failure_class or "").strip() or "weak_applicability_boundary"
        blocked_scaffolds = list(active_bundle.blocked_scaffold_signatures)
        cleaned_scaffold_signature = str(scaffold_signature or "").strip()
        if cleaned_failure_class in {
            "weak_applicability_boundary",
            "bad_routing",
            "bad_fallback_ordering",
            "validator_miss",
        } and cleaned_scaffold_signature and cleaned_scaffold_signature not in blocked_scaffolds:
            blocked_scaffolds.append(cleaned_scaffold_signature)

        dangerous_patterns = [
            str(reason or "").strip()
            for reason in failure_reasons
            if str(reason or "").strip().startswith("dangerous_overreach:")
        ]
        applicability_conditions = list(active_bundle.applicability_conditions)
        validator_expectations = list(active_bundle.validator_expectations)
        repair_policy = list(active_bundle.repair_policy)
        forbidden_overreach_patterns = list(active_bundle.forbidden_overreach_patterns)
        for pattern in dangerous_patterns:
            if pattern not in forbidden_overreach_patterns:
                forbidden_overreach_patterns.append(pattern)

        forbidden_relation_families = list(active_bundle.forbidden_relation_families)
        trusted_incorrect_completion = any(
            str(reason or "").strip() == "trusted_incorrect_completion"
            for reason in failure_reasons
        )
        if cleaned_failure_class in {
            "forbidden_relation_family_miss",
            "validator_miss",
        }:
            for relation_name in relation_names:
                relation_text = str(relation_name or "").strip()
                if not relation_text:
                    continue
                if cleaned_failure_class == "validator_miss" and not (
                    relation_text.startswith("type.") or dangerous_patterns
                ):
                    continue
                if relation_text not in forbidden_relation_families:
                    forbidden_relation_families.append(relation_text)
        if cleaned_failure_class == "weak_applicability_boundary" and trusted_incorrect_completion:
            for relation_name in relation_names:
                relation_text = str(relation_name or "").strip()
                if relation_text and relation_text not in forbidden_relation_families:
                    forbidden_relation_families.append(relation_text)

        if cleaned_failure_class == "weak_applicability_boundary":
            tightened_expectation = "tighten_answer_target_semantics"
            if tightened_expectation not in validator_expectations:
                validator_expectations.append(tightened_expectation)
            narrowed_condition = "avoid_broad_family_applicability"
            if narrowed_condition not in applicability_conditions:
                applicability_conditions.append(narrowed_condition)
        if cleaned_failure_class == "validator_miss":
            validator_guard = "reject_known_dangerous_overreach_patterns"
            if validator_guard not in validator_expectations:
                validator_expectations.append(validator_guard)
        if cleaned_failure_class in {"bad_routing", "bad_fallback_ordering"}:
            retry_guard = "deprioritize_failed_scaffold_before_family_switch"
            if retry_guard not in repair_policy:
                repair_policy.append(retry_guard)

        updated_bundle = FamilyPolicyBundle(
            family_name=active_bundle.family_name,
            version=active_bundle.version,
            renderer_name=active_bundle.renderer_name,
            applicability_conditions=_dedupe_strings(applicability_conditions),
            validator_expectations=_dedupe_strings(validator_expectations),
            repair_policy=_dedupe_strings(repair_policy),
            allowed_materialization_modes=active_bundle.allowed_materialization_modes,
            blocked_scaffold_signatures=_dedupe_strings(blocked_scaffolds),
            forbidden_overreach_patterns=_dedupe_strings(forbidden_overreach_patterns),
            forbidden_relation_families=_dedupe_strings(forbidden_relation_families),
        )
        fields_changed = [
            field_name
            for field_name in (
                "applicability_conditions",
                "validator_expectations",
                "repair_policy",
                "blocked_scaffold_signatures",
                "forbidden_overreach_patterns",
                "forbidden_relation_families",
            )
            if getattr(updated_bundle, field_name) != getattr(active_bundle, field_name)
        ]
        if not fields_changed:
            return None

        existing_versions = payload.get("versions") or {}
        updated_bundle_payload = updated_bundle.as_dict()
        for version_name, version_payload in existing_versions.items():
            if not isinstance(version_payload, Mapping):
                continue
            if (version_payload.get("bundle") or {}) == updated_bundle_payload:
                if str(version_payload.get("status") or "").strip() == "candidate":
                    return FamilyPolicyCandidateUpdate(
                        family_name=family_name,
                        base_version=active_version,
                        candidate_version=str(version_name),
                        bundle=bundle_from_dict(updated_bundle_payload),
                        fields_changed=tuple(fields_changed),
                        reason_for_change=str(
                            version_payload.get("reason_for_change") or "duplicate_candidate"
                        ).strip(),
                        trigger_context=dict(trigger_context),
                    )

        suffix = 1
        while True:
            candidate_version = f"{active_version}__cand{suffix:04d}"
            if candidate_version not in existing_versions:
                break
            suffix += 1

        candidate_bundle = FamilyPolicyBundle(
            family_name=updated_bundle.family_name,
            version=candidate_version,
            renderer_name=updated_bundle.renderer_name,
            applicability_conditions=updated_bundle.applicability_conditions,
            validator_expectations=updated_bundle.validator_expectations,
            repair_policy=updated_bundle.repair_policy,
            allowed_materialization_modes=updated_bundle.allowed_materialization_modes,
            blocked_scaffold_signatures=updated_bundle.blocked_scaffold_signatures,
            forbidden_overreach_patterns=updated_bundle.forbidden_overreach_patterns,
            forbidden_relation_families=updated_bundle.forbidden_relation_families,
        )
        reason_parts = list(dangerous_patterns)
        if not reason_parts:
            reason_parts = _dedupe_strings(
                [
                    f"family_failure_class:{cleaned_failure_class}",
                    *[
                        cleaned_reason
                        for cleaned_reason in (
                            str(failure_reason or "").strip()
                            for failure_reason in failure_reasons
                        )
                        if cleaned_reason
                    ],
                ]
            )[:4]
        reason = "family_failure:" + ",".join(reason_parts)
        now = datetime.now(UTC).isoformat()
        existing_versions[candidate_version] = {
            "status": "candidate",
            "parent_version": active_version,
            "bundle": candidate_bundle.as_dict(),
            "fields_changed": list(fields_changed),
            "reason_for_change": reason,
            "evaluation_results": {},
            "promotion_decision": {
                "decision": "pending",
                "reason": "awaiting_regression_gate",
            },
            "created_at": now,
            "updated_at": now,
            "trigger_context": {
                **dict(trigger_context),
                "failure_class": cleaned_failure_class,
            },
        }
        payload["versions"] = existing_versions
        self._save_family_payload(family_name, payload)
        return FamilyPolicyCandidateUpdate(
            family_name=family_name,
            base_version=active_version,
            candidate_version=candidate_version,
            bundle=candidate_bundle,
            fields_changed=tuple(fields_changed),
            reason_for_change=reason,
            trigger_context=dict(trigger_context),
        )

    def update_candidate_evaluation(
        self,
        family_name: str,
        *,
        candidate_version: str,
        evaluation_results: Mapping[str, Any],
    ) -> None:
        payload = self._load_family_payload(family_name)
        versions = payload.get("versions") or {}
        version_payload = versions.get(candidate_version)
        if not isinstance(version_payload, dict):
            raise KeyError(f"unknown_family_candidate:{family_name}:{candidate_version}")
        version_payload["evaluation_results"] = dict(evaluation_results)
        version_payload["updated_at"] = datetime.now(UTC).isoformat()
        self._save_family_payload(family_name, payload)

    def promote_candidate(
        self,
        family_name: str,
        *,
        candidate_version: str,
        evaluation_results: Mapping[str, Any],
        promotion_reason: str,
    ) -> None:
        payload = self._load_family_payload(family_name)
        versions = payload.get("versions") or {}
        active_version = str(payload.get("active_version") or "").strip()
        if active_version and isinstance(versions.get(active_version), dict):
            versions[active_version]["status"] = "archived"
            versions[active_version]["updated_at"] = datetime.now(UTC).isoformat()
        candidate_payload = versions.get(candidate_version)
        if not isinstance(candidate_payload, dict):
            raise KeyError(f"unknown_family_candidate:{family_name}:{candidate_version}")
        candidate_payload["status"] = "active"
        candidate_payload["evaluation_results"] = dict(evaluation_results)
        candidate_payload["promotion_decision"] = {
            "decision": "promoted",
            "reason": promotion_reason,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        candidate_payload["updated_at"] = datetime.now(UTC).isoformat()
        payload["active_version"] = candidate_version
        self._save_family_payload(family_name, payload)

    def reject_candidate(
        self,
        family_name: str,
        *,
        candidate_version: str,
        evaluation_results: Mapping[str, Any],
        rejection_reason: str,
    ) -> None:
        payload = self._load_family_payload(family_name)
        versions = payload.get("versions") or {}
        candidate_payload = versions.get(candidate_version)
        if not isinstance(candidate_payload, dict):
            raise KeyError(f"unknown_family_candidate:{family_name}:{candidate_version}")
        candidate_payload["status"] = "rejected"
        candidate_payload["evaluation_results"] = dict(evaluation_results)
        candidate_payload["promotion_decision"] = {
            "decision": "rejected",
            "reason": rejection_reason,
            "rejected_at": datetime.now(UTC).isoformat(),
        }
        candidate_payload["updated_at"] = datetime.now(UTC).isoformat()
        self._save_family_payload(family_name, payload)

    def get_pending_candidates(self, family_name: str) -> list[dict[str, Any]]:
        payload = self._load_family_payload(family_name)
        pending: list[dict[str, Any]] = []
        for version_name, version_payload in (payload.get("versions") or {}).items():
            if not isinstance(version_payload, Mapping):
                continue
            if str(version_payload.get("status") or "").strip() != "candidate":
                continue
            pending.append(
                {
                    "candidate_version": str(version_name),
                    **dict(version_payload),
                }
            )
        pending.sort(key=lambda item: str(item.get("created_at") or ""))
        return pending


def build_family_policy_store(
    *,
    baseline_bundles: Mapping[str, FamilyPolicyBundle],
    store_path: Path | None = None,
) -> FamilyPolicyStore:
    return FamilyPolicyStore(
        store_path=store_path or resolve_family_policy_store_path(),
        baseline_bundles=baseline_bundles,
    )


def summarize_sample_metrics(sample_result: Mapping[str, Any]) -> dict[str, Any]:
    sample_status = str(sample_result.get("sample_status") or "").strip()
    evaluation_outcome = str(sample_result.get("evaluation_outcome") or "").strip()
    wrong_completed = int(sample_status == "completed" and evaluation_outcome != "correct")
    correct_completed = int(sample_status == "completed" and evaluation_outcome == "correct")
    dangerous_overreach_count = int(sample_result.get("dangerous_overreach_count") or 0)
    if correct_completed:
        score = 3
    elif wrong_completed:
        score = 0
    else:
        score = 1
    return {
        "sample_index": str(sample_result.get("sample_index") or "").strip(),
        "sample_status": sample_status,
        "evaluation_outcome": evaluation_outcome,
        "wrong_completed": wrong_completed,
        "correct_completed": correct_completed,
        "dangerous_overreach_count": dangerous_overreach_count,
        "material_improvement_score": score,
    }


__all__ = [
    "ENV_ENABLE_EVOLUTION",
    "ENV_ENABLE_PROMOTION",
    "ENV_ENABLED_FAMILIES",
    "ENV_OVERRIDE_VERSIONS",
    "ENV_STORE_PATH",
    "FamilyPolicyCandidateUpdate",
    "FamilyPolicyStore",
    "build_family_policy_store",
    "bundle_from_dict",
    "family_policy_enabled_for",
    "family_policy_evolution_enabled",
    "family_policy_promotion_enabled",
    "parse_override_versions",
    "resolve_family_policy_store_path",
    "summarize_sample_metrics",
]
