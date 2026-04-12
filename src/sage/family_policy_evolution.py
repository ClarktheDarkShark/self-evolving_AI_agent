from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.sage.family_contracts import get_family_contract
from src.sage.policy_contracts import FamilyPolicyBundle


ENV_ENABLE_EVOLUTION = "SAGE_ENABLE_FAMILY_POLICY_EVOLUTION"
ENV_ENABLE_PROMOTION = "SAGE_ENABLE_FAMILY_POLICY_PROMOTION"
ENV_ENABLED_FAMILIES = "SAGE_FAMILY_POLICY_EVOLUTION_FAMILIES"
ENV_STORE_PATH = "SAGE_FAMILY_POLICY_STORE_PATH"
ENV_OVERRIDE_VERSIONS = "SAGE_FAMILY_POLICY_OVERRIDE_VERSIONS"
ENV_COMPARE_LOCK_FAMILY = "SAGE_FAMILY_POLICY_COMPARE_LOCK_FAMILY"
ENV_STRICT_UPDATE_MAPPING = "SAGE_FAMILY_POLICY_STRICT_UPDATE_MAPPING"
ENV_DEDUP_SIGNATURE = "SAGE_FAMILY_POLICY_DEDUP_SIGNATURE"
ENV_STRUCTURED_SUCCESS_FEATURES = "SAGE_FAMILY_POLICY_STRUCTURED_SUCCESS_FEATURES"


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
    normalized_reasons = [reason.lower() for reason in cleaned_reasons]

    def _has_reason_fragment(*fragments: str) -> bool:
        for fragment in fragments:
            cleaned_fragment = str(fragment or "").strip().lower()
            if cleaned_fragment and any(
                cleaned_fragment in reason for reason in normalized_reasons
            ):
                return True
        return False

    if dangerous_overreach or any(
        reason.startswith("dangerous_overreach:") for reason in cleaned_reasons
    ):
        return "validator_miss"
    if any("query_uses_unplanned_predicate:" in reason for reason in cleaned_reasons):
        return "forbidden_relation_family_miss"
    if cleaned_family in {
        "count_over_direct_relation",
        "count_over_joined_set",
        "multi_anchor_intersection",
        "shared_type_intersection",
    } and _has_reason_fragment(
        "repairable_bad_count_set",
        "count_answer_target_unenforced",
        "count_set_weak_or_broken",
        "trusted_incorrect_completion",
    ):
        return "weak_applicability_boundary"
    if cleaned_family == "single_anchor_lookup" and _has_reason_fragment(
        "repairable_weak_grounding",
        "repairable_anchor_path_empty",
        "repairable_anchor_not_found",
        "anchor_path_empty",
        "anchor_not_found",
    ):
        return "bad_routing"
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
    output_dir = Path(os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/sage_runtime"))
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


def compare_locked_family_name() -> str:
    return str(os.environ.get(ENV_COMPARE_LOCK_FAMILY) or "").strip()


def strict_update_mapping_enabled() -> bool:
    return os.environ.get(ENV_STRICT_UPDATE_MAPPING) == "1"


def candidate_signature_dedup_enabled() -> bool:
    return os.environ.get(ENV_DEDUP_SIGNATURE) == "1"


def structured_success_features_enabled() -> bool:
    return os.environ.get(ENV_STRUCTURED_SUCCESS_FEATURES) == "1"


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


def _reason_contains_any(
    failure_reasons: Sequence[str],
    *fragments: str,
) -> bool:
    lowered_reasons = [str(item or "").strip().lower() for item in failure_reasons]
    for fragment in fragments:
        cleaned_fragment = str(fragment or "").strip().lower()
        if cleaned_fragment and any(cleaned_fragment in reason for reason in lowered_reasons):
            return True
    return False


def version_reuse_compatible(source_version: str, active_version: str) -> bool:
    cleaned_source = str(source_version or "").strip()
    cleaned_active = str(active_version or "").strip()
    if not cleaned_source or not cleaned_active:
        return False
    return cleaned_active == cleaned_source or cleaned_active.startswith(
        f"{cleaned_source}__cand"
    )


def _constructive_policy_updates(
    *,
    family_name: str,
    failure_class: str,
    failure_reasons: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    applicability_conditions: list[str] = []
    validator_expectations: list[str] = []
    repair_policy: list[str] = []

    if failure_class == "weak_applicability_boundary":
        applicability_conditions.append("require_explicit_answer_role_alignment")
        validator_expectations.append(
            "verify_answer_target_semantics_not_just_executability"
        )
        repair_policy.append("prefer_structural_rebinding_before_relation_bans")
    if failure_class in {"bad_routing", "bad_fallback_ordering"}:
        repair_policy.extend(
            (
                "change_scaffold_family_after_dead_relation_evidence",
                "prefer_unused_grounded_relations_before_retrying",
            )
        )
    if family_name == "count_over_direct_relation":
        if failure_class in {
            "weak_applicability_boundary",
            "bad_routing",
            "bad_fallback_ordering",
        }:
            validator_expectations.append(
                "verify_count_targets_requested_entity_set"
            )
            repair_policy.append(
                "switch_to_joined_count_when_downstream_filter_or_projection_exists"
            )
    elif family_name == "count_over_joined_set":
        if failure_class in {
            "weak_applicability_boundary",
            "bad_routing",
            "bad_fallback_ordering",
        }:
            validator_expectations.append(
                "verify_count_targets_requested_entity_set"
            )
            repair_policy.append("preserve_joined_count_target_semantics_during_repairs")
    elif family_name in {"single_anchor_lookup", "single_anchor_chain_lookup"}:
        if failure_class in {"weak_applicability_boundary", "bad_routing"}:
            validator_expectations.append(
                "verify_answer_target_semantics_not_just_executability"
            )
        if failure_class == "weak_applicability_boundary":
            validator_expectations.append(
                "verify_projected_entity_matches_question_target"
            )
        if _reason_contains_any(
            failure_reasons,
            "repairable_anchor_path_empty",
            "anchor_path_empty",
        ):
            validator_expectations.append(
                "require_direct_single_anchor_answer_path"
            )
    elif family_name == "superlative_chain":
        if failure_class in {"weak_applicability_boundary", "bad_routing"}:
            validator_expectations.append(
                "verify_superlative_selection_basis_is_explicit"
            )
            repair_policy.append("preserve_candidate_set_and_ordering_path_together")
    elif family_name in {"multi_anchor_intersection", "shared_type_intersection"}:
        if failure_class == "weak_applicability_boundary":
            validator_expectations.append(
                "preserve_all_anchor_constraints_on_same_answer_variable"
            )
    if family_name == "count_over_joined_set" and _reason_contains_any(
        failure_reasons,
        "count_answer_target_unenforced",
        "count_query_counts_wrong_variable",
        "count_repair_dropped_preserved_relation_family",
        "repair:preserve_live_count_target_family_after_pivot",
        "dangerous_overreach:weak_count_semantics",
    ):
        validator_expectations.append("require_count_answer_target_preservation")

    return {
        "applicability_conditions": _dedupe_strings(applicability_conditions),
        "validator_expectations": _dedupe_strings(validator_expectations),
        "repair_policy": _dedupe_strings(repair_policy),
    }


def _normalize_role_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _allowed_update_fields_for_failure(
    *,
    family_name: str,
    failure_class: str,
    failure_reasons: Sequence[str] = (),
) -> frozenset[str]:
    cleaned_family = str(family_name or "").strip()
    cleaned_failure = str(failure_class or "").strip()
    if cleaned_failure == "bad_routing":
        if cleaned_family in {"single_anchor_lookup", "single_anchor_chain_lookup"}:
            if not _reason_contains_any(
                failure_reasons,
                "repairable_anchor_path_empty",
                "anchor_path_empty",
                "anchor_paths_live_but_projection_empty",
                "query_shape:single_anchor_lookup",
                "family_compare_locked_query_shape",
                "single_anchor_chain_lookup",
            ):
                return frozenset()
            return frozenset({"validator_expectations", "blocked_scaffold_signatures"})
        return frozenset({"repair_policy", "blocked_scaffold_signatures"})
    if cleaned_failure == "bad_fallback_ordering":
        return frozenset({"repair_policy"})
    if cleaned_failure in {"validator_miss", "forbidden_relation_family_miss"}:
        if cleaned_family in {"single_anchor_lookup", "single_anchor_chain_lookup"}:
            if cleaned_failure == "validator_miss":
                if not _reason_contains_any(
                    failure_reasons,
                    "entity_answer_target_unenforced",
                    "dangerous_overreach:weak_entity_semantics",
                    "generic_type_result_overbroad_for_answer_target",
                ):
                    return frozenset()
                return frozenset(
                    {
                        "validator_expectations",
                        "blocked_scaffold_signatures",
                        "forbidden_overreach_patterns",
                    }
                )
            return frozenset({"forbidden_relation_families"})
        return frozenset(
            {
                "validator_expectations",
                "blocked_scaffold_signatures",
                "forbidden_overreach_patterns",
                "forbidden_relation_families",
            }
        )
    if cleaned_failure == "weak_applicability_boundary":
        if cleaned_family in {
            "count_over_direct_relation",
            "count_over_joined_set",
            "multi_anchor_intersection",
            "shared_type_intersection",
        }:
            return frozenset({"validator_expectations"})
        if cleaned_family in {"single_anchor_lookup", "single_anchor_chain_lookup"}:
            if not _reason_contains_any(
                failure_reasons,
                "repairable_anchor_path_empty",
                "anchor_path_empty",
                "anchor_paths_live_but_projection_empty",
                "repairable_weak_grounding",
                "entity_answer_target_unenforced",
                "dangerous_overreach:weak_entity_semantics",
                "grounded_single_anchor_empty_result",
                "anchor_not_found",
            ):
                return frozenset()
            return frozenset({"validator_expectations"})
        return frozenset({"applicability_conditions", "validator_expectations"})
    return frozenset()


def build_candidate_signature(
    *,
    bundle: FamilyPolicyBundle,
    fields_changed: Sequence[str],
) -> dict[str, Any]:
    field_names = (
        "applicability_conditions",
        "validator_expectations",
        "repair_policy",
        "blocked_scaffold_signatures",
        "forbidden_overreach_patterns",
        "forbidden_relation_families",
    )
    cleaned_validators = {
        str(item or "").strip()
        for item in bundle.validator_expectations
        if str(item or "").strip()
    }
    validator_groups = (
        (
            "single_anchor_direct_path",
            {
                "require_direct_single_anchor_answer_path",
                "verify_projected_entity_matches_question_target",
            },
        ),
        (
            "answer_target_semantics",
            {
                "verify_answer_target_semantics_not_just_executability",
                "tighten_answer_target_semantics",
            },
        ),
        (
            "count_answer_target_preservation",
            {
                "verify_count_targets_requested_entity_set",
                "require_count_answer_target_preservation",
            },
        ),
        (
            "dangerous_overreach_rejection",
            {"reject_known_dangerous_overreach_patterns"},
        ),
    )
    validator_tags: list[str] = []
    covered_validators: set[str] = set()
    for tag, members in validator_groups:
        if cleaned_validators & members:
            validator_tags.append(tag)
            covered_validators.update(members)
    for item in sorted(cleaned_validators - covered_validators):
        validator_tags.append(f"validator:{item}")

    blocked_scaffold_tags: list[str] = []
    for raw_value in bundle.blocked_scaffold_signatures:
        cleaned = str(raw_value or "").strip()
        if not cleaned:
            continue
        parts = [part for part in cleaned.split("|") if part]
        family_name = parts[0] if parts else cleaned
        relation_parts = [part for part in parts[1:] if "." in part]
        normalized = (
            "|".join([family_name, *relation_parts])
            if relation_parts
            else cleaned
        )
        if normalized not in blocked_scaffold_tags:
            blocked_scaffold_tags.append(normalized)

    return {
        "family_name": str(bundle.family_name or "").strip(),
        "renderer_name": str(bundle.renderer_name or "").strip(),
        "fields_changed": [
            str(field_name or "").strip()
            for field_name in fields_changed
            if str(field_name or "").strip()
        ],
        "field_payloads": {
            field_name: [
                str(item or "").strip()
                for item in getattr(bundle, field_name)
                if str(item or "").strip()
            ]
            for field_name in field_names
        },
        "effective_constraint_signature": {
            "validator_tags": validator_tags,
            "blocked_scaffold_tags": blocked_scaffold_tags,
            "forbidden_relation_families": sorted(
                {
                    str(item or "").strip()
                    for item in bundle.forbidden_relation_families
                    if str(item or "").strip()
                }
            ),
            "forbidden_overreach_patterns": sorted(
                {
                    str(item or "").strip()
                    for item in bundle.forbidden_overreach_patterns
                    if str(item or "").strip()
                }
            ),
        },
    }


def build_success_plan_archetype(
    query_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(query_plan, Mapping):
        return {}

    query_shape = str(query_plan.get("query_shape") or "").strip().lower()
    answer_mode = str(query_plan.get("answer_mode") or "").strip().lower()
    join_structure = (
        dict(query_plan.get("join_structure"))
        if isinstance(query_plan.get("join_structure"), Mapping)
        else {}
    )
    join_type = str(join_structure.get("type") or "").strip().lower()

    anchor_roles: list[str] = []
    for item in query_plan.get("anchored_entities") or []:
        if not isinstance(item, Mapping):
            continue
        role = _normalize_role_token(item.get("role"))
        if role and role not in anchor_roles:
            anchor_roles.append(role)

    anchor_constraints: list[str] = []
    for item in join_structure.get("anchor_constraints") or []:
        if not isinstance(item, Mapping):
            continue
        anchor_role = _normalize_role_token(item.get("anchor_role"))
        target_variable = str(item.get("constrains_variable") or "").strip()
        if anchor_role and target_variable:
            anchor_constraints.append(f"{anchor_role}->{target_variable}")

    relation_role_skeleton: list[str] = []
    for relation_path in query_plan.get("relation_paths") or []:
        if not isinstance(relation_path, Mapping):
            continue
        from_role = _normalize_role_token(relation_path.get("from_role"))
        to_role = _normalize_role_token(relation_path.get("to_role"))
        grounding_source = str(relation_path.get("grounding_source") or "").strip().lower()
        if from_role and to_role:
            relation_role_skeleton.append(
                f"{from_role}->{to_role}"
                + (f":{grounding_source}" if grounding_source else "")
            )

    candidate_set_variable = str(query_plan.get("candidate_set_variable") or "").strip()
    count_set_variable = str(query_plan.get("count_set_variable") or "").strip()
    shared_answer_variable = str(query_plan.get("shared_answer_variable") or "").strip()

    structural_notes: list[str] = []
    anchor_like_roles = [role for role in anchor_roles if role in {"anchor", "anchor_a", "anchor_b"}]
    if len(anchor_like_roles) >= 2:
        structural_notes.append("preserve_multiple_anchor_constraints")
    if any(item.startswith("constraint_value->") for item in anchor_constraints):
        structural_notes.append("explicit_constraint_value_binding")
    if any(item.startswith("shared_type->") or item.startswith("type_set->") for item in anchor_constraints):
        structural_notes.append("explicit_type_constraint_binding")
    if count_set_variable and candidate_set_variable and count_set_variable != candidate_set_variable:
        structural_notes.append("count_target_distinct_from_candidate_set")
    if shared_answer_variable and shared_answer_variable == candidate_set_variable:
        structural_notes.append("shared_answer_equals_candidate_set")
    if query_shape == "count_over_joined_set" and join_type == "count":
        structural_notes.append("count_join_requires_explicit_joined_set")

    pattern_signature_parts = [
        query_shape,
        answer_mode,
        join_type,
        ",".join(anchor_roles),
        ",".join(anchor_constraints),
        ",".join(relation_role_skeleton),
        candidate_set_variable,
        count_set_variable,
        shared_answer_variable,
        ",".join(structural_notes),
    ]
    pattern_signature = "|".join(part for part in pattern_signature_parts if part)

    payload = {
        "query_shape": query_shape,
        "answer_mode": answer_mode,
        "anchor_count": len(anchor_roles),
        "relation_depth": len(relation_role_skeleton),
        "join_type": join_type,
        "anchor_roles": anchor_roles,
        "anchor_constraints": anchor_constraints,
        "relation_role_skeleton": relation_role_skeleton[:4],
        "candidate_set_variable": candidate_set_variable,
        "count_set_variable": count_set_variable,
        "shared_answer_variable": shared_answer_variable,
        "structural_notes": structural_notes,
        "pattern_signature": pattern_signature,
    }
    if structured_success_features_enabled():
        grounding_sources = sorted(
            {
                str(relation_path.get("grounding_source") or "").strip().lower()
                for relation_path in query_plan.get("relation_paths") or []
                if isinstance(relation_path, Mapping)
                and str(relation_path.get("grounding_source") or "").strip()
            }
        )
        relation_signatures = [
            "->".join(
                part
                for part in (
                    _normalize_role_token(relation_path.get("from_role")),
                    str(relation_path.get("relation") or "").strip(),
                    _normalize_role_token(relation_path.get("to_role")),
                    str(relation_path.get("grounding_source") or "").strip().lower(),
                )
                if part
            )
            for relation_path in query_plan.get("relation_paths") or []
            if isinstance(relation_path, Mapping)
        ]
        anchor_binding_modes: list[str] = []
        for item in query_plan.get("anchored_entities") or []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("resolved_entity_id") or "").strip():
                anchor_binding_modes.append("resolved_entity_id")
            chosen_alias = str(item.get("chosen_alias") or "").strip()
            surface = str(item.get("surface") or "").strip()
            if chosen_alias and chosen_alias.startswith("m."):
                anchor_binding_modes.append("freebase_mid_anchor")
            elif chosen_alias and surface and chosen_alias.lower() != surface.lower():
                anchor_binding_modes.append("surface_alias_anchor")
        payload.update(
            {
                "grounding_sources": grounding_sources,
                "relation_signatures": relation_signatures[:4],
                "anchor_binding_modes": list(_dedupe_strings(anchor_binding_modes)),
                "answer_target_present": bool(
                    str(query_plan.get("answer_target_phrase") or "").strip()
                ),
            }
        )
    return payload

def merge_success_plan_archetypes(
    existing_archetypes: Sequence[Mapping[str, Any]],
    new_archetype: Mapping[str, Any],
    *,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_signatures: list[str] = []

    def _append(item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            return
        signature = str(item.get("pattern_signature") or "").strip()
        if not signature or signature in seen_signatures:
            return
        seen_signatures.append(signature)
        merged.append(dict(item))

    for archetype in existing_archetypes:
        _append(archetype)
    _append(new_archetype)
    return merged[-max_items:]


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [
            _json_safe_value(item)
            for item in sorted(value, key=lambda item: str(item))
        ]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


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
            json.dumps(
                _json_safe_value(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
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

    def get_tool_evolution_context(self, family_name: str) -> dict[str, Any]:
        payload = self._load_family_payload(family_name)
        context = payload.get("tool_evolution_context") or {}
        return dict(context) if isinstance(context, Mapping) else {}

    def set_tool_evolution_context(
        self,
        family_name: str,
        *,
        source_version: str,
        preferred_patterns: Sequence[Mapping[str, Any]] | None = None,
        avoid_patterns: Sequence[Mapping[str, Any]] | None = None,
        last_signal: Mapping[str, Any] | None = None,
    ) -> None:
        payload = self._load_family_payload(family_name)
        context: dict[str, Any] = {
            "source_version": str(source_version or "").strip(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if preferred_patterns:
            context["preferred_patterns"] = [dict(item) for item in preferred_patterns if isinstance(item, Mapping)]
        if avoid_patterns:
            context["avoid_patterns"] = [dict(item) for item in avoid_patterns if isinstance(item, Mapping)]
        if last_signal:
            context["last_signal"] = dict(last_signal)
        payload["tool_evolution_context"] = context
        self._save_family_payload(family_name, payload)

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
            "bad_routing",
            "bad_fallback_ordering",
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
        if cleaned_failure_class in {
            "forbidden_relation_family_miss",
            "validator_miss",
        }:
            for relation_name in relation_names:
                relation_text = str(relation_name or "").strip()
                if not relation_text:
                    continue
                if cleaned_failure_class == "validator_miss" and not relation_text.startswith(
                    "type."
                ):
                    continue
                if relation_text not in forbidden_relation_families:
                    forbidden_relation_families.append(relation_text)
        constructive_updates = _constructive_policy_updates(
            family_name=family_name,
            failure_class=cleaned_failure_class,
            failure_reasons=failure_reasons,
        )
        applicability_conditions.extend(
            constructive_updates["applicability_conditions"]
        )
        validator_expectations.extend(
            constructive_updates["validator_expectations"]
        )
        repair_policy.extend(constructive_updates["repair_policy"])

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
            if cleaned_scaffold_signature and cleaned_scaffold_signature not in blocked_scaffolds:
                blocked_scaffolds.append(cleaned_scaffold_signature)
        if cleaned_failure_class in {"bad_routing", "bad_fallback_ordering"}:
            retry_guard = "deprioritize_failed_scaffold_before_family_switch"
            if retry_guard not in repair_policy:
                repair_policy.append(retry_guard)

        if strict_update_mapping_enabled():
            allowed_fields = _allowed_update_fields_for_failure(
                family_name=family_name,
                failure_class=cleaned_failure_class,
                failure_reasons=failure_reasons,
            )
            if not allowed_fields:
                return None
            if "applicability_conditions" not in allowed_fields:
                applicability_conditions = list(active_bundle.applicability_conditions)
            if "validator_expectations" not in allowed_fields:
                validator_expectations = list(active_bundle.validator_expectations)
            if "repair_policy" not in allowed_fields:
                repair_policy = list(active_bundle.repair_policy)
            if "blocked_scaffold_signatures" not in allowed_fields:
                blocked_scaffolds = list(active_bundle.blocked_scaffold_signatures)
            if "forbidden_overreach_patterns" not in allowed_fields:
                forbidden_overreach_patterns = list(
                    active_bundle.forbidden_overreach_patterns
                )
            if "forbidden_relation_families" not in allowed_fields:
                forbidden_relation_families = list(
                    active_bundle.forbidden_relation_families
                )

        family_contract = get_family_contract(family_name)
        runtime_consumable_fields = set(
            family_contract.runtime_consumable_fields if family_contract is not None else ()
        )
        if runtime_consumable_fields:
            if "applicability_conditions" not in runtime_consumable_fields:
                applicability_conditions = list(active_bundle.applicability_conditions)
            if "validator_expectations" in runtime_consumable_fields:
                runtime_validator_expectations = set(
                    family_contract.runtime_validator_expectations
                    if family_contract is not None
                    else ()
                )
                validator_expectations = list(active_bundle.validator_expectations) + [
                    expectation
                    for expectation in validator_expectations
                    if expectation not in active_bundle.validator_expectations
                    and expectation in runtime_validator_expectations
                ]
            else:
                validator_expectations = list(active_bundle.validator_expectations)
            if "repair_policy" not in runtime_consumable_fields:
                repair_policy = list(active_bundle.repair_policy)
            if "blocked_scaffold_signatures" not in runtime_consumable_fields:
                blocked_scaffolds = list(active_bundle.blocked_scaffold_signatures)
            if "forbidden_overreach_patterns" not in runtime_consumable_fields:
                forbidden_overreach_patterns = list(
                    active_bundle.forbidden_overreach_patterns
                )
            if "forbidden_relation_families" not in runtime_consumable_fields:
                forbidden_relation_families = list(
                    active_bundle.forbidden_relation_families
                )

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

        candidate_signature = build_candidate_signature(
            bundle=updated_bundle,
            fields_changed=fields_changed,
        )

        existing_versions = payload.get("versions") or {}
        updated_bundle_payload = updated_bundle.as_dict()
        for version_name, version_payload in existing_versions.items():
            if not isinstance(version_payload, Mapping):
                continue
            existing_signature = version_payload.get("candidate_signature")
            if not isinstance(existing_signature, Mapping):
                existing_bundle_payload = version_payload.get("bundle")
                if isinstance(existing_bundle_payload, Mapping):
                    try:
                        existing_signature = build_candidate_signature(
                            bundle=bundle_from_dict(existing_bundle_payload),
                            fields_changed=version_payload.get("fields_changed") or [],
                        )
                    except Exception:
                        existing_signature = None
            existing_effective_signature = (
                dict(existing_signature.get("effective_constraint_signature") or {})
                if isinstance(existing_signature, Mapping)
                else {}
            )
            candidate_effective_signature = dict(
                candidate_signature.get("effective_constraint_signature") or {}
            )
            if (
                candidate_signature_dedup_enabled()
                and isinstance(existing_signature, Mapping)
                and (
                    dict(existing_signature) == candidate_signature
                    or (
                        existing_effective_signature
                        and existing_effective_signature
                        == candidate_effective_signature
                    )
                )
            ):
                if str(version_payload.get("status") or "").strip() == "candidate":
                    return FamilyPolicyCandidateUpdate(
                        family_name=family_name,
                        base_version=active_version,
                        candidate_version=str(version_name),
                        bundle=bundle_from_dict(updated_bundle_payload),
                        fields_changed=tuple(fields_changed),
                        reason_for_change=str(
                            version_payload.get("reason_for_change")
                            or "duplicate_candidate"
                        ).strip(),
                        trigger_context=dict(trigger_context),
                    )
                return None
            existing_bundle_payload = version_payload.get("bundle") or {}
            if isinstance(existing_bundle_payload, Mapping):
                existing_bundle_payload = dict(existing_bundle_payload)
                existing_bundle_payload.pop("version", None)
            updated_bundle_payload_no_version = dict(updated_bundle_payload)
            updated_bundle_payload_no_version.pop("version", None)
            if existing_bundle_payload == updated_bundle_payload_no_version:
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
                return None

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
            "candidate_signature": candidate_signature,
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
        trusted_success_metadata = payload.get("trusted_success_bank_metadata")
        if isinstance(trusted_success_metadata, dict) and (
            str(trusted_success_metadata.get("source_version") or "").strip()
            == active_version
        ):
            trusted_success_metadata["source_version"] = candidate_version
            evaluation_context = trusted_success_metadata.get("evaluation_context")
            if isinstance(evaluation_context, dict) and (
                str(evaluation_context.get("source_version") or "").strip()
                == active_version
            ):
                evaluation_context["source_version"] = candidate_version
        tool_evolution_context = payload.get("tool_evolution_context")
        if isinstance(tool_evolution_context, dict) and (
            str(tool_evolution_context.get("source_version") or "").strip()
            == active_version
        ):
            tool_evolution_context["source_version"] = candidate_version
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
    "ENV_DEDUP_SIGNATURE",
    "ENV_COMPARE_LOCK_FAMILY",
    "ENV_ENABLE_EVOLUTION",
    "ENV_ENABLE_PROMOTION",
    "ENV_ENABLED_FAMILIES",
    "ENV_OVERRIDE_VERSIONS",
    "ENV_STRICT_UPDATE_MAPPING",
    "ENV_STORE_PATH",
    "FamilyPolicyCandidateUpdate",
    "FamilyPolicyStore",
    "build_candidate_signature",
    "build_family_policy_store",
    "bundle_from_dict",
    "candidate_signature_dedup_enabled",
    "compare_locked_family_name",
    "family_policy_enabled_for",
    "family_policy_evolution_enabled",
    "family_policy_promotion_enabled",
    "parse_override_versions",
    "resolve_family_policy_store_path",
    "strict_update_mapping_enabled",
    "summarize_sample_metrics",
]
