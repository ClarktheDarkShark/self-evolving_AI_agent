from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FamilyContract:
    family_name: str
    required_slots: tuple[str, ...] = ()
    optional_slots: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    allowed_exhaustion_states: tuple[str, ...] = ()
    permitted_repair_classes: tuple[str, ...] = ()
    handoff_expectations: tuple[str, ...] = ()
    runtime_consumable_fields: tuple[str, ...] = ()
    runtime_validator_expectations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "invariants": list(self.invariants),
            "allowed_exhaustion_states": list(self.allowed_exhaustion_states),
            "permitted_repair_classes": list(self.permitted_repair_classes),
            "handoff_expectations": list(self.handoff_expectations),
            "runtime_consumable_fields": list(self.runtime_consumable_fields),
            "runtime_validator_expectations": list(self.runtime_validator_expectations),
        }


@dataclass(frozen=True)
class FamilyAttemptOutcome:
    family_name: str
    query_shape: str
    completion_state: str
    stop_reason: str
    primary_failure_kind: str
    plausibility_verdict: str = ""
    verdict_reasons: tuple[str, ...] = ()
    execution_success: bool = False
    binding_count: int | None = None
    scalar_count: str | None = None
    repair_attempt_count: int = 0
    repair_budget_exhausted: bool = False
    stagnation_detected: bool = False
    candidate_score: int | None = None
    grounding_summary: Mapping[str, Any] = field(default_factory=dict)
    trust_contract: Mapping[str, Any] = field(default_factory=dict)
    family_contract: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "query_shape": self.query_shape,
            "completion_state": self.completion_state,
            "stop_reason": self.stop_reason,
            "primary_failure_kind": self.primary_failure_kind,
            "plausibility_verdict": self.plausibility_verdict,
            "verdict_reasons": list(self.verdict_reasons),
            "execution_success": self.execution_success,
            "binding_count": self.binding_count,
            "scalar_count": self.scalar_count,
            "repair_attempt_count": self.repair_attempt_count,
            "repair_budget_exhausted": self.repair_budget_exhausted,
            "stagnation_detected": self.stagnation_detected,
            "candidate_score": self.candidate_score,
            "grounding_summary": dict(self.grounding_summary),
            "trust_contract": dict(self.trust_contract),
            "family_contract": dict(self.family_contract),
        }


_COMMON_EXHAUSTION_STATES = (
    "accepted_completion",
    "switch_family",
    "grounding_miss",
    "validator_blocked",
    "repair_budget_exhausted",
    "repeated_stagnation",
    "solver_handoff_failure",
    "infrastructure_failure",
)

_COMMON_RUNTIME_FIELDS = (
    "validator_expectations",
    "blocked_scaffold_signatures",
    "forbidden_relation_families",
)

_FAMILY_CONTRACTS = {
    "single_anchor_lookup": FamilyContract(
        family_name="single_anchor_lookup",
        required_slots=("anchor", "answer"),
        optional_slots=("constraint_value", "type_set"),
        invariants=(
            "projected answer must remain anchored to a grounded single-anchor scaffold",
            "projected entity must be structurally bound",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=(
            "alias_repair",
            "anchor_entity_repair",
            "structural_rewrite",
            "plan_refresh",
        ),
        handoff_expectations=("entity_id", "entity_set"),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "require_direct_single_anchor_answer_path",
            "verify_projected_entity_matches_question_target",
            "verify_answer_target_semantics_not_just_executability",
            "tighten_answer_target_semantics",
        ),
    ),
    "single_anchor_chain_lookup": FamilyContract(
        family_name="single_anchor_chain_lookup",
        required_slots=("anchor", "answer"),
        optional_slots=("constraint_value", "type_set"),
        invariants=(
            "chain path must preserve anchor semantics",
            "projected entity must be structurally bound",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=(
            "alias_repair",
            "anchor_entity_repair",
            "structural_rewrite",
            "plan_refresh",
        ),
        handoff_expectations=("entity_id", "entity_set"),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "require_direct_single_anchor_answer_path",
            "verify_projected_entity_matches_question_target",
            "verify_answer_target_semantics_not_just_executability",
            "tighten_answer_target_semantics",
        ),
    ),
    "multi_anchor_intersection": FamilyContract(
        family_name="multi_anchor_intersection",
        required_slots=("anchor_a", "anchor_b", "shared_answer"),
        optional_slots=("constraint_value", "type_set"),
        invariants=(
            "all anchors must constrain the same answer scaffold",
            "projected shared answer must not degrade to a generic ontology dump",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=("structural_rewrite", "plan_refresh"),
        handoff_expectations=("entity_id", "entity_set"),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "preserve_all_anchor_constraints_on_same_answer_variable",
        ),
    ),
    "shared_type_intersection": FamilyContract(
        family_name="shared_type_intersection",
        required_slots=("anchor_a", "anchor_b", "shared_type"),
        optional_slots=("shared_answer",),
        invariants=(
            "shared type projection must stay within the anchored scaffold",
            "generic shared-type dumps are not acceptable answers",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=("structural_rewrite", "plan_refresh"),
        handoff_expectations=("entity_id", "entity_set"),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "preserve_all_anchor_constraints_on_same_answer_variable",
        ),
    ),
    "containment_or_ownership_lookup": FamilyContract(
        family_name="containment_or_ownership_lookup",
        required_slots=("anchor", "answer"),
        optional_slots=("constraint_value", "type_set"),
        invariants=(
            "projected entity must stay within the ownership or containment relation family",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=(
            "alias_repair",
            "anchor_entity_repair",
            "structural_rewrite",
            "plan_refresh",
        ),
        handoff_expectations=("entity_id", "entity_set"),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "verify_answer_target_semantics_not_just_executability",
            "tighten_answer_target_semantics",
        ),
    ),
    "count_over_direct_relation": FamilyContract(
        family_name="count_over_direct_relation",
        required_slots=("anchor", "count_set"),
        optional_slots=("constraint_value", "type_set"),
        invariants=(
            "counted variable must be structurally bound",
            "counted set must preserve answer-target semantics",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=(
            "alias_repair",
            "anchor_entity_repair",
            "structural_rewrite",
            "plan_refresh",
        ),
        handoff_expectations=("count_scalar",),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "verify_count_targets_requested_entity_set",
        ),
    ),
    "count_over_joined_set": FamilyContract(
        family_name="count_over_joined_set",
        required_slots=("candidate_set", "count_set"),
        optional_slots=("anchor", "anchor_a", "anchor_b", "constraint_value", "type_set"),
        invariants=(
            "joined candidate set must be explicit before counting",
            "count target must remain aligned with the requested answer target",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=("structural_rewrite", "plan_refresh"),
        handoff_expectations=("count_scalar",),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "verify_count_targets_requested_entity_set",
            "require_count_answer_target_preservation",
        ),
    ),
    "superlative_chain": FamilyContract(
        family_name="superlative_chain",
        required_slots=("candidate_set", "ordering_attribute"),
        optional_slots=("answer", "shared_answer"),
        invariants=(
            "candidate set and ordering path must both be grounded",
            "ORDER BY and LIMIT 1 must remain explicit",
        ),
        allowed_exhaustion_states=_COMMON_EXHAUSTION_STATES,
        permitted_repair_classes=("structural_rewrite", "plan_refresh"),
        handoff_expectations=("entity_id", "entity_set", "scalar_literal", "text_literal"),
        runtime_consumable_fields=_COMMON_RUNTIME_FIELDS,
        runtime_validator_expectations=(
            "verify_superlative_selection_basis_is_explicit",
        ),
    ),
}


def get_family_contract(family_name: str) -> FamilyContract | None:
    return _FAMILY_CONTRACTS.get(str(family_name or "").strip())


def iter_family_contracts() -> tuple[FamilyContract, ...]:
    return tuple(_FAMILY_CONTRACTS.values())


__all__ = [
    "FamilyAttemptOutcome",
    "FamilyContract",
    "get_family_contract",
    "iter_family_contracts",
]
