from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class FamilyPolicyBundle:
    family_name: str
    version: str
    renderer_name: str
    applicability_conditions: tuple[str, ...] = ()
    validator_expectations: tuple[str, ...] = ()
    repair_policy: tuple[str, ...] = ()
    allowed_materialization_modes: tuple[str, ...] = ()
    blocked_scaffold_signatures: tuple[str, ...] = ()
    forbidden_overreach_patterns: tuple[str, ...] = ()
    forbidden_relation_families: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "version": self.version,
            "renderer_name": self.renderer_name,
            "applicability_conditions": list(self.applicability_conditions),
            "validator_expectations": list(self.validator_expectations),
            "repair_policy": list(self.repair_policy),
            "allowed_materialization_modes": list(self.allowed_materialization_modes),
            "blocked_scaffold_signatures": list(self.blocked_scaffold_signatures),
            "forbidden_overreach_patterns": list(self.forbidden_overreach_patterns),
            "forbidden_relation_families": list(self.forbidden_relation_families),
        }


@dataclass(frozen=True)
class TrustContractEvaluation:
    plan_consistency_passed: bool
    execution_shape_passed: bool
    plausibility_validation_passed: bool
    adapter_safety_passed: bool
    materialization_allowed: bool
    denial_reasons: tuple[str, ...] = ()
    dangerous_overreach: bool = False
    dangerous_overreach_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_consistency_passed": self.plan_consistency_passed,
            "execution_shape_passed": self.execution_shape_passed,
            "plausibility_validation_passed": self.plausibility_validation_passed,
            "adapter_safety_passed": self.adapter_safety_passed,
            "materialization_allowed": self.materialization_allowed,
            "denial_reasons": list(self.denial_reasons),
            "dangerous_overreach": self.dangerous_overreach,
            "dangerous_overreach_reasons": list(self.dangerous_overreach_reasons),
        }


@dataclass(frozen=True)
class AttemptDecisionRecord:
    selected_family: Optional[str]
    family_bundle_version: Optional[str]
    renderer_name: Optional[str]
    generation_source: str
    selection_reasons: tuple[str, ...] = ()
    stay_in_family_evidence: tuple[str, ...] = ()
    switch_family_evidence: tuple[str, ...] = ()
    trust_contract: Mapping[str, Any] = field(default_factory=dict)
    materialization_allowed: Optional[bool] = None
    materialization_denial_reasons: tuple[str, ...] = ()
    dangerous_overreach: bool = False
    dangerous_overreach_reasons: tuple[str, ...] = ()
    family_contract: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_family": self.selected_family,
            "family_bundle_version": self.family_bundle_version,
            "renderer_name": self.renderer_name,
            "generation_source": self.generation_source,
            "selection_reasons": list(self.selection_reasons),
            "stay_in_family_evidence": list(self.stay_in_family_evidence),
            "switch_family_evidence": list(self.switch_family_evidence),
            "trust_contract": dict(self.trust_contract),
            "materialization_allowed": self.materialization_allowed,
            "materialization_denial_reasons": list(self.materialization_denial_reasons),
            "dangerous_overreach": self.dangerous_overreach,
            "dangerous_overreach_reasons": list(self.dangerous_overreach_reasons),
            "family_contract": dict(self.family_contract),
        }


def build_trust_contract_evaluation(
    *,
    plan_consistency_passed: bool,
    execution_shape_passed: bool,
    plausibility_validation_passed: bool,
    adapter_safety_passed: bool,
    extra_denial_reasons: tuple[str, ...] | list[str] = (),
    dangerous_overreach_reasons: tuple[str, ...] | list[str] = (),
) -> TrustContractEvaluation:
    denial_reasons: list[str] = []
    if not plan_consistency_passed:
        denial_reasons.append("plan_consistency_failed")
    if not execution_shape_passed:
        denial_reasons.append("execution_shape_mismatch")
    if not plausibility_validation_passed:
        denial_reasons.append("plausibility_validation_failed")
    if not adapter_safety_passed:
        denial_reasons.append("adapter_safety_failed")
    for reason in extra_denial_reasons:
        cleaned = str(reason or "").strip()
        if cleaned and cleaned not in denial_reasons:
            denial_reasons.append(cleaned)
    dangerous_reasons = tuple(
        cleaned
        for cleaned in (str(reason or "").strip() for reason in dangerous_overreach_reasons)
        if cleaned
    )
    return TrustContractEvaluation(
        plan_consistency_passed=plan_consistency_passed,
        execution_shape_passed=execution_shape_passed,
        plausibility_validation_passed=plausibility_validation_passed,
        adapter_safety_passed=adapter_safety_passed,
        materialization_allowed=not denial_reasons,
        denial_reasons=tuple(denial_reasons),
        dangerous_overreach=bool(dangerous_reasons),
        dangerous_overreach_reasons=dangerous_reasons,
    )


__all__ = [
    "AttemptDecisionRecord",
    "FamilyPolicyBundle",
    "TrustContractEvaluation",
    "build_trust_contract_evaluation",
]
