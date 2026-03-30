"""
PAL Plausibility Validator
==========================
Leakage-safe, heuristic validator for PAL candidates *after* execution.

Signals used (all observable without the ground-truth answer):
  - query plan fields: answer_mode, strategy, anchored_entities,
    relation_paths, allow_exploratory_predicates
  - query text structure: ORDER BY, FILTER, anchor literals, predicates
  - execution result shape: binding count, emptiness, boolean presence
  - original entity list: anchor coverage in WHERE clause
  - anchor probe results: live KG existence checks for entity names

Signals NOT used:
  - expected answers
  - benchmark labels
  - evaluation outcomes
  - any hidden task solution information

Verdict categories
------------------
accepted
repairable_weak_grounding             – empty result + weak / exploratory plan
repairable_grounded_empty_result      – grounded multi-anchor plan returned empty
repairable_anchor_not_found           – live probe shows anchor name does not bind
repairable_anchor_path_empty          – anchor binds but relation path is empty
repairable_bad_join                   – multi-anchor plan missing an anchor literal in WHERE
repairable_bad_superlative_structure  – superlative strategy without ORDER BY
repairable_bad_count_set              – count plan with failed anchor probe or weak path
repairable_bad_projection             – projection / answer-mode mismatch detected
rejected_unsupported_predicate        – non-Freebase ontology used in WHERE
rejected_unbounded_exploration        – (reserved)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VERDICT_ACCEPTED = "accepted"
VERDICT_REPAIRABLE_WEAK_GROUNDING = "repairable_weak_grounding"
VERDICT_REPAIRABLE_GROUNDED_EMPTY = "repairable_grounded_empty_result"
VERDICT_REPAIRABLE_ANCHOR_NOT_FOUND = "repairable_anchor_not_found"
VERDICT_REPAIRABLE_ANCHOR_PATH_EMPTY = "repairable_anchor_path_empty"
VERDICT_REPAIRABLE_BAD_JOIN = "repairable_bad_join"
VERDICT_REPAIRABLE_BAD_SUPERLATIVE = "repairable_bad_superlative_structure"
VERDICT_REPAIRABLE_BAD_COUNT_SET = "repairable_bad_count_set"
VERDICT_REPAIRABLE_BAD_PROJECTION = "repairable_bad_projection"
VERDICT_REJECTED_UNSUPPORTED_PREDICATE = "rejected_unsupported_predicate"
VERDICT_REJECTED_UNBOUNDED = "rejected_unbounded_exploration"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AnchorProbeResult:
    """Result of a live KG existence probe for one anchor alias."""

    anchor_name: str
    """The alias / surface form that was probed."""

    entity_count: int
    """How many KG entities match this name (0 = not found, -1 = probe failed)."""

    path_count: int | None = None
    """Answers reachable via the primary relation (None = not probed)."""

    relation_probed: str | None = None
    """Which relation was used for the path probe (None = not probed)."""

    anchor_position: str | None = None
    """Whether the anchor was probed on the subject or object side of the triple."""

    resolved_entity_id: str | None = None
    """A uniquely resolved Freebase entity id for this anchor, when probing found one."""

    @property
    def found(self) -> bool:
        """True if at least one entity with this name exists in the KG."""
        return self.entity_count > 0

    @property
    def path_produces_results(self) -> bool | None:
        """True/False if path was probed; None otherwise."""
        if self.path_count is None:
            return None
        return self.path_count > 0


@dataclass
class PlausibilityVerdict:
    verdict: str
    reasons: list[str] = field(default_factory=list)

    @property
    def is_accepted(self) -> bool:
        return self.verdict == VERDICT_ACCEPTED

    @property
    def is_repairable(self) -> bool:
        return self.verdict.startswith("repairable_")

    @property
    def is_rejected(self) -> bool:
        return self.verdict.startswith("rejected_")


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_pal_execution(
    *,
    query_plan: Mapping[str, Any],
    query_text: str,
    result_dict: Mapping[str, Any] | None,
    entities: Sequence[str],
    anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
) -> PlausibilityVerdict:
    """
    Heuristic plausibility check for a PAL candidate after execution on the
    real KG.  Returns a PlausibilityVerdict classifying the candidate and
    explaining why it was accepted, flagged as repairable, or rejected.

    All signals are derived from observable artifacts only — no expected
    answers or benchmark labels are consulted.

    Parameters
    ----------
    query_plan
        Normalised plan dict from the plan generator.
    query_text
        The SPARQL query string extracted from the generated code.
    result_dict
        The JSON payload returned by the SPARQL endpoint, or None if
        execution failed.
    entities
        The raw entity list from the task question (for anchor coverage
        checks).
    anchor_probe_results
        Optional live KG probes for each anchor alias.  When provided,
        entity-existence and path-plausibility signals are activated.
    """
    answer_mode: str = str(query_plan.get("answer_mode") or "entity").lower()
    strategy: str = str(query_plan.get("strategy") or "").lower()
    query_shape: str = str(query_plan.get("query_shape") or "").lower()
    allow_exploratory: bool = bool(query_plan.get("allow_exploratory_predicates", False))
    anchored_entities: list[Any] = list(query_plan.get("anchored_entities") or [])
    relation_paths: list[Any] = list(query_plan.get("relation_paths") or [])
    join_structure: Mapping[str, Any] = (
        query_plan.get("join_structure")
        if isinstance(query_plan.get("join_structure"), Mapping)
        else {}
    )
    shared_answer_variable = str(query_plan.get("shared_answer_variable") or "").strip()
    candidate_set_variable = str(query_plan.get("candidate_set_variable") or "").strip()
    count_set_variable = str(query_plan.get("count_set_variable") or "").strip()
    ordering_attribute: Mapping[str, Any] = (
        query_plan.get("ordering_attribute")
        if isinstance(query_plan.get("ordering_attribute"), Mapping)
        else {}
    )
    ordering_direction = str(query_plan.get("ordering_direction") or "").strip().lower()
    qt: str = str(query_text or "")

    soft_reasons: list[str] = []

    # A plan is "weak" if predicates are exploratory guesses or if it is
    # structurally missing key grounding elements.
    plan_is_weak: bool = (
        allow_exploratory
        or not anchored_entities
        or not relation_paths
    )

    is_multi_anchor_strategy: bool = _is_multi_anchor_plan(
        query_shape=query_shape,
        strategy=strategy,
        anchored_entities=anchored_entities,
        join_structure=join_structure,
    )
    is_superlative_strategy: bool = _is_superlative_plan(
        query_shape=query_shape,
        strategy=strategy,
    )
    multi_anchor_issues = _collect_multi_anchor_issues(
        query_shape=query_shape,
        anchored_entities=anchored_entities,
        join_structure=join_structure,
        shared_answer_variable=shared_answer_variable,
        candidate_set_variable=candidate_set_variable,
        count_set_variable=count_set_variable,
        relation_paths=relation_paths,
    )
    count_structure_issues = _collect_count_structure_issues(
        answer_mode=answer_mode,
        query_shape=query_shape,
        query_text=qt,
        anchored_entities=anchored_entities,
        relation_paths=relation_paths,
        join_structure=join_structure,
        shared_answer_variable=shared_answer_variable,
        candidate_set_variable=candidate_set_variable,
        count_set_variable=count_set_variable,
    )
    superlative_issues = _collect_superlative_structure_issues(
        query_shape=query_shape,
        query_text=qt,
        relation_paths=relation_paths,
        candidate_set_variable=candidate_set_variable,
        ordering_attribute=ordering_attribute,
        ordering_direction=ordering_direction,
    )

    # ------------------------------------------------------------------ #
    # Signal 1 — Unsupported (non-Freebase) predicates in query text      #
    # Hard reject: model hallucinated a non-Freebase ontology.            #
    # ------------------------------------------------------------------ #
    unsupported = _detect_unsupported_predicates(qt)
    if unsupported:
        return PlausibilityVerdict(
            verdict=VERDICT_REJECTED_UNSUPPORTED_PREDICATE,
            reasons=[f"unsupported_predicate_in_query:{p}" for p in unsupported[:3]],
        )

    # ------------------------------------------------------------------ #
    # Signal 2 — Multi-anchor plan with a missing anchor literal in WHERE  #
    # Checks that each anchor entity's surface form appears as a quoted   #
    # string in the query.  Absent → one-sided join.                      #
    # ------------------------------------------------------------------ #
    if multi_anchor_issues:
        return PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_JOIN,
            reasons=multi_anchor_issues
            + ["repair:align_join_structure_and_anchor_constraints_on_one_shared_answer_set"],
        )

    expected_anchor_literals = _collect_expected_anchor_literals(
        query_plan=query_plan,
        fallback_entities=entities,
    )
    if is_multi_anchor_strategy and len(expected_anchor_literals) >= 2:
        missing = _find_missing_anchor_literals(
            qt,
            expected_anchor_literals[:2],
            anchor_probe_results=anchor_probe_results,
        )
        if missing:
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_JOIN,
                reasons=(
                    [f"multi_anchor_missing_anchor_literal:{m!r}" for m in missing]
                    + [
                        "repair:bind_both_anchor_entities_explicitly_in_WHERE_as_string_literals"
                    ]
                ),
            )

    # ------------------------------------------------------------------ #
    # Signal 3 — Superlative strategy without ORDER BY                    #
    # superlative_chain MUST order the candidate set.                     #
    # ------------------------------------------------------------------ #
    if is_superlative_strategy and superlative_issues:
        return PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_SUPERLATIVE,
            reasons=superlative_issues
            + ["repair:add_grounded_candidate_set_ordering_attribute_and_ORDER_BY"],
        )

    # ------------------------------------------------------------------ #
    # Signal 4 — Live anchor probe signals                                #
    # (Only active when anchor_probe_results are provided.)               #
    # These run before result-shape checks so a broken anchor is caught   #
    # early regardless of what the combined query returned.               #
    # ------------------------------------------------------------------ #
    if anchor_probe_results:
        ambiguous_surface_bound = [
            result
            for result in anchor_probe_results
            if result.entity_count > 1
            and (result.path_count or 0) > 0
            and str(result.resolved_entity_id or "").strip()
            and not _query_pins_resolved_anchor_entity(
                query_text=qt,
                resolved_entity_id=str(result.resolved_entity_id or "").strip(),
            )
        ]
        if ambiguous_surface_bound:
            reasons = [
                (
                    f"ambiguous_anchor_surface_binding:{result.anchor_name!r}:"
                    f"{result.entity_count}"
                )
                for result in ambiguous_surface_bound
            ]
            reasons += [
                "repair:bind_anchor_to_resolved_entity_id_before_accepting_result"
            ]
            if is_multi_anchor_strategy:
                return PlausibilityVerdict(
                    verdict=VERDICT_REPAIRABLE_BAD_JOIN,
                    reasons=reasons,
                )
            if answer_mode == "count":
                return PlausibilityVerdict(
                    verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                    reasons=reasons,
                )
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_ANCHOR_PATH_EMPTY,
                reasons=reasons,
            )

        relation_path_roles = {
            _normalize_contract_role(relation_path.get(endpoint_role))
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
            for endpoint_role in ("from_role", "to_role")
        }
        anchor_probe_optional = (
            bool({"type_set", "shared_type"} & relation_path_roles)
            and not bool({"anchor", "anchor_a", "anchor_b"} & relation_path_roles)
        )
        not_found = [r for r in anchor_probe_results if r.entity_count == 0]
        if not_found:
            if anchor_probe_optional:
                soft_reasons += [
                    f"anchor_probe_optional_not_found:{r.anchor_name!r}"
                    for r in not_found
                ]
            else:
                reasons = [f"anchor_not_found:{r.anchor_name!r}" for r in not_found]
                reasons += [
                    f"probe_count:{r.anchor_name!r}={r.entity_count}" for r in not_found
                ]
                if answer_mode == "count":
                    reasons += ["count_set_anchor_not_found"]
                    return PlausibilityVerdict(
                        verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET, reasons=reasons
                    )
                return PlausibilityVerdict(
                    verdict=VERDICT_REPAIRABLE_ANCHOR_NOT_FOUND, reasons=reasons
                )

        probe_failed = [r for r in anchor_probe_results if r.entity_count == -1]
        path_empty = [
            r
            for r in anchor_probe_results
            if r.path_count is not None and r.path_count == 0 and r.entity_count > 0
        ]
        if path_empty:
            reasons = [
                f"anchor_path_empty:{r.anchor_name!r}:{r.relation_probed}" for r in path_empty
            ]
            if answer_mode == "count":
                reasons += ["count_set_path_empty"]
                return PlausibilityVerdict(
                    verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET, reasons=reasons
                )
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_ANCHOR_PATH_EMPTY, reasons=reasons
            )

        if probe_failed:
            soft_reasons += [
                f"anchor_probe_failed:{r.anchor_name!r}" for r in probe_failed
            ]

    # ------------------------------------------------------------------ #
    # Execution-result signals (only run when we have a result)           #
    # ------------------------------------------------------------------ #
    if result_dict is not None:
        bindings: list[Any] = (
            result_dict.get("results", {}).get("bindings", [])  # type: ignore[union-attr]
            if isinstance(result_dict, Mapping)
            else []
        )
        binding_count: int = len(bindings) if isinstance(bindings, list) else 0
        has_boolean: bool = (
            "boolean" in result_dict if isinstance(result_dict, Mapping) else False
        )
        is_empty: bool = binding_count == 0 and not has_boolean
        scalar_count = (
            _extract_scalar_count_value(result_dict)
            if answer_mode == "count"
            else None
        )

        if scalar_count is not None and count_structure_issues:
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=count_structure_issues
                + [f"count_scalar_returned:{scalar_count}"]
                + ["repair:count_the_joined_grounded_set_not_a_weak_candidate_pool"],
            )

        if (
            scalar_count == 0
            and _plan_uses_only_generic_type_relations(relation_paths)
            and any(
                reason.startswith("anchor_probe_optional_not_found:")
                for reason in soft_reasons
            )
        ):
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=soft_reasons
                + [
                    "generic_type_only_zero_count_plan",
                    "repair:strengthen_type_grounding_before_accepting_zero_count",
                ],
            )

        live_anchor_paths = [
            result
            for result in (anchor_probe_results or [])
            if result.entity_count > 0 and (result.path_count or 0) > 0
        ]
        if (
            answer_mode == "count"
            and scalar_count == 0
            and query_shape == "count_over_joined_set"
            and is_multi_anchor_strategy
            and not plan_is_weak
            and len(live_anchor_paths) >= 2
        ):
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=[
                    "count_query_zero_with_live_anchor_paths",
                    "count_scalar_returned:0",
                    "repair:change_the_joined_relation_family_or_anchor_binding_before_accepting_zero_count",
                ],
            )

        if (
            answer_mode == "count"
            and scalar_count == 0
            and _has_dynamic_type_filtered_zero_count(relation_paths)
        ):
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=[
                    "count_query_dynamic_chain_too_weak",
                    "count_query_unverified_type_constraint",
                    "count_scalar_returned:0",
                    "repair:bind_type_constraint_with_name_or_alias_variants_before_accepting_zero_count",
                ],
            )

        if (
            answer_mode == "count"
            and scalar_count == 0
            and _count_query_has_unplanned_dynamic_type_filter(
                query_text=qt,
                relation_paths=relation_paths,
            )
        ):
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=[
                    "count_query_dynamic_chain_too_weak",
                    "count_query_unverified_type_constraint",
                    "count_query_unplanned_type_constraint",
                    "count_scalar_returned:0",
                    "repair:align_the_dynamic_count_path_and_type_constraint_before_accepting_zero_count",
                ],
            )

        if (
            answer_mode == "count"
            and scalar_count == 0
            and _has_weak_dynamic_count_chain(
                anchored_entities=anchored_entities,
                relation_paths=relation_paths,
            )
        ):
            reasons = [
                "count_query_dynamic_chain_too_weak",
                "count_scalar_returned:0",
            ]
            if _has_type_constraint_relation(relation_paths):
                reasons.append("count_query_unverified_type_constraint")
            reasons.append(
                "repair:prefer_the_shortest_grounded_anchor_to_count_set_chain_before_adding_dynamic_type_filters"
            )
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=reasons,
            )

        ambiguous_low_support_dynamic_count = [
            result
            for result in (anchor_probe_results or [])
            if result.entity_count > 1
            and (result.path_count or 0) <= 1
            and str(result.resolved_entity_id or "").strip()
            and _query_pins_resolved_anchor_entity(
                query_text=qt,
                resolved_entity_id=str(result.resolved_entity_id or "").strip(),
            )
        ]
        if (
            answer_mode == "count"
            and scalar_count is not None
            and ambiguous_low_support_dynamic_count
            and _has_dynamic_only_count_relation_paths(relation_paths)
        ):
            reasons = [
                (
                    f"ambiguous_anchor_surface_binding:{result.anchor_name!r}:"
                    f"{result.entity_count}"
                )
                for result in ambiguous_low_support_dynamic_count
            ]
            reasons += [
                (
                    f"count_anchor_path_low_support:{result.anchor_name!r}:"
                    f"{result.path_count or 0}:{result.relation_probed or ''}"
                )
                for result in ambiguous_low_support_dynamic_count
            ]
            reasons += [
                f"count_scalar_returned:{scalar_count}",
                "count_query_dynamic_chain_too_weak",
                "repair:verify_resolved_anchor_entity_and_relation_family_before_accepting_low_support_count",
            ]
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                reasons=reasons,
            )

        # -------------------------------------------------------------- #
        # Signal 5 — Grounded multi-anchor empty result                   #
        #                                                                  #
        # NEW (tighter): a grounded multi-anchor plan that returns zero   #
        # bindings defaults to repairable regardless of anchor probe      #
        # results, because multi-anchor intersections almost never have   #
        # a genuinely empty answer — they are far more likely broken.     #
        # -------------------------------------------------------------- #
        if is_empty and is_multi_anchor_strategy and not plan_is_weak:
            if anchor_probe_results:
                live_anchor_paths = [
                    result
                    for result in anchor_probe_results
                    if result.entity_count > 0 and (result.path_count or 0) > 0
                ]
            else:
                live_anchor_paths = []
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_GROUNDED_EMPTY,
                reasons=[
                    "grounded_multi_anchor_empty_result",
                    *(
                        ["anchor_paths_live_but_join_overlap_empty"]
                        if len(live_anchor_paths) >= 2
                        else []
                    ),
                    "query_shape:" + (query_shape or strategy[:80]),
                    "repair:verify_each_anchor_alias_and_relation_path_independently",
                ],
            )

        if is_empty and is_superlative_strategy and not plan_is_weak:
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_BAD_SUPERLATIVE,
                reasons=[
                    "grounded_superlative_empty_result",
                    "query_shape:" + (query_shape or strategy[:80]),
                    "repair:verify_candidate_set_and_ordering_path_before_ORDER_BY",
                ],
            )

        # -------------------------------------------------------------- #
        # Signal 6 — Empty result with a weak / exploratory plan          #
        #                                                                  #
        # Only flag when the plan is structurally weak.  Grounded single- #
        # anchor plans with empty results are NOT flagged here (empty     #
        # may be a valid answer).                                         #
        # -------------------------------------------------------------- #
        if is_empty and plan_is_weak:
            if answer_mode == "count":
                return PlausibilityVerdict(
                    verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
                    reasons=[
                        "count_query_returned_empty",
                        (
                            "plan_exploratory:predicates_are_ungrounded_guesses"
                            if allow_exploratory
                            else "plan_weak:no_anchored_entities_or_relation_paths"
                        ),
                        *count_structure_issues,
                        "repair:verify_anchor_entity_name_and_counted_relation_path",
                    ],
                )
            return PlausibilityVerdict(
                verdict=VERDICT_REPAIRABLE_WEAK_GROUNDING,
                reasons=[
                    "execution_result_empty",
                    (
                        "plan_exploratory:predicates_are_ungrounded_guesses"
                        if allow_exploratory
                        else "plan_weak:no_anchored_entities_or_relation_paths"
                    ),
                    "repair:try_alternative_anchor_alias_or_grounded_relation",
                ],
            )

        # -------------------------------------------------------------- #
        # Signal 7 — Large unfiltered result set (soft warning only)     #
        # -------------------------------------------------------------- #
        if answer_mode == "entity" and binding_count > 40 and not allow_exploratory:
            if not re.search(r"\bFILTER\b", qt, flags=re.IGNORECASE):
                soft_reasons.append(
                    f"result_suspicious_large_unfiltered:{binding_count}_bindings_no_FILTER"
                )

    return PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=soft_reasons)


def _normalize_contract_token(raw_value: Any) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    value = value.lstrip("?")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value


def _normalize_contract_role(raw_value: Any) -> str:
    token = _normalize_contract_token(raw_value)
    role_map = {
        "anchorentity": "anchor",
        "anchor_entity": "anchor",
        "anchora": "anchor_a",
        "anchor_a": "anchor_a",
        "anchorb": "anchor_b",
        "anchor_b": "anchor_b",
        "candidate": "candidate_set",
        "candidate_set": "candidate_set",
        "count_set": "count_set",
        "counted_set": "count_set",
        "constraint_value": "constraint_value",
        "attribute_value": "constraint_value",
        "answer_entity": "answer",
        "shared_answer": "shared_answer",
        "ordering_attribute": "ordering_attribute",
        "shared_type": "shared_type",
        "type_set": "type_set",
    }
    return role_map.get(token, token)


def _query_mentions_variable(query_text: str, variable_name: str) -> bool:
    normalized_variable = _normalize_contract_token(variable_name)
    if not normalized_variable:
        return False
    return bool(
        re.search(
            rf"\?{re.escape(normalized_variable)}\b",
            query_text,
            flags=re.IGNORECASE,
        )
    )


def _extract_count_aggregate_variables(query_text: str) -> list[str]:
    variables: list[str] = []
    for match in re.finditer(
        r"COUNT\s*\(\s*(?:DISTINCT\s+)?\?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        query_text,
        flags=re.IGNORECASE,
    ):
        variable = _normalize_contract_token(match.group(1))
        if variable and variable not in variables:
            variables.append(variable)
    return variables


def _count_set_token_is_structural(
    count_set_variable: str,
    relation_paths: Sequence[Any],
) -> bool:
    count_set_token = _normalize_contract_token(count_set_variable)
    if not count_set_token:
        return False
    for relation_path in relation_paths:
        if not isinstance(relation_path, Mapping):
            continue
        if count_set_token in {
            _normalize_contract_token(relation_path.get("from")),
            _normalize_contract_token(relation_path.get("to")),
        }:
            return True
        if "count_set" in {
            _normalize_contract_role(relation_path.get("from_role")),
            _normalize_contract_role(relation_path.get("to_role")),
        }:
            return True
    return False


def _extract_anchor_roles(anchored_entities: Sequence[Any]) -> list[str]:
    roles: list[str] = []
    for anchored_entity in anchored_entities:
        if not isinstance(anchored_entity, Mapping):
            continue
        normalized_role = _normalize_contract_role(anchored_entity.get("role"))
        if normalized_role:
            roles.append(normalized_role)
    return roles


def _anchor_roles_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if left == "anchor" and right in {"anchor_a", "anchor_b"}:
        return True
    if right == "anchor" and left in {"anchor_a", "anchor_b"}:
        return True
    return False


def _is_multi_anchor_plan(
    *,
    query_shape: str,
    strategy: str,
    anchored_entities: Sequence[Any],
    join_structure: Mapping[str, Any],
) -> bool:
    if query_shape in {
        "multi_anchor_intersection",
        "count_over_joined_set",
        "shared_type_intersection",
    }:
        return True
    join_type = str(join_structure.get("type") or "").strip().lower()
    if join_type in {"intersection", "shared_type"}:
        return True
    anchor_roles = _extract_anchor_roles(anchored_entities)
    anchor_like_count = sum(
        1 for role in anchor_roles if role in {"anchor", "anchor_a", "anchor_b"}
    )
    if anchor_like_count >= 2:
        return True
    return any(
        kw in strategy
        for kw in ("multi_anchor", "count_over_joined", "two_anchor", "intersect")
    )


def _is_superlative_plan(*, query_shape: str, strategy: str) -> bool:
    return query_shape == "superlative_chain" or "superlative" in strategy


def _collect_multi_anchor_issues(
    *,
    query_shape: str,
    anchored_entities: Sequence[Any],
    join_structure: Mapping[str, Any],
    shared_answer_variable: str,
    candidate_set_variable: str,
    count_set_variable: str,
    relation_paths: Sequence[Any],
) -> list[str]:
    if not _is_multi_anchor_plan(
        query_shape=query_shape,
        strategy="",
        anchored_entities=anchored_entities,
        join_structure=join_structure,
    ):
        return []

    issues: list[str] = []
    anchor_roles = _extract_anchor_roles(anchored_entities)
    anchor_constraints = [
        constraint
        for constraint in (join_structure.get("anchor_constraints") or [])
        if isinstance(constraint, Mapping)
    ]
    if not anchor_constraints:
        issues.append("join_structure_missing_anchor_constraints")
    else:
        if len(anchor_constraints) < min(2, len(anchor_roles) or 2):
            issues.append("join_structure_anchor_constraint_count_mismatch")
        constraint_roles = [
            _normalize_contract_role(constraint.get("anchor_role"))
            for constraint in anchor_constraints
        ]
        if anchor_roles and not all(role == "anchor" for role in constraint_roles):
            missing_roles = [
                anchor_role
                for anchor_role in anchor_roles[:2]
                if not any(
                    _anchor_roles_equivalent(anchor_role, constraint_role)
                    for constraint_role in constraint_roles
                )
            ]
            if missing_roles:
                issues.append(
                    "join_structure_missing_anchor_roles:" + ",".join(missing_roles)
                )

    primary_targets = {
        _normalize_contract_token(shared_answer_variable),
        _normalize_contract_token(candidate_set_variable),
        _normalize_contract_token(count_set_variable),
    }
    primary_targets.discard("")
    if query_shape == "count_over_joined_set" and "shared_answer" in primary_targets:
        primary_targets.add(_normalize_contract_token(shared_answer_variable))
    shared_target = next(iter(primary_targets), "")
    allowed_constraint_targets = set(primary_targets)
    for relation_path in relation_paths:
        if not isinstance(relation_path, Mapping):
            continue
        from_role = _normalize_contract_role(relation_path.get("from_role"))
        to_role = _normalize_contract_role(relation_path.get("to_role"))
        from_variable = _normalize_contract_token(relation_path.get("from"))
        to_variable = _normalize_contract_token(relation_path.get("to"))
        if from_role in {"shared_answer", "answer", "candidate_set", "count_set"}:
            if from_variable:
                allowed_constraint_targets.add(from_variable)
            if to_variable:
                allowed_constraint_targets.add(to_variable)
        if to_role in {"shared_answer", "answer", "candidate_set", "count_set"}:
            if to_variable:
                allowed_constraint_targets.add(to_variable)
            if from_variable:
                allowed_constraint_targets.add(from_variable)
    constraint_targets = [
        _normalize_contract_token(constraint.get("constrains_variable"))
        for constraint in anchor_constraints
        if _normalize_contract_token(constraint.get("constrains_variable"))
    ]
    if query_shape != "shared_type_intersection":
        if not shared_target:
            issues.append("join_structure_missing_shared_answer_variable")
        else:
            drifting_targets = [
                target
                for target in constraint_targets
                if target not in allowed_constraint_targets
            ]
            if drifting_targets:
                issues.append(
                    "join_structure_drifting_constraints:" + ",".join(sorted(set(drifting_targets)))
                )

    if query_shape == "shared_type_intersection":
        type_roles = {
            _normalize_contract_role(relation_path.get("from_role"))
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
        } | {
            _normalize_contract_role(relation_path.get("to_role"))
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
        }
        if not {"shared_type", "type_set"} & type_roles:
            issues.append("shared_type_structure_missing_type_roles")
    return issues


def _collect_expected_anchor_literals(
    *,
    query_plan: Mapping[str, Any],
    fallback_entities: Sequence[str],
) -> list[str]:
    planned_literals: list[str] = []
    for anchored_entity in query_plan.get("anchored_entities") or []:
        if not isinstance(anchored_entity, Mapping):
            continue
        role = _normalize_contract_role(anchored_entity.get("role"))
        if role not in {"anchor", "anchor_a", "anchor_b"}:
            continue
        literal = str(
            anchored_entity.get("chosen_alias")
            or anchored_entity.get("surface")
            or ""
        ).strip()
        if literal:
            planned_literals.append(literal)
    if planned_literals:
        return planned_literals
    return [str(entity or "").strip() for entity in fallback_entities if str(entity or "").strip()]


def _collect_count_structure_issues(
    *,
    answer_mode: str,
    query_shape: str,
    query_text: str,
    anchored_entities: Sequence[Any],
    relation_paths: Sequence[Any],
    join_structure: Mapping[str, Any],
    shared_answer_variable: str,
    candidate_set_variable: str,
    count_set_variable: str,
) -> list[str]:
    if answer_mode != "count":
        return []

    issues: list[str] = []
    if not count_set_variable:
        issues.append("count_set_variable_missing")

    if not re.search(r"\bCOUNT\s*\(", query_text, flags=re.IGNORECASE):
        issues.append("count_query_missing_COUNT_aggregate")
    count_set_token = _normalize_contract_token(count_set_variable)
    counted_variables = _extract_count_aggregate_variables(query_text)
    if (
        count_set_token
        and counted_variables
        and _count_set_token_is_structural(count_set_variable, relation_paths)
        and _query_mentions_variable(query_text, count_set_variable)
        and count_set_token not in counted_variables
    ):
        issues.append(
            "count_query_counts_wrong_variable:" + ",".join(counted_variables[:2])
        )

    if not relation_paths:
        issues.append("count_query_missing_relation_paths")
    else:
        grounding_sources = {
            str(relation_path.get("grounding_source") or "").strip().lower()
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
        }
        if grounding_sources and grounding_sources <= {"exploratory"}:
            issues.append("count_query_all_relation_paths_exploratory")
        anchor_roles = {
            _normalize_contract_role(
                anchored_entity.get("role") if isinstance(anchored_entity, Mapping) else ""
            )
            for anchored_entity in anchored_entities
        }
        exploratory_constraint_paths = [
            relation_path
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
            and str(relation_path.get("grounding_source") or "").strip().lower() == "exploratory"
            and (
                _normalize_contract_role(relation_path.get("from_role")) == "constraint_value"
                or _normalize_contract_role(relation_path.get("to_role")) == "constraint_value"
            )
        ]
        if exploratory_constraint_paths and anchor_roles <= {"anchor", "anchor_a", "anchor_b"}:
            issues.append("count_query_has_unanchored_exploratory_constraint")

    if query_shape == "count_over_joined_set":
        if not candidate_set_variable and not shared_answer_variable:
            issues.append("count_join_missing_candidate_set_variable")
        issues.extend(
            _collect_multi_anchor_issues(
                query_shape=query_shape,
                anchored_entities=anchored_entities,
                join_structure=join_structure,
                shared_answer_variable=shared_answer_variable,
                candidate_set_variable=candidate_set_variable,
                count_set_variable=count_set_variable,
                relation_paths=relation_paths,
            )
        )
    elif query_shape == "count_over_direct_relation" and not anchored_entities:
        issues.append("count_query_missing_anchor_binding")

    deduped_issues: list[str] = []
    for issue in issues:
        if issue not in deduped_issues:
            deduped_issues.append(issue)
    return deduped_issues


def _collect_superlative_structure_issues(
    *,
    query_shape: str,
    query_text: str,
    relation_paths: Sequence[Any],
    candidate_set_variable: str,
    ordering_attribute: Mapping[str, Any],
    ordering_direction: str,
) -> list[str]:
    if not _is_superlative_plan(query_shape=query_shape, strategy=""):
        return []

    issues: list[str] = []
    if not candidate_set_variable:
        issues.append("superlative_missing_candidate_set_variable")

    relation = str(ordering_attribute.get("relation") or "").strip()
    source_variable = str(ordering_attribute.get("source_variable") or "").strip()
    attribute_variable = str(ordering_attribute.get("attribute_variable") or "").strip()
    if not relation or not source_variable or not attribute_variable:
        issues.append("ordering_attribute_incomplete")
    else:
        matched_path = next(
            (
                relation_path
                for relation_path in relation_paths
                if isinstance(relation_path, Mapping)
                and str(relation_path.get("relation") or "").strip() == relation
            ),
            None,
        )
        if matched_path is None:
            issues.append("ordering_attribute_path_missing")
        elif str(matched_path.get("grounding_source") or "").strip().lower() == "exploratory":
            issues.append("ordering_attribute_path_exploratory")

    if ordering_direction not in {"max", "min"}:
        issues.append("ordering_direction_missing")
    if not re.search(r"\bORDER\s+BY\b", query_text, flags=re.IGNORECASE):
        issues.append("superlative_strategy_missing_ORDER_BY")
    return issues


def _extract_scalar_count_value(result_dict: Mapping[str, Any] | None) -> int | None:
    if not isinstance(result_dict, Mapping):
        return None
    bindings = result_dict.get("results", {}).get("bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 1:
        return None
    binding = bindings[0]
    if not isinstance(binding, Mapping) or len(binding) != 1:
        return None
    cell = next(iter(binding.values()))
    if not isinstance(cell, Mapping):
        return None
    raw_value = str(cell.get("value") or "").strip()
    if not raw_value:
        return None
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return None


def _query_pins_resolved_anchor_entity(
    *,
    query_text: str,
    resolved_entity_id: str,
) -> bool:
    entity_id = str(resolved_entity_id or "").strip()
    if not entity_id:
        return False
    return f"fb:{entity_id}" in str(query_text or "")


def _plan_uses_only_generic_type_relations(
    relation_paths: Sequence[Any],
) -> bool:
    generic_relations = {"type.object.type", "type.type.instance"}
    seen_relation = False
    for relation_path in relation_paths or ():
        if not isinstance(relation_path, Mapping):
            continue
        relation = str(relation_path.get("relation") or "").strip()
        if not relation:
            continue
        seen_relation = True
        if relation not in generic_relations:
            return False
    return seen_relation


def _has_weak_dynamic_count_chain(
    *,
    anchored_entities: Sequence[Any],
    relation_paths: Sequence[Any],
) -> bool:
    normalized_paths = [
        relation_path
        for relation_path in relation_paths
        if isinstance(relation_path, Mapping)
    ]
    if len(normalized_paths) < 3:
        return False
    grounding_sources = {
        str(relation_path.get("grounding_source") or "").strip().lower()
        for relation_path in normalized_paths
    }
    if not grounding_sources <= {"dynamic_probe", "exploratory"}:
        return False
    if "dynamic_probe" not in grounding_sources:
        return False
    anchor_roles = {
        _normalize_contract_role(
            anchored_entity.get("role") if isinstance(anchored_entity, Mapping) else ""
        )
        for anchored_entity in anchored_entities
    }
    if anchor_roles - {"anchor", "anchor_a", "anchor_b"}:
        return False
    return any(
        str(relation_path.get("relation") or "").strip() == "type.type.instance"
        for relation_path in normalized_paths
    ) or "exploratory" in grounding_sources or len(normalized_paths) >= 4


def _has_type_constraint_relation(
    relation_paths: Sequence[Any],
) -> bool:
    normalized_paths = [
        relation_path
        for relation_path in relation_paths
        if isinstance(relation_path, Mapping)
    ]
    for relation_path in normalized_paths:
        relation = str(relation_path.get("relation") or "").strip()
        from_role = _normalize_contract_role(relation_path.get("from_role"))
        to_role = _normalize_contract_role(relation_path.get("to_role"))
        if relation in {"type.type.instance", "type.object.type"}:
            return True
        if {"type_set", "shared_type"} & {from_role, to_role}:
            return True
    return False


def _has_dynamic_type_filtered_zero_count(
    relation_paths: Sequence[Any],
) -> bool:
    normalized_paths = [
        relation_path
        for relation_path in relation_paths
        if isinstance(relation_path, Mapping)
    ]
    if len(normalized_paths) < 2:
        return False
    type_paths = [
        relation_path
        for relation_path in normalized_paths
        if _has_type_constraint_relation([relation_path])
    ]
    non_type_paths = [
        relation_path
        for relation_path in normalized_paths
        if relation_path not in type_paths
    ]
    if not type_paths or not non_type_paths:
        return False
    dynamic_like_sources = {"dynamic_probe", "exploratory"}
    type_sources = {
        str(relation_path.get("grounding_source") or "").strip().lower()
        for relation_path in type_paths
    }
    non_type_sources = {
        str(relation_path.get("grounding_source") or "").strip().lower()
        for relation_path in non_type_paths
    }
    return (
        bool(type_sources)
        and bool(non_type_sources)
        and type_sources <= dynamic_like_sources
        and non_type_sources <= dynamic_like_sources
    )


def _count_query_has_unplanned_dynamic_type_filter(
    *,
    query_text: str,
    relation_paths: Sequence[Any],
) -> bool:
    normalized_paths = [
        relation_path
        for relation_path in relation_paths
        if isinstance(relation_path, Mapping)
    ]
    if not normalized_paths:
        return False
    if _has_type_constraint_relation(normalized_paths):
        return False
    grounding_sources = {
        str(relation_path.get("grounding_source") or "").strip().lower()
        for relation_path in normalized_paths
    }
    if not grounding_sources or not grounding_sources <= {"dynamic_probe", "exploratory"}:
        return False
    return bool(
        re.search(r"\bfb:type\.object\.type\b", query_text)
        or re.search(r"\bfb:type\.type\.instance\b", query_text)
    )


def _has_dynamic_only_count_relation_paths(
    relation_paths: Sequence[Any],
) -> bool:
    normalized_paths = [
        relation_path
        for relation_path in relation_paths
        if isinstance(relation_path, Mapping)
    ]
    if not normalized_paths:
        return False
    grounding_sources = {
        str(relation_path.get("grounding_source") or "").strip().lower()
        for relation_path in normalized_paths
    }
    return bool(grounding_sources) and grounding_sources <= {"dynamic_probe", "exploratory"}


# ---------------------------------------------------------------------------
# Repair feedback builder
# ---------------------------------------------------------------------------

def build_repair_feedback(verdict: PlausibilityVerdict) -> list[str]:
    """
    Convert a non-accepted plausibility verdict into structured feedback
    strings for the next code-generation attempt.

    All feedback is leakage-safe: structural / plausibility issues only.
    """
    if verdict.is_accepted:
        return []

    feedback: list[str] = []

    if verdict.verdict == VERDICT_REPAIRABLE_ANCHOR_NOT_FOUND:
        not_found = [
            r.split("anchor_not_found:", 1)[-1].strip("'\"")
            for r in verdict.reasons
            if r.startswith("anchor_not_found:")
        ]
        feedback += [
            "plausibility_feedback:anchor_entity_not_found_in_KG — the probed anchor alias returned no KG entities",
        ]
        for name in not_found:
            feedback.append(f"anchor_not_found:{name}")
        feedback += [
            "repair_hint:try_alternative_alias — use a different surface form or casing for the anchor entity (e.g. title-case, without articles, singular/plural, standard Freebase label)",
            "repair_hint:check_alias_candidates — check the alias_candidates list in the grounding card and try the next-best option",
        ]

    elif verdict.verdict == VERDICT_REPAIRABLE_ANCHOR_PATH_EMPTY:
        path_empty = [
            r
            for r in verdict.reasons
            if r.startswith("anchor_path_empty:")
        ]
        feedback += [
            "plausibility_feedback:anchor_path_produces_no_results — anchor entity binds but the primary relation returns nothing",
        ]
        for item in path_empty:
            feedback.append(item)
        feedback += [
            "repair_hint:try_alternative_relation — the grounded relation may have the wrong direction or wrong Freebase namespace; check the grounding card for alternative relation paths",
            "repair_hint:verify_relation_direction — try the reverse direction (forward ↔ reverse) for the primary relation path",
        ]

    elif verdict.verdict == VERDICT_REPAIRABLE_GROUNDED_EMPTY:
        feedback += [
            "plausibility_feedback:grounded_multi_anchor_empty_result — grounded multi-anchor query returned no bindings",
            "repair_hint:decompose_and_verify — check each anchor independently before combining; verify anchor A binds, verify anchor B binds, verify each relation path produces results",
            "repair_hint:check_alias_correctness — one or more anchor aliases may not match the Freebase label; try alternative surface forms from the grounding card alias_candidates",
            "repair_hint:check_relation_direction — one or more relation paths may have the wrong direction or namespace; inspect the grounding card carefully",
        ]
        if "anchor_paths_live_but_join_overlap_empty" in verdict.reasons:
            feedback += [
                "plausibility_feedback:join_overlap_empty — each anchor path is live, but the combined scaffold is empty",
                "repair_hint:change_scaffold_family_not_query_wording — do not just restate the same relation family; change the scaffold or add a grounded bridge to a different shared answer set",
                "repair_hint:prefer_unused_grounded_relations — when the current grounded family is empty, prefer grounded relations not used in the failed scaffold",
            ]

    elif verdict.verdict == VERDICT_REPAIRABLE_WEAK_GROUNDING:
        feedback += [
            "plausibility_feedback:execution_result_empty — the query returned no bindings",
            "repair_hint:verify_anchor_entity_name — the entity name in the WHERE clause must exactly match the Freebase label (check casing, articles, alternate surface forms)",
            "repair_hint:verify_relation_path — confirm the grounded fb: relation actually connects the anchor to the answer type in Freebase; try the best-ranked alternative from the grounding card if the first choice failed",
        ]

    elif verdict.verdict == VERDICT_REPAIRABLE_BAD_JOIN:
        missing = [
            r.split(":", 1)[-1].strip("'\"")
            for r in verdict.reasons
            if "missing_anchor_literal:" in r
            or "multi_anchor_missing_anchor_literal:" in r
        ]
        feedback += [
            "plausibility_feedback:multi_anchor_join_incomplete — at least one anchor entity is not bound in WHERE",
        ]
        for m in missing:
            feedback.append(f"missing_anchor_not_in_query:{m}")
        feedback += [
            "repair_hint:bind_both_anchors — add an explicit ?anchor fb:type.object.name '...'@en binding (or FILTER) for EACH anchor entity; do not omit any anchor from the WHERE clause",
        ]

    elif verdict.verdict == VERDICT_REPAIRABLE_BAD_SUPERLATIVE:
        feedback += [
            "plausibility_feedback:superlative_ordering_absent — strategy=superlative_chain requires ORDER BY + LIMIT 1",
            "repair_hint:add_ordering — after building the candidate set, bind the ordering attribute (?date / ?length / ?count) via the appropriate relation, then add ORDER BY ?ordering_attr DESC LIMIT 1 (or ASC for earliest/smallest)",
        ]

    elif verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET:
        anchor_issues = [r for r in verdict.reasons if "anchor_not_found" in r or "anchor_path_empty" in r or "count_set" in r]
        feedback += [
            "plausibility_feedback:count_set_weak_or_broken — the counted candidate set is empty or the anchor/path is invalid",
        ]
        feedback += anchor_issues[:4]
        if "count_query_unverified_type_constraint" in verdict.reasons:
            feedback += [
                "plausibility_feedback:type_constraint_unverified — the category/type filter may not bind to a real Freebase type entity",
                "repair_hint:treat_question_category_as_type_constraint — bind the category/type phrase as a separate type node instead of folding it into the anchor relation",
                "repair_hint:try_type_alias_variants — when binding a type/category node, try singular and title-cased variants of the category phrase and allow common.topic.alias as well as type.object.name",
                "repair_hint:prefer_candidate_to_type_binding — if the candidate set is already bound, prefer ?candidate fb:type.object.type ?type over starting from an unverified type node",
            ]
        feedback += [
            "repair_hint:verify_anchor_and_counted_relation — confirm the anchor entity name exactly matches the Freebase label and the fb: relation produces the counted set",
            "repair_hint:try_alternative_anchor_alias — if the anchor name does not bind, try alternative forms from the grounding card alias_candidates",
        ]

    elif verdict.verdict == VERDICT_REPAIRABLE_BAD_PROJECTION:
        feedback += [
            "plausibility_feedback:answer_projection_mismatch — query projection does not match the answer_mode in the plan",
            "repair_hint:fix_projection — entity mode: project ?answer first; count mode: project (COUNT(DISTINCT ?answer) AS ?count); boolean mode: use ASK",
        ]

    elif verdict.verdict == VERDICT_REJECTED_UNSUPPORTED_PREDICATE:
        bad_preds = [
            r.split(":", 1)[-1]
            for r in verdict.reasons
            if "unsupported_predicate_in_query:" in r
        ]
        feedback += [
            "plausibility_feedback:unsupported_predicate — query uses non-Freebase ontology predicates; the endpoint only understands fb:* predicates",
        ]
        for p in bad_preds[:3]:
            feedback.append(f"bad_predicate_namespace_detected:{p}")
        feedback += [
            "repair_hint:use_only_fb_predicates — every predicate in the WHERE clause MUST use the fb: prefix (http://rdf.freebase.com/ns/). Do not use schema:, owl:, rdfs: (except rdfs:label), wikidata:, or dbpedia: predicates",
        ]

    elif verdict.verdict == VERDICT_REJECTED_UNBOUNDED:
        feedback += [
            "plausibility_feedback:unbounded_exploration — query performs unguarded graph traversal without anchor constraints",
            "repair_hint:add_anchor_constraints — bind the anchor entity first (?anchor fb:type.object.name '...'@en), then use specific grounded predicates from the grounding card",
        ]

    else:
        feedback += [f"plausibility_feedback:{r}" for r in verdict.reasons]

    return feedback


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_unsupported_predicates(query_text: str) -> list[str]:
    """Return names of non-Freebase predicate namespaces found in the query."""
    qt = query_text or ""
    found: list[str] = []
    if re.search(r"\bschema:", qt, flags=re.IGNORECASE):
        found.append("schema:")
    if re.search(r"http://schema\.org", qt):
        found.append("schema.org_IRI")
    if re.search(r"\bowl:", qt, flags=re.IGNORECASE):
        found.append("owl:")
    if re.search(r"\brdfs:(?!label\b)", qt, flags=re.IGNORECASE):
        found.append("rdfs:non-label")
    if re.search(r"wikidata\.org|dbpedia\.org", qt, flags=re.IGNORECASE):
        found.append("external_ontology_IRI")
    return found


def _find_missing_anchor_literals(
    query_text: str,
    entities: Sequence[str],
    *,
    anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
) -> list[str]:
    """Return anchor entities absent from either quoted literals or resolved mid pins."""
    qt = query_text or ""
    missing: list[str] = []
    probe_results = list(anchor_probe_results or [])
    for idx, entity in enumerate(entities):
        surface = str(entity or "").strip()
        if not surface:
            continue
        resolved_entity_id = ""
        if idx < len(probe_results):
            resolved_entity_id = str(
                getattr(probe_results[idx], "resolved_entity_id", "") or ""
            ).strip()
        found = _query_mentions_anchor_binding(
            query_text=qt,
            literal=surface,
            resolved_entity_id=resolved_entity_id,
        )
        if not found:
            missing.append(surface)
    return missing


def _looks_like_freebase_mid(value: str) -> bool:
    return bool(re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", str(value or "").strip()))


def _query_mentions_anchor_binding(
    *,
    query_text: str,
    literal: str,
    resolved_entity_id: str = "",
) -> bool:
    qt = str(query_text or "")
    surface = str(literal or "").strip()
    resolved_id = str(resolved_entity_id or "").strip()
    if surface and re.search(r'["\']' + re.escape(surface) + r'["\']', qt, flags=re.IGNORECASE):
        return True
    if resolved_id and f"fb:{resolved_id}" in qt:
        return True
    if _looks_like_freebase_mid(surface) and f"fb:{surface}" in qt:
        return True
    return False
