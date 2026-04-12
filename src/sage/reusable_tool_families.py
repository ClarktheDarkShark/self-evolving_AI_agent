from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from src.sage.family_contracts import FamilyContract, get_family_contract
from src.sage.family_policy_evolution import (
    build_family_policy_store,
    compare_locked_family_name,
    family_policy_enabled_for,
)
from src.sage.policy_contracts import FamilyPolicyBundle


_ANCHOR_ROLES = {"anchor", "anchor_a", "anchor_b"}
_ENTITY_QUERY_SHAPES = {
    "single_anchor_lookup",
    "single_anchor_chain_lookup",
    "multi_anchor_intersection",
    "shared_type_intersection",
    "containment_or_ownership_lookup",
}
_SUPPORTED_QUERY_SHAPES = {
    *_ENTITY_QUERY_SHAPES,
    "count_over_direct_relation",
    "count_over_joined_set",
    "superlative_chain",
}
_GENERIC_NAMED_ROLE_TOKENS = {
    "constraint_value",
    "shared_type",
    "type_set",
    "anchor_value",
    "type",
    "value",
}
_CONSTRAINT_NODE_ROLES = {
    "constraint_value",
    "shared_type",
    "type_set",
    "anchor_value",
}
_VARIABLE_NODE_ROLES = {
    "answer",
    "shared_answer",
    "candidate_set",
    "count_set",
    "ordering_attribute",
}
_FAMILY_POLICY_VERSION = "2026-03-31"
_FAMILY_POLICY_BUNDLES = {
    "single_anchor_lookup": FamilyPolicyBundle(
        family_name="single_anchor_lookup",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="entity",
        applicability_conditions=(
            "query_shape=single_anchor_lookup",
            "entity projection resolved",
            "at least one grounded relation path",
        ),
        validator_expectations=(
            "projected answer entity must be structurally bound",
            "result must not collapse to a generic type dump",
        ),
        repair_policy=(
            "prefer same-family anchor alias and relation repair before switching",
        ),
        allowed_materialization_modes=("entity_id", "entity_set"),
    ),
    "single_anchor_chain_lookup": FamilyPolicyBundle(
        family_name="single_anchor_chain_lookup",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="entity",
        applicability_conditions=(
            "query_shape=single_anchor_chain_lookup",
            "entity projection resolved",
            "chain path grounded",
        ),
        validator_expectations=(
            "chain must preserve anchor semantics",
            "projected entity must be structurally bound",
        ),
        repair_policy=(
            "repair within chain family before switching to broader lookup",
        ),
        allowed_materialization_modes=("entity_id", "entity_set"),
    ),
    "multi_anchor_intersection": FamilyPolicyBundle(
        family_name="multi_anchor_intersection",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="entity",
        applicability_conditions=(
            "query_shape=multi_anchor_intersection",
            "multiple anchor constraints converge on one answer set",
        ),
        validator_expectations=(
            "all anchors must bind in the same answer scaffold",
            "generic shared-type dumps are not acceptable answers",
        ),
        repair_policy=(
            "switch family when join evidence shows a projected-answer or shared-type pivot is required",
        ),
        allowed_materialization_modes=("entity_id", "entity_set"),
    ),
    "shared_type_intersection": FamilyPolicyBundle(
        family_name="shared_type_intersection",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="entity",
        applicability_conditions=(
            "query_shape=shared_type_intersection",
            "shared type node explicitly represented",
        ),
        validator_expectations=(
            "shared types must not be generic ontology dumps",
        ),
        repair_policy=(
            "prefer domain-specific shared-type relations before broad generic-type fallback",
        ),
        allowed_materialization_modes=("entity_id", "entity_set"),
    ),
    "containment_or_ownership_lookup": FamilyPolicyBundle(
        family_name="containment_or_ownership_lookup",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="entity",
        applicability_conditions=(
            "query_shape=containment_or_ownership_lookup",
            "ownership or containment relation grounded",
        ),
        validator_expectations=(
            "projected entity must stay anchored to the ownership relation family",
        ),
        repair_policy=(
            "repair anchor binding before switching to generic lookup",
        ),
        allowed_materialization_modes=("entity_id", "entity_set"),
    ),
    "count_over_direct_relation": FamilyPolicyBundle(
        family_name="count_over_direct_relation",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="count",
        applicability_conditions=(
            "query_shape=count_over_direct_relation",
            "count_set_variable resolved",
            "counted relation grounded",
        ),
        validator_expectations=(
            "counted variable must be structurally bound",
            "answer-target semantics must be enforced in the counted set",
        ),
        repair_policy=(
            "repair direct count relation family before escalating to joined-count family",
        ),
        allowed_materialization_modes=("count_scalar",),
    ),
    "count_over_joined_set": FamilyPolicyBundle(
        family_name="count_over_joined_set",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="count",
        applicability_conditions=(
            "query_shape=count_over_joined_set",
            "joined candidate set defined before count",
        ),
        validator_expectations=(
            "joined set must be explicit and semantically aligned",
            "count must target the joined set, not an auxiliary bridge",
        ),
        repair_policy=(
            "switch family when join overlap repeatedly fails or count target semantics drift",
        ),
        allowed_materialization_modes=("count_scalar",),
    ),
    "superlative_chain": FamilyPolicyBundle(
        family_name="superlative_chain",
        version=_FAMILY_POLICY_VERSION,
        renderer_name="superlative",
        applicability_conditions=(
            "query_shape=superlative_chain",
            "ordering attribute resolved",
            "candidate set explicit",
        ),
        validator_expectations=(
            "candidate set and ordering path must both be grounded",
            "ORDER BY and LIMIT 1 must be present",
        ),
        repair_policy=(
            "repair ordering path within family before degrading to generic lookup",
        ),
        allowed_materialization_modes=(
            "entity_id",
            "entity_set",
            "scalar_literal",
            "text_literal",
        ),
    ),
}


@dataclass(frozen=True)
class ReusableToolSelection:
    family_name: str
    renderer_name: str
    fit_score: int
    reasons: tuple[str, ...]
    policy_bundle: FamilyPolicyBundle


@dataclass(frozen=True)
class _NodeSpec:
    term: str
    binding_clause: str | None = None


def select_reusable_tool(
    query_plan: Mapping[str, Any],
) -> Optional[ReusableToolSelection]:
    query_shape = _normalize_token(query_plan.get("query_shape"))
    answer_mode = _normalize_token(query_plan.get("answer_mode"))
    if query_shape not in _SUPPORTED_QUERY_SHAPES:
        return None
    locked_family = compare_locked_family_name()
    if locked_family and query_shape != locked_family:
        return None

    relation_paths = [
        path
        for path in (query_plan.get("relation_paths") or [])
        if isinstance(path, Mapping) and str(path.get("relation") or "").strip()
    ]
    if not relation_paths:
        return None

    if query_shape in {
        "count_over_direct_relation",
        "count_over_joined_set",
    }:
        if answer_mode not in {"", "count"}:
            return None
        count_var = _resolve_count_variable_name(query_plan=query_plan)
        if not count_var:
            return None
        policy_bundle = _get_family_policy_bundle(query_shape, renderer_name="count")
        if not _bundle_allows_query_plan(policy_bundle=policy_bundle, query_plan=query_plan):
            return None
        return ReusableToolSelection(
            family_name=query_shape,
            renderer_name="count",
            fit_score=100,
            reasons=("exact_query_shape_match", "count_projection_resolved"),
            policy_bundle=policy_bundle,
        )

    if query_shape == "superlative_chain":
        if answer_mode not in {"", "entity", "literal"}:
            return None
        ordering_var = _resolve_ordering_variable_name(query_plan=query_plan)
        if not ordering_var:
            return None
        answer_var = ""
        if answer_mode in {"", "entity"}:
            answer_var = _resolve_entity_variable_name(query_plan=query_plan)
        if answer_mode in {"", "entity"} and not answer_var:
            return None
        policy_bundle = _get_family_policy_bundle(
            query_shape,
            renderer_name="superlative",
        )
        if not _bundle_allows_query_plan(policy_bundle=policy_bundle, query_plan=query_plan):
            return None
        return ReusableToolSelection(
            family_name=query_shape,
            renderer_name="superlative",
            fit_score=95,
            reasons=(
                "exact_query_shape_match",
                "ordering_projection_resolved",
                (
                    "entity_projection_resolved"
                    if answer_mode in {"", "entity"}
                    else "literal_projection_via_ordering_attribute"
                ),
            ),
            policy_bundle=policy_bundle,
        )

    if answer_mode not in {"", "entity"}:
        return None
    answer_var = _resolve_entity_variable_name(query_plan=query_plan)
    if not answer_var:
        return None
    policy_bundle = _get_family_policy_bundle(query_shape, renderer_name="entity")
    if not _bundle_allows_query_plan(policy_bundle=policy_bundle, query_plan=query_plan):
        return None
    return ReusableToolSelection(
        family_name=query_shape,
        renderer_name="entity",
        fit_score=100,
        reasons=("exact_query_shape_match", "entity_projection_resolved"),
        policy_bundle=policy_bundle,
    )


def get_reusable_family_policy_bundle(family_name: str) -> Optional[FamilyPolicyBundle]:
    normalized_family = str(family_name or "").strip()
    baseline_bundle = _FAMILY_POLICY_BUNDLES.get(normalized_family)
    if baseline_bundle is None:
        return None
    locked_family = compare_locked_family_name()
    if locked_family and normalized_family != locked_family:
        return None
    if not family_policy_enabled_for(normalized_family):
        return baseline_bundle
    try:
        store = build_family_policy_store(baseline_bundles=_FAMILY_POLICY_BUNDLES)
        return store.get_active_bundle(normalized_family) or baseline_bundle
    except Exception:
        return baseline_bundle


def get_reusable_family_contract(family_name: str) -> Optional[FamilyContract]:
    return get_family_contract(family_name)


def iter_reusable_family_policy_bundles() -> tuple[FamilyPolicyBundle, ...]:
    return tuple(
        get_reusable_family_policy_bundle(family_name) or bundle
        for family_name, bundle in _FAMILY_POLICY_BUNDLES.items()
    )


def get_baseline_reusable_family_policy_bundles() -> Mapping[str, FamilyPolicyBundle]:
    return dict(_FAMILY_POLICY_BUNDLES)


def _get_family_policy_bundle(
    family_name: str,
    *,
    renderer_name: str,
) -> FamilyPolicyBundle:
    existing = get_reusable_family_policy_bundle(family_name)
    if existing is not None:
        return existing
    return FamilyPolicyBundle(
        family_name=family_name,
        version=_FAMILY_POLICY_VERSION,
        renderer_name=renderer_name,
    )


def _bundle_allows_query_plan(
    *,
    policy_bundle: FamilyPolicyBundle,
    query_plan: Mapping[str, Any],
) -> bool:
    scaffold_signature = _build_scaffold_signature(query_plan=query_plan)
    if (
        scaffold_signature
        and scaffold_signature in set(policy_bundle.blocked_scaffold_signatures)
    ):
        return False
    forbidden_relations = set(policy_bundle.forbidden_relation_families)
    if forbidden_relations:
        relation_names = set(_relation_names_from_query_plan(query_plan=query_plan))
        if relation_names & forbidden_relations:
            return False
    return True


def _relation_names_from_query_plan(*, query_plan: Mapping[str, Any]) -> list[str]:
    relation_names: list[str] = []
    for relation_path in query_plan.get("relation_paths") or []:
        if not isinstance(relation_path, Mapping):
            continue
        relation_name = str(relation_path.get("relation") or "").strip()
        if relation_name and relation_name not in relation_names:
            relation_names.append(relation_name)
    return relation_names


def _build_scaffold_signature(*, query_plan: Mapping[str, Any]) -> str:
    query_shape = str(query_plan.get("query_shape") or "").strip().lower()
    shared_answer_variable = str(
        query_plan.get("shared_answer_variable")
        or query_plan.get("candidate_set_variable")
        or query_plan.get("count_set_variable")
        or ""
    ).strip()
    relation_names = _relation_names_from_query_plan(query_plan=query_plan)
    if not relation_names:
        return ""
    signature_parts = [query_shape or "other", shared_answer_variable or "unknown"]
    signature_parts.extend(sorted(relation_names))
    return "|".join(signature_parts)


def render_reusable_tool(
    *,
    query_plan: Mapping[str, Any],
    selection: ReusableToolSelection,
) -> str:
    if selection.renderer_name == "count":
        query_text = _render_count_query(query_plan=query_plan)
    elif selection.renderer_name == "superlative":
        query_text = _render_superlative_query(query_plan=query_plan)
    elif selection.renderer_name == "entity":
        query_text = _render_entity_query(query_plan=query_plan)
    else:
        raise ValueError(f"unsupported_reusable_renderer:{selection.renderer_name}")
    return _wrap_query_as_sage_program(query_text)


def _render_count_query(*, query_plan: Mapping[str, Any]) -> str:
    body_lines = _build_query_body_lines(query_plan=query_plan, count_mode=True)
    count_var_name = _resolve_count_variable_name(query_plan=query_plan) or "count_set"
    count_var = f"?{count_var_name}"
    select_clause = f"SELECT (COUNT(DISTINCT {count_var}) AS ?count) WHERE {{"
    return _build_query_text(
        select_clause=select_clause,
        body_lines=body_lines,
        trailing_lines=["} LIMIT 50"],
    )


def _render_entity_query(*, query_plan: Mapping[str, Any]) -> str:
    body_lines = _build_query_body_lines(query_plan=query_plan, count_mode=False)
    answer_var_name = _resolve_entity_variable_name(query_plan=query_plan) or "answer"
    answer_var = f"?{answer_var_name}"
    body_lines.append(
        f"  OPTIONAL {{ {answer_var} fb:type.object.name ?{answer_var_name}_name . }}"
    )
    select_clause = (
        f"SELECT DISTINCT {answer_var} ?{answer_var_name}_name WHERE {{"
    )
    return _build_query_text(
        select_clause=select_clause,
        body_lines=body_lines,
        trailing_lines=["}"],
    )


def _render_superlative_query(*, query_plan: Mapping[str, Any]) -> str:
    answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()
    answer_var_name = _resolve_entity_variable_name(query_plan=query_plan) or "answer"
    candidate_var_name = (
        _normalize_token(query_plan.get("candidate_set_variable")) or answer_var_name
    )
    ordering_var_name = (
        _resolve_ordering_variable_name(query_plan=query_plan) or "ordering_attribute"
    )
    answer_var = f"?{answer_var_name}"
    ordering_var = f"?{ordering_var_name}"
    direction = str(query_plan.get("ordering_direction") or "").strip().lower()
    order_clause = "DESC" if direction == "max" else "ASC"

    if answer_mode == "literal":
        body_lines = _build_query_body_lines(query_plan=query_plan, count_mode=False)
        return _build_query_text(
            select_clause=f"SELECT DISTINCT {ordering_var} WHERE {{",
            body_lines=body_lines,
            trailing_lines=[f"}} ORDER BY {order_clause}({ordering_var}) LIMIT 1"],
        )

    relation_paths = [
        dict(path)
        for path in (query_plan.get("relation_paths") or [])
        if isinstance(path, Mapping) and str(path.get("relation") or "").strip()
    ]
    ordering_index = next(
        (
            index
            for index, path in enumerate(relation_paths)
            if "ordering_attribute"
            in {
                _normalize_token(path.get("from_role")),
                _normalize_token(path.get("to_role")),
            }
        ),
        -1,
    )
    if answer_var_name != candidate_var_name and ordering_index >= 0:
        inner_paths: list[dict[str, Any]] = []
        outer_paths: list[dict[str, Any]] = []
        for index, path in enumerate(relation_paths):
            roles = {
                _normalize_token(path.get("from_role")),
                _normalize_token(path.get("to_role")),
            }
            if index <= ordering_index or roles & (_ANCHOR_ROLES | {"ordering_attribute"}):
                inner_paths.append(path)
            else:
                outer_paths.append(path)
        if outer_paths:
            inner_plan = dict(query_plan)
            inner_plan["relation_paths"] = inner_paths
            inner_plan["projection"] = [candidate_var_name]
            inner_body_lines = _build_query_body_lines(
                query_plan=inner_plan,
                count_mode=False,
            )

            outer_plan = dict(query_plan)
            outer_plan["relation_paths"] = outer_paths
            outer_plan["projection"] = [answer_var_name]
            outer_body_lines = [
                "  {",
                f"    SELECT DISTINCT ?{candidate_var_name} {ordering_var} WHERE {{",
            ]
            for line in inner_body_lines:
                stripped = line[2:] if line.startswith("  ") else line
                outer_body_lines.append(f"    {stripped}")
            outer_body_lines.append(
                f"    }} ORDER BY {order_clause}({ordering_var}) LIMIT 1"
            )
            outer_body_lines.append("  }")
            outer_body_lines.extend(
                _build_query_body_lines(query_plan=outer_plan, count_mode=False)
            )
            outer_body_lines.append(
                f"  OPTIONAL {{ {answer_var} fb:type.object.name ?{answer_var_name}_name . }}"
            )
            select_clause = (
                f"SELECT DISTINCT {answer_var} ?{answer_var_name}_name WHERE {{"
            )
            return _build_query_text(
                select_clause=select_clause,
                body_lines=outer_body_lines,
                trailing_lines=["}"],
            )

    body_lines = _build_query_body_lines(query_plan=query_plan, count_mode=False)
    body_lines.append(
        f"  OPTIONAL {{ {answer_var} fb:type.object.name ?{answer_var_name}_name . }}"
    )
    select_clause = (
        f"SELECT DISTINCT {answer_var} ?{answer_var_name}_name WHERE {{"
    )
    return _build_query_text(
        select_clause=select_clause,
        body_lines=body_lines,
        trailing_lines=[f"}} ORDER BY {order_clause}({ordering_var}) LIMIT 1"],
    )


def _build_query_text(
    *,
    select_clause: str,
    body_lines: Sequence[str],
    trailing_lines: Sequence[str],
) -> str:
    rendered_lines = [
        "PREFIX fb: <http://rdf.freebase.com/ns/>",
        select_clause,
        *body_lines,
        *trailing_lines,
    ]
    return "\n".join(rendered_lines)


def _build_query_body_lines(
    *,
    query_plan: Mapping[str, Any],
    count_mode: bool,
) -> list[str]:
    query_shape = _normalize_token(query_plan.get("query_shape"))
    relation_paths = [
        dict(path)
        for path in (query_plan.get("relation_paths") or [])
        if isinstance(path, Mapping) and str(path.get("relation") or "").strip()
    ]
    relation_paths = _expand_anchor_specific_relation_paths(
        query_plan=query_plan,
        relation_paths=relation_paths,
    )
    relation_paths = _promote_projection_relation_roles(
        query_plan=query_plan,
        relation_paths=relation_paths,
    )
    endpoint_stats = _count_endpoint_occurrences(relation_paths)
    anchor_entities_by_role = {
        _normalize_token(entity.get("role")): dict(entity)
        for entity in (query_plan.get("anchored_entities") or [])
        if isinstance(entity, Mapping) and _normalize_token(entity.get("role"))
    }
    repeated_role_endpoint_specs = _derive_repeated_role_endpoint_specs(
        query_plan=query_plan,
        relation_paths=relation_paths,
    )
    path_endpoint_overrides = _derive_joined_set_endpoint_overrides(
        query_plan=query_plan,
        relation_paths=relation_paths,
        anchor_entities_by_role=anchor_entities_by_role,
    )
    path_endpoint_overrides.update(
        _derive_direct_count_chain_endpoint_overrides(
            query_plan=query_plan,
            relation_paths=relation_paths,
            anchor_entities_by_role=anchor_entities_by_role,
        )
    )
    node_specs: dict[tuple[str, str], _NodeSpec] = {}

    def _resolve_node(raw_endpoint: Any, role: Any) -> _NodeSpec:
        endpoint_text = str(raw_endpoint or "").strip()
        role_token = _normalize_token(role)
        cache_key = (endpoint_text, role_token)
        cached = node_specs.get(cache_key)
        if cached is not None:
            return cached

        repeated_role_spec = repeated_role_endpoint_specs.get(cache_key)
        if repeated_role_spec is not None:
            node_specs[cache_key] = repeated_role_spec
            return repeated_role_spec

        node_spec = _build_node_spec(
            query_plan=query_plan,
            endpoint_text=endpoint_text,
            role_token=role_token,
            endpoint_stats=endpoint_stats,
            anchor_entities_by_role=anchor_entities_by_role,
            count_mode=count_mode,
        )
        node_specs[cache_key] = node_spec
        return node_spec

    triple_lines: list[str] = []
    seen_triples: set[str] = set()
    for index, path in enumerate(relation_paths):
        relation = str(path.get("relation") or "").strip()
        direction = _normalize_token(path.get("direction"))
        from_spec = path_endpoint_overrides.get((index, "from")) or _resolve_node(
            path.get("from"),
            path.get("from_role"),
        )
        to_spec = path_endpoint_overrides.get((index, "to")) or _resolve_node(
            path.get("to"),
            path.get("to_role"),
        )
        from_role = _normalize_token(path.get("from_role"))
        to_role = _normalize_token(path.get("to_role"))
        render_reversed = (
            _normalize_token(query_plan.get("query_shape")) == "count_over_direct_relation"
            and direction == "reverse"
            and from_role in _ANCHOR_ROLES
            and to_role
            in {
                "answer",
                "shared_answer",
                "candidate_set",
                "count_set",
                "constraint_value",
            }
        )
        subject_term = to_spec.term if render_reversed else from_spec.term
        object_term = from_spec.term if render_reversed else to_spec.term
        triple_line = f"  {subject_term} fb:{relation} {object_term} ."
        if triple_line in seen_triples:
            continue
        seen_triples.add(triple_line)
        triple_lines.append(triple_line)

    binding_lines: list[str] = []
    seen_bindings: set[str] = set()
    all_node_specs = [*node_specs.values(), *path_endpoint_overrides.values()]
    for node_spec in all_node_specs:
        binding_clause = str(node_spec.binding_clause or "").strip()
        if not binding_clause or binding_clause in seen_bindings:
            continue
        seen_bindings.add(binding_clause)
        for line in binding_clause.splitlines():
            binding_lines.append(line)

    return [*binding_lines, *triple_lines]


def _derive_repeated_role_endpoint_specs(
    *,
    query_plan: Mapping[str, Any],
    relation_paths: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], _NodeSpec]:
    overrides: dict[tuple[str, str], _NodeSpec] = {}
    explicit_variable_tokens = _collect_explicit_variable_tokens(query_plan)
    for path in relation_paths:
        from_role = _normalize_token(path.get("from_role"))
        to_role = _normalize_token(path.get("to_role"))
        if from_role != to_role or from_role not in _VARIABLE_NODE_ROLES:
            continue
        if from_role == "ordering_attribute":
            continue

        from_token = _normalize_token(path.get("from"))
        to_token = _normalize_token(path.get("to"))
        if not from_token or not to_token or from_token == to_token:
            continue

        canonical_token = _resolve_variable_name(
            query_plan=query_plan,
            endpoint_text=str(path.get("from") or ""),
            role_token=from_role,
            count_mode=False,
        )
        if from_token != canonical_token and to_token != canonical_token:
            variable_name = _build_named_node_variable_name(
                str(path.get("to") or ""),
                from_role,
            )
            overrides[(to_token, from_role)] = _NodeSpec(term=f"?{variable_name}")
            continue
        for endpoint_token, endpoint_text, other_token in (
            (from_token, str(path.get("from") or ""), to_token),
            (to_token, str(path.get("to") or ""), from_token),
        ):
            if endpoint_token == canonical_token or other_token != canonical_token:
                continue
            if endpoint_token in _GENERIC_NAMED_ROLE_TOKENS or endpoint_token in explicit_variable_tokens:
                continue
            variable_name = _build_named_node_variable_name(endpoint_text, from_role)
            overrides[(endpoint_token, from_role)] = _NodeSpec(term=f"?{variable_name}")
    return overrides


def _derive_direct_count_chain_endpoint_overrides(
    *,
    query_plan: Mapping[str, Any],
    relation_paths: Sequence[Mapping[str, Any]],
    anchor_entities_by_role: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[int, str], _NodeSpec]:
    if _normalize_token(query_plan.get("query_shape")) != "count_over_direct_relation":
        return {}

    anchored_tokens = {
        _normalize_token(entity.get("surface"))
        for entity in anchor_entities_by_role.values()
        if isinstance(entity, Mapping)
    } | {
        _normalize_token(entity.get("chosen_alias"))
        for entity in anchor_entities_by_role.values()
        if isinstance(entity, Mapping)
    }
    structural_tokens = {
        _normalize_token(query_plan.get("candidate_set_variable")),
        _normalize_token(query_plan.get("count_set_variable")),
        _normalize_token(query_plan.get("shared_answer_variable")),
    }
    structural_tokens.discard("")
    occurrences: dict[str, list[tuple[int, str]]] = {}
    for index, relation_path in enumerate(relation_paths):
        for side in ("from", "to"):
            endpoint_token = _normalize_token(relation_path.get(side))
            if (
                not endpoint_token
                or endpoint_token in _ANCHOR_ROLES
                or endpoint_token in _GENERIC_NAMED_ROLE_TOKENS
                or endpoint_token in anchored_tokens
                or endpoint_token in structural_tokens
            ):
                continue
            occurrences.setdefault(endpoint_token, []).append((index, side))

    overrides: dict[tuple[int, str], _NodeSpec] = {}
    for endpoint_token, token_occurrences in occurrences.items():
        if len(token_occurrences) < 2:
            continue
        shared_spec = _NodeSpec(term=f"?{endpoint_token}")
        for index, side in token_occurrences:
            overrides[(index, side)] = shared_spec
    return overrides


def _expand_anchor_specific_relation_paths(
    *,
    query_plan: Mapping[str, Any],
    relation_paths: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    query_shape = _normalize_token(query_plan.get("query_shape"))
    if query_shape == "count_over_joined_set":
        anchor_constraints = [
            dict(item)
            for item in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or [])
            if isinstance(item, Mapping)
        ]
        if not anchor_constraints:
            return [dict(path) for path in relation_paths]

        anchored_entities_by_role = {
            _normalize_token(item.get("role")): dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping) and _normalize_token(item.get("role"))
        }
        anchored_endpoint_tokens = {
            _normalize_token(value)
            for entity in anchored_entities_by_role.values()
            if isinstance(entity, Mapping)
            for value in (
                entity.get("surface"),
                entity.get("chosen_alias"),
                entity.get("resolved_entity_id"),
                entity.get("role"),
            )
            if _normalize_token(value)
        }
        single_anchor_joined_count = len(anchored_entities_by_role) <= 1
        expanded_paths: list[dict[str, Any]] = []
        for relation_path in relation_paths:
            generic_anchor_sides = [
                side
                for side in ("from", "to")
                if _normalize_token(relation_path.get(f"{side}_role")) == "anchor"
                and (
                    not single_anchor_joined_count
                    or _normalize_token(relation_path.get(side)) in _ANCHOR_ROLES
                    or _normalize_token(relation_path.get(side)) in anchored_endpoint_tokens
                )
            ]
            if len(generic_anchor_sides) != 1:
                expanded_paths.append(dict(relation_path))
                continue

            side = generic_anchor_sides[0]
            matching_constraints = [
                constraint
                for constraint in anchor_constraints
                if _constraint_semantically_matches_relation_path(
                    relation_path=relation_path,
                    anchor_constraint=constraint,
                )
            ]
            if not matching_constraints:
                expanded_paths.append(dict(relation_path))
                continue

            emitted = False
            for constraint in matching_constraints:
                anchor_role = _normalize_token(constraint.get("anchor_role"))
                if anchor_role not in anchored_entities_by_role:
                    continue
                cloned_path = dict(relation_path)
                cloned_path[side] = anchor_role
                cloned_path[f"{side}_role"] = anchor_role
                expanded_paths.append(cloned_path)
                emitted = True
            if not emitted:
                expanded_paths.append(dict(relation_path))
        return expanded_paths

    if query_shape not in {"multi_anchor_intersection", "shared_type_intersection"}:
        return [dict(path) for path in relation_paths]

    anchor_constraints = [
        dict(item)
        for item in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or [])
        if isinstance(item, Mapping)
    ]
    if not anchor_constraints:
        return [dict(path) for path in relation_paths]

    anchored_entities_by_role = {
        _normalize_token(item.get("role")): dict(item)
        for item in (query_plan.get("anchored_entities") or [])
        if isinstance(item, Mapping) and _normalize_token(item.get("role"))
    }

    expanded_paths: list[dict[str, Any]] = []
    for relation_path in relation_paths:
        generic_sides = [
            side
            for side in ("from", "to")
            if _path_side_is_generic_constraint_endpoint(
                endpoint_text=str(relation_path.get(side) or ""),
                role_token=_normalize_token(relation_path.get(f"{side}_role")),
            )
        ]
        if len(generic_sides) != 1:
            expanded_paths.append(dict(relation_path))
            continue

        matching_constraints = [
            constraint
            for constraint in anchor_constraints
            if _constraint_semantically_matches_relation_path(
                relation_path=relation_path,
                anchor_constraint=constraint,
            )
        ]
        if not matching_constraints:
            expanded_paths.append(dict(relation_path))
            continue

        side = generic_sides[0]
        emitted = False
        for constraint in matching_constraints:
            anchor_role = _normalize_token(constraint.get("anchor_role"))
            if not anchor_role:
                continue
            if anchor_role not in anchored_entities_by_role and anchor_role not in _CONSTRAINT_NODE_ROLES:
                continue
            cloned_path = dict(relation_path)
            cloned_path[side] = anchor_role
            cloned_path[f"{side}_role"] = anchor_role
            expanded_paths.append(cloned_path)
            emitted = True
        if not emitted:
            expanded_paths.append(dict(relation_path))
    return expanded_paths


def _path_side_is_generic_constraint_endpoint(
    *,
    endpoint_text: str,
    role_token: str,
) -> bool:
    endpoint_token = _normalize_token(endpoint_text)
    if role_token in _CONSTRAINT_NODE_ROLES:
        return True
    if (
        role_token == "anchor"
        and endpoint_token
        and endpoint_token not in _ANCHOR_ROLES
    ):
        return True
    return endpoint_token in _GENERIC_NAMED_ROLE_TOKENS


def _constraint_semantically_matches_relation_path(
    *,
    relation_path: Mapping[str, Any],
    anchor_constraint: Mapping[str, Any],
) -> bool:
    notes = str(anchor_constraint.get("notes") or "")
    relation = str(relation_path.get("relation") or "").strip()
    if not notes:
        return False

    relation_hints = {hint.lower() for hint in _extract_relation_hints_from_notes(notes)}
    if relation and relation.lower() in relation_hints:
        return True

    note_text = re.sub(r"[^a-z0-9]+", " ", notes.lower())
    relation_parts = [part.strip() for part in relation.split(".") if part.strip()]
    relation_phrases = []
    if relation_parts:
        relation_phrases.append(relation_parts[-1].replace("_", " ").strip())
    if len(relation_parts) >= 2:
        relation_phrases.append(
            " ".join(
                part.replace("_", " ").strip()
                for part in relation_parts[-2:]
                if part.strip()
            ).strip()
        )
    for phrase in relation_phrases:
        if phrase and len(phrase) >= 4 and re.search(
            rf"\b{re.escape(phrase)}\b",
            note_text,
        ):
            return True

    for side in ("from", "to"):
        endpoint_text = str(relation_path.get(side) or "")
        role_token = _normalize_token(relation_path.get(f"{side}_role"))
        if not _path_side_is_generic_constraint_endpoint(
            endpoint_text=endpoint_text,
            role_token=role_token,
        ):
            continue
        if _note_mentions_generic_constraint_endpoint(
            note_text=note_text,
            endpoint_text=endpoint_text,
        ):
            return True
    return False


def _note_mentions_generic_constraint_endpoint(
    *,
    note_text: str,
    endpoint_text: str,
) -> bool:
    endpoint_token = _normalize_token(endpoint_text)
    if (
        endpoint_token in _GENERIC_NAMED_ROLE_TOKENS
        or any(
            endpoint_token == f"{generic_token}_{suffix}"
            or endpoint_token.startswith(f"{generic_token}_")
            for generic_token in _GENERIC_NAMED_ROLE_TOKENS
            for suffix in ("2", "3", "4", "5")
        )
    ):
        return False
    endpoint_phrase = re.sub(r"[^a-z0-9]+", " ", endpoint_token).strip()
    if not endpoint_phrase:
        return False
    candidate_phrases = [endpoint_phrase]
    if endpoint_phrase.endswith(" value"):
        candidate_phrases.append(endpoint_phrase[: -len(" value")].strip())
    for phrase in candidate_phrases:
        if len(phrase) < 4:
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", note_text):
            return True
    return False


def _promote_projection_relation_roles(
    *,
    query_plan: Mapping[str, Any],
    relation_paths: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    answer_mode = _normalize_token(query_plan.get("answer_mode"))
    if answer_mode != "entity":
        return [dict(path) for path in relation_paths]

    projection = [
        _normalize_token(item)
        for item in (query_plan.get("projection") or [])
        if _normalize_token(item)
    ]
    if not projection:
        return [dict(path) for path in relation_paths]

    projection_token = projection[0]
    shared_answer_token = _normalize_token(query_plan.get("shared_answer_variable"))
    candidate_set_token = _normalize_token(query_plan.get("candidate_set_variable"))
    if projection_token in {"", shared_answer_token, candidate_set_token}:
        return [dict(path) for path in relation_paths]

    promoted: list[dict[str, Any]] = []
    for relation_path in relation_paths:
        updated = dict(relation_path)
        for side in ("from", "to"):
            endpoint_token = _normalize_token(updated.get(side))
            side_role = _normalize_token(updated.get(f"{side}_role"))
            if endpoint_token != projection_token:
                continue
            if side_role in {"shared_answer", "candidate_set"}:
                updated[f"{side}_role"] = "answer"
        promoted.append(updated)
    return promoted


def _derive_joined_set_endpoint_overrides(
    *,
    query_plan: Mapping[str, Any],
    relation_paths: Sequence[Mapping[str, Any]],
    anchor_entities_by_role: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[int, str], _NodeSpec]:
    if _normalize_token(query_plan.get("query_shape")) != "count_over_joined_set":
        return {}

    overrides: dict[tuple[int, str], _NodeSpec] = {}
    counted_variable_tokens = {
        _normalize_token(query_plan.get("candidate_set_variable")),
        _normalize_token(query_plan.get("count_set_variable")),
        _normalize_token(query_plan.get("shared_answer_variable")),
    }
    counted_variable_tokens.discard("")
    has_explicit_constraint_entities = any(
        role in _CONSTRAINT_NODE_ROLES for role in anchor_entities_by_role
    )
    anchored_endpoint_tokens = {
        _normalize_token(value)
        for entity in anchor_entities_by_role.values()
        if isinstance(entity, Mapping)
        for value in (
            entity.get("surface"),
            entity.get("chosen_alias"),
            entity.get("resolved_entity_id"),
            entity.get("role"),
        )
        if _normalize_token(value)
    }
    candidate_constraint_paths_by_relation: dict[str, list[tuple[int, str]]] = {}
    anchor_constraint_paths_by_relation: dict[str, list[tuple[int, str, str]]] = {}
    ordered_candidate_constraint_paths: list[tuple[int, str, str]] = []
    auxiliary_bridge_occurrences: dict[str, list[tuple[int, str, str]]] = {}
    structural_bridge_occurrences: dict[str, list[tuple[int, str]]] = {}
    misclassified_anchor_bridge_occurrences: dict[str, list[tuple[int, str]]] = {}
    constraint_endpoint_tokens_by_path: dict[tuple[int, str], str] = {}
    constraint_endpoint_occurrence_counts: dict[str, int] = {}

    for index, relation_path in enumerate(relation_paths):
        relation = str(relation_path.get("relation") or "").strip()
        if not relation:
            continue
        from_role = _normalize_token(relation_path.get("from_role"))
        to_role = _normalize_token(relation_path.get("to_role"))
        from_side_is_constraint = from_role in _CONSTRAINT_NODE_ROLES
        to_side_is_constraint = to_role in _CONSTRAINT_NODE_ROLES
        if from_side_is_constraint == to_side_is_constraint:
            for side in ("from", "to"):
                side_role = _normalize_token(relation_path.get(f"{side}_role"))
                endpoint_token = _normalize_token(relation_path.get(side))
                if (
                    side_role in {"candidate_set", "count_set", "shared_answer", "answer"}
                    and endpoint_token
                    and endpoint_token not in counted_variable_tokens
                    and endpoint_token not in _GENERIC_NAMED_ROLE_TOKENS
                    and endpoint_token not in anchored_endpoint_tokens
                ):
                    structural_bridge_occurrences.setdefault(endpoint_token, []).append(
                        (index, side)
                    )
                elif (
                    side_role in _ANCHOR_ROLES
                    and endpoint_token
                    and endpoint_token not in _ANCHOR_ROLES
                    and endpoint_token not in _GENERIC_NAMED_ROLE_TOKENS
                    and endpoint_token not in anchored_endpoint_tokens
                ):
                    misclassified_anchor_bridge_occurrences.setdefault(
                        endpoint_token,
                        [],
                    ).append((index, side))
                if side_role not in {"candidate_set", "count_set", "shared_answer"}:
                    continue
                if (
                    not endpoint_token
                    or endpoint_token in counted_variable_tokens
                    or endpoint_token in _GENERIC_NAMED_ROLE_TOKENS
                ):
                    continue
                other_side = "to" if side == "from" else "from"
                other_role = _normalize_token(relation_path.get(f"{other_side}_role"))
                if other_role not in _ANCHOR_ROLES | {"candidate_set", "count_set", "shared_answer"}:
                    continue
                auxiliary_bridge_occurrences.setdefault(
                    f"{relation}|{endpoint_token}",
                    [],
                ).append((index, side, other_role))
            continue
        constraint_side = "from" if from_side_is_constraint else "to"
        other_role = to_role if from_side_is_constraint else from_role
        constraint_endpoint_token = _normalize_token(relation_path.get(constraint_side))
        if constraint_endpoint_token:
            constraint_endpoint_tokens_by_path[(index, constraint_side)] = (
                constraint_endpoint_token
            )
            constraint_endpoint_occurrence_counts[constraint_endpoint_token] = (
                constraint_endpoint_occurrence_counts.get(constraint_endpoint_token, 0) + 1
            )
        if other_role in {"candidate_set", "count_set", "shared_answer"}:
            candidate_constraint_paths_by_relation.setdefault(relation, []).append(
                (index, constraint_side)
            )
            ordered_candidate_constraint_paths.append((index, constraint_side, relation))
        elif other_role in _ANCHOR_ROLES:
            anchor_constraint_paths_by_relation.setdefault(relation, []).append(
                (index, constraint_side, other_role)
            )

    for relation, candidate_paths in candidate_constraint_paths_by_relation.items():
        anchor_paths = anchor_constraint_paths_by_relation.get(relation) or []
        if not anchor_paths:
            continue
        shared_var = _build_shared_constraint_variable_name(
            relation=relation,
            relation_paths=relation_paths,
            candidate_paths=candidate_paths,
            anchor_paths=anchor_paths,
        )
        shared_spec = _NodeSpec(term=f"?{shared_var}")
        for index, constraint_side in candidate_paths:
            overrides[(index, constraint_side)] = shared_spec
        for index, constraint_side, _anchor_role in anchor_paths:
            overrides[(index, constraint_side)] = shared_spec

    for relation_key, occurrences in auxiliary_bridge_occurrences.items():
        endpoint_token = relation_key.split("|", 1)[1]
        touches_anchor = any(other_role in _ANCHOR_ROLES for _idx, _side, other_role in occurrences)
        touches_candidate_side = any(
            other_role in {"candidate_set", "count_set", "shared_answer"}
            for _idx, _side, other_role in occurrences
        )
        if not (touches_anchor and touches_candidate_side):
            continue
        shared_spec = _NodeSpec(term=f"?{endpoint_token}")
        for index, side, _other_role in occurrences:
            overrides.setdefault((index, side), shared_spec)

    for endpoint_token, anchor_occurrences in misclassified_anchor_bridge_occurrences.items():
        structural_occurrences = structural_bridge_occurrences.get(endpoint_token) or []
        if not structural_occurrences:
            continue
        shared_spec = _NodeSpec(term=f"?{endpoint_token}")
        for index, side in structural_occurrences:
            overrides.setdefault((index, side), shared_spec)
        for index, side in anchor_occurrences:
            overrides.setdefault((index, side), shared_spec)

    anchor_constraints = (
        (query_plan.get("join_structure") or {}).get("anchor_constraints") or []
    )
    for anchor_constraint in anchor_constraints:
        if not isinstance(anchor_constraint, Mapping):
            continue
        anchor_role = _normalize_token(anchor_constraint.get("anchor_role"))
        relation_hints = _extract_relation_hints_from_notes(
            str(anchor_constraint.get("notes") or "")
        )
        if not relation_hints:
            relation_hints = []

        if anchor_role in _ANCHOR_ROLES:
            matched_relation_hint = any(
                role == anchor_role
                for relation in relation_hints
                for _idx, _side, role in anchor_constraint_paths_by_relation.get(
                    relation, ()
                )
            )
            anchor_entity = anchor_entities_by_role.get(anchor_role)
            anchor_spec = _build_anchor_node_spec(
                anchor_role=anchor_role,
                anchor_entity=anchor_entity,
            )
            if anchor_spec is None:
                continue
            for relation in relation_hints:
                for index, constraint_side in candidate_constraint_paths_by_relation.get(
                    relation,
                    (),
                ):
                    if has_explicit_constraint_entities:
                        continue
                    # A relation-hinted anchor constraint should keep its anchor
                    # binding even when another constraint edge reuses the same
                    # lexical endpoint token.
                    overrides.setdefault((index, constraint_side), anchor_spec)
                    matched_relation_hint = True
            if matched_relation_hint or has_explicit_constraint_entities:
                continue

            for index, constraint_side, relation in ordered_candidate_constraint_paths:
                if (index, constraint_side) in overrides:
                    continue
                if anchor_constraint_paths_by_relation.get(relation):
                    continue
                constraint_endpoint_token = constraint_endpoint_tokens_by_path.get(
                    (index, constraint_side),
                    "",
                )
                if (
                    constraint_endpoint_token
                    and constraint_endpoint_token not in _GENERIC_NAMED_ROLE_TOKENS
                    and constraint_endpoint_occurrence_counts.get(
                        constraint_endpoint_token, 0
                    )
                    > 1
                ):
                    continue
                overrides[(index, constraint_side)] = anchor_spec
                break
            continue

        if anchor_role not in _CONSTRAINT_NODE_ROLES or has_explicit_constraint_entities:
            continue
        constraint_spec = _build_constraint_anchor_note_spec(
            anchor_role=anchor_role,
            anchor_constraint=anchor_constraint,
        )
        if constraint_spec is None:
            continue
        for relation in relation_hints:
            for index, constraint_side in candidate_constraint_paths_by_relation.get(
                relation,
                (),
            ):
                # Explicit answer-class / constraint guidance should be able to
                # override a tentative anchor-hinted binding on the same
                # relation when both appear in the repaired plan.
                overrides[(index, constraint_side)] = constraint_spec
    return overrides


def _build_shared_constraint_variable_name(
    *,
    relation: str,
    relation_paths: Sequence[Mapping[str, Any]],
    candidate_paths: Sequence[tuple[int, str]],
    anchor_paths: Sequence[tuple[int, str, str]],
) -> str:
    endpoint_tokens: list[str] = []
    for index, constraint_side in candidate_paths:
        endpoint_tokens.append(
            _normalize_token(relation_paths[index].get(constraint_side))  # type: ignore[index]
        )
    for index, constraint_side, _anchor_role in anchor_paths:
        endpoint_tokens.append(
            _normalize_token(relation_paths[index].get(constraint_side))  # type: ignore[index]
        )
    endpoint_tokens = [
        token
        for token in endpoint_tokens
        if token and token not in _GENERIC_NAMED_ROLE_TOKENS
    ]
    if endpoint_tokens:
        return endpoint_tokens[0]
    relation_tail = _normalize_token(str(relation or "").split(".")[-1])
    return relation_tail or "shared_constraint"


def _extract_relation_hints_from_notes(notes: str) -> list[str]:
    text = str(notes or "")
    seen: set[str] = set()
    hints: list[str] = []
    for match in re.finditer(r"\b[a-z0-9_]+\.[a-z0-9_]+\.[a-z0-9_]+\b", text, flags=re.IGNORECASE):
        relation = match.group(0).strip()
        lowered = relation.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        hints.append(relation)
    return hints


def _build_anchor_node_spec(
    *,
    anchor_role: str,
    anchor_entity: Mapping[str, Any] | None,
) -> _NodeSpec | None:
    role_token = _normalize_token(anchor_role)
    if role_token not in _ANCHOR_ROLES:
        return None
    if anchor_entity is None:
        return _NodeSpec(term=f"?{role_token}")
    resolved_mid = str(
        anchor_entity.get("resolved_entity_id")
        or anchor_entity.get("chosen_alias")
        or ""
    ).strip()
    if _looks_like_mid(resolved_mid):
        return _NodeSpec(term=f"fb:{resolved_mid}")
    labels = _dedupe_labels(
        [
            str(anchor_entity.get("chosen_alias") or "").strip(),
            str(anchor_entity.get("surface") or "").strip(),
        ]
    )
    if not labels:
        return _NodeSpec(term=f"?{role_token}")
    return _NodeSpec(
        term=f"?{role_token}",
        binding_clause=_build_anchor_binding_clause(
            variable_name=role_token,
            labels=labels,
        ),
    )


def _build_constraint_anchor_note_spec(
    *,
    anchor_role: str,
    anchor_constraint: Mapping[str, Any],
) -> _NodeSpec | None:
    role_token = _normalize_token(anchor_role)
    if role_token not in _CONSTRAINT_NODE_ROLES:
        return None
    notes = str(anchor_constraint.get("notes") or "")
    if not notes:
        return None
    match = re.search(r"'([^']+)'|\"([^\"]+)\"", notes)
    if match is None:
        return None
    label = str(match.group(1) or match.group(2) or "").strip()
    if not label:
        return None
    if _looks_like_mid(label):
        return _NodeSpec(term=f"fb:{label}")
    variable_name = _build_named_node_variable_name(label, role_token)
    return _NodeSpec(
        term=f"?{variable_name}",
        binding_clause=_build_name_binding_clause(
            variable_name=variable_name,
            label=label,
        ),
    )


def _build_node_spec(
    *,
    query_plan: Mapping[str, Any],
    endpoint_text: str,
    role_token: str,
    endpoint_stats: Mapping[str, int],
    anchor_entities_by_role: Mapping[str, Mapping[str, Any]],
    count_mode: bool,
) -> _NodeSpec:
    endpoint_token = _normalize_token(endpoint_text)
    inferred_role = _infer_structural_role_from_endpoint(
        query_plan=query_plan,
        endpoint_token=endpoint_token,
    )
    if inferred_role and (
        role_token not in _ANCHOR_ROLES | _CONSTRAINT_NODE_ROLES
        or (role_token == "anchor" and inferred_role in _VARIABLE_NODE_ROLES | _CONSTRAINT_NODE_ROLES)
    ):
        role_token = inferred_role

    if role_token in _CONSTRAINT_NODE_ROLES:
        constrained_entity = anchor_entities_by_role.get(role_token)
        if constrained_entity is not None:
            resolved_mid = str(
                constrained_entity.get("resolved_entity_id")
                or constrained_entity.get("chosen_alias")
                or ""
            ).strip()
            if _looks_like_mid(resolved_mid):
                return _NodeSpec(term=f"fb:{resolved_mid}")
            labels = _dedupe_labels(
                [
                    str(constrained_entity.get("chosen_alias") or "").strip(),
                    str(constrained_entity.get("surface") or "").strip(),
                ]
            )
            variable_name = role_token
            if labels:
                return _NodeSpec(
                    term=f"?{variable_name}",
                    binding_clause=_build_anchor_binding_clause(
                        variable_name=variable_name,
                        labels=labels,
                    ),
                )
    specific_anchor_role = (
        endpoint_token if endpoint_token in _ANCHOR_ROLES else role_token
    )
    if _looks_like_mid(endpoint_text):
        return _NodeSpec(term=f"fb:{endpoint_text}")

    if specific_anchor_role in _ANCHOR_ROLES:
        anchor_entity = anchor_entities_by_role.get(specific_anchor_role)
        if anchor_entity is not None:
            resolved_mid = str(
                anchor_entity.get("resolved_entity_id")
                or anchor_entity.get("chosen_alias")
                or ""
            ).strip()
            if _looks_like_mid(resolved_mid):
                return _NodeSpec(term=f"fb:{resolved_mid}")
            labels = _dedupe_labels(
                [
                    str(anchor_entity.get("chosen_alias") or "").strip(),
                    str(anchor_entity.get("surface") or "").strip(),
                ]
            )
            var_name = specific_anchor_role
            return _NodeSpec(
                term=f"?{var_name}",
                binding_clause=_build_anchor_binding_clause(
                    variable_name=var_name,
                    labels=labels,
                ),
            )
        return _NodeSpec(
            term=f"?{specific_anchor_role or 'anchor'}",
            binding_clause=_build_name_binding_clause(
                variable_name=specific_anchor_role or "anchor",
                label=endpoint_text,
            ),
        )

    if role_token in _VARIABLE_NODE_ROLES:
        return _NodeSpec(
            term=f"?{_resolve_variable_name(query_plan=query_plan, endpoint_text=endpoint_text, role_token=role_token, count_mode=count_mode)}"
        )

    explicit_variable_tokens = _collect_explicit_variable_tokens(query_plan)
    if role_token in {"constraint_value", "shared_type", "type_set", "anchor_value"}:
        if (
            endpoint_token
            and endpoint_token not in _GENERIC_NAMED_ROLE_TOKENS
            and endpoint_token not in explicit_variable_tokens
            and endpoint_stats.get(endpoint_token, 0) <= 1
        ):
            variable_name = _build_named_node_variable_name(endpoint_text, role_token)
            return _NodeSpec(
                term=f"?{variable_name}",
                binding_clause=_build_name_binding_clause(
                    variable_name=variable_name,
                    label=endpoint_text,
                ),
            )
        return _NodeSpec(
            term=f"?{_resolve_variable_name(query_plan=query_plan, endpoint_text=endpoint_text, role_token=role_token, count_mode=count_mode)}"
        )

    if endpoint_token and endpoint_token not in explicit_variable_tokens:
        variable_name = _build_named_node_variable_name(endpoint_text, role_token or "node")
        return _NodeSpec(
            term=f"?{variable_name}",
            binding_clause=_build_name_binding_clause(
                variable_name=variable_name,
                label=endpoint_text,
            ),
        )

    return _NodeSpec(
        term=f"?{_resolve_variable_name(query_plan=query_plan, endpoint_text=endpoint_text, role_token=role_token, count_mode=count_mode)}"
    )


def _infer_structural_role_from_endpoint(
    *,
    query_plan: Mapping[str, Any],
    endpoint_token: str,
) -> str:
    if not endpoint_token:
        return ""
    role_candidates = {
        _normalize_token(query_plan.get("shared_answer_variable")): "shared_answer",
        _normalize_token(query_plan.get("candidate_set_variable")): "candidate_set",
        _normalize_token(query_plan.get("count_set_variable")): "count_set",
        _normalize_token((query_plan.get("ordering_attribute") or {}).get("attribute_variable")): "ordering_attribute",
        _normalize_token((query_plan.get("ordering_attribute") or {}).get("source_variable")): "candidate_set",
    }
    if endpoint_token in role_candidates and role_candidates[endpoint_token]:
        return role_candidates[endpoint_token]
    if endpoint_token in _VARIABLE_NODE_ROLES | _CONSTRAINT_NODE_ROLES:
        return endpoint_token
    return ""


def _resolve_variable_name(
    *,
    query_plan: Mapping[str, Any],
    endpoint_text: str,
    role_token: str,
    count_mode: bool,
) -> str:
    projection = [
        str(item).strip()
        for item in (query_plan.get("projection") or [])
        if str(item).strip()
    ]
    if role_token == "answer":
        if count_mode:
            candidate_set_var = str(query_plan.get("candidate_set_variable") or "").strip()
            shared_answer_var = str(query_plan.get("shared_answer_variable") or "").strip()
            return (
                _normalize_token(candidate_set_var)
                or _normalize_token(shared_answer_var)
                or _normalize_token(endpoint_text)
                or "answer"
            )
        if projection:
            return _normalize_token(projection[0]) or "answer"
        shared_answer_var = str(query_plan.get("shared_answer_variable") or "").strip()
        return _normalize_token(shared_answer_var) or _normalize_token(endpoint_text) or "answer"
    if role_token == "shared_answer":
        shared_answer_var = str(query_plan.get("shared_answer_variable") or "").strip()
        endpoint_token = _normalize_token(endpoint_text)
        shared_answer_token = _normalize_token(shared_answer_var)
        if (
            endpoint_token
            and shared_answer_token
            and endpoint_token.startswith(shared_answer_token)
            and endpoint_token != shared_answer_token
        ):
            return endpoint_token
        return _normalize_token(shared_answer_var) or _normalize_token(endpoint_text) or "shared_answer"
    if role_token == "candidate_set":
        query_shape = _normalize_token(query_plan.get("query_shape"))
        candidate_set_var = str(query_plan.get("candidate_set_variable") or "").strip()
        endpoint_token = _normalize_token(endpoint_text)
        candidate_set_token = _normalize_token(candidate_set_var)
        if query_shape == "shared_type_intersection":
            shared_answer_var = str(query_plan.get("shared_answer_variable") or "").strip()
            shared_answer_token = _normalize_token(shared_answer_var)
            return shared_answer_token or candidate_set_token or endpoint_token or "shared_answer"
        if (
            endpoint_token
            and candidate_set_token
            and endpoint_token.startswith(candidate_set_token)
            and endpoint_token != candidate_set_token
        ):
            return endpoint_token
        return candidate_set_token or endpoint_token or "candidate_set"
    if role_token == "count_set":
        query_shape = _normalize_token(query_plan.get("query_shape"))
        count_set_var = str(query_plan.get("count_set_variable") or "").strip()
        candidate_set_var = str(query_plan.get("candidate_set_variable") or "").strip()
        endpoint_token = _normalize_token(endpoint_text)
        count_set_token = _normalize_token(count_set_var)
        candidate_set_token = _normalize_token(candidate_set_var)
        if count_set_token in {"count", "result_count"}:
            count_set_token = ""
        if (
            query_shape == "count_over_direct_relation"
            and endpoint_token
            and _count_set_endpoint_is_terminal_leaf(
                query_plan=query_plan,
                endpoint_token=endpoint_token,
            )
        ):
            return endpoint_token
        if (
            query_shape == "count_over_joined_set"
            and candidate_set_token
            and not count_set_token
        ):
            if endpoint_token and _count_set_endpoint_is_terminal_leaf(
                query_plan=query_plan,
                endpoint_token=endpoint_token,
            ):
                return endpoint_token
            return candidate_set_token
        if (
            endpoint_token
            and endpoint_token not in {"count", "result_count"}
            and endpoint_token != candidate_set_token
        ):
            return count_set_token or endpoint_token
        return (
            count_set_token
            or candidate_set_token
            or endpoint_token
            or "count_set"
        )
    if role_token == "ordering_attribute":
        endpoint_token = _normalize_token(endpoint_text)
        attribute_var = str(
            (query_plan.get("ordering_attribute") or {}).get("attribute_variable") or ""
        ).strip()
        attribute_token = _normalize_token(attribute_var)
        if endpoint_token and attribute_token and endpoint_token != attribute_token:
            return endpoint_token
        return attribute_token or endpoint_token or "ordering_attribute"
    if role_token == "type_set":
        endpoint_token = _normalize_token(endpoint_text)
        return endpoint_token or "type_set"
    if role_token == "shared_type":
        endpoint_token = _normalize_token(endpoint_text)
        return endpoint_token or "shared_type"
    if role_token == "constraint_value":
        return _normalize_token(endpoint_text) or "constraint_value"
    if role_token == "anchor_value":
        return _normalize_token(endpoint_text) or "anchor_value"
    return _normalize_token(endpoint_text) or role_token or "node"


def _resolve_entity_variable_name(*, query_plan: Mapping[str, Any]) -> str:
    query_shape = _normalize_token(query_plan.get("query_shape"))
    relation_paths = [
        relation_path
        for relation_path in (query_plan.get("relation_paths") or [])
        if isinstance(relation_path, Mapping)
    ]
    projection = [
        str(item).strip()
        for item in (query_plan.get("projection") or [])
        if str(item).strip()
    ]
    if projection:
        projection_token = _normalize_token(projection[0])
        if _token_is_structural_in_relation_paths(projection_token, relation_paths):
            return projection_token
        if query_shape == "superlative_chain":
            candidate_set_var = str(query_plan.get("candidate_set_variable") or "").strip()
            candidate_set_token = _normalize_token(candidate_set_var)
            if (
                candidate_set_token
                and _token_is_structural_in_relation_paths(candidate_set_token, relation_paths)
            ):
                return candidate_set_token
        return projection_token

    if query_shape == "shared_type_intersection":
        for relation_path in (query_plan.get("relation_paths") or []):
            if not isinstance(relation_path, Mapping):
                continue
            for endpoint_key, role_key in (("from", "from_role"), ("to", "to_role")):
                if _normalize_token(relation_path.get(role_key)) == "shared_type":
                    return _normalize_token(relation_path.get(endpoint_key))

    shared_answer_var = str(query_plan.get("shared_answer_variable") or "").strip()
    candidate_set_var = str(query_plan.get("candidate_set_variable") or "").strip()
    return (
        _normalize_token(shared_answer_var)
        or _normalize_token(candidate_set_var)
        or "answer"
    )


def _resolve_count_variable_name(*, query_plan: Mapping[str, Any]) -> str:
    query_shape = _normalize_token(query_plan.get("query_shape"))
    count_set_var = str(query_plan.get("count_set_variable") or "").strip()
    candidate_set_var = str(query_plan.get("candidate_set_variable") or "").strip()
    shared_answer_var = str(query_plan.get("shared_answer_variable") or "").strip()
    count_set_token = _normalize_token(count_set_var)
    candidate_set_token = _normalize_token(candidate_set_var)
    shared_answer_token = _normalize_token(shared_answer_var)
    relation_paths = [
        relation_path
        for relation_path in (query_plan.get("relation_paths") or [])
        if isinstance(relation_path, Mapping)
    ]
    explicit_count_role_tokens: list[str] = []
    for relation_path in relation_paths:
        for endpoint_key, role_key in (("from", "from_role"), ("to", "to_role")):
            if _normalize_token(relation_path.get(role_key)) != "count_set":
                continue
            endpoint_token = _normalize_token(relation_path.get(endpoint_key))
            if endpoint_token and endpoint_token not in {"count", "result_count"}:
                explicit_count_role_tokens.append(endpoint_token)
    if query_shape == "count_over_direct_relation":
        for endpoint_token in explicit_count_role_tokens:
            if _count_set_endpoint_is_terminal_leaf(
                query_plan=query_plan,
                endpoint_token=endpoint_token,
            ):
                return endpoint_token
        candidate_set_paths = [
            relation_path
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
            and (
                _normalize_token(relation_path.get("from_role")) == "candidate_set"
                or _normalize_token(relation_path.get("to_role")) == "candidate_set"
            )
        ]
        if (
            candidate_set_paths
            and candidate_set_token
            and _token_is_structural_in_relation_paths(candidate_set_token, relation_paths)
        ):
            return candidate_set_token
    if query_shape == "count_over_joined_set" and candidate_set_token and count_set_token in {"", "count", "result_count"}:
        for endpoint_token in explicit_count_role_tokens:
            if _count_set_endpoint_is_terminal_leaf(
                query_plan=query_plan,
                endpoint_token=endpoint_token,
            ):
                return endpoint_token
        return candidate_set_token
    if query_shape == "count_over_joined_set":
        if (
            shared_answer_token
            and _token_is_structural_in_relation_paths(shared_answer_token, relation_paths)
        ):
            return shared_answer_token
        if (
            candidate_set_token
            and _token_is_structural_in_relation_paths(candidate_set_token, relation_paths)
        ):
            return candidate_set_token
    for endpoint_token in explicit_count_role_tokens:
        if _token_is_structural_in_relation_paths(endpoint_token, relation_paths):
            return endpoint_token
    return (
        (
            count_set_token
            if count_set_token
            and count_set_token not in {"count", "result_count"}
            and _token_is_structural_in_relation_paths(count_set_token, relation_paths)
            else ""
        )
        or (
            candidate_set_token
            if candidate_set_token
            and _token_is_structural_in_relation_paths(candidate_set_token, relation_paths)
            else ""
        )
        or candidate_set_token
        or (
            shared_answer_token
            if shared_answer_token
            and _token_is_structural_in_relation_paths(shared_answer_token, relation_paths)
            else ""
        )
        or shared_answer_token
        or "count_set"
    )


def _resolve_ordering_variable_name(*, query_plan: Mapping[str, Any]) -> str:
    relation_paths = [
        relation_path
        for relation_path in (query_plan.get("relation_paths") or [])
        if isinstance(relation_path, Mapping)
    ]
    ordering_attribute = query_plan.get("ordering_attribute") or {}
    if isinstance(ordering_attribute, Mapping):
        attribute_var = str(ordering_attribute.get("attribute_variable") or "").strip()
        if attribute_var:
            current_token = _normalize_token(attribute_var)
            seen_tokens: set[str] = set()
            while current_token and current_token not in seen_tokens:
                seen_tokens.add(current_token)
                next_token = ""
                for relation_path in relation_paths:
                    roles = {
                        _normalize_token(relation_path.get("from_role")),
                        _normalize_token(relation_path.get("to_role")),
                    }
                    if "ordering_attribute" not in roles:
                        continue
                    from_token = _normalize_token(relation_path.get("from"))
                    to_token = _normalize_token(relation_path.get("to"))
                    if from_token == current_token and to_token and to_token != current_token:
                        next_token = to_token
                        break
                    if (
                        to_token == current_token
                        and _normalize_token(relation_path.get("direction")) == "reverse"
                        and from_token
                        and from_token != current_token
                    ):
                        next_token = from_token
                        break
                if not next_token:
                    return current_token
                current_token = next_token
            return current_token or _normalize_token(attribute_var)
    for relation_path in relation_paths:
        for endpoint_key, role_key in (("from", "from_role"), ("to", "to_role")):
            if _normalize_token(relation_path.get(role_key)) == "ordering_attribute":
                return _normalize_token(relation_path.get(endpoint_key))
    return ""


def _count_set_endpoint_is_terminal_leaf(
    *,
    query_plan: Mapping[str, Any],
    endpoint_token: str,
) -> bool:
    normalized_endpoint = _normalize_token(endpoint_token)
    if not normalized_endpoint:
        return False
    saw_count_set_leaf = False
    for relation_path in (query_plan.get("relation_paths") or []):
        if not isinstance(relation_path, Mapping):
            continue
        if (
            _normalize_token(relation_path.get("from")) == normalized_endpoint
            and _normalize_token(relation_path.get("from_role"))
            in _VARIABLE_NODE_ROLES | {"anchor"}
        ):
            return False
        if _normalize_token(relation_path.get("to_role")) != "count_set":
            continue
        if _normalize_token(relation_path.get("to")) == normalized_endpoint:
            saw_count_set_leaf = True
    return saw_count_set_leaf


def _collect_explicit_variable_tokens(query_plan: Mapping[str, Any]) -> set[str]:
    tokens = {
        _normalize_token(query_plan.get("shared_answer_variable")),
        _normalize_token(query_plan.get("candidate_set_variable")),
        _normalize_token(query_plan.get("count_set_variable")),
        _normalize_token((query_plan.get("ordering_attribute") or {}).get("source_variable")),
        _normalize_token((query_plan.get("ordering_attribute") or {}).get("attribute_variable")),
    }
    projection = query_plan.get("projection") or []
    if isinstance(projection, Sequence):
        tokens.update(_normalize_token(item) for item in projection)
    return {token for token in tokens if token}


def _count_endpoint_occurrences(
    relation_paths: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in relation_paths:
        for key in ("from", "to"):
            token = _normalize_token(path.get(key))
            if not token:
                continue
            counts[token] = counts.get(token, 0) + 1
    return counts


def _token_is_structural_in_relation_paths(
    token: str,
    relation_paths: Sequence[Mapping[str, Any]],
) -> bool:
    normalized_token = _normalize_token(token)
    if not normalized_token:
        return False
    for relation_path in relation_paths:
        if not isinstance(relation_path, Mapping):
            continue
        if normalized_token in {
            _normalize_token(relation_path.get("from")),
            _normalize_token(relation_path.get("to")),
            _normalize_token(relation_path.get("from_role")),
            _normalize_token(relation_path.get("to_role")),
        }:
            return True
    return False


def _build_anchor_binding_clause(
    *,
    variable_name: str,
    labels: Sequence[str],
) -> str:
    deduped_labels = [label for label in labels if label]
    if not deduped_labels:
        return ""
    primary_label = deduped_labels[0]
    if len(deduped_labels) == 1:
        return _build_name_binding_clause(
            variable_name=variable_name,
            label=primary_label,
        )
    alias_label = deduped_labels[1]
    label_var = f"?{variable_name}_label"
    return "\n".join(
        [
            "  {",
            f"    ?{variable_name} fb:type.object.name {label_var} .",
            f"    FILTER(LCASE(STR({label_var})) = {json.dumps(primary_label.lower())})",
            "  } UNION {",
            f"    ?{variable_name} fb:common.topic.alias {label_var} .",
            f"    FILTER(LCASE(STR({label_var})) = {json.dumps(alias_label.lower())})",
            "  }",
        ]
    )


def _build_name_binding_clause(*, variable_name: str, label: str) -> str:
    safe_label = str(label or "").strip()
    if not safe_label:
        return ""
    label_var = f"?{variable_name}_label"
    return "\n".join(
        [
            f"  ?{variable_name} fb:type.object.name {label_var} .",
            f"  FILTER(LCASE(STR({label_var})) = {json.dumps(safe_label.lower())})",
        ]
    )


def _build_named_node_variable_name(label: str, role_token: str) -> str:
    label_token = _normalize_token(label)
    role_suffix = _normalize_token(role_token)
    if label_token and role_suffix and not label_token.endswith(role_suffix):
        return f"{label_token}_{role_suffix}"
    return label_token or role_suffix or "node"


def _wrap_query_as_sage_program(query_text: str) -> str:
    indented_query = "\n".join(f"    {line}" for line in query_text.splitlines())
    return "\n".join(
        [
            "###QUERY_START",
            "from SPARQLWrapper import SPARQLWrapper, JSON",
            "",
            "def solve(endpoint_url):",
            "    sparql = SPARQLWrapper(endpoint_url)",
            '    query = """',
            indented_query,
            '    """',
            "    sparql.setQuery(query)",
            "    sparql.setReturnFormat(JSON)",
            "    return sparql.query().convert()",
            "###QUERY_END",
        ]
    )


def _dedupe_labels(labels: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(text)
    return deduped


def _looks_like_mid(raw_text: Any) -> bool:
    value = str(raw_text or "").strip()
    return bool(re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", value))


def _normalize_token(raw_value: Any) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    value = value.lstrip("?")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value
