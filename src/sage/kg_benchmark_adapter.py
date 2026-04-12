from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


_FREEBASE_NS_PREFIX = "http://rdf.freebase.com/ns/"
_MID_PATTERN = re.compile(r"[mg]\.[A-Za-z0-9_]+")
_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_DATE_PATTERN = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?)?")
_COUNT_VAR_HINT_PATTERN = re.compile(r"(?:^|_)(?:count|counts?|result_count)(?:$|_)")
SAGE_BENCHMARK_BRIDGE_TOOL_NAME = "sage_benchmark_bridge_macro"
_BRIDGE_BANNED_TOKENS = (
    "SPARQLWrapper",
    "requests.",
    "urllib.",
    "http://",
    "https://",
    "get_relations",
    "get_neighbors",
    "intersection(",
    "count(",
    "get_attributes",
    "argmax(",
    "argmin(",
)


@dataclass(frozen=True)
class PalExecutionArtifact:
    raw_result: Any
    artifact_type: str
    value: Any
    is_structured: bool
    source: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkAdapterContext:
    task_question: str
    run_id: str
    state_dir: str
    bridge_tool_name: str = SAGE_BENCHMARK_BRIDGE_TOOL_NAME
    family_name: str = ""
    query_shape: str = ""


@dataclass(frozen=True)
class BenchmarkMaterialization:
    materialization_type: str
    needs_bridge: bool
    bridge_action: Optional[str]
    bridge_tool_name: Optional[str]
    bridge_payload: Optional[Mapping[str, Any]]
    final_variable: Optional[str]
    final_answer_text: Optional[str]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    determinism_level: str = "unknown"
    tool_status: str = "failed"
    semantic_description: Optional[str] = None
    solves_task: bool = False
    trusted_for_materialization: bool = False
    useful_intermediate: bool = False
    intermediate_variables: tuple[str, ...] = ()
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class PalBenchmarkAdaptation:
    artifact: PalExecutionArtifact
    materialization: BenchmarkMaterialization


def classify_execution_artifact(raw_result: Any) -> PalExecutionArtifact:
    if not isinstance(raw_result, Mapping):
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="unresolved",
            value=None,
            is_structured=False,
            source="raw_execution",
            diagnostics={"reason": "raw_result_not_mapping"},
        )

    if str(raw_result.get("sage_execution_status") or "").strip().lower() == "error":
        failure_kind = str(raw_result.get("sage_failure_kind") or "execution_error")
        diagnostics = dict(raw_result.get("sage_diagnostics") or {})
        diagnostics["failure_kind"] = failure_kind
        diagnostics.setdefault(
            "error",
            str(raw_result.get("sage_error") or "sage_execution_failed"),
        )
        diagnostics.setdefault(
            "endpoint_url",
            str(raw_result.get("endpoint_url") or ""),
        )
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="unresolved",
            value=failure_kind,
            is_structured=True,
            source="raw_execution_failure",
            diagnostics=diagnostics,
        )

    if "boolean" in raw_result:
        boolean_value = bool(raw_result.get("boolean"))
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="boolean",
            value=boolean_value,
            is_structured=True,
            source="raw_execution",
            diagnostics={"result_kind": "boolean"},
        )

    results_payload = raw_result.get("results")
    if not isinstance(results_payload, Mapping) or "bindings" not in results_payload:
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="unresolved",
            value=None,
            is_structured=False,
            source="raw_execution",
            diagnostics={"reason": "results_bindings_missing"},
        )

    bindings = results_payload.get("bindings", [])
    if not isinstance(bindings, Sequence):
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="unresolved",
            value=None,
            is_structured=False,
            source="raw_execution",
            diagnostics={"reason": "bindings_not_sequence"},
        )
    if len(bindings) == 0:
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="empty",
            value=[],
            is_structured=True,
            source="raw_execution",
            diagnostics={"binding_count": 0},
        )

    answer_var, diagnostics = _select_answer_var(raw_result, bindings)
    if answer_var is not None:
        bindings_missing_selected_var = [
            binding
            for binding in bindings
            if isinstance(binding, Mapping) and answer_var not in binding
        ]
        if bindings_missing_selected_var:
            diagnostics["reason"] = "selected_head_var_missing_from_bindings"
            diagnostics["available_binding_vars"] = sorted(
                {
                    str(key)
                    for binding in bindings
                    if isinstance(binding, Mapping)
                    for key in binding.keys()
                }
            )
            return PalExecutionArtifact(
                raw_result=raw_result,
                artifact_type="unresolved",
                value=None,
                is_structured=True,
                source="raw_execution",
                diagnostics=diagnostics,
            )
    normalized_values = _extract_normalized_values(bindings, answer_var)
    diagnostics["binding_count"] = len(bindings)
    diagnostics["normalized_value_count"] = len(normalized_values)
    diagnostics["value_preview"] = [
        str(item["value"]) for item in normalized_values[:5]
    ]
    diagnostics["row_preview"] = _build_binding_row_preview(
        bindings,
        answer_var=answer_var,
        head_vars=diagnostics.get("head_vars"),
    )
    if not normalized_values:
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="empty",
            value=[],
            is_structured=True,
            source="raw_execution",
            diagnostics=diagnostics,
        )

    kinds = {item["kind"] for item in normalized_values}
    if kinds == {"entity_id"}:
        entity_values = [item["value"] for item in normalized_values]
        artifact_type = "entity_id" if len(entity_values) == 1 else "entity_set"
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type=artifact_type,
            value=entity_values[0] if artifact_type == "entity_id" else entity_values,
            is_structured=True,
            source="raw_execution",
            diagnostics=diagnostics,
        )
    if kinds == {"numeric"} and len(normalized_values) == 1:
        selected_var_token = _normalize_variable_hint(answer_var)
        artifact_type = (
            "count_scalar"
            if _looks_like_count_variable(selected_var_token)
            else "scalar_literal"
        )
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type=artifact_type,
            value=normalized_values[0]["value"],
            is_structured=True,
            source="raw_execution",
            diagnostics=diagnostics,
        )
    if kinds == {"boolean"} and len(normalized_values) == 1:
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="boolean",
            value=normalized_values[0]["value"],
            is_structured=True,
            source="raw_execution",
            diagnostics=diagnostics,
        )
    if kinds <= {"numeric", "date", "text_literal"}:
        literal_values = [item["value"] for item in normalized_values]
        artifact_type = "scalar_literal" if any(
            item["kind"] in {"numeric", "date"} for item in normalized_values
        ) else "text_literal"
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type=artifact_type,
            value=literal_values[0] if len(literal_values) == 1 else literal_values,
            is_structured=True,
            source="raw_execution",
            diagnostics=diagnostics,
        )

    diagnostics["value_kinds"] = sorted(kinds)
    diagnostics["normalized_values_preview"] = [
        item["value"] for item in normalized_values[:5]
    ]
    return PalExecutionArtifact(
        raw_result=raw_result,
        artifact_type="unresolved",
        value=None,
        is_structured=True,
        source="raw_execution",
        diagnostics=diagnostics,
    )


def materialize_benchmark_artifact(
    artifact: PalExecutionArtifact,
    *,
    context: BenchmarkAdapterContext,
) -> BenchmarkMaterialization:
    if artifact.artifact_type in {"unresolved", "empty"}:
        failure_reason = str(
            artifact.diagnostics.get("failure_kind")
            or artifact.diagnostics.get("reason")
            or artifact.artifact_type
            or "unresolved_artifact"
        ).strip()
        return BenchmarkMaterialization(
            materialization_type=(
                "empty_failure" if artifact.artifact_type == "empty" else "unresolved_failure"
            ),
            needs_bridge=False,
            bridge_action=None,
            bridge_tool_name=None,
            bridge_payload=None,
            final_variable=None,
            final_answer_text=None,
            diagnostics={
                "artifact_type": artifact.artifact_type,
                "artifact_source": artifact.source,
                "failure_kind": artifact.diagnostics.get("failure_kind"),
            },
            confidence=0.0,
            determinism_level="none",
            tool_status="failed",
            semantic_description=_describe_artifact_semantics(artifact),
            solves_task=False,
            trusted_for_materialization=False,
            useful_intermediate=False,
            failure_reason=failure_reason,
        )

    materialization_diagnostics = {
        "artifact_type": artifact.artifact_type,
        "artifact_source": artifact.source,
    }
    family_name = str(context.family_name or "").strip()
    if family_name:
        materialization_diagnostics["family_name"] = family_name
    query_shape = str(context.query_shape or "").strip()
    if query_shape:
        materialization_diagnostics["query_shape"] = query_shape
    selected_query_variable = str(
        artifact.diagnostics.get("selected_var") or ""
    ).strip()
    if selected_query_variable:
        materialization_diagnostics["selected_query_variable"] = selected_query_variable
    binding_count = artifact.diagnostics.get("binding_count")
    if isinstance(binding_count, int):
        materialization_diagnostics["binding_count"] = binding_count
    unique_value_count = artifact.diagnostics.get("normalized_value_count")
    if not isinstance(unique_value_count, int):
        unique_value_count = artifact.diagnostics.get("parsed_value_count")
    if isinstance(unique_value_count, int):
        materialization_diagnostics["unique_value_count"] = unique_value_count
    value_preview = artifact.diagnostics.get("value_preview")
    if isinstance(value_preview, Sequence) and not isinstance(value_preview, (str, bytes)):
        materialization_diagnostics["value_preview"] = [
            str(item).strip()
            for item in value_preview
            if str(item).strip()
        ][:5]
    row_preview = artifact.diagnostics.get("row_preview")
    if isinstance(row_preview, Sequence) and not isinstance(row_preview, (str, bytes)):
        compact_rows: list[dict[str, str]] = []
        for row in row_preview:
            if not isinstance(row, Mapping):
                continue
            compact_row: dict[str, str] = {}
            for key, value in row.items():
                cleaned_key = str(key or "").strip()
                cleaned_value = str(value or "").strip()
                if cleaned_key and cleaned_value:
                    compact_row[cleaned_key] = cleaned_value
            if compact_row:
                compact_rows.append(compact_row)
        if compact_rows:
            materialization_diagnostics["row_preview"] = compact_rows[:3]

    trusted_for_materialization = artifact.source == "raw_execution"
    useful_intermediate = (
        artifact.source != "raw_execution"
        and artifact.artifact_type
        in {
            "entity_id",
            "entity_set",
            "count_scalar",
            "boolean",
            "scalar_literal",
            "text_literal",
        }
    )
    tool_status = "success" if trusted_for_materialization else (
        "partial" if useful_intermediate else "failed"
    )
    bridge_payload = {
        "sage_artifact_type": artifact.artifact_type,
        "sage_artifact_value": artifact.value,
        "sage_artifact_source": artifact.source,
        "sage_artifact_diagnostics": dict(artifact.diagnostics),
        "sage_family_name": family_name,
        "sage_query_shape": query_shape,
        "run_id": context.run_id,
        "state_dir": context.state_dir,
    }
    bridge_action = (
        f"Action: execute_macro({json.dumps(context.bridge_tool_name)}, "
        f"{json.dumps(bridge_payload, ensure_ascii=False)})"
    )
    return BenchmarkMaterialization(
        materialization_type="bridge_action",
        needs_bridge=True,
        bridge_action=bridge_action,
        bridge_tool_name=context.bridge_tool_name,
        bridge_payload=bridge_payload,
        final_variable=None,
        final_answer_text=None,
        diagnostics=materialization_diagnostics,
        confidence=_confidence_for_source(artifact.source),
        determinism_level=_determinism_for_source(artifact.source),
        tool_status=tool_status,
        semantic_description=_describe_artifact_semantics(artifact),
        solves_task=trusted_for_materialization,
        trusted_for_materialization=trusted_for_materialization,
        useful_intermediate=useful_intermediate,
        failure_reason=None,
    )


def adapt_sage_result_to_benchmark(
    *,
    raw_result: Any,
    solver_output: Optional[str],
    context: BenchmarkAdapterContext,
    emit_event: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> PalBenchmarkAdaptation:
    _emit(
        emit_event,
        {
            "event": "sage_adapter_raw_result_received",
            "sage_adapter": True,
            "run_id": context.run_id,
            "raw_result_kind": type(raw_result).__name__,
            "has_solver_output": bool((solver_output or "").strip()),
        },
    )
    raw_artifact = classify_execution_artifact(raw_result)
    selected_artifact = raw_artifact
    if (
        raw_artifact.artifact_type == "unresolved"
        and raw_artifact.source != "raw_execution_failure"
    ):
        parsed_solver_artifact = _classify_solver_output(solver_output)
        if parsed_solver_artifact is not None and _solver_fallback_is_allowed(
            parsed_solver_artifact
        ):
            selected_artifact = parsed_solver_artifact
            _emit(
                emit_event,
                {
                    "event": "sage_adapter_solver_fallback_used",
                    "sage_adapter": True,
                    "run_id": context.run_id,
                    "fallback_source": parsed_solver_artifact.source,
                    "artifact_type": parsed_solver_artifact.artifact_type,
                },
            )
        elif parsed_solver_artifact is not None:
            _emit(
                emit_event,
                {
                    "event": "sage_adapter_solver_fallback_rejected",
                    "sage_adapter": True,
                    "run_id": context.run_id,
                    "fallback_source": parsed_solver_artifact.source,
                    "artifact_type": parsed_solver_artifact.artifact_type,
                },
            )
    _emit(
        emit_event,
        {
            "event": "sage_adapter_artifact_classified",
            "sage_adapter": True,
            "run_id": context.run_id,
            "artifact_type": selected_artifact.artifact_type,
            "source": selected_artifact.source,
            "is_structured": selected_artifact.is_structured,
            "diagnostics": dict(selected_artifact.diagnostics),
        },
    )
    materialization = materialize_benchmark_artifact(
        selected_artifact,
        context=context,
    )
    _emit(
        emit_event,
        {
            "event": "sage_adapter_materialization_selected",
            "sage_adapter": True,
            "run_id": context.run_id,
            "artifact_type": selected_artifact.artifact_type,
            "materialization_type": materialization.materialization_type,
            "determinism_level": materialization.determinism_level,
            "confidence": materialization.confidence,
        },
    )
    if materialization.needs_bridge and materialization.bridge_action:
        _emit(
            emit_event,
            {
                "event": "sage_adapter_bridge_emitted",
                "sage_adapter": True,
                "run_id": context.run_id,
                "bridge_tool_name": materialization.bridge_tool_name,
                "artifact_type": selected_artifact.artifact_type,
            },
        )
    return PalBenchmarkAdaptation(
        artifact=selected_artifact,
        materialization=materialization,
    )


def build_sage_benchmark_bridge_tool_code() -> str:
    return """from src.tasks.instance.knowledge_graph.api import Variable
import json


_TYPE_MAP = {
    "entity_id": "sage.entity_id",
    "entity_set": "sage.entity_set",
    "count_scalar": "sage.scalar",
    "boolean": "sage.boolean",
    "scalar_literal": "sage.scalar_literal",
    "text_literal": "sage.text_literal",
    "empty": "sage.empty",
}


def run(payload: dict) -> dict:
    variable_list = payload.get("variable_list")
    artifact_type = str(payload.get("sage_artifact_type") or "").strip()
    artifact_value = payload.get("sage_artifact_value")
    if not isinstance(variable_list, list) or not artifact_type:
        return {
            "status": "ERROR",
            "final_variable": None,
            "observation": "SAGE benchmark bridge received an invalid payload.",
        }
    if artifact_type == "unresolved":
        return {
            "status": "ERROR",
            "final_variable": None,
            "observation": "SAGE benchmark bridge refuses to materialize unresolved SAGE execution artifacts.",
        }

    program = json.dumps(
        {
            "kind": artifact_type,
            "value": artifact_value,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    variable = Variable(
        type=_TYPE_MAP.get(artifact_type, "sage.empty"),
        program=program,
    )
    variable_list.append(variable)
    final_pointer = f"#{len(variable_list) - 1}"
    semantic_description = str(
        payload.get("sage_semantic_description")
        or ("SAGE artifact materialized as " + artifact_type)
    ).strip()
    artifact_source = str(payload.get("sage_artifact_source") or "").strip() or None
    selected_query_variable = str(
        payload.get("sage_selected_query_variable") or ""
    ).strip() or None
    binding_count = payload.get("sage_binding_count")
    unique_value_count = payload.get("sage_unique_value_count")
    raw_value_preview = payload.get("sage_value_preview")
    value_preview = []
    if isinstance(raw_value_preview, list):
        for item in raw_value_preview:
            cleaned = str(item or "").strip()
            if cleaned:
                value_preview.append(cleaned)
    raw_row_preview = payload.get("sage_row_preview")
    row_preview = []
    if isinstance(raw_row_preview, list):
        for row in raw_row_preview:
            if not isinstance(row, dict):
                continue
            compact_row = {}
            for key, value in row.items():
                cleaned_key = str(key or "").strip()
                cleaned_value = str(value or "").strip()
                if cleaned_key and cleaned_value:
                    compact_row[cleaned_key] = cleaned_value
            if compact_row:
                row_preview.append(compact_row)
    tool_status = str(payload.get("sage_tool_status") or "success").strip().upper()
    if tool_status not in {"SUCCESS", "PARTIAL", "ERROR", "FAILED"}:
        tool_status = "SUCCESS"
    solves_task = bool(payload.get("sage_solves_task"))
    trusted_for_materialization = bool(payload.get("sage_trusted_for_materialization"))
    failure_reason = str(payload.get("sage_failure_reason") or "").strip() or None
    confidence = payload.get("sage_confidence")
    proof_hint = str(payload.get("sage_proof_hint") or "").strip() or None
    answer_cardinality_hint = (
        str(payload.get("sage_answer_cardinality_hint") or "").strip() or None
    )
    raw_relation_summary = payload.get("sage_relation_summary")
    relation_summary = []
    if isinstance(raw_relation_summary, list):
        for item in raw_relation_summary:
            cleaned = str(item or "").strip()
            if cleaned:
                relation_summary.append(cleaned)
    selection_basis = str(payload.get("sage_selection_basis") or "").strip() or None
    completeness_hint = (
        str(payload.get("sage_completeness_hint") or "").strip() or None
    )
    repair_caveat = str(payload.get("sage_repair_caveat") or "").strip() or None
    return {
        "status": "ERROR" if tool_status == "FAILED" else tool_status,
        "final_variable": final_pointer,
        "observation": (
            "SAGE benchmark bridge materialized "
            + artifact_type
            + " into a benchmark variable."
        ),
        "semantic_description": semantic_description,
        "solves_task": solves_task,
        "trusted_for_materialization": trusted_for_materialization,
        "intermediate_variables": [final_pointer],
        "failure_reason": failure_reason,
        "confidence": confidence,
        "artifact_type": artifact_type,
        "artifact_source": artifact_source,
        "selected_query_variable": selected_query_variable,
        "binding_count": binding_count,
        "unique_value_count": unique_value_count,
        "value_preview": value_preview[:5],
        "row_preview": row_preview[:3],
        "proof_hint": proof_hint,
        "answer_cardinality_hint": answer_cardinality_hint,
        "relation_summary": relation_summary[:4],
        "selection_basis": selection_basis,
        "completeness_hint": completeness_hint,
        "repair_caveat": repair_caveat,
    }
"""


def assert_bridge_tool_code_narrow(tool_code: str) -> None:
    code = str(tool_code or "")
    violations = [
        token
        for token in _BRIDGE_BANNED_TOKENS
        if token.lower() in code.lower()
    ]
    if violations:
        raise ValueError(
            "bridge_tool_contract_violation:" + ",".join(sorted(set(violations)))
        )


def evaluate_adapter_safety(
    *,
    artifact: PalExecutionArtifact,
    materialization: BenchmarkMaterialization,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if artifact.artifact_type in {"unresolved", "empty"}:
        reasons.append(f"adapter_rejected_artifact_type:{artifact.artifact_type}")
    if artifact.source != "raw_execution":
        reasons.append(f"adapter_non_raw_source:{artifact.source}")
    if materialization.materialization_type != "bridge_action":
        reasons.append(
            f"adapter_materialization_type:{materialization.materialization_type}"
        )
    if not materialization.needs_bridge:
        reasons.append("adapter_bridge_not_required")
    if materialization.final_answer_text:
        reasons.append("adapter_final_answer_text_forbidden")
    return (not reasons, tuple(reasons))


def _classify_solver_output(solver_output: Optional[str]) -> Optional[PalExecutionArtifact]:
    if not isinstance(solver_output, str):
        return None
    answer_text = solver_output.strip()
    if not answer_text:
        return None
    if answer_text.startswith("Final Answer:"):
        answer_text = answer_text[len("Final Answer:") :].strip()
    if not answer_text:
        return PalExecutionArtifact(
            raw_result=None,
            artifact_type="empty",
            value=[],
            is_structured=False,
            source="parsed_solver",
            diagnostics={"reason": "empty_solver_answer"},
        )
    if "<SEP>" in answer_text:
        values = [item.strip() for item in answer_text.split("<SEP>") if item.strip()]
        return _classify_literal_values(
            values,
            source="parsed_solver",
            raw_result=solver_output,
        )
    return _classify_literal_values(
        [answer_text],
        source="solver_fallback",
        raw_result=solver_output,
    )


def _classify_literal_values(
    values: Sequence[str],
    *,
    source: str,
    raw_result: Any,
) -> PalExecutionArtifact:
    normalized_values = []
    for value in values:
        normalized_value = _normalize_literal_value(value)
        normalized_values.append(
            {
                "kind": normalized_value["kind"],
                "value": normalized_value["value"],
            }
        )
    kinds = {item["kind"] for item in normalized_values}
    diagnostics = {
        "parsed_value_count": len(normalized_values),
        "parsed_kinds": sorted(kinds),
    }
    if kinds == {"entity_id"}:
        entity_values = [item["value"] for item in normalized_values]
        artifact_type = "entity_id" if len(entity_values) == 1 else "entity_set"
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type=artifact_type,
            value=entity_values[0] if artifact_type == "entity_id" else entity_values,
            is_structured=False,
            source=source,
            diagnostics=diagnostics,
        )
    if kinds == {"numeric"} and len(normalized_values) == 1:
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="count_scalar",
            value=normalized_values[0]["value"],
            is_structured=False,
            source=source,
            diagnostics=diagnostics,
        )
    if kinds == {"boolean"} and len(normalized_values) == 1:
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type="boolean",
            value=normalized_values[0]["value"],
            is_structured=False,
            source=source,
            diagnostics=diagnostics,
        )
    if kinds <= {"numeric", "date", "text_literal"}:
        literal_values = [item["value"] for item in normalized_values]
        artifact_type = "scalar_literal" if any(
            item["kind"] in {"numeric", "date"} for item in normalized_values
        ) else "text_literal"
        return PalExecutionArtifact(
            raw_result=raw_result,
            artifact_type=artifact_type,
            value=literal_values[0] if len(literal_values) == 1 else literal_values,
            is_structured=False,
            source=source,
            diagnostics=diagnostics,
        )
    return PalExecutionArtifact(
        raw_result=raw_result,
        artifact_type="unresolved",
        value=values[0] if len(values) == 1 else list(values),
        is_structured=False,
        source=source,
        diagnostics=diagnostics,
    )


def _select_answer_var(
    raw_result: Mapping[str, Any],
    bindings: Sequence[Any],
) -> tuple[Optional[str], dict[str, Any]]:
    head_vars = raw_result.get("head", {}).get("vars", [])
    diagnostics: dict[str, Any] = {
        "head_vars": list(head_vars) if isinstance(head_vars, list) else [],
    }
    if isinstance(head_vars, list) and head_vars:
        diagnostics["selected_var"] = str(head_vars[0])
        return str(head_vars[0]), diagnostics
    for binding in bindings:
        if isinstance(binding, Mapping) and binding:
            first_key = next(iter(binding.keys()))
            diagnostics["selected_var"] = str(first_key)
            return str(first_key), diagnostics
    diagnostics["selected_var"] = None
    return None, diagnostics


def _normalize_variable_hint(raw_value: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw_value or "").strip().lower()).strip("_")


def _looks_like_count_variable(variable_hint: str) -> bool:
    hint = _normalize_variable_hint(variable_hint)
    return bool(hint and _COUNT_VAR_HINT_PATTERN.search(hint))


def _extract_normalized_values(
    bindings: Sequence[Any],
    answer_var: Optional[str],
) -> list[dict[str, Any]]:
    normalized_values: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        selected_key = answer_var if answer_var in binding else next(iter(binding.keys()), None)
        if selected_key is None:
            continue
        cell = binding.get(selected_key)
        if not isinstance(cell, Mapping):
            continue
        normalized = _normalize_sparql_cell(cell)
        if normalized is None:
            continue
        dedupe_key = (normalized["kind"], str(normalized["value"]))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_values.append(normalized)
    return normalized_values


def _build_binding_row_preview(
    bindings: Sequence[Any],
    *,
    answer_var: Optional[str],
    head_vars: Any,
) -> list[dict[str, str]]:
    ordered_head_vars = [
        str(item).strip()
        for item in (head_vars or [])
        if str(item).strip()
    ]
    preview: list[dict[str, str]] = []
    seen_rows: set[tuple[tuple[str, str], ...]] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        ordered_keys: list[str] = []
        if answer_var and answer_var in binding:
            ordered_keys.append(answer_var)
        for key in ordered_head_vars:
            if key in binding and key not in ordered_keys:
                ordered_keys.append(key)
        for key in binding.keys():
            cleaned_key = str(key or "").strip()
            if cleaned_key and cleaned_key not in ordered_keys:
                ordered_keys.append(cleaned_key)
        row: dict[str, str] = {}
        for key in ordered_keys[:4]:
            cell = binding.get(key)
            if not isinstance(cell, Mapping):
                continue
            normalized = _normalize_sparql_cell(cell)
            if normalized is not None:
                cleaned_value = str(normalized["value"]).strip()
            else:
                cleaned_value = str(cell.get("value") or "").strip()
            cleaned_key = str(key or "").strip()
            if cleaned_key and cleaned_value:
                row[cleaned_key] = cleaned_value
        if not row:
            continue
        row_signature = tuple(row.items())
        if row_signature in seen_rows:
            continue
        seen_rows.add(row_signature)
        preview.append(row)
        if len(preview) >= 3:
            break
    return preview


def _normalize_sparql_cell(cell: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    raw_value = str(cell.get("value") or "").strip()
    if not raw_value:
        return None
    cell_type = str(cell.get("type") or "").strip().lower()
    if cell_type == "uri":
        entity_id = _normalize_entity_id(raw_value)
        if entity_id:
            return {"kind": "entity_id", "value": entity_id}
    return _normalize_literal_value(raw_value, datatype=str(cell.get("datatype") or ""))


def _normalize_literal_value(
    raw_value: str,
    *,
    datatype: str = "",
) -> dict[str, Any]:
    entity_id = _normalize_entity_id(raw_value)
    if entity_id:
        return {"kind": "entity_id", "value": entity_id}
    lowered_value = raw_value.lower()
    if lowered_value in {"true", "false"}:
        return {"kind": "boolean", "value": lowered_value == "true"}
    if _looks_numeric(raw_value, datatype):
        return {"kind": "numeric", "value": raw_value}
    if _DATE_PATTERN.fullmatch(raw_value):
        return {"kind": "date", "value": raw_value}
    return {"kind": "text_literal", "value": raw_value}


def _normalize_entity_id(raw_value: str) -> Optional[str]:
    value = raw_value.strip()
    if value.startswith(_FREEBASE_NS_PREFIX):
        value = value.replace(_FREEBASE_NS_PREFIX, "", 1)
    if _MID_PATTERN.fullmatch(value):
        return value
    return None
def _looks_numeric(raw_value: str, datatype: str) -> bool:
    datatype_lower = datatype.lower()
    if any(
        token in datatype_lower
        for token in ("integer", "float", "double", "decimal", "int")
    ):
        return True
    return _NUMERIC_PATTERN.fullmatch(raw_value) is not None


def _confidence_for_source(source: str) -> float:
    if source == "raw_execution":
        return 1.0
    if source == "raw_execution_failure":
        return 1.0
    if source == "parsed_solver":
        return 0.6
    if source == "solver_fallback":
        return 0.3
    return 0.1


def _determinism_for_source(source: str) -> str:
    if source == "raw_execution":
        return "deterministic_raw"
    if source == "raw_execution_failure":
        return "deterministic_failure"
    if source == "parsed_solver":
        return "parsed_solver"
    if source == "solver_fallback":
        return "solver_fallback"
    return "unknown"


def _solver_fallback_is_allowed(artifact: PalExecutionArtifact) -> bool:
    return False


def _describe_artifact_semantics(artifact: PalExecutionArtifact) -> str:
    artifact_type = str(artifact.artifact_type or "").strip()
    details: list[str] = []
    selected_var = str(artifact.diagnostics.get("selected_var") or "").strip()
    if selected_var:
        details.append(f"query variable '{selected_var}'")
    binding_count = artifact.diagnostics.get("binding_count")
    if isinstance(binding_count, int):
        details.append(f"{binding_count} raw bindings")
    unique_value_count = artifact.diagnostics.get("normalized_value_count")
    if not isinstance(unique_value_count, int):
        unique_value_count = artifact.diagnostics.get("parsed_value_count")
    if isinstance(unique_value_count, int):
        details.append(f"{unique_value_count} unique values")

    def _with_details(label: str) -> str:
        if not details:
            return label
        return f"{label} ({', '.join(details)})"

    if artifact_type == "count_scalar":
        return _with_details("count result returned by the SAGE query")
    if artifact_type == "entity_id":
        return _with_details("single entity id returned by the SAGE query")
    if artifact_type == "entity_set":
        return _with_details("bounded set of entity ids returned by the SAGE query")
    if artifact_type == "boolean":
        return _with_details("boolean result returned by the SAGE query")
    if artifact_type in {"scalar_literal", "text_literal"}:
        return _with_details("literal value returned by the SAGE query")
    if artifact_type == "empty":
        return "SAGE query returned an empty result"
    if artifact_type == "unresolved":
        failure_kind = str(artifact.diagnostics.get("failure_kind") or "").strip()
        if failure_kind:
            return f"SAGE query did not yield a trustworthy artifact ({failure_kind})"
        return "SAGE query did not yield a trustworthy artifact"
    return "SAGE query produced an execution artifact"


def _emit(
    emit_event: Optional[Callable[[Mapping[str, Any]], None]],
    payload: Mapping[str, Any],
) -> None:
    if emit_event is None:
        return
    try:
        emit_event(payload)
    except Exception:
        return


__all__ = [
    "BenchmarkAdapterContext",
    "BenchmarkMaterialization",
    "SAGE_BENCHMARK_BRIDGE_TOOL_NAME",
    "PalBenchmarkAdaptation",
    "PalExecutionArtifact",
    "adapt_sage_result_to_benchmark",
    "assert_bridge_tool_code_narrow",
    "build_sage_benchmark_bridge_tool_code",
    "classify_execution_artifact",
    "evaluate_adapter_safety",
    "materialize_benchmark_artifact",
]
