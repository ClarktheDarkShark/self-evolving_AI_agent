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
PAL_BENCHMARK_BRIDGE_TOOL_NAME = "pal_benchmark_bridge_macro"
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
    bridge_tool_name: str = PAL_BENCHMARK_BRIDGE_TOOL_NAME


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

    if str(raw_result.get("pal_execution_status") or "").strip().lower() == "error":
        failure_kind = str(raw_result.get("pal_failure_kind") or "execution_error")
        diagnostics = dict(raw_result.get("pal_diagnostics") or {})
        diagnostics["failure_kind"] = failure_kind
        diagnostics.setdefault(
            "error",
            str(raw_result.get("pal_error") or "pal_execution_failed"),
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
        )

    bridge_payload = {
        "pal_artifact_type": artifact.artifact_type,
        "pal_artifact_value": artifact.value,
        "pal_artifact_source": artifact.source,
        "pal_artifact_diagnostics": dict(artifact.diagnostics),
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
        diagnostics={
            "artifact_type": artifact.artifact_type,
            "artifact_source": artifact.source,
        },
        confidence=_confidence_for_source(artifact.source),
        determinism_level=_determinism_for_source(artifact.source),
    )


def adapt_pal_result_to_benchmark(
    *,
    raw_result: Any,
    solver_output: Optional[str],
    context: BenchmarkAdapterContext,
    emit_event: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> PalBenchmarkAdaptation:
    _emit(
        emit_event,
        {
            "event": "pal_adapter_raw_result_received",
            "pal_adapter": True,
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
                    "event": "pal_adapter_solver_fallback_used",
                    "pal_adapter": True,
                    "run_id": context.run_id,
                    "fallback_source": parsed_solver_artifact.source,
                    "artifact_type": parsed_solver_artifact.artifact_type,
                },
            )
        elif parsed_solver_artifact is not None:
            _emit(
                emit_event,
                {
                    "event": "pal_adapter_solver_fallback_rejected",
                    "pal_adapter": True,
                    "run_id": context.run_id,
                    "fallback_source": parsed_solver_artifact.source,
                    "artifact_type": parsed_solver_artifact.artifact_type,
                },
            )
    _emit(
        emit_event,
        {
            "event": "pal_adapter_artifact_classified",
            "pal_adapter": True,
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
            "event": "pal_adapter_materialization_selected",
            "pal_adapter": True,
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
                "event": "pal_adapter_bridge_emitted",
                "pal_adapter": True,
                "run_id": context.run_id,
                "bridge_tool_name": materialization.bridge_tool_name,
                "artifact_type": selected_artifact.artifact_type,
            },
        )
    return PalBenchmarkAdaptation(
        artifact=selected_artifact,
        materialization=materialization,
    )


def build_pal_benchmark_bridge_tool_code() -> str:
    return """from src.tasks.instance.knowledge_graph.api import Variable
import json


_TYPE_MAP = {
    "entity_id": "pal.entity_id",
    "entity_set": "pal.entity_set",
    "count_scalar": "pal.scalar",
    "boolean": "pal.boolean",
    "scalar_literal": "pal.scalar_literal",
    "text_literal": "pal.text_literal",
    "empty": "pal.empty",
}


def run(payload: dict) -> dict:
    variable_list = payload.get("variable_list")
    artifact_type = str(payload.get("pal_artifact_type") or "").strip()
    artifact_value = payload.get("pal_artifact_value")
    if not isinstance(variable_list, list) or not artifact_type:
        return {
            "status": "ERROR",
            "final_variable": None,
            "observation": "PAL benchmark bridge received an invalid payload.",
        }
    if artifact_type == "unresolved":
        return {
            "status": "ERROR",
            "final_variable": None,
            "observation": "PAL benchmark bridge refuses to materialize unresolved PAL execution artifacts.",
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
        type=_TYPE_MAP.get(artifact_type, "pal.empty"),
        program=program,
    )
    variable_list.append(variable)
    final_pointer = f"#{len(variable_list) - 1}"
    return {
        "status": "SUCCESS",
        "final_variable": final_pointer,
        "observation": (
            "PAL benchmark bridge materialized "
            + artifact_type
            + " into a benchmark variable."
        ),
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
    "PAL_BENCHMARK_BRIDGE_TOOL_NAME",
    "PalBenchmarkAdaptation",
    "PalExecutionArtifact",
    "adapt_pal_result_to_benchmark",
    "assert_bridge_tool_code_narrow",
    "build_pal_benchmark_bridge_tool_code",
    "classify_execution_artifact",
    "evaluate_adapter_safety",
    "materialize_benchmark_artifact",
]
