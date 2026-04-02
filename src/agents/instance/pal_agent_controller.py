from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from typing_extensions import override

from src.agents.agent import Agent
from src.agents.exceptions import AgentUnknownException
from src.language_models import LanguageModel
from src.pal.family_policy_evolution import (
    build_family_policy_store,
    classify_family_failure,
    family_policy_enabled_for,
)
from src.pal.invoker import (
    build_pal_execution_failure_payload,
    execute_pal_code_with_result,
)
from src.pal.kg_benchmark_adapter import (
    BenchmarkAdapterContext,
    BenchmarkMaterialization,
    PAL_BENCHMARK_BRIDGE_TOOL_NAME,
    adapt_pal_result_to_benchmark,
    assert_bridge_tool_code_narrow,
    build_pal_benchmark_bridge_tool_code,
    evaluate_adapter_safety,
)
from src.pal.policy_contracts import (
    AttemptDecisionRecord,
    TrustContractEvaluation,
    build_trust_contract_evaluation,
)
from src.pal.parser import extract_and_validate_code
from src.pal.plausibility_validator import (
    AnchorProbeResult,
    PlausibilityVerdict,
    build_repair_feedback,
    validate_pal_execution,
)
from src.pal.prompts import (
    GENERATOR_CODE_SYSTEM_PROMPT,
    GENERATOR_PLAN_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    SOLVER_SYSTEM_PROMPT,
)
from src.pal.reusable_tool_families import (
    get_baseline_reusable_family_policy_bundles,
    get_reusable_family_policy_bundle,
    render_reusable_tool,
    select_reusable_tool,
)
from src.self_evolving_agent.controller import SelfEvolvingController
from src.self_evolving_agent.tool_registry import get_registry
from src.typings import ChatHistory, ChatHistoryItem, Role
from src.utils.output_paths import prefix_filename

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeoutError

try:
    from SPARQLWrapper import SPARQLWrapper as _ProbeSPARQLWrapper
    from SPARQLWrapper import JSON as _ProbeSPARQLJSON
    _SPARQL_PROBE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ProbeSPARQLWrapper = None  # type: ignore[assignment,misc]
    _ProbeSPARQLJSON = None  # type: ignore[assignment]
    _SPARQL_PROBE_AVAILABLE = False


class PALAgentController(Agent):
    _PAL_QUERY_TOOL_BASE_NAME = "pal_sparql_query_tool"
    _PAL_QUERY_PLAN_MAX_ATTEMPTS = 2
    _PAL_QUERY_CODE_MAX_ATTEMPTS = 3
    # Max repair iterations *after* the initial candidate (total attempts = 1 + this)
    _PAL_REPAIR_MAX_ATTEMPTS = 3
    _PAL_REPAIR_STAGNATION_REPEAT_LIMIT = 2
    _SUPPORTED_QUERY_SHAPES = {
        "single_anchor_lookup",
        "single_anchor_chain_lookup",
        "multi_anchor_intersection",
        "shared_type_intersection",
        "count_over_direct_relation",
        "count_over_joined_set",
        "superlative_chain",
        "containment_or_ownership_lookup",
        "other",
    }
    _STABLE_RELATION_ROLES = {
        "anchor",
        "anchor_a",
        "anchor_b",
        "anchor_value",
        "answer",
        "shared_answer",
        "candidate_set",
        "count_set",
        "constraint_value",
        "ordering_attribute",
        "shared_type",
        "type_set",
    }
    _CANONICAL_ROLE_SYNONYMS = {
        "anchorentity": "anchor",
        "anchor_entity": "anchor",
        "anchorvalue": "anchor_value",
        "anchor_value": "anchor_value",
        "anchora": "anchor_a",
        "anchor_a": "anchor_a",
        "anchorb": "anchor_b",
        "anchor_b": "anchor_b",
        "answerentity": "answer",
        "answer_entity": "answer",
        "answerset": "answer",
        "answer_set": "answer",
        "attributevalue": "constraint_value",
        "attribute_value": "constraint_value",
        "candidate": "candidate_set",
        "candidateanswer": "candidate_set",
        "candidate_answer": "candidate_set",
        "candidateanswerset": "candidate_set",
        "candidate_answer_set": "candidate_set",
        "candidateset": "candidate_set",
        "candidate_set": "candidate_set",
        "constraintvalue": "constraint_value",
        "constraint_value": "constraint_value",
        "countedset": "count_set",
        "counted_set": "count_set",
        "countset": "count_set",
        "count_set": "count_set",
        "order_attribute": "ordering_attribute",
        "orderattribute": "ordering_attribute",
        "orderingattribute": "ordering_attribute",
        "ordering_attribute": "ordering_attribute",
        "orderingvalue": "ordering_attribute",
        "ordering_value": "ordering_attribute",
        "result": "answer",
        "resultset": "answer",
        "result_set": "answer",
        "sharedanswer": "shared_answer",
        "shared_answer": "shared_answer",
        "sharedtype": "shared_type",
        "shared_type": "shared_type",
        "sortattribute": "ordering_attribute",
        "sort_attribute": "ordering_attribute",
        "typeintersection": "shared_type",
        "type_intersection": "shared_type",
        "type": "type_set",
        "typeentity": "type_set",
        "typeset": "type_set",
        "type_set": "type_set",
        "class": "type_set",
        "category": "type_set",
        "kind": "type_set",
        "value": "constraint_value",
    }
    _VALUE_ROLE_HINTS = (
        "attribute",
        "brand",
        "category",
        "class",
        "country",
        "date",
        "disease",
        "duration",
        "genre",
        "ingredient",
        "kingdom",
        "length",
        "manufacturer",
        "moiety",
        "parent",
        "profession",
        "source",
        "symptom",
        "temperament",
        "texture",
        "type",
        "value",
    )
    _ORDERING_ROLE_HINTS = (
        "age",
        "date",
        "duration",
        "first",
        "introduced",
        "introduction",
        "latest",
        "length",
        "longest",
        "newest",
        "oldest",
        "time",
        "year",
    )

    def __init__(
        self,
        language_model: LanguageModel,
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._language_model = language_model
        self._inference_config_dict = dict(inference_config_dict or {})
        self._kwargs = kwargs
        self._pending_macro_runs: dict[str, dict[str, Any]] = {}
        self._registered_bridge_tools: set[str] = set()
        self._tool_invoked_in_last_inference = None
        self._manual_fallback_agent: Optional[SelfEvolvingController] = None
        self._manual_fallback_active_runs: set[str] = set()

    @override
    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        self._tool_invoked_in_last_inference = None
        pipeline_stage = "start"
        generated_tool_name: Optional[str] = None
        try:
            last_user_content = chat_history.get_item_deep_copy(-1).content
            if self._manual_fallback_active_for_current_run():
                return self._delegate_to_manual_solver(chat_history=chat_history)
            if macro_pointer := self._extract_macro_pointer(last_user_content):
                self._log_macro_result(last_user_content)
                return ChatHistoryItem(
                    role=Role.AGENT,
                    content=f"Final Answer: {macro_pointer}",
                )
            if "Macro result:" in (last_user_content or ""):
                self._log_macro_result(last_user_content)
                self._activate_manual_fallback(chat_history=chat_history)
                return self._delegate_to_manual_solver(chat_history=chat_history)

            task_question = last_user_content
            generated_tool_name = self._build_query_tool_name(task_question)
            question_text, question_entities = self._split_task_question(task_question)
            question_target_phrase = self._extract_answer_target_phrase(question_text)
            question_interpretation = self._build_question_interpretation(
                question_text=question_text,
                explicit_entities=question_entities,
                answer_target_phrase=question_target_phrase,
            )
            interpreted_grounding_entities = (
                self._extract_grounding_entities_from_question_interpretation(
                    question_interpretation
                )
            )
            has_structured_non_entity_inputs = any(
                isinstance(item, Mapping)
                and str(item.get("kind") or "").strip()
                in {"class_phrase", "type_constraint", "shared_attribute"}
                for item in (question_interpretation.get("question_inputs") or [])
            )
            grounding_entities = interpreted_grounding_entities or (
                []
                if has_structured_non_entity_inputs
                else list(question_entities)
            )
            domain_hints = self._infer_domain_hints(question_text)
            relation_grounding = self._build_grounded_relation_candidates_with_dynamic_fallback(
                task_question=task_question,
                entities=grounding_entities,
                answer_target_phrase=question_target_phrase,
                domain_hints=domain_hints,
                question_interpretation=question_interpretation,
            )
            question_interpretation = self._refine_question_interpretation_with_grounding(
                question_text=question_text,
                answer_target_phrase=question_target_phrase,
                question_interpretation=question_interpretation,
                grounded_relation_candidates=relation_grounding,
            )
            interpreted_grounding_entities = (
                self._extract_grounding_entities_from_question_interpretation(
                    question_interpretation
                )
            )
            has_structured_non_entity_inputs = any(
                isinstance(item, Mapping)
                and str(item.get("kind") or "").strip()
                in {"class_phrase", "type_constraint", "shared_attribute"}
                for item in (question_interpretation.get("question_inputs") or [])
            )
            grounding_entities = interpreted_grounding_entities or (
                []
                if has_structured_non_entity_inputs
                else list(question_entities)
            )
            grounding_card = self._build_pal_grounding_card(
                task_question,
                relation_grounding=relation_grounding,
                question_interpretation=question_interpretation,
            )
            self._emit_generated_tools_event(
                {
                    "event": "pal_question_interpretation",
                    "mode": "pal",
                    "question_inputs": question_interpretation.get("question_inputs")
                    or [],
                    "preferred_scaffolds": question_interpretation.get(
                        "preferred_scaffolds"
                    )
                    or [],
                    "grounding_entities": grounding_entities,
                }
            )
            pipeline_stage = "orchestrator"
            self._emit_generated_tools_event(
                {
                    "event": "toolgen_attempt",
                    "mode": "pal",
                    "requested_mode": "pal",
                    "max_rounds": 1,
                    "prompt_chars": len(task_question),
                    "system_prompt_name": "ORCHESTRATOR_SYSTEM_PROMPT",
                    "stage": "orchestrator",
                }
            )
            orchestrator_raw = self._run_text_prompt(
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                user_prompt=task_question,
            )
            action = self._parse_orchestrator_action(orchestrator_raw)
            self._emit_generated_tools_event(
                {
                    "event": "toolgen_exec_payload_semantics",
                    "target_concept_source": "pal_question",
                    "domain_hints_source": "pal_question",
                    "attribute_target_concept_source": "pal_question",
                    "has_tool_plan": False,
                    "target_concept": None,
                    "execution_style": "sparql_query",
                    "preferred_tool_mode": "full_solve",
                    "fallback_strategies": ["adapter_bridge"],
                    "has_entity_target_concepts": bool(grounding_entities),
                    "entity_target_concepts_count": len(grounding_entities),
                    "has_domain_hints": bool(domain_hints or question_target_phrase),
                    "entity_target_concepts": grounding_entities,
                    "domain_hints": domain_hints,
                    "target_concept": question_target_phrase,
                    "action": action,
                }
            )

            if action != "generate_tool":
                self._emit_generated_tools_event(
                    {
                        "event": "pal_orchestrator_action_overridden",
                        "mode": "pal",
                        "original_action": action,
                        "forced_action": "generate_tool",
                    }
                )
                action = "generate_tool"

            if action == "generate_tool":
                pipeline_stage = "plan"
                query_plan = self._generate_pal_query_plan(
                    task_question=task_question,
                    grounding_card=grounding_card,
                    relation_grounding=relation_grounding,
                    generated_tool_name=generated_tool_name,
                )
                pipeline_stage = "generator"
                # Bounded repair loop: generate → execute → plausibility-check → repair
                generated_code, invocation_result, _repair_loop_log = (
                    self._run_pal_repair_loop(
                        task_question=task_question,
                        grounding_card=grounding_card,
                        query_plan=query_plan,
                        generated_tool_name=generated_tool_name,
                        question_entities=grounding_entities,
                        relation_grounding=relation_grounding,
                    )
                )
                self._ensure_repair_loop_accepted(
                    generated_tool_name=generated_tool_name,
                    query_plan=query_plan,
                    repair_loop_log=_repair_loop_log,
                )
                pipeline_stage = "parser"
                query_artifacts = self._capture_pal_query_artifacts(
                    generated_tool_name=generated_tool_name,
                    generated_code=generated_code,
                    extracted_query_text=self._extract_sparql_query_text(generated_code),
                    query_plan=query_plan,
                    query_validation_errors=[],
                )
                self._emit_generated_tools_event(
                    {
                        "event": "toolgen_static_check",
                        "mode": "pal",
                        "round": _repair_loop_log.get("total_attempts", 1),
                        "tool_name": generated_tool_name,
                        "ok": True,
                        "code_len": len(generated_code),
                        "repair_attempts": _repair_loop_log.get("total_attempts", 1) - 1,
                        "repair_used": _repair_loop_log.get("repair_used", False),
                        "final_verdict": _repair_loop_log.get("final_verdict"),
                    }
                )
                pipeline_stage = "invoker"
                query_artifacts = self._refresh_query_artifacts_from_invocation(
                    generated_tool_name=generated_tool_name,
                    query_artifacts=query_artifacts,
                    invocation_result=invocation_result,
                )
                projected_variables = self._resolve_projected_variables(
                    query_text=query_artifacts.get("query_text", ""),
                    invocation_result=invocation_result,
                )
                if not invocation_result.success or invocation_result.payload is None:
                    failure_kind = (
                        invocation_result.failure_kind or "execution_error"
                    )
                    failure_endpoint_url = self._resolve_invocation_endpoint_url(
                        invocation_result
                    )
                    failure_payload = build_pal_execution_failure_payload(
                        invocation_result,
                        endpoint_url=failure_endpoint_url,
                    )
                    self._emit_generated_tools_event(
                        {
                            "event": "toolgen_smoke_test",
                            "mode": "pal",
                            "round": 1,
                            "tool_name": generated_tool_name,
                            "ok": False,
                            "endpoint_url": failure_endpoint_url,
                            "projected_variables": projected_variables,
                            "failure_kind": failure_kind,
                            "error": invocation_result.error,
                            "diagnostics": dict(invocation_result.diagnostics),
                        }
                    )
                    self._emit_generated_tools_event(
                        self._build_pal_query_execution_event(
                            generated_tool_name=generated_tool_name,
                            query_artifacts=query_artifacts,
                            invocation_result=invocation_result,
                            result_payload=None,
                            projected_variables=projected_variables,
                            result_kind="failed",
                        )
                    )
                    self._emit_pal_invoker_failure_event(
                        generated_tool_name=generated_tool_name,
                        invocation_result=invocation_result,
                    )
                    pipeline_stage = "adapter"
                    materialization = self._adapt_pal_result(
                        task_question=task_question,
                        raw_result=failure_payload,
                        solver_output="",
                    )
                    failure_verdict = PlausibilityVerdict(
                        verdict=str(failure_kind or "execution_failure"),
                        reasons=[
                            str(invocation_result.error or failure_kind or "execution_failure")
                        ],
                    )
                    trust_contract = self._evaluate_materialization_trust_contract(
                        generated_code=generated_code,
                        query_plan=query_plan,
                        query_text=query_artifacts.get("query_text", ""),
                        result_dict=None,
                        verdict=failure_verdict,
                        materialization=materialization,
                        artifact_type=str(
                            (materialization.diagnostics or {}).get("artifact_type") or ""
                        ).strip(),
                        artifact_source=str(
                            (materialization.diagnostics or {}).get("artifact_source") or ""
                        ).strip(),
                    )
                    self._emit_final_attempt_decision(
                        generated_tool_name=generated_tool_name,
                        loop_log=_repair_loop_log,
                        trust_contract=trust_contract,
                        tool_result_semantics=self._build_tool_result_semantics(
                            materialization=materialization,
                            trust_contract=trust_contract,
                        ),
                    )
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_controller_execution_fallback_used",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "failure_kind": failure_kind,
                            "artifact_type": materialization.diagnostics.get(
                                "artifact_type"
                            ),
                            "artifact_source": materialization.diagnostics.get(
                                "artifact_source"
                            ),
                        }
                    )
                    if not trust_contract.materialization_allowed:
                        self._maybe_record_family_policy_candidate(
                            generated_tool_name=generated_tool_name,
                            query_plan=query_plan,
                            failure_reason=(
                                "pal_materialization_contract_failed:"
                                + ",".join(trust_contract.denial_reasons)
                            ),
                            repair_loop_log=_repair_loop_log,
                            trust_contract=trust_contract,
                        )
                        self._activate_manual_fallback(
                            chat_history=chat_history,
                            advisory_text=self._build_tool_advisory_text(
                                materialization=materialization,
                                trust_contract=trust_contract,
                            ),
                        )
                        return self._delegate_to_manual_solver(chat_history=chat_history)
                    tool_result_semantics = self._build_tool_result_semantics(
                        materialization=materialization,
                        trust_contract=trust_contract,
                    )
                    return ChatHistoryItem(
                        role=Role.AGENT,
                        content=self._materialize_adapter_response(
                            task_question=task_question,
                            materialization=materialization,
                            generated_tool_name=generated_tool_name,
                            tool_result_semantics=tool_result_semantics,
                        ),
                    )
                result_dict = invocation_result.payload
                self._emit_generated_tools_event(
                        {
                            "event": "toolgen_smoke_test",
                            "mode": "pal",
                        "round": 1,
                            "tool_name": generated_tool_name,
                            "ok": True,
                            "endpoint_url": self._get_runtime_sparql_endpoint(),
                            "projected_variables": projected_variables,
                            "binding_count": self._get_result_binding_count(result_dict),
                            "result_summary": self._summarize_text(
                                json.dumps(result_dict, ensure_ascii=False, default=str),
                                max_len=240,
                            ),
                        }
                    )
                self._emit_generated_tools_event(
                    self._build_pal_query_execution_event(
                        generated_tool_name=generated_tool_name,
                        query_artifacts=query_artifacts,
                        invocation_result=invocation_result,
                        result_payload=result_dict,
                        projected_variables=projected_variables,
                        result_kind=self._classify_pal_result_kind(result_dict),
                    )
                )
                pipeline_stage = "solver"
                solver_prompt = SOLVER_SYSTEM_PROMPT.format(
                    raw_sparql_json=json.dumps(result_dict, ensure_ascii=False),
                    original_question=task_question,
                )
                solver_output = self._normalize_final_answer_output(
                    self._run_text_prompt(
                        system_prompt=solver_prompt,
                        user_prompt=task_question,
                    )
                )
                self._emit_generated_tools_event(
                    {
                        "event": "toolgen_validation_result",
                        "phase": "solver",
                        "mode": "pal",
                        "round": 1,
                        "tool_name": generated_tool_name,
                        "binding_count": self._get_result_binding_count(result_dict),
                        "answer_line": self._summarize_text(solver_output, max_len=180),
                    }
                )
                pipeline_stage = "adapter"
                materialization = self._adapt_pal_result(
                    task_question=task_question,
                    raw_result=result_dict,
                    solver_output=solver_output,
                )
                final_verdict = PlausibilityVerdict(
                    verdict=str(
                        _repair_loop_log.get("final_verdict")
                        or _repair_loop_log.get("last_verdict")
                        or ""
                    ).strip(),
                    reasons=list(_repair_loop_log.get("last_reasons") or []),
                )
                trust_contract = self._evaluate_materialization_trust_contract(
                    generated_code=generated_code,
                    query_plan=query_plan,
                    query_text=query_artifacts.get("query_text", ""),
                    result_dict=result_dict,
                    verdict=final_verdict,
                    materialization=materialization,
                    artifact_type=str(
                        (materialization.diagnostics or {}).get("artifact_type") or ""
                    ).strip(),
                    artifact_source=str(
                        (materialization.diagnostics or {}).get("artifact_source") or ""
                    ).strip(),
                )
                self._emit_final_attempt_decision(
                    generated_tool_name=generated_tool_name,
                    loop_log=_repair_loop_log,
                    trust_contract=trust_contract,
                    tool_result_semantics=self._build_tool_result_semantics(
                        materialization=materialization,
                        trust_contract=trust_contract,
                    ),
                )
                if not trust_contract.materialization_allowed:
                    self._maybe_record_family_policy_candidate(
                        generated_tool_name=generated_tool_name,
                        query_plan=query_plan,
                        failure_reason=(
                            "pal_materialization_contract_failed:"
                            + ",".join(trust_contract.denial_reasons)
                        ),
                        repair_loop_log=_repair_loop_log,
                        trust_contract=trust_contract,
                    )
                    self._activate_manual_fallback(
                        chat_history=chat_history,
                        advisory_text=self._build_tool_advisory_text(
                            materialization=materialization,
                            trust_contract=trust_contract,
                        ),
                    )
                    return self._delegate_to_manual_solver(chat_history=chat_history)
                tool_result_semantics = self._build_tool_result_semantics(
                    materialization=materialization,
                    trust_contract=trust_contract,
                )
                return ChatHistoryItem(
                    role=Role.AGENT,
                    content=self._materialize_adapter_response(
                        task_question=task_question,
                        materialization=materialization,
                        generated_tool_name=generated_tool_name,
                        tool_result_semantics=tool_result_semantics,
                    ),
                )

            raise AgentUnknownException("pal_orchestrator_refused_generate_tool")
        except Exception as e:
            self._emit_generated_tools_event(
                {
                    "event": "pal_adapter_failed" if pipeline_stage == "adapter" else "tool_generation_failed",
                    "phase": pipeline_stage,
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "error": str(e),
                }
            )
            if pipeline_stage == "parser":
                self._emit_generated_tools_event(
                    {
                        "event": "toolgen_static_check",
                        "mode": "pal",
                        "round": 1,
                        "tool_name": generated_tool_name,
                        "ok": False,
                        "error": str(e),
                    }
                )
            elif pipeline_stage == "invoker":
                self._emit_generated_tools_event(
                    {
                        "event": "toolgen_smoke_test",
                        "mode": "pal",
                        "round": 1,
                        "tool_name": generated_tool_name,
                        "ok": False,
                        "failure_kind": "execution_error",
                        "error": str(e),
                    }
                )
            raise AgentUnknownException(str(e)) from e

    def _adapt_pal_result(
        self,
        *,
        task_question: str,
        raw_result: Any,
        solver_output: str,
    ) -> BenchmarkMaterialization:
        adaptation = adapt_pal_result_to_benchmark(
            raw_result=raw_result,
            solver_output=solver_output,
            context=self._build_adapter_context(task_question),
            emit_event=self._emit_generated_tools_event,
        )
        return adaptation.materialization

    def _materialize_adapter_response(
        self,
        *,
        task_question: str,
        materialization: BenchmarkMaterialization,
        generated_tool_name: Optional[str] = None,
        tool_result_semantics: Optional[Mapping[str, Any]] = None,
    ) -> str:
        artifact_type = str(
            (materialization.diagnostics or {}).get("artifact_type") or ""
        ).strip()
        if materialization.needs_bridge:
            tool_name = self._ensure_bridge_tool(
                materialization.bridge_tool_name or PAL_BENCHMARK_BRIDGE_TOOL_NAME
            )
            payload = dict(materialization.bridge_payload or {})
            payload.setdefault("run_id", self._get_run_id())
            payload.setdefault("state_dir", self._get_macro_state_dir())
            payload.setdefault(
                "pal_semantic_description",
                str(
                    (tool_result_semantics or {}).get("tool_result_semantic_description")
                    or materialization.semantic_description
                    or ""
                ).strip(),
            )
            payload.setdefault(
                "pal_solves_task",
                bool(
                    (tool_result_semantics or {}).get("tool_result_solves_task")
                    or materialization.solves_task
                ),
            )
            payload.setdefault(
                "pal_trusted_for_materialization",
                bool(
                    (tool_result_semantics or {}).get(
                        "tool_result_trusted_for_materialization"
                    )
                ),
            )
            payload.setdefault(
                "pal_tool_status",
                str(
                    (tool_result_semantics or {}).get("tool_result_status")
                    or (
                        "success"
                        if (
                            materialization.solves_task
                            and materialization.trusted_for_materialization
                        )
                        else (
                            "partial" if materialization.useful_intermediate else "failed"
                        )
                    )
                    or materialization.tool_status
                ).strip(),
            )
            payload.setdefault(
                "pal_failure_reason",
                str(
                    (tool_result_semantics or {}).get("tool_result_failure_reason")
                    or materialization.failure_reason
                    or ""
                ).strip(),
            )
            payload.setdefault("pal_confidence", materialization.confidence)
            run_id = str(payload.get("run_id") or self._get_run_id())
            self._pending_macro_runs[run_id] = {
                "tool_name": tool_name,
                "query_tool_name": generated_tool_name,
                "run_id": run_id,
                "state_dir": payload.get("state_dir"),
                "asked_for": task_question,
                "artifact_type": payload.get("pal_artifact_type"),
            }
            if generated_tool_name:
                self._tool_invoked_in_last_inference = (
                    f"{generated_tool_name} -> {tool_name}"
                )
            else:
                self._tool_invoked_in_last_inference = tool_name
            self._emit_generated_tools_event(
                {
                    "event": "pal_macro_invoke_requested",
                    "tool_name": tool_name,
                    "run_id": run_id,
                    "state_dir": payload.get("state_dir"),
                    "artifact_type": payload.get("pal_artifact_type"),
                    "artifact_source": payload.get("pal_artifact_source"),
                    "determinism_level": materialization.determinism_level,
                    "confidence": materialization.confidence,
                    "asked_for": self._summarize_text(task_question, max_len=160),
                }
            )
            return (
                f"Action: execute_macro({json.dumps(tool_name)}, "
                f"{json.dumps(payload, ensure_ascii=False)})"
            )
        if materialization.final_answer_text:
            return materialization.final_answer_text
        if artifact_type == "unresolved":
            raise AgentUnknownException("pal_adapter_unresolved_artifact")
        raise AgentUnknownException("pal_adapter_empty_materialization")

    def _build_attempt_decision_record(
        self,
        *,
        query_plan: Mapping[str, Any],
        generation_source: str,
        reusable_selection: Any = None,
        verdict: PlausibilityVerdict | None = None,
        trust_contract: TrustContractEvaluation | None = None,
    ) -> AttemptDecisionRecord:
        answer_mode = str(query_plan.get("answer_mode") or "").strip().lower()
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()
        selected_family = (
            str(getattr(reusable_selection, "family_name", "") or "").strip() or None
        )
        renderer_name = (
            str(getattr(reusable_selection, "renderer_name", "") or "").strip() or None
        )
        family_bundle = (
            getattr(reusable_selection, "policy_bundle", None)
            or get_reusable_family_policy_bundle(selected_family or "")
        )
        selection_reasons = tuple(
            str(reason or "").strip()
            for reason in (getattr(reusable_selection, "reasons", ()) or ())
            if str(reason or "").strip()
        )
        stay_in_family_evidence: list[str] = []
        switch_family_evidence: list[str] = []
        if selected_family:
            stay_in_family_evidence.append(f"query_shape_match:{query_shape}")
            if answer_mode:
                stay_in_family_evidence.append(f"answer_mode_match:{answer_mode}")
            if family_bundle is not None:
                stay_in_family_evidence.extend(
                    f"bundle_expectation:{item}"
                    for item in family_bundle.validator_expectations[:2]
                )
        else:
            switch_family_evidence.append("no_reusable_family_selected")
        if verdict is not None:
            if verdict.is_accepted:
                stay_in_family_evidence.append(f"validator_accepts_family:{verdict.verdict}")
            else:
                switch_family_evidence.append(f"validator_verdict:{verdict.verdict}")
            for reason in verdict.reasons:
                cleaned_reason = str(reason or "").strip()
                if not cleaned_reason:
                    continue
                if cleaned_reason.startswith("repair_hint:"):
                    continue
                if cleaned_reason.startswith("dangerous_overreach:"):
                    switch_family_evidence.append(cleaned_reason)
                elif cleaned_reason.startswith("anchor_not_found"):
                    switch_family_evidence.append(cleaned_reason)
                elif cleaned_reason.startswith("anchor_path_empty"):
                    switch_family_evidence.append(cleaned_reason)
                elif cleaned_reason.startswith("count_answer_target_unenforced"):
                    switch_family_evidence.append(cleaned_reason)
                elif cleaned_reason.startswith("entity_result_hits_limit_ceiling"):
                    switch_family_evidence.append(cleaned_reason)
        if trust_contract is not None and trust_contract.materialization_allowed:
            stay_in_family_evidence.append("trust_contract:materialization_allowed")
        dangerous_overreach_reasons = list(
            trust_contract.dangerous_overreach_reasons if trust_contract is not None else ()
        )
        if verdict is not None:
            dangerous_overreach_reasons.extend(
                self._collect_dangerous_overreach_reasons(verdict=verdict)
            )
        dangerous_overreach_reasons = list(dict.fromkeys(dangerous_overreach_reasons))
        trust_contract_payload = (
            trust_contract.as_dict() if trust_contract is not None else {}
        )
        return AttemptDecisionRecord(
            selected_family=selected_family,
            family_bundle_version=(
                family_bundle.version if family_bundle is not None else None
            ),
            renderer_name=renderer_name,
            generation_source=generation_source,
            selection_reasons=selection_reasons,
            stay_in_family_evidence=tuple(dict.fromkeys(stay_in_family_evidence)),
            switch_family_evidence=tuple(dict.fromkeys(switch_family_evidence)),
            trust_contract=trust_contract_payload,
            materialization_allowed=(
                trust_contract.materialization_allowed if trust_contract is not None else None
            ),
            materialization_denial_reasons=(
                trust_contract.denial_reasons if trust_contract is not None else ()
            ),
            dangerous_overreach=bool(dangerous_overreach_reasons),
            dangerous_overreach_reasons=tuple(dangerous_overreach_reasons),
        )

    def _execution_shape_matches_query_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        result_dict: Mapping[str, Any] | None,
        artifact_type: str,
    ) -> tuple[bool, tuple[str, ...]]:
        answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()
        reasons: list[str] = []
        if answer_mode == "count":
            if artifact_type != "count_scalar":
                reasons.append(f"expected_count_scalar:{artifact_type}")
        elif answer_mode == "entity":
            if artifact_type not in {"entity_id", "entity_set"}:
                reasons.append(f"expected_entity_artifact:{artifact_type}")
        elif answer_mode == "boolean":
            if artifact_type != "boolean":
                reasons.append(f"expected_boolean_artifact:{artifact_type}")
        elif answer_mode == "literal":
            if artifact_type not in {"scalar_literal", "text_literal"}:
                reasons.append(f"expected_literal_artifact:{artifact_type}")
        if result_dict is None and artifact_type not in {"unresolved", "empty"}:
            reasons.append("result_dict_missing_for_structured_artifact")
        return (not reasons, tuple(reasons))

    def _collect_dangerous_overreach_reasons(
        self,
        *,
        verdict: PlausibilityVerdict,
    ) -> tuple[str, ...]:
        reasons = [
            str(reason or "").strip()
            for reason in verdict.reasons
            if str(reason or "").strip().startswith("dangerous_overreach:")
        ]
        if str(verdict.verdict or "").strip() == "rejected_dangerous_overreach":
            reasons.insert(0, "dangerous_overreach:validator_rejected")
        return tuple(dict.fromkeys(reasons))

    def _get_family_policy_store(self):
        return build_family_policy_store(
            baseline_bundles=get_baseline_reusable_family_policy_bundles()
        )

    def _load_family_success_archetypes(
        self,
        *,
        family_name: str,
    ) -> list[dict[str, Any]]:
        cleaned_family = str(family_name or "").strip()
        if not cleaned_family:
            return []
        try:
            store = self._get_family_policy_store()
            metadata = store.get_trusted_success_bank_metadata(cleaned_family)
        except Exception:
            return []
        evaluation_context = metadata.get("evaluation_context") or {}
        if not isinstance(evaluation_context, Mapping):
            return []
        active_bundle = get_reusable_family_policy_bundle(cleaned_family)
        source_version = str(metadata.get("source_version") or "").strip()
        active_version = str(getattr(active_bundle, "version", "") or "").strip()
        if source_version and active_version and source_version != active_version:
            return []
        return [
            dict(item)
            for item in (evaluation_context.get("success_plan_archetypes") or [])
            if isinstance(item, Mapping)
        ]

    def _format_family_success_archetypes(
        self,
        *,
        family_name: str,
    ) -> list[str]:
        archetypes = self._load_family_success_archetypes(family_name=family_name)
        formatted: list[str] = []
        for archetype in archetypes[:2]:
            anchor_roles = ",".join(
                str(item).strip()
                for item in (archetype.get("anchor_roles") or [])
                if str(item).strip()
            )
            anchor_constraints = " | ".join(
                str(item).strip()
                for item in (archetype.get("anchor_constraints") or [])
                if str(item).strip()
            )
            relation_roles = " | ".join(
                str(item).strip()
                for item in (archetype.get("relation_role_skeleton") or [])
                if str(item).strip()
            )
            structural_notes = ",".join(
                str(item).strip()
                for item in (archetype.get("structural_notes") or [])
                if str(item).strip()
            )
            summary_bits = [
                f"anchor_roles={anchor_roles or 'none'}",
                f"anchor_constraints={anchor_constraints or 'none'}",
                f"relation_roles={relation_roles or 'none'}",
            ]
            if structural_notes:
                summary_bits.append(f"structural_notes={structural_notes}")
            formatted.append("; ".join(summary_bits))
        return formatted

    def _collect_family_policy_failure_reasons(
        self,
        *,
        failure_reason: str,
        repair_loop_log: Mapping[str, Any] | None = None,
        trust_contract: TrustContractEvaluation | None = None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        cleaned_failure = str(failure_reason or "").strip()
        if cleaned_failure:
            reasons.append(cleaned_failure)
        if repair_loop_log is not None:
            reasons.extend(
                str(reason or "").strip()
                for reason in (repair_loop_log.get("last_reasons") or [])
                if str(reason or "").strip()
            )
            best_candidate = repair_loop_log.get("best_executing_candidate") or {}
            reasons.extend(
                str(reason or "").strip()
                for reason in (best_candidate.get("verdict_reasons") or [])
                if str(reason or "").strip()
            )
        if trust_contract is not None:
            reasons.extend(
                str(reason or "").strip()
                for reason in trust_contract.denial_reasons
                if str(reason or "").strip()
            )
            reasons.extend(
                str(reason or "").strip()
                for reason in trust_contract.dangerous_overreach_reasons
                if str(reason or "").strip()
            )
        return tuple(dict.fromkeys(reason for reason in reasons if reason))

    def _maybe_record_family_policy_candidate(
        self,
        *,
        generated_tool_name: str,
        query_plan: Mapping[str, Any],
        failure_reason: str,
        repair_loop_log: Mapping[str, Any] | None = None,
        trust_contract: TrustContractEvaluation | None = None,
    ) -> None:
        attempt_decisions = list((repair_loop_log or {}).get("attempt_decisions") or [])
        final_attempt = dict(attempt_decisions[-1] if attempt_decisions else {})
        trigger_query_plan = self._resolve_family_policy_trigger_query_plan(
            query_plan=query_plan,
            repair_loop_log=repair_loop_log,
        )
        trigger_query_shape = str(
            trigger_query_plan.get("query_shape") or ""
        ).strip()
        selected_family = str(
            final_attempt.get("selected_family")
            or trigger_query_shape
            or ""
        ).strip()
        if (
            trigger_query_shape
            and trigger_query_shape != selected_family
            and family_policy_enabled_for(trigger_query_shape)
        ):
            selected_family = trigger_query_shape
        if not selected_family or not family_policy_enabled_for(selected_family):
            return
        relation_names = self._relation_names_from_plan(trigger_query_plan)
        scaffold_signature = self._build_scaffold_signature(trigger_query_plan)
        failure_reasons = self._collect_family_policy_failure_reasons(
            failure_reason=failure_reason,
            repair_loop_log=repair_loop_log,
            trust_contract=trust_contract,
        )
        failure_class = classify_family_failure(
            family_name=selected_family,
            sample_status="agent_unknown_error",
            evaluation_outcome="incorrect",
            relation_names=relation_names,
            failure_reasons=failure_reasons,
            dangerous_overreach=bool(
                getattr(trust_contract, "dangerous_overreach", False)
            ),
        )
        if not scaffold_signature and not failure_reasons:
            return
        try:
            store = self._get_family_policy_store()
            candidate = store.create_candidate_update(
                family_name=selected_family,
                scaffold_signature=scaffold_signature,
                relation_names=relation_names,
                failure_reasons=failure_reasons,
                failure_class=failure_class,
                trigger_context={
                    "task_name": getattr(
                        getattr(self, "_current_session", None), "task_name", None
                    ),
                    "sample_index": getattr(
                        getattr(self, "_current_session", None), "sample_index", None
                    ),
                    "tool_name": generated_tool_name,
                    "failure_reason": str(failure_reason or "").strip(),
                    "selected_family": selected_family,
                    "bundle_version": final_attempt.get("family_bundle_version"),
                    "query_shape": trigger_query_plan.get("query_shape"),
                    "scaffold_signature": scaffold_signature,
                    "relation_names": relation_names,
                    "failure_class": failure_class,
                },
            )
        except Exception as exc:
            self._emit_generated_tools_event(
                {
                    "event": "pal_family_policy_candidate_failed",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "family_name": selected_family,
                    "error": str(exc),
                }
            )
            return
        if candidate is None:
            return
        self._emit_generated_tools_event(
            {
                "event": "pal_family_policy_candidate_created",
                "mode": "pal",
                "tool_name": generated_tool_name,
                "family_name": selected_family,
                "base_version": candidate.base_version,
                "candidate_version": candidate.candidate_version,
                "fields_changed": list(candidate.fields_changed),
                "reason_for_change": candidate.reason_for_change,
                "trigger_context": dict(candidate.trigger_context),
            }
        )

    def _resolve_family_policy_trigger_query_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        repair_loop_log: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        final_verdict = str((repair_loop_log or {}).get("final_verdict") or "").strip()
        if final_verdict in {"accepted", "accepted_best_effort"}:
            return query_plan
        final_attempt_query_plan = (repair_loop_log or {}).get("final_attempt_query_plan")
        if isinstance(final_attempt_query_plan, Mapping):
            return final_attempt_query_plan
        return query_plan

    def _evaluate_materialization_trust_contract(
        self,
        *,
        generated_code: str,
        query_plan: Mapping[str, Any],
        query_text: str,
        result_dict: Mapping[str, Any] | None,
        verdict: PlausibilityVerdict,
        materialization: BenchmarkMaterialization,
        artifact_type: str,
        artifact_source: str,
    ) -> TrustContractEvaluation:
        plan_validation_errors = self._validate_pal_query_candidate(
            raw_output=generated_code,
            generated_code=generated_code,
            query_text=query_text,
            query_texts=self._extract_sparql_query_texts(generated_code),
            query_plan=query_plan,
        )
        plan_consistency_passed = not bool(plan_validation_errors)
        execution_shape_passed, execution_shape_errors = (
            self._execution_shape_matches_query_plan(
                query_plan=query_plan,
                result_dict=result_dict,
                artifact_type=artifact_type,
            )
        )
        plausibility_validation_passed = bool(getattr(verdict, "is_accepted", False))
        adapter_safety_passed, adapter_safety_errors = evaluate_adapter_safety(
            artifact=type("ArtifactProxy", (), {
                "artifact_type": artifact_type,
                "source": artifact_source,
            })(),
            materialization=materialization,
        )
        return build_trust_contract_evaluation(
            plan_consistency_passed=plan_consistency_passed,
            execution_shape_passed=execution_shape_passed,
            plausibility_validation_passed=plausibility_validation_passed,
            adapter_safety_passed=adapter_safety_passed,
            extra_denial_reasons=tuple(
                list(plan_validation_errors)
                + list(execution_shape_errors)
                + list(adapter_safety_errors)
            ),
            dangerous_overreach_reasons=self._collect_dangerous_overreach_reasons(
                verdict=verdict
            ),
        )

    def _emit_final_attempt_decision(
        self,
        *,
        generated_tool_name: str,
        loop_log: Mapping[str, Any],
        trust_contract: TrustContractEvaluation,
        tool_result_semantics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        attempt_decisions = list(loop_log.get("attempt_decisions") or [])
        final_payload = dict(attempt_decisions[-1] if attempt_decisions else {})
        final_payload.update(
            {
                "trust_contract": trust_contract.as_dict(),
                "materialization_allowed": trust_contract.materialization_allowed,
                "materialization_denial_reasons": list(
                    trust_contract.denial_reasons
                ),
                "dangerous_overreach": trust_contract.dangerous_overreach,
                "dangerous_overreach_reasons": list(
                    trust_contract.dangerous_overreach_reasons
                ),
            }
        )
        if tool_result_semantics:
            final_payload.update(dict(tool_result_semantics))
        self._emit_generated_tools_event(
            {
                "event": "pal_attempt_decision_finalized",
                "mode": "pal",
                "tool_name": generated_tool_name,
                **final_payload,
            }
        )

    def _run_text_prompt(self, *, system_prompt: str, user_prompt: str) -> str:
        prompt_history = ChatHistory()
        prompt_history.inject(
            ChatHistoryItem(role=Role.USER, content=user_prompt)
        )
        response = self._language_model.inference(
            [prompt_history],
            self._inference_config_dict,
            system_prompt,
        )[0]
        return response.content or ""

    def _parse_orchestrator_action(self, raw_output: str) -> str:
        stripped_output = (raw_output or "").strip()
        candidate_payloads = [stripped_output]
        json_match = re.search(r"\{.*\}", stripped_output, flags=re.DOTALL)
        if json_match is not None:
            candidate_payloads.append(json_match.group(0))
        for payload in candidate_payloads:
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            action = str(parsed.get("action") or "").strip()
            if action == "generate_tool":
                return action
        return "generate_tool"

    def _emit_pal_invoker_failure_event(
        self,
        *,
        generated_tool_name: Optional[str],
        invocation_result: Any,
    ) -> None:
        failure_kind = str(
            getattr(invocation_result, "failure_kind", None) or "execution_error"
        )
        event_name = (
            "pal_invoker_endpoint_unavailable"
            if failure_kind == "endpoint_unavailable"
            else "pal_invoker_transport_error"
            if failure_kind in {"transport_error", "endpoint_timeout"}
            else "pal_invoker_execution_error"
        )
        self._emit_generated_tools_event(
            {
                "event": event_name,
                "mode": "pal",
                "tool_name": generated_tool_name,
                "failure_kind": failure_kind,
                "endpoint_url": self._resolve_invocation_endpoint_url(
                    invocation_result
                ),
                "error": getattr(invocation_result, "error", None),
                "diagnostics": dict(
                    getattr(invocation_result, "diagnostics", {}) or {}
                ),
            }
        )

    def _resolve_invocation_endpoint_url(self, invocation_result: Any) -> str:
        diagnostics = getattr(invocation_result, "diagnostics", {}) or {}
        endpoint_url = diagnostics.get("endpoint_url")
        if isinstance(endpoint_url, str) and endpoint_url.strip():
            return endpoint_url
        return self._get_runtime_sparql_endpoint()

    def _normalize_final_answer_output(self, raw_output: str) -> str:
        text = (raw_output or "").strip()
        if not text:
            return "Final Answer: "
        if text.startswith("Final Answer:"):
            return text
        return f"Final Answer: {text}"

    def _build_query_tool_name(self, task_question: str) -> str:
        sample_index = getattr(
            getattr(self, "_current_session", None), "sample_index", "unknown"
        )
        question_hash = hashlib.sha256(task_question.encode("utf-8")).hexdigest()[:12]
        return f"{self._PAL_QUERY_TOOL_BASE_NAME}_{sample_index}_{question_hash}"

    def _emit_generated_tools_event(self, payload: Mapping[str, Any]) -> None:
        event_payload = dict(payload)
        event_payload.setdefault("timestamp", datetime.now(UTC).isoformat())
        task_name = getattr(getattr(self, "_current_session", None), "task_name", None)
        sample_index = getattr(
            getattr(self, "_current_session", None), "sample_index", None
        )
        if task_name is not None:
            event_payload.setdefault("task_name", str(task_name))
        if sample_index is not None:
            event_payload.setdefault("sample_index", str(sample_index))
        try:
            registry = get_registry()
            listeners = getattr(registry, "_event_listeners", None)
            if isinstance(listeners, list) and listeners:
                registry._notify(event_payload)
                return
        except Exception:
            pass
        self._append_generated_tools_log_fallback(event_payload)

    def _append_generated_tools_log_fallback(
        self, payload: Mapping[str, Any]
    ) -> None:
        log_path = self._get_generated_tools_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(payload)
        entry["t"] = self._next_generated_tools_log_seq(log_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")

    def _get_generated_tools_log_path(self) -> Path:
        output_dir = Path(
            os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/pal_runtime")
        )
        return output_dir / prefix_filename("generated_tools.log")

    def _next_generated_tools_log_seq(self, log_path: Path) -> int:
        if not log_path.exists():
            return 1
        try:
            with log_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict) and isinstance(parsed.get("t"), int):
                    return int(parsed["t"]) + 1
        except Exception:
            return 1
        return 1

    def _sha1_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _summarize_text(self, text: str, *, max_len: int = 120) -> dict[str, Any]:
        safe_text = str(text or "")
        preview = safe_text if len(safe_text) <= max_len else safe_text[: max_len - 3] + "..."
        return {
            "len": len(safe_text),
            "sha1": self._sha1_text(safe_text),
            "preview": preview,
        }

    def _build_repair_loop_stagnation_signature(
        self,
        *,
        query_plan: Mapping[str, Any],
        query_text: str,
        verdict: PlausibilityVerdict,
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
        invocation_result: Any,
    ) -> str:
        probe_signature = [
            {
                "anchor": result.anchor_name,
                "resolved_entity_id": result.resolved_entity_id,
                "entity_count": result.entity_count,
                "path_count": result.path_count,
                "relation_probed": result.relation_probed,
                "anchor_position": result.anchor_position,
            }
            for result in (anchor_probe_results or [])
        ]
        signature_payload = {
            "query_shape": query_plan.get("query_shape"),
            "answer_mode": query_plan.get("answer_mode"),
            "strategy": query_plan.get("strategy"),
            "relation_paths": query_plan.get("relation_paths") or [],
            "query_text_sha1": self._sha1_text(str(query_text or "")),
            "execution_success": bool(getattr(invocation_result, "success", False)),
            "failure_kind": getattr(invocation_result, "failure_kind", None),
            "verdict": verdict.verdict,
            "verdict_reasons": list(verdict.reasons),
            "anchor_probes": probe_signature,
        }
        return self._sha1_text(json.dumps(signature_payload, sort_keys=True, default=str))

    def _get_result_binding_count(self, result_dict: Mapping[str, Any]) -> int:
        bindings = result_dict.get("results", {}).get("bindings", [])
        if isinstance(bindings, Sequence):
            return len(bindings)
        return 0

    def _extract_scalar_count_value(self, result_dict: Mapping[str, Any] | None) -> int | None:
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

    def _classify_pal_result_kind(self, result_dict: Mapping[str, Any]) -> str:
        if "boolean" in result_dict:
            return "boolean"
        if self._get_result_binding_count(result_dict) == 0:
            return "empty"
        return "bindings"

    def _format_prompt_template(
        self,
        *,
        template: str,
        **fields: Any,
    ) -> str:
        escaped_template = template.replace("{", "{{").replace("}", "}}")
        for field_name in fields:
            escaped_template = escaped_template.replace(
                "{{" + field_name + "}}",
                "{" + field_name + "}",
            )
        return escaped_template.format(**fields)

    def _generate_pal_query_plan(
        self,
        *,
        task_question: str,
        grounding_card: str,
        relation_grounding: Sequence[Mapping[str, str]],
        generated_tool_name: str,
        extra_plan_feedback: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        last_error = ""
        plan_feedback: list[str] = self._canonicalize_feedback_items(
            [
                str(item).strip()
                for item in (extra_plan_feedback or [])
                if str(item).strip()
            ]
        )
        dead_scaffold_signatures = {
            item.split(":", 1)[1].strip()
            for item in plan_feedback
            if item.startswith("dead_scaffold_signature:")
            and item.split(":", 1)[1].strip()
        }
        for attempt in range(1, self._PAL_QUERY_PLAN_MAX_ATTEMPTS + 1):
            plan_prompt = self._format_prompt_template(
                template=GENERATOR_PLAN_SYSTEM_PROMPT,
                ontology_card=grounding_card,
                plan_feedback=self._build_validation_feedback(plan_feedback),
                task_question=task_question,
            )
            plan_raw = self._run_text_prompt(
                system_prompt=plan_prompt,
                user_prompt=task_question,
            )
            try:
                query_plan = self._parse_pal_query_plan(plan_raw)
                query_plan = self._apply_question_scaffold_plan_rewrites(
                    task_question=task_question,
                    query_plan=query_plan,
                )
                question_text, _ = self._split_task_question(task_question)
                answer_target_phrase = self._extract_answer_target_phrase(question_text)
                if answer_target_phrase and not str(
                    query_plan.get("answer_target_phrase") or ""
                ).strip():
                    query_plan = self._normalize_pal_query_plan(
                        {
                            **dict(query_plan),
                            "answer_target_phrase": answer_target_phrase,
                        }
                    )
                active_family_bundle = get_reusable_family_policy_bundle(
                    str(query_plan.get("query_shape") or "").strip()
                )
                if active_family_bundle is not None:
                    dead_scaffold_signatures.update(
                        str(item).strip()
                        for item in active_family_bundle.blocked_scaffold_signatures
                        if str(item).strip()
                    )
                plan_validation_errors = self._validate_query_plan_grounding(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                )
                if plan_validation_errors:
                    raise ValueError(",".join(plan_validation_errors))
                plan_signature = self._build_scaffold_signature(query_plan)
                if plan_signature and plan_signature in dead_scaffold_signatures:
                    raise ValueError(
                        f"query_plan_reused_dead_scaffold_signature:{plan_signature}"
                    )
            except Exception as exc:
                last_error = str(exc)
                plan_feedback = self._canonicalize_feedback_items(
                    [
                        str(item).strip()
                        for item in [*(extra_plan_feedback or []), last_error]
                        if str(item).strip()
                    ]
                )
                self._emit_generated_tools_event(
                    {
                        "event": "pal_query_plan_rejected",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "error": last_error,
                        "raw_plan_summary": self._summarize_text(plan_raw, max_len=240),
                    }
                )
                continue
            plan_artifact_path = self._capture_pal_query_plan_artifact(
                generated_tool_name=generated_tool_name,
                query_plan=query_plan,
            )
            self._emit_generated_tools_event(
                {
                    "event": "pal_query_plan_generated",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "attempt": attempt,
                    "plan_artifact_path": str(plan_artifact_path),
                    "answer_mode": query_plan.get("answer_mode"),
                    "answer_type": query_plan.get("answer_type"),
                    "query_shape": query_plan.get("query_shape"),
                    "strategy": query_plan.get("strategy"),
                    "allow_exploratory_predicates": query_plan.get(
                        "allow_exploratory_predicates"
                    ),
                    "shared_answer_variable": query_plan.get("shared_answer_variable"),
                    "candidate_set_variable": query_plan.get("candidate_set_variable"),
                    "count_set_variable": query_plan.get("count_set_variable"),
                    "ordering_attribute": query_plan.get("ordering_attribute"),
                    "ordering_direction": query_plan.get("ordering_direction"),
                    "join_structure": query_plan.get("join_structure"),
                    "relation_path_count": len(query_plan.get("relation_paths", [])),
                    "relation_paths": query_plan.get("relation_paths", []),
                    "projection": list(query_plan.get("projection", [])),
                    "plan_summary": self._summarize_text(
                        json.dumps(query_plan, ensure_ascii=False, sort_keys=True),
                        max_len=240,
                    ),
                }
            )
            return query_plan
        raise ValueError(f"pal_query_plan_invalid:{last_error or 'unknown_error'}")

    def _parse_pal_query_plan(self, raw_output: str) -> dict[str, Any]:
        json_payload = self._extract_json_object(raw_output)
        try:
            parsed = json.loads(json_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid_query_plan_json:{exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("query_plan_must_be_json_object")
        return self._normalize_pal_query_plan(parsed)

    def _normalize_pal_query_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        answer_mode = str(plan.get("answer_mode") or "").strip().lower()
        if answer_mode not in {"entity", "count", "boolean", "literal"}:
            raise ValueError("query_plan_invalid_answer_mode")
        answer_type = str(plan.get("answer_type") or answer_mode).strip().lower()
        if answer_type not in {"entity", "count", "boolean", "literal"}:
            answer_type = answer_mode
        answer_target_phrase = str(plan.get("answer_target_phrase") or "").strip()

        raw_anchored_entities = self._normalize_plan_items(
            plan.get("anchored_entities"),
            required_keys=("surface", "chosen_alias", "role"),
        )
        normalized_aliases = self._normalize_plan_items(
            plan.get("normalized_aliases"),
            required_keys=("surface", "chosen_alias", "reason"),
        )
        query_shape = self._normalize_query_shape_value(plan.get("query_shape"))
        if not query_shape:
            if answer_mode == "count":
                query_shape = (
                    "count_over_joined_set"
                    if len(raw_anchored_entities) > 1
                    else "count_over_direct_relation"
                )
            elif len(raw_anchored_entities) > 1:
                query_shape = "multi_anchor_intersection"
            else:
                query_shape = "single_anchor_lookup"
        elif answer_mode == "count" and query_shape not in {
            "count_over_direct_relation",
            "count_over_joined_set",
        }:
            query_shape = (
                "count_over_joined_set"
                if len(raw_anchored_entities) > 1
                else "count_over_direct_relation"
            )
        ordering_attribute = self._normalize_ordering_attribute(
            plan.get("ordering_attribute")
        )
        ordering_direction = str(plan.get("ordering_direction") or "").strip().lower()
        if ordering_direction not in {"max", "min", "none"}:
            ordering_direction = "none"
        if (
            ordering_direction in {"max", "min"}
            and (
                str(ordering_attribute.get("relation") or "").strip()
                or str(ordering_attribute.get("source_variable") or "").strip()
            )
        ):
            query_shape = "superlative_chain"
        anchored_entities = self._normalize_anchored_entities(
            raw_anchored_entities,
            query_shape=query_shape,
        )
        shared_answer_variable = str(plan.get("shared_answer_variable") or "").strip()
        candidate_set_variable = str(plan.get("candidate_set_variable") or "").strip()
        count_set_variable = str(plan.get("count_set_variable") or "").strip()
        join_structure = self._normalize_join_structure(
            plan.get("join_structure"),
            query_shape=query_shape,
            anchored_entities=anchored_entities,
            shared_answer_variable=shared_answer_variable,
            candidate_set_variable=candidate_set_variable,
            count_set_variable=count_set_variable,
        )
        raw_relation_paths = plan.get("relation_paths")
        relation_paths = self._normalize_plan_relation_paths(
            raw_relation_paths,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            anchored_entities=anchored_entities,
            shared_answer_variable=shared_answer_variable,
            candidate_set_variable=candidate_set_variable,
            count_set_variable=count_set_variable,
            ordering_attribute=ordering_attribute,
            allow_exploratory=bool(plan.get("allow_exploratory_predicates", False)),
        )
        relation_paths = self._expand_projected_answer_intersection_paths(
            query_shape=query_shape,
            shared_answer_variable=shared_answer_variable,
            join_structure=join_structure,
            relation_paths=relation_paths,
        )
        projection = self._normalize_string_list(plan.get("projection"))
        if query_shape == "superlative_chain":
            answer_mode = "entity"
            answer_type = "entity"
            relation_paths = self._normalize_plan_relation_paths(
                raw_relation_paths,
                query_shape=query_shape,
                answer_mode=answer_mode,
                answer_target_phrase=answer_target_phrase,
                anchored_entities=anchored_entities,
                shared_answer_variable=shared_answer_variable,
                candidate_set_variable=candidate_set_variable,
                count_set_variable=count_set_variable,
                ordering_attribute=ordering_attribute,
                allow_exploratory=bool(plan.get("allow_exploratory_predicates", False)),
            )
            relation_paths = self._expand_projected_answer_intersection_paths(
                query_shape=query_shape,
                shared_answer_variable=shared_answer_variable,
                join_structure=join_structure,
                relation_paths=relation_paths,
            )
            answer_projection_variable = (
                self._resolve_superlative_answer_projection_variable(
                    relation_paths=relation_paths,
                )
            )
            if answer_projection_variable:
                shared_answer_variable = answer_projection_variable
                projection = [answer_projection_variable]
            elif candidate_set_variable:
                shared_answer_variable = candidate_set_variable
                projection = [candidate_set_variable]
        if self._literal_mode_projects_semantic_shared_answer(
            answer_mode=answer_mode,
            answer_type=answer_type,
            query_shape=query_shape,
            projection=projection,
            shared_answer_variable=shared_answer_variable,
        ):
            answer_mode = "entity"
            answer_type = "entity"
        if self._literal_superlative_projects_ordering_attribute(
            answer_mode=answer_mode,
            answer_type=answer_type,
            query_shape=query_shape,
            projection=projection,
            candidate_set_variable=candidate_set_variable,
            ordering_attribute=ordering_attribute,
        ):
            answer_mode = "entity"
            answer_type = "entity"
            if candidate_set_variable:
                shared_answer_variable = candidate_set_variable
                projection = [candidate_set_variable]
            relation_paths = self._normalize_plan_relation_paths(
                plan.get("relation_paths"),
                query_shape=query_shape,
                answer_mode=answer_mode,
                answer_target_phrase=answer_target_phrase,
                anchored_entities=anchored_entities,
                shared_answer_variable=shared_answer_variable,
                candidate_set_variable=candidate_set_variable,
                count_set_variable=count_set_variable,
                ordering_attribute=ordering_attribute,
                allow_exploratory=bool(plan.get("allow_exploratory_predicates", False)),
            )
        relation_paths = self._ensure_distinct_constraint_value_variables(
            relation_paths=relation_paths
        )
        ordering_attribute, relation_paths = self._normalize_terminal_ordering_attribute_path(
            ordering_attribute=ordering_attribute,
            relation_paths=relation_paths,
        )
        if answer_mode == "entity" and projection:
            first_projection = projection[0].lower()
            if "name" in first_projection or "label" in first_projection:
                raise ValueError("query_plan_entity_projection_must_start_with_entity")
        if answer_mode == "count" and projection and not any(
            "count" in item.lower() or item.lower().startswith("num")
            for item in projection
        ):
            raise ValueError("query_plan_count_projection_missing_count_var")
        if not anchored_entities and not relation_paths:
            raise ValueError("query_plan_missing_grounded_structure")

        strategy = str(plan.get("strategy") or "").strip() or "grounded_relation_lookup"
        plan_rationale = self._normalize_string_list(plan.get("plan_rationale"))
        allow_exploratory_predicates = bool(
            plan.get("allow_exploratory_predicates", False)
        )
        if allow_exploratory_predicates and not plan_rationale:
            raise ValueError("query_plan_exploratory_requires_rationale")

        return {
            "answer_type": answer_type,
            "answer_mode": answer_mode,
            "answer_target_phrase": answer_target_phrase,
            "query_shape": query_shape,
            "anchored_entities": anchored_entities,
            "normalized_aliases": normalized_aliases,
            "shared_answer_variable": shared_answer_variable,
            "candidate_set_variable": candidate_set_variable,
            "count_set_variable": count_set_variable,
            "ordering_attribute": ordering_attribute,
            "ordering_direction": ordering_direction,
            "join_structure": join_structure,
            "relation_paths": relation_paths,
            "projection": projection,
            "allow_exploratory_predicates": allow_exploratory_predicates,
            "strategy": strategy,
            "plan_rationale": plan_rationale,
        }

    def _ensure_distinct_constraint_value_variables(
        self,
        *,
        relation_paths: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized_paths: list[dict[str, str]] = []
        used_constraint_tokens: dict[str, int] = {}
        generic_constraint_tokens = {
            "constraint",
            "constraint_value",
            "constraintvalue",
            "anchor_value",
            "anchorvalue",
            "value",
        }
        for raw_path in relation_paths:
            path = dict(raw_path)
            from_role = self._normalize_relation_role(path.get("from_role"))
            to_role = self._normalize_relation_role(path.get("to_role"))
            constraint_side = ""
            if to_role == "constraint_value":
                constraint_side = "to"
            elif from_role == "constraint_value":
                constraint_side = "from"
            if not constraint_side:
                normalized_paths.append(path)
                continue
            endpoint_value = str(path.get(constraint_side) or "").strip()
            endpoint_token = self._normalize_variable_token(endpoint_value)
            if not endpoint_token:
                normalized_paths.append(path)
                continue
            if endpoint_token not in generic_constraint_tokens:
                normalized_paths.append(path)
                continue
            seen_count = used_constraint_tokens.get(endpoint_token, 0)
            used_constraint_tokens[endpoint_token] = seen_count + 1
            if seen_count > 0:
                path[constraint_side] = f"{endpoint_token}_{seen_count + 1}"
            normalized_paths.append(path)
        return normalized_paths

    def _literal_mode_projects_semantic_shared_answer(
        self,
        *,
        answer_mode: str,
        answer_type: str,
        query_shape: str,
        projection: Sequence[str],
        shared_answer_variable: str,
    ) -> bool:
        if answer_mode != "literal" or answer_type not in {"literal", answer_mode}:
            return False
        if query_shape not in {"multi_anchor_intersection", "shared_type_intersection"}:
            return False
        if not projection or not shared_answer_variable:
            return False
        projection_token = self._normalize_variable_token(projection[0])
        shared_answer_token = self._normalize_variable_token(shared_answer_variable)
        if not projection_token or projection_token != shared_answer_token:
            return False
        if self._projection_token_looks_literal_label(projection_token):
            return False
        return True

    def _projection_token_looks_literal_label(self, projection_token: str) -> bool:
        if not projection_token:
            return False
        parts = {part for part in projection_token.split("_") if part}
        literal_hints = {
            "alias",
            "date",
            "description",
            "label",
            "name",
            "string",
            "text",
            "title",
            "year",
        }
        return bool(parts & literal_hints)

    def _literal_superlative_projects_ordering_attribute(
        self,
        *,
        answer_mode: str,
        answer_type: str,
        query_shape: str,
        projection: Sequence[str],
        candidate_set_variable: str,
        ordering_attribute: Mapping[str, str],
    ) -> bool:
        if answer_mode != "literal" or answer_type not in {"literal", answer_mode}:
            return False
        if query_shape != "superlative_chain":
            return False
        candidate_token = self._normalize_variable_token(candidate_set_variable)
        ordering_token = self._normalize_variable_token(
            ordering_attribute.get("attribute_variable")
        )
        if not candidate_token or not ordering_token or not projection:
            return False
        projection_token = self._normalize_variable_token(projection[0])
        return projection_token == ordering_token and projection_token != candidate_token

    def _normalize_terminal_ordering_attribute_path(
        self,
        *,
        ordering_attribute: Mapping[str, str],
        relation_paths: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        normalized_ordering_attribute = {
            str(key): str(value)
            for key, value in dict(ordering_attribute or {}).items()
            if str(value).strip()
        }
        normalized_relation_paths = [
            dict(relation_path)
            for relation_path in relation_paths
            if isinstance(relation_path, Mapping)
        ]
        current_variable = str(
            normalized_ordering_attribute.get("attribute_variable") or ""
        ).strip()
        current_token = self._normalize_variable_token(current_variable)
        if not current_token or not normalized_relation_paths:
            return normalized_ordering_attribute, normalized_relation_paths

        seen_tokens: set[str] = set()
        latest_variable = current_variable
        while current_token and current_token not in seen_tokens:
            seen_tokens.add(current_token)
            next_variable = ""
            next_token = ""
            for relation_path in normalized_relation_paths:
                from_token = self._normalize_variable_token(relation_path.get("from"))
                to_token = self._normalize_variable_token(relation_path.get("to"))
                roles = {
                    self._normalize_relation_role(relation_path.get("from_role")),
                    self._normalize_relation_role(relation_path.get("to_role")),
                }
                if "ordering_attribute" not in roles:
                    continue
                if from_token == current_token and to_token and to_token != current_token:
                    next_variable = str(relation_path.get("to") or "").strip()
                    next_token = to_token
                    relation_path["from_role"] = "ordering_attribute"
                    relation_path["to_role"] = "ordering_attribute"
                    break
                if (
                    to_token == current_token
                    and self._normalize_variable_token(relation_path.get("direction"))
                    == "reverse"
                    and from_token
                    and from_token != current_token
                ):
                    next_variable = str(relation_path.get("from") or "").strip()
                    next_token = from_token
                    relation_path["from_role"] = "ordering_attribute"
                    relation_path["to_role"] = "ordering_attribute"
                    break
            if not next_token:
                break
            latest_variable = next_variable or latest_variable
            current_token = next_token

        if latest_variable and latest_variable != current_variable:
            normalized_ordering_attribute["attribute_variable"] = latest_variable
        return normalized_ordering_attribute, normalized_relation_paths

    def _apply_question_scaffold_plan_rewrites(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        question_text, question_entities = self._split_task_question(task_question)
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        interpretation = self._build_question_interpretation(
            question_text=question_text,
            explicit_entities=question_entities,
            answer_target_phrase=answer_target_phrase,
        )
        scaffold_names = {
            str(item.get("name") or "").strip()
            for item in (interpretation.get("preferred_scaffolds") or [])
            if isinstance(item, Mapping)
        }

        rewritten_plan = dict(query_plan)
        if "count_shared_attribute" in scaffold_names:
            rewritten_plan = self._rewrite_count_shared_attribute_plan(
                query_plan=rewritten_plan,
                answer_target_phrase=answer_target_phrase,
                question_interpretation=interpretation,
            )
        rewritten_plan = self._rewrite_broadcast_producer_of_anchor_count_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
        )
        rewritten_plan = self._rewrite_cvg_release_region_intersection_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
            question_interpretation=interpretation,
        )
        rewritten_plan = self._rewrite_joined_anchor_constraint_paths(
            query_plan=rewritten_plan,
        )
        if "count_shared_attribute" not in scaffold_names:
            rewritten_plan = self._rewrite_joined_count_answer_target_hint_plan(
                task_question=task_question,
                query_plan=rewritten_plan,
                question_interpretation=interpretation,
            )
        rewritten_plan = self._rewrite_fictional_character_constraint_count_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
            question_interpretation=interpretation,
        )
        rewritten_plan = self._rewrite_exhibition_subject_type_count_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
        )
        rewritten_plan = self._rewrite_condition_example_parent_disease_count_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
        )
        rewritten_plan = self._rewrite_single_anchor_answer_target_count_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
            question_interpretation=interpretation,
        )
        rewritten_plan = self._rewrite_single_anchor_attribute_qualifier_lookup_plan(
            task_question=task_question,
            query_plan=rewritten_plan,
        )
        return rewritten_plan

    def _rewrite_single_anchor_attribute_qualifier_lookup_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "entity":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() not in {
            "single_anchor_lookup",
            "single_anchor_chain_lookup",
        }:
            return dict(query_plan)

        question_text, _ = self._split_task_question(task_question)
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return dict(query_plan)
        if not self._answer_target_qualifier_is_encoded_in_answer_relation(
            answer_target_phrase=answer_target_phrase,
            relation_candidates=relation_paths,
        ):
            return dict(query_plan)

        removable_paths: list[dict[str, Any]] = []
        for relation_path in relation_paths:
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            roles = {from_role, to_role}
            if roles & {"type_set", "shared_type", "constraint_value"} and roles & {
                "candidate_set",
                "shared_answer",
                "answer",
            }:
                removable_paths.append(relation_path)
        if not removable_paths:
            return dict(query_plan)

        rewritten_paths = [
            relation_path
            for relation_path in relation_paths
            if relation_path not in removable_paths
        ]
        if not any(
            "answer"
            in {
                self._normalize_relation_role(path.get("from_role")),
                self._normalize_relation_role(path.get("to_role")),
            }
            for path in rewritten_paths
        ):
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["relation_paths"] = rewritten_paths
        strategy = str(rewritten_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The answer relation already encodes the answer-target qualifier semantics, so drop the redundant type/class filter and treat the qualifier as a relation-selection hint."
        )
        rewritten_plan["strategy"] = (
            f"{strategy} Remove the redundant answer-target class filter because the projected answer relation already carries that qualifier semantics."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_broadcast_producer_of_anchor_count_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_joined_set":
            return dict(query_plan)

        question_text, _ = self._split_task_question(task_question)
        lower_text = str(question_text or "").lower()
        if "producer of" not in lower_text:
            return dict(query_plan)

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return dict(query_plan)

        producer_bridge_index: Optional[int] = None
        producer_projection_index: Optional[int] = None
        for index, relation_path in enumerate(relation_paths):
            relation = str(relation_path.get("relation") or "").strip()
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            if (
                relation == "broadcast.producer.produces"
                and to_role == "anchor_b"
                and from_role in {"constraint_value", "candidate_set"}
            ):
                producer_bridge_index = index
            elif (
                relation == "broadcast.producer.produces"
                and from_role in {"constraint_value", "candidate_set"}
                and to_role == "candidate_set"
            ):
                producer_projection_index = index
        if producer_bridge_index is None or producer_projection_index is None:
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_paths = [
            dict(path)
            for path in (rewritten_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        rewritten_paths[producer_bridge_index] = {
            "relation": "broadcast.content.producer",
            "direction": "forward",
            "from": "anchor_b",
            "to": "producer",
            "from_role": "anchor_b",
            "to_role": "constraint_value",
            "grounding_source": "curated",
            "reason": (
                "Producer-of rewrite: bind the producer directly from anchor_b via "
                "broadcast.content.producer before projecting the other content items "
                "that share that producer."
            ),
        }
        rewritten_paths[producer_projection_index]["from"] = "producer"
        rewritten_paths[producer_projection_index]["from_role"] = "constraint_value"
        rewritten_paths[producer_projection_index]["to"] = (
            str(query_plan.get("candidate_set_variable") or "").strip() or "content"
        )
        rewritten_paths[producer_projection_index]["to_role"] = "candidate_set"
        rewritten_plan["relation_paths"] = rewritten_paths
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Producer-of rewrite: use broadcast.content.producer to bind the producer of anchor_b directly, then reuse broadcast.producer.produces only for the projected content set."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_fictional_character_constraint_count_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_joined_set":
            return dict(query_plan)

        question_text, explicit_entities = self._split_task_question(task_question)
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        if "character" not in str(answer_target_phrase or "").lower():
            return dict(query_plan)

        interpretation = question_interpretation or self._build_question_interpretation(
            question_text=question_text,
            explicit_entities=explicit_entities,
            answer_target_phrase=answer_target_phrase,
        )
        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        constraint_entity = next(
            (
                item
                for item in anchored_entities
                if self._normalize_relation_role(item.get("role")) == "constraint_value"
            ),
            None,
        )
        species_entity = next(
            (
                item
                for item in anchored_entities
                if self._normalize_relation_role(item.get("role")) in {"anchor", "anchor_a", "anchor_b"}
            ),
            None,
        )
        if constraint_entity is None or species_entity is None:
            return dict(query_plan)

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if not relation_paths:
            return dict(query_plan)

        character_variable = (
            self._normalize_variable_token(query_plan.get("shared_answer_variable"))
            or self._normalize_variable_token(query_plan.get("candidate_set_variable"))
            or "character"
        )
        species_role = self._normalize_relation_role(species_entity.get("role")) or "anchor_b"
        species_alias = str(
            species_entity.get("resolved_entity_id")
            or species_entity.get("chosen_alias")
            or species_entity.get("surface")
            or ""
        ).strip()
        constraint_alias = str(
            constraint_entity.get("chosen_alias")
            or constraint_entity.get("surface")
            or ""
        ).strip()
        if not species_alias or not constraint_alias:
            return dict(query_plan)

        rewritten_paths: list[dict[str, Any]] = []
        saw_species = False
        saw_occupation = False
        saw_world_characters = False
        synthesized_species_path: dict[str, Any] | None = None
        for relation_path in relation_paths:
            relation = str(relation_path.get("relation") or "").strip()
            if relation == "fictional_universe.fictional_character.appears_in":
                continue
            rewritten_path = dict(relation_path)
            if relation == "fictional_universe.fictional_character.occupation":
                rewritten_path["direction"] = "forward"
                rewritten_path["from"] = character_variable
                rewritten_path["from_role"] = "candidate_set"
                rewritten_path["to"] = constraint_alias
                rewritten_path["to_role"] = "constraint_value"
                rewritten_path["grounding_source"] = "curated"
                saw_occupation = True
            elif relation == "fictional_universe.fictional_character.species":
                rewritten_path["direction"] = "forward"
                rewritten_path["from"] = character_variable
                rewritten_path["from_role"] = "candidate_set"
                rewritten_path["to"] = species_alias
                rewritten_path["to_role"] = species_role
                saw_species = True
            elif relation == "fictional_universe.fictional_universe.characters":
                saw_world_characters = True
                continue
            elif relation == "fictional_universe.fictional_universe.species":
                synthesized_species_path = {
                    "relation": "fictional_universe.fictional_character.species",
                    "direction": "forward",
                    "from": character_variable,
                    "to": species_alias,
                    "from_role": "candidate_set",
                    "to_role": species_role,
                    "grounding_source": "exploratory",
                    "reason": (
                        "Rewrite fictional-world species detour into a direct "
                        "fictional-character species filter so the same character "
                        "set is constrained by both occupation and species."
                    ),
                }
                continue
            rewritten_paths.append(rewritten_path)

        if not saw_species and synthesized_species_path is not None and saw_world_characters:
            rewritten_paths.append(synthesized_species_path)
            saw_species = True

        if not saw_species or not saw_occupation:
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["allow_exploratory_predicates"] = False
        rewritten_plan["shared_answer_variable"] = character_variable
        rewritten_plan["candidate_set_variable"] = character_variable
        rewritten_plan["count_set_variable"] = character_variable
        rewritten_plan["relation_paths"] = rewritten_paths
        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": character_variable,
                    "notes": "Constrain the fictional character set by the explicit occupation value.",
                },
                {
                    "anchor_role": species_role,
                    "constrains_variable": character_variable,
                    "notes": "Constrain the same fictional character set by the explicit species anchor.",
                },
            ],
        }
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Fictional-character rewrite: bind the explicit occupation and species constraints directly on the same fictional-character candidate set and drop the redundant film-type bridge."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        rewritten_plan["strategy"] = (
            f"Count fictional characters by applying the explicit occupation constraint "
            f"({constraint_alias}) and species constraint ({species_alias}) directly to "
            f"the same character set."
        )
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_exhibition_subject_type_count_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_direct_relation":
            return dict(query_plan)

        question_text, _ = self._split_task_question(task_question)
        lower_text = str(question_text or "").lower()
        if "exhibition subject" not in lower_text and "exhibition subjects" not in lower_text:
            return dict(query_plan)

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
            and self._normalize_relation_role(item.get("role")) in {"anchor", "anchor_a", "anchor_b"}
        ]
        if len(anchored_entities) != 1:
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["query_shape"] = "count_over_direct_relation"
        rewritten_plan["shared_answer_variable"] = "subject"
        rewritten_plan["candidate_set_variable"] = "exhibition"
        rewritten_plan["count_set_variable"] = "subject"
        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "exhibition_type",
                    "notes": "Retrieve the exhibition type of the anchor exhibition first, then enumerate exhibitions of that type before counting their subjects.",
                }
            ],
        }
        rewritten_plan["relation_paths"] = [
            {
                "relation": "exhibitions.exhibition.exhibition_types",
                "direction": "forward",
                "from": "anchor",
                "to": "exhibition_type",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "reason": "Retrieve the type of the anchored exhibition.",
            },
            {
                "relation": "exhibitions.type_of_exhibition.exhibitions_of_this_type",
                "direction": "forward",
                "from": "exhibition_type",
                "to": "exhibition",
                "from_role": "candidate_set",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "reason": "Enumerate exhibitions that share the same exhibition type.",
            },
            {
                "relation": "exhibitions.exhibition.subjects",
                "direction": "forward",
                "from": "exhibition",
                "to": "subject",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
                "reason": "Count exhibition subjects across the exhibitions in that type.",
            },
        ]
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Exhibition-subject rewrite: count subjects across exhibitions that share the anchored exhibition's type rather than only the anchor exhibition itself."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_condition_example_parent_disease_count_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_direct_relation":
            return dict(query_plan)

        question_text, _ = self._split_task_question(task_question)
        lower_text = str(question_text or "").lower()
        if "conditions such as" not in lower_text or "prevention factor" not in lower_text:
            return dict(query_plan)

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if not any(
            str(path.get("relation") or "").strip() == "medicine.disease.prevention_factors"
            for path in relation_paths
        ):
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["candidate_set_variable"] = "candidate_set"
        rewritten_plan["count_set_variable"] = "count_set"
        rewritten_plan["shared_answer_variable"] = "candidate_set"
        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "Treat the named disease as the parent disease class, then count prevention factors of its child conditions.",
                }
            ],
        }
        rewritten_plan["relation_paths"] = [
            {
                "relation": "medicine.disease.parent_disease",
                "direction": "reverse",
                "from": "parent_disease",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "reason": "Conditions-such-as rewrite: retrieve child diseases of the named parent disease.",
            },
            {
                "relation": "medicine.disease.prevention_factors",
                "direction": "forward",
                "from": "candidate_set",
                "to": "count_set",
                "from_role": "candidate_set",
                "to_role": "answer",
                "grounding_source": "curated",
                "reason": "Count prevention factors attached to the child-disease set.",
            },
        ]
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Conditions-such-as rewrite: interpret the named disease as the parent disease anchor and count prevention factors over the resulting child-condition set."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_joined_anchor_constraint_paths(
        self,
        *,
        query_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_joined_set":
            return dict(query_plan)

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        anchor_roles = {
            self._normalize_relation_role(item.get("role"))
            for item in anchored_entities
            if self._normalize_relation_role(item.get("role")) in {"anchor_a", "anchor_b"}
        }
        if len(anchor_roles) < 2:
            return dict(query_plan)

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return dict(query_plan)
        explicit_anchor_roles = {
            role
            for relation_path in relation_paths
            for role in (
                self._normalize_relation_role(relation_path.get("from_role")),
                self._normalize_relation_role(relation_path.get("to_role")),
            )
            if role in anchor_roles
        }
        if anchor_roles.issubset(explicit_anchor_roles):
            return dict(query_plan)

        anchor_constraints = [
            dict(item)
            for item in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or [])
            if isinstance(item, Mapping)
        ]
        if not anchor_constraints:
            return dict(query_plan)

        relation_hints_by_anchor: dict[str, list[str]] = {}
        for constraint in anchor_constraints:
            anchor_role = self._normalize_relation_role(constraint.get("anchor_role"))
            if anchor_role not in anchor_roles:
                continue
            note_text = str(constraint.get("notes") or "")
            relation_hints = [
                match.group(0).strip()
                for match in re.finditer(
                    r"\b[a-z0-9_]+\.[a-z0-9_]+\.[a-z0-9_]+\b",
                    note_text,
                    flags=re.IGNORECASE,
                )
            ]
            if relation_hints:
                relation_hints_by_anchor[anchor_role] = relation_hints

        if not relation_hints_by_anchor:
            return dict(query_plan)

        existing_anchor_relations = {
            (
                self._normalize_relation_role(path.get("from_role")),
                str(path.get("relation") or "").strip(),
            )
            for path in relation_paths
            if self._normalize_relation_role(path.get("from_role")) in anchor_roles
        } | {
            (
                self._normalize_relation_role(path.get("to_role")),
                str(path.get("relation") or "").strip(),
            )
            for path in relation_paths
            if self._normalize_relation_role(path.get("to_role")) in anchor_roles
        }

        rewritten_paths = list(relation_paths)
        appended = False
        for anchor_role, relation_hints in relation_hints_by_anchor.items():
            for relation_hint in relation_hints:
                if (anchor_role, relation_hint) in existing_anchor_relations:
                    continue
                for relation_path in relation_paths:
                    relation = str(relation_path.get("relation") or "").strip()
                    if relation != relation_hint:
                        continue
                    from_role = self._normalize_relation_role(relation_path.get("from_role"))
                    to_role = self._normalize_relation_role(relation_path.get("to_role"))
                    if (
                        from_role in {"candidate_set", "count_set", "shared_answer"}
                        and to_role == "constraint_value"
                    ):
                        cloned = dict(relation_path)
                        cloned["from"] = anchor_role
                        cloned["from_role"] = anchor_role
                        rewritten_paths.append(cloned)
                        existing_anchor_relations.add((anchor_role, relation_hint))
                        appended = True
                        break
                    if (
                        to_role in {"candidate_set", "count_set", "shared_answer"}
                        and from_role == "constraint_value"
                    ):
                        cloned = dict(relation_path)
                        cloned["to"] = anchor_role
                        cloned["to_role"] = anchor_role
                        rewritten_paths.append(cloned)
                        existing_anchor_relations.add((anchor_role, relation_hint))
                        appended = True
                        break
        if not appended:
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["relation_paths"] = rewritten_paths
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Joined-count rewrite: materialize missing anchored constraint paths from join notes so each named anchor is explicitly represented in relation_paths."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_single_anchor_answer_target_count_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() not in {
            "count_over_direct_relation",
            "count_over_joined_set",
        }:
            return dict(query_plan)

        question_text, explicit_entities = self._split_task_question(task_question)
        interpretation = question_interpretation or self._build_question_interpretation(
            question_text=question_text,
            explicit_entities=explicit_entities,
            answer_target_phrase=self._extract_answer_target_phrase(question_text),
        )
        anchor_question_inputs = [
            item
            for item in (interpretation.get("question_inputs") or [])
            if isinstance(item, Mapping)
            and self._normalize_question_input_role(item.get("role_hint")) in {
                "anchor",
                "anchor_a",
                "anchor_b",
            }
        ]
        non_anchor_semantic_inputs = [
            item
            for item in (interpretation.get("question_inputs") or [])
            if isinstance(item, Mapping)
            and str(item.get("kind") or "").strip() not in {"named_entity", "answer_target"}
        ]
        anchor_is_attribute_or_class_value = any(
            self._normalize_question_input_kind(item.get("kind"))
            in {"attribute_value", "class_phrase", "type_constraint"}
            or str(item.get("reason") or "").strip().startswith("explicit_entity:attribute_value.")
            for item in anchor_question_inputs
        )

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
            and self._normalize_relation_role(item.get("role")) in {"anchor", "anchor_a", "anchor_b"}
        ]
        if len(anchored_entities) != 1:
            return dict(query_plan)
        if not anchor_is_attribute_or_class_value:
            return dict(query_plan)
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role")) or "anchor"

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return dict(query_plan)

        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        answer_target_token = self._normalize_variable_token(
            self._singularize_phrase(answer_target_phrase)
        )
        explicit_entity_tokens = {
            self._normalize_variable_token(entity)
            for entity in explicit_entities
            if self._normalize_variable_token(entity)
        }
        counted_variable_tokens = {
            self._normalize_variable_token(query_plan.get(field_name))
            for field_name in (
                "shared_answer_variable",
                "candidate_set_variable",
                "count_set_variable",
            )
            if self._normalize_variable_token(query_plan.get(field_name))
        }

        def _path_touches_counted_set(path: Mapping[str, Any]) -> bool:
            roles = {
                self._normalize_relation_role(path.get("from_role")),
                self._normalize_relation_role(path.get("to_role")),
            }
            tokens = {
                self._normalize_variable_token(path.get("from")),
                self._normalize_variable_token(path.get("to")),
            }
            return bool(
                roles & {"candidate_set", "count_set", "shared_answer", "answer"}
                or tokens & counted_variable_tokens
            )

        anchor_paths: list[dict[str, Any]] = []
        answer_target_filter_paths: list[dict[str, Any]] = []
        other_non_anchor_paths: list[dict[str, Any]] = []

        for path in relation_paths:
            from_role = self._normalize_relation_role(path.get("from_role"))
            to_role = self._normalize_relation_role(path.get("to_role"))
            touches_counted_set = _path_touches_counted_set(path)
            if (
                not non_anchor_semantic_inputs
                and touches_counted_set
                and "constraint_value" in {from_role, to_role}
            ):
                answer_target_filter_paths.append(path)
                continue
            relation_name = str(path.get("relation") or "").strip()
            constraint_raw = ""
            if from_role == "constraint_value":
                constraint_raw = str(path.get("from") or "").strip()
            elif to_role == "constraint_value":
                constraint_raw = str(path.get("to") or "").strip()
            constraint_token = self._normalize_variable_token(
                self._singularize_phrase(constraint_raw)
            )
            type_filter_token = ""
            if relation_name.startswith("type_filter:"):
                type_filter_token = self._normalize_variable_token(
                    self._singularize_phrase(relation_name.split(":", 1)[1].replace("_", " "))
                )
            filter_tokens = {
                self._normalize_variable_token(
                    self._singularize_phrase(str(path.get(endpoint_name) or "").strip())
                )
                for endpoint_name in ("from", "to")
            }
            filter_tokens.discard("")
            if (
                not non_anchor_semantic_inputs
                and touches_counted_set
                and (
                    (
                        constraint_token
                        and constraint_token == answer_target_token
                        and constraint_token not in explicit_entity_tokens
                    )
                    or (
                        type_filter_token
                        and type_filter_token == answer_target_token
                        and type_filter_token not in explicit_entity_tokens
                    )
                    or any(
                        filter_token
                        and filter_token == answer_target_token
                        and filter_token not in explicit_entity_tokens
                        for filter_token in filter_tokens
                    )
                )
            ):
                answer_target_filter_paths.append(path)
            elif self._path_touches_anchor_role(path, anchor_role=anchor_role) and touches_counted_set:
                anchor_paths.append(path)
            else:
                other_non_anchor_paths.append(path)

        if len(anchor_paths) != 1 or not answer_target_filter_paths or other_non_anchor_paths:
            return dict(query_plan)

        rewritten_anchor_path = copy.deepcopy(anchor_paths[0])
        from_role = self._normalize_relation_role(rewritten_anchor_path.get("from_role"))
        to_role = self._normalize_relation_role(rewritten_anchor_path.get("to_role"))
        from_token = self._normalize_variable_token(rewritten_anchor_path.get("from"))
        to_token = self._normalize_variable_token(rewritten_anchor_path.get("to"))
        if from_role == anchor_role and (
            to_role in {"candidate_set", "shared_answer", "count_set"}
            or to_token in counted_variable_tokens
        ):
            count_variable = (
                self._normalize_variable_token(rewritten_anchor_path.get("to"))
                or "count_set"
            )
            rewritten_anchor_path["to"] = count_variable
            rewritten_anchor_path["to_role"] = "count_set"
        elif to_role == anchor_role and (
            from_role in {"candidate_set", "shared_answer", "count_set"}
            or from_token in counted_variable_tokens
        ):
            count_variable = (
                self._normalize_variable_token(rewritten_anchor_path.get("from"))
                or "count_set"
            )
            rewritten_anchor_path["from"] = count_variable
            rewritten_anchor_path["from_role"] = "count_set"
        else:
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["query_shape"] = "count_over_direct_relation"
        rewritten_plan["shared_answer_variable"] = count_variable
        rewritten_plan["candidate_set_variable"] = count_variable
        rewritten_plan["count_set_variable"] = count_variable
        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": count_variable,
                    "notes": (
                        f"{anchor_role} directly constrains the counted set; the answer "
                        "target is a semantic description, not a second KG filter."
                    ),
                }
            ],
        }
        rewritten_plan["relation_paths"] = [rewritten_anchor_path]
        rewritten_plan["projection"] = ["count"]
        rewritten_plan["normalized_aliases"] = [
            {
                **dict(item),
                "reason": (
                    "semantic answer target phrase used only for relation selection"
                    if self._clean_question_input_surface(item.get("surface"))
                    == self._clean_question_input_surface(answer_target_phrase)
                    else str(item.get("reason") or "").strip()
                ),
            }
            for item in (rewritten_plan.get("normalized_aliases") or [])
            if isinstance(item, Mapping)
        ]
        rewritten_plan["plan_rationale"] = [
            "Single-anchor count rewrite: use one grounded relation from the anchor to the counted set.",
            "The answer target is a semantic description of the counted set, not a second KG filter.",
            "Do not add extra candidate-to-constraint paths unless a separate question input explicitly requires them.",
        ]
        rewritten_plan["strategy"] = (
            f"Count the distinct {count_variable} reached directly from the anchor "
            f"via {str(rewritten_anchor_path.get('relation') or '').strip()}. "
            "Treat the answer target only as a relation-selection hint."
        ).strip()
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_joined_count_answer_target_hint_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_joined_set":
            return dict(query_plan)

        question_text, explicit_entities = self._split_task_question(task_question)
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return dict(query_plan)
        if not self._count_answer_target_head_is_encoded_in_relation(
            answer_target_phrase=answer_target_phrase,
            relation_candidates=relation_paths,
        ):
            return dict(query_plan)

        answer_target_token = self._normalize_variable_token(
            self._singularize_phrase(answer_target_phrase)
        )
        explicit_entity_tokens = {
            self._normalize_variable_token(self._singularize_phrase(entity))
            for entity in explicit_entities
            if self._normalize_variable_token(self._singularize_phrase(entity))
        }
        removable_paths: list[dict[str, Any]] = []
        for relation_path in relation_paths:
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            roles = {from_role, to_role}
            if roles.isdisjoint({"candidate_set", "count_set", "shared_answer", "answer"}):
                continue
            if roles.isdisjoint({"type_set", "shared_type", "constraint_value"}):
                continue
            filter_tokens = {
                self._normalize_variable_token(
                    self._singularize_phrase(str(relation_path.get(endpoint_name) or "").strip())
                )
                for endpoint_name in ("from", "to")
            }
            filter_tokens.discard("")
            if (
                answer_target_token
                and any(
                    self._token_matches_answer_target(
                        filter_token,
                        answer_target_phrase,
                    )
                    for filter_token in filter_tokens
                )
                and not (filter_tokens & explicit_entity_tokens)
            ):
                removable_paths.append(relation_path)
        if not removable_paths:
            return dict(query_plan)

        rewritten_paths = [
            relation_path
            for relation_path in relation_paths
            if relation_path not in removable_paths
        ]
        remaining_roles = {
            self._normalize_relation_role(path.get(endpoint_role))
            for path in rewritten_paths
            for endpoint_role in ("from_role", "to_role")
        }
        rewritten_constraints = [
            dict(item)
            for item in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or [])
            if isinstance(item, Mapping)
            and self._normalize_relation_role(item.get("anchor_role")) in remaining_roles
        ]
        if not rewritten_constraints:
            rewritten_constraints = [
                dict(item)
                for item in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or [])
                if isinstance(item, Mapping)
            ]
        original_anchor_roles = {
            self._normalize_relation_role(item.get("role"))
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
            and self._normalize_relation_role(item.get("role")) in {"anchor", "anchor_a", "anchor_b"}
        }
        if len(original_anchor_roles) > 1:
            rewritten_path_anchor_roles = {
                self._normalize_relation_role(path.get(endpoint_role))
                for path in rewritten_paths
                for endpoint_role in ("from_role", "to_role")
                if self._normalize_relation_role(path.get(endpoint_role)) in original_anchor_roles
            }
            if not original_anchor_roles.issubset(rewritten_path_anchor_roles):
                return dict(query_plan)
            original_constraint_roles = {
                self._normalize_relation_role(item.get("anchor_role"))
                for item in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or [])
                if isinstance(item, Mapping)
                and self._normalize_relation_role(item.get("anchor_role")) in original_anchor_roles
            }
            rewritten_constraint_roles = {
                self._normalize_relation_role(item.get("anchor_role"))
                for item in rewritten_constraints
                if isinstance(item, Mapping)
                and self._normalize_relation_role(item.get("anchor_role")) in original_anchor_roles
            }
            if original_constraint_roles and not original_constraint_roles.issubset(
                rewritten_constraint_roles
            ):
                return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["relation_paths"] = rewritten_paths
        rewritten_plan["join_structure"] = {
            "type": str((query_plan.get("join_structure") or {}).get("type") or "count"),
            "anchor_constraints": rewritten_constraints,
        }
        strategy = str(rewritten_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Joined-count rewrite: the remaining grounded relations already encode the answer-target head semantics, so drop the redundant answer-target class filter."
        )
        rewritten_plan["strategy"] = (
            f"{strategy} Remove the redundant answer-target class filter because the remaining grounded relations already identify the counted answer class."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _count_shared_attribute_targets_shared_values(
        self,
        *,
        answer_target_phrase: str,
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        target_phrase = str(answer_target_phrase or "").strip()
        _, target_head = self._split_answer_target_compound_phrase(target_phrase)
        target_phrase = target_head or target_phrase
        shared_surfaces = [
            str(item.get("surface") or "").strip()
            for item in ((question_interpretation or {}).get("question_inputs") or [])
            if isinstance(item, Mapping)
            and str(item.get("kind") or "").strip() == "shared_attribute"
            and str(item.get("surface") or "").strip()
        ]
        for surface in shared_surfaces:
            surface_token = self._normalize_variable_token(surface)
            if surface_token and self._token_matches_answer_target(
                surface_token, target_phrase
            ):
                return True
        return False

    def _rewrite_count_shared_attribute_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        answer_target_phrase: str,
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return dict(query_plan)
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_joined_set":
            return dict(query_plan)

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return dict(query_plan)

        shared_value_token = ""
        shared_value_label = ""
        candidate_token = ""
        candidate_label = ""
        preferred_shared_value_token = ""
        preferred_shared_value_label = ""
        preferred_candidate_token = ""
        preferred_candidate_label = ""
        preferred_bridge_relation = ""
        bridge_candidates: dict[str, dict[str, str]] = {}
        bridge_relations_by_value: dict[str, str] = {}
        anchor_value_hits: dict[str, set[str]] = {}
        anchor_value_paths: list[dict[str, str]] = []
        anchor_roles = {"anchor", "anchor_a", "anchor_b"}
        shared_attribute_tokens = {
            self._normalize_variable_token(item.get("surface"))
            for item in ((question_interpretation or {}).get("question_inputs") or [])
            if isinstance(item, Mapping)
            and str(item.get("kind") or "").strip() == "shared_attribute"
            and self._normalize_variable_token(item.get("surface"))
        }
        generic_structural_tokens = {"candidate_set", "count_set", "shared_answer", "answer"}
        shared_answer_label = str(query_plan.get("shared_answer_variable") or "").strip()
        shared_answer_token = self._normalize_variable_token(shared_answer_label)

        for relation_path in relation_paths:
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            from_raw = str(relation_path.get("from") or "").strip()
            to_raw = str(relation_path.get("to") or "").strip()
            from_token = self._normalize_variable_token(from_raw)
            to_token = self._normalize_variable_token(to_raw)
            relation = str(relation_path.get("relation") or "").strip()

            if shared_attribute_tokens and not preferred_shared_value_token:
                for endpoint_token, endpoint_label, endpoint_role, other_token, other_label, other_role in (
                    (from_token, from_raw, from_role, to_token, to_raw, to_role),
                    (to_token, to_raw, to_role, from_token, from_raw, from_role),
                ):
                    if (
                        endpoint_token not in shared_attribute_tokens
                        or endpoint_role
                        not in {
                            "constraint_value",
                            "count_set",
                            "candidate_set",
                            "shared_answer",
                            "answer",
                        }
                    ):
                        continue
                    preferred_shared_value_token = endpoint_token
                    preferred_shared_value_label = endpoint_label
                    preferred_bridge_relation = relation
                    if (
                        other_token
                        and other_token != endpoint_token
                        and (
                            other_role
                            in {
                                "constraint_value",
                                "count_set",
                                "candidate_set",
                                "shared_answer",
                                "answer",
                            }
                            or self._token_matches_answer_target(
                                other_token,
                                answer_target_phrase,
                            )
                        )
                    ):
                        preferred_candidate_token = other_token
                        preferred_candidate_label = other_label
                    break

            if from_role in anchor_roles and to_role in {"constraint_value", "count_set"} and to_token:
                anchor_value_hits.setdefault(to_token, set()).add(from_role)
                anchor_value_paths.append(
                    {"relation": relation, "value_token": to_token, "value_label": to_raw}
                )
            if to_role in anchor_roles and from_role in {"constraint_value", "count_set"} and from_token:
                anchor_value_hits.setdefault(from_token, set()).add(to_role)
                anchor_value_paths.append(
                    {"relation": relation, "value_token": from_token, "value_label": from_raw}
                )

            if (
                from_role in {"shared_answer", "answer", "candidate_set"}
                and to_role in {"constraint_value", "count_set"}
                and from_token
                and to_token
            ):
                bridge_candidates.setdefault(
                    to_token,
                    {"value_label": to_raw, "candidate_token": from_token, "candidate_label": from_raw},
                )
                bridge_relations_by_value[to_token] = relation
            elif (
                to_role in {"shared_answer", "answer", "candidate_set"}
                and from_role in {"constraint_value", "count_set"}
                and from_token
                and to_token
            ):
                bridge_candidates.setdefault(
                    from_token,
                    {"value_label": from_raw, "candidate_token": to_token, "candidate_label": to_raw},
                )
                bridge_relations_by_value[from_token] = relation

        for value_token, relation in list(bridge_relations_by_value.items()):
            if value_token in anchor_value_hits:
                continue
            matching_anchor_roles = {
                anchor_role
                for anchor_path in anchor_value_paths
                if anchor_path.get("relation") == relation
                for anchor_role in anchor_value_hits.get(
                    str(anchor_path.get("value_token") or ""), set()
                )
            }
            if matching_anchor_roles:
                anchor_value_hits[value_token] = matching_anchor_roles

        ranked_value_tokens = sorted(
            anchor_value_hits,
            key=lambda token: (
                1 if token in bridge_candidates else 0,
                len(anchor_value_hits.get(token, set())),
                token,
            ),
            reverse=True,
        )
        if not preferred_shared_value_token and not ranked_value_tokens:
            return dict(query_plan)

        if preferred_shared_value_token:
            shared_value_token = preferred_shared_value_token
            shared_value_label = preferred_shared_value_label
            candidate_token = preferred_candidate_token
            candidate_label = preferred_candidate_label
            if not candidate_token:
                normalized_answer_target = str(answer_target_phrase or "").strip()
                _, target_head = self._split_answer_target_compound_phrase(
                    normalized_answer_target
                )
                target_candidates = [
                    phrase
                    for phrase in (
                        normalized_answer_target,
                        str(target_head or "").strip(),
                    )
                    if phrase
                ]
                for relation_path in relation_paths:
                    from_token = self._normalize_variable_token(relation_path.get("from"))
                    to_token = self._normalize_variable_token(relation_path.get("to"))
                    from_raw = str(relation_path.get("from") or "").strip()
                    to_raw = str(relation_path.get("to") or "").strip()
                    if (
                        from_token == shared_value_token
                        and to_token
                        and to_token != shared_value_token
                        and (
                            any(
                                self._token_matches_answer_target(
                                    to_token,
                                    candidate_phrase,
                                )
                                for candidate_phrase in target_candidates
                            )
                            or (
                                to_token in generic_structural_tokens
                                and shared_answer_token
                                and any(
                                    self._token_matches_answer_target(
                                        shared_answer_token,
                                        candidate_phrase,
                                    )
                                    for candidate_phrase in target_candidates
                                )
                            )
                        )
                    ):
                        if to_token in generic_structural_tokens and shared_answer_token:
                            candidate_token = shared_answer_token
                            candidate_label = shared_answer_label
                        else:
                            candidate_token = to_token
                            candidate_label = to_raw
                        break
                    if (
                        to_token == shared_value_token
                        and from_token
                        and from_token != shared_value_token
                        and (
                            any(
                                self._token_matches_answer_target(
                                    from_token,
                                    candidate_phrase,
                                )
                                for candidate_phrase in target_candidates
                            )
                            or (
                                from_token in generic_structural_tokens
                                and shared_answer_token
                                and any(
                                    self._token_matches_answer_target(
                                        shared_answer_token,
                                        candidate_phrase,
                                    )
                                    for candidate_phrase in target_candidates
                                )
                            )
                        )
                    ):
                        if from_token in generic_structural_tokens and shared_answer_token:
                            candidate_token = shared_answer_token
                            candidate_label = shared_answer_label
                        else:
                            candidate_token = from_token
                            candidate_label = from_raw
                        break
        else:
            shared_value_token = ranked_value_tokens[0]
            shared_value_label = (
                bridge_candidates.get(shared_value_token, {}).get("value_label")
                or next(
                    (
                        str(path.get("to") or "").strip()
                        for path in relation_paths
                        if self._normalize_variable_token(path.get("to")) == shared_value_token
                    ),
                    "",
                )
                or next(
                    (
                        str(path.get("from") or "").strip()
                        for path in relation_paths
                        if self._normalize_variable_token(path.get("from")) == shared_value_token
                    ),
                    "",
                )
            )
            candidate_token = bridge_candidates.get(shared_value_token, {}).get("candidate_token", "")
            candidate_label = bridge_candidates.get(shared_value_token, {}).get("candidate_label", "")

        if not shared_value_label:
            return dict(query_plan)

        rewritten_plan = copy.deepcopy(dict(query_plan))
        count_shared_values = self._count_shared_attribute_targets_shared_values(
            answer_target_phrase=answer_target_phrase,
            question_interpretation=question_interpretation,
        )
        counted_label = shared_value_label if count_shared_values else candidate_label
        if not counted_label:
            return dict(query_plan)

        rewritten_plan["shared_answer_variable"] = (
            shared_value_label if count_shared_values else counted_label
        )
        rewritten_plan["count_set_variable"] = counted_label
        if candidate_label:
            rewritten_plan["candidate_set_variable"] = candidate_label

        rewritten_paths: list[dict[str, Any]] = []
        shared_value_relation = (
            preferred_bridge_relation
            or bridge_relations_by_value.get(shared_value_token, "")
        )
        for relation_path in relation_paths:
            rewritten_path = dict(relation_path)
            from_role = self._normalize_relation_role(rewritten_path.get("from_role"))
            to_role = self._normalize_relation_role(rewritten_path.get("to_role"))
            from_token = self._normalize_variable_token(rewritten_path.get("from"))
            to_token = self._normalize_variable_token(rewritten_path.get("to"))

            if candidate_token:
                if (
                    from_token == candidate_token
                    and from_role
                    in {
                        "constraint_value",
                        "count_set",
                        "shared_answer",
                        "answer",
                        "candidate_set",
                    }
                    and from_role not in anchor_roles
                ):
                    rewritten_path["from_role"] = "candidate_set"
                if (
                    to_token == candidate_token
                    and to_role
                    in {
                        "constraint_value",
                        "count_set",
                        "shared_answer",
                        "answer",
                        "candidate_set",
                    }
                    and to_role not in anchor_roles
                ):
                    rewritten_path["to_role"] = "candidate_set"
                if (
                    from_token in generic_structural_tokens
                    and from_role in {"constraint_value", "count_set", "shared_answer", "answer", "candidate_set"}
                    and from_role not in anchor_roles
                    and candidate_label
                ):
                    rewritten_path["from"] = candidate_label
                    rewritten_path["from_role"] = "candidate_set"
                if (
                    to_token in generic_structural_tokens
                    and to_role in {"constraint_value", "count_set", "shared_answer", "answer", "candidate_set"}
                    and to_role not in anchor_roles
                    and candidate_label
                ):
                    rewritten_path["to"] = candidate_label
                    rewritten_path["to_role"] = "candidate_set"

            if (
                from_token == shared_value_token
                and from_role not in anchor_roles
                and from_role in {"constraint_value", "shared_answer", "answer", "candidate_set", "count_set"}
            ):
                rewritten_path["from"] = shared_value_label
                rewritten_path["from_role"] = (
                    "count_set" if count_shared_values else "constraint_value"
                )
            if (
                to_token == shared_value_token
                and to_role not in anchor_roles
                and to_role in {"constraint_value", "shared_answer", "answer", "candidate_set", "count_set"}
            ):
                rewritten_path["to"] = shared_value_label
                rewritten_path["to_role"] = (
                    "count_set" if count_shared_values else "constraint_value"
                )
            elif (
                str(rewritten_path.get("relation") or "").strip()
                == shared_value_relation
                and to_role in {"constraint_value", "count_set"}
                and self._normalize_relation_role(rewritten_path.get("from_role")) in anchor_roles
            ):
                rewritten_path["to"] = shared_value_label
                rewritten_path["to_role"] = (
                    "count_set" if count_shared_values else "constraint_value"
                )
            elif (
                str(rewritten_path.get("relation") or "").strip()
                == shared_value_relation
                and from_role in {"constraint_value", "count_set"}
                and self._normalize_relation_role(rewritten_path.get("to_role")) in anchor_roles
            ):
                rewritten_path["from"] = shared_value_label
                rewritten_path["from_role"] = (
                    "count_set" if count_shared_values else "constraint_value"
                )

            rewritten_paths.append(rewritten_path)

        anchor_constraints = [
            dict(constraint)
            for constraint in (
                (rewritten_plan.get("join_structure") or {}).get("anchor_constraints") or []
            )
            if isinstance(constraint, Mapping)
        ]
        rewritten_constraints: list[dict[str, Any]] = []
        for constraint in anchor_constraints:
            anchor_role = self._normalize_relation_role(constraint.get("anchor_role"))
            target_variable = shared_value_label
            touches_shared_value = any(
                anchor_role in {
                    self._normalize_relation_role(path.get("from_role")),
                    self._normalize_relation_role(path.get("to_role")),
                }
                and shared_value_token in {
                    self._normalize_variable_token(path.get("from")),
                    self._normalize_variable_token(path.get("to")),
                }
                for path in rewritten_paths
            )
            if touches_shared_value:
                target_variable = shared_value_label
            elif candidate_label:
                target_variable = candidate_label
            rewritten_constraint = dict(constraint)
            rewritten_constraint["constrains_variable"] = target_variable
            rewritten_constraints.append(rewritten_constraint)

        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": rewritten_constraints,
        }
        rewritten_plan["relation_paths"] = rewritten_paths
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        if count_shared_values:
            plan_rationale.append(
                f"Shared-attribute count scaffold: count the shared attribute/value set {shared_value_label} rather than the intermediate entity bridge."
            )
            rewritten_plan["strategy"] = (
                f"{str(rewritten_plan.get('strategy') or '').strip()} Count the shared attribute/value set {shared_value_label} instead of the intermediate candidate entity set."
            ).strip()
        else:
            plan_rationale.append(
                f"Shared-attribute count scaffold: keep {shared_value_label} as an intermediate shared-attribute filter and count the matching candidate entity set {counted_label}."
            )
            rewritten_plan["strategy"] = (
                f"{str(rewritten_plan.get('strategy') or '').strip()} Keep the shared attribute/value set {shared_value_label} only as an intermediate filter and count the candidate entity set {counted_label}."
            ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _rewrite_cvg_release_region_intersection_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "multi_anchor_intersection":
            return dict(query_plan)
        if str(query_plan.get("answer_mode") or "").strip().lower() != "entity":
            return dict(query_plan)

        question_text, _ = self._split_task_question(task_question)
        lower_text = str(question_text or "").lower()
        if "released where" not in lower_text or "developed by" not in lower_text:
            return dict(query_plan)

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) < 2:
            return dict(query_plan)

        role_to_entity: dict[str, dict[str, Any]] = {}
        for anchored_entity in anchored_entities:
            role = self._normalize_relation_role(anchored_entity.get("role"))
            if role in {"anchor", "anchor_a", "anchor_b"} and role not in role_to_entity:
                role_to_entity[role] = anchored_entity
        if "anchor_a" not in role_to_entity and "anchor" in role_to_entity:
            role_to_entity["anchor_a"] = role_to_entity["anchor"]
        if "anchor_b" not in role_to_entity:
            remaining = [
                entity
                for entity in anchored_entities
                if entity is not role_to_entity.get("anchor_a")
            ]
            if remaining:
                role_to_entity["anchor_b"] = remaining[0]
        if "anchor_a" not in role_to_entity or "anchor_b" not in role_to_entity:
            return dict(query_plan)

        answer_variable = self._extract_primary_projection_variable(query_plan) or "answer"
        if self._normalize_variable_token(answer_variable) in {
            "",
            "answer",
            "shared_answer",
            "candidate_set",
            "count_set",
        }:
            answer_variable = "answer"
        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["answer_target_phrase"] = (
            str(query_plan.get("answer_target_phrase") or "").strip() or "region"
        )
        rewritten_plan["shared_answer_variable"] = "candidate_set"
        rewritten_plan["candidate_set_variable"] = "candidate_set"
        rewritten_plan["allow_exploratory_predicates"] = False
        rewritten_plan["projection"] = [answer_variable]
        rewritten_plan["join_structure"] = {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "candidate_set",
                    "notes": (
                        "The first anchor constrains the shared game-version set via "
                        "the distribution-system relation."
                    ),
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "candidate_set",
                    "notes": (
                        "The second anchor constrains the same shared game-version set "
                        "via the developer relation."
                    ),
                },
            ],
        }
        rewritten_plan["relation_paths"] = [
            {
                "relation": "cvg.computer_game_distribution_system.games_distributed",
                "direction": "reverse",
                "from": "anchor_a",
                "to": "candidate_set",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "reason": (
                    "Retrieve the shared game-version set distributed by the first anchor."
                ),
            },
            {
                "relation": "cvg.cvg_developer.game_versions_developed",
                "direction": "reverse",
                "from": "anchor_b",
                "to": "candidate_set",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "reason": (
                    "Constrain the same shared game-version set to the versions "
                    "developed by the second anchor."
                ),
            },
            {
                "relation": "cvg.game_version.regions",
                "direction": "forward",
                "from": "candidate_set",
                "to": answer_variable,
                "from_role": "candidate_set",
                "to_role": "answer",
                "grounding_source": "curated",
                "reason": "Project the intersected game-version set to its release regions.",
            },
        ]
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Release-region CVG rewrite: intersect the grounded game-version set "
            "constrained by both anchors first, then project that shared game-version "
            "set to release regions."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        rewritten_plan["strategy"] = (
            "Intersect the grounded game-version set constrained by the first anchor "
            "and second anchor, then project that shared game-version set to regions."
        )
        return self._normalize_pal_query_plan(rewritten_plan)

    def _normalize_anchored_entities(
        self,
        raw_entities: Sequence[Mapping[str, Any]],
        *,
        query_shape: str,
    ) -> list[dict[str, str]]:
        normalized_entities: list[dict[str, str]] = []
        total = len(raw_entities)
        for index, anchored_entity in enumerate(raw_entities):
            role = self._normalize_anchored_entity_role(
                raw_role=anchored_entity.get("role"),
                index=index,
                total=total,
                query_shape=query_shape,
            )
            normalized_entities.append(
                {
                    "surface": anchored_entity["surface"],
                    "chosen_alias": anchored_entity["chosen_alias"],
                    "role": role,
                }
            )
        return normalized_entities

    def _normalize_anchored_entity_role(
        self,
        *,
        raw_role: Any,
        index: int,
        total: int,
        query_shape: str,
    ) -> str:
        normalized_role = self._normalize_relation_role(raw_role)
        if normalized_role in self._STABLE_RELATION_ROLES:
            if (
                normalized_role == "anchor"
                and query_shape in {"multi_anchor_intersection", "count_over_joined_set", "shared_type_intersection"}
                and total > 1
            ):
                return "anchor_a" if index == 0 else "anchor_b" if index == 1 else "constraint_value"
            return normalized_role
        if query_shape == "superlative_chain":
            return "anchor_value"
        if query_shape in {"multi_anchor_intersection", "count_over_joined_set", "shared_type_intersection"}:
            if index == 0:
                return "anchor_a"
            if index == 1:
                return "anchor_b"
            return "constraint_value"
        return "anchor"

    def _normalize_query_shape_value(self, raw_value: Any) -> str:
        query_shape = str(raw_value or "").strip().lower()
        if query_shape == "single_anchor_direct_lookup":
            query_shape = "single_anchor_lookup"
        return query_shape if query_shape in self._SUPPORTED_QUERY_SHAPES else ""

    def _normalize_join_structure(
        self,
        raw_value: Any,
        *,
        query_shape: str,
        anchored_entities: Sequence[Mapping[str, Any]],
        shared_answer_variable: str,
        candidate_set_variable: str,
        count_set_variable: str,
    ) -> dict[str, Any]:
        if not isinstance(raw_value, Mapping):
            return {}
        join_type = str(raw_value.get("type") or "").strip().lower()
        anchor_constraints: list[dict[str, str]] = []
        for index, raw_constraint in enumerate(raw_value.get("anchor_constraints") or []):
            if not isinstance(raw_constraint, Mapping):
                continue
            anchor_role = self._normalize_join_anchor_role(
                raw_role=raw_constraint.get("anchor_role"),
                index=index,
                query_shape=query_shape,
                anchored_entities=anchored_entities,
            )
            constrains_variable = self._normalize_join_constraint_variable(
                raw_value=raw_constraint.get("constrains_variable"),
                shared_answer_variable=shared_answer_variable,
                candidate_set_variable=candidate_set_variable,
                count_set_variable=count_set_variable,
            )
            notes = str(raw_constraint.get("notes") or "").strip()
            if not anchor_role and not constrains_variable:
                continue
            anchor_constraints.append(
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": constrains_variable,
                    "notes": notes,
                }
            )
        normalized_join_structure: dict[str, Any] = {}
        if join_type:
            normalized_join_structure["type"] = join_type
        if anchor_constraints:
            normalized_join_structure["anchor_constraints"] = anchor_constraints
        return normalized_join_structure

    def _normalize_join_anchor_role(
        self,
        *,
        raw_role: Any,
        index: int,
        query_shape: str,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> str:
        normalized_role = self._normalize_relation_role(raw_role)
        if normalized_role in self._STABLE_RELATION_ROLES:
            return normalized_role
        if index < len(anchored_entities):
            anchored_role = self._normalize_relation_role(
                anchored_entities[index].get("role")
            )
            if anchored_role:
                return anchored_role
        if query_shape == "superlative_chain":
            return "anchor_value"
        if query_shape in {"multi_anchor_intersection", "count_over_joined_set", "shared_type_intersection"}:
            if index == 0:
                return "anchor_a"
            if index == 1:
                return "anchor_b"
            return "constraint_value"
        return "anchor"

    def _normalize_join_constraint_variable(
        self,
        *,
        raw_value: Any,
        shared_answer_variable: str,
        candidate_set_variable: str,
        count_set_variable: str,
    ) -> str:
        raw_text = str(raw_value or "").strip()
        token = self._normalize_variable_token(raw_text)
        if token in {"shared_answer", "sharedanswer", "answer"} and shared_answer_variable:
            return shared_answer_variable
        if token in {
            "candidate",
            "candidate_set",
            "candidateanswer",
            "candidateanswerset",
            "candidateset",
        } and candidate_set_variable:
            return candidate_set_variable
        if token in {"count", "count_set", "counted_set", "countset"} and count_set_variable:
            return count_set_variable
        return raw_text

    def _expand_projected_answer_intersection_paths(
        self,
        *,
        query_shape: str,
        shared_answer_variable: str,
        join_structure: Mapping[str, Any],
        relation_paths: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        if query_shape != "multi_anchor_intersection":
            return [dict(item) for item in relation_paths]

        shared_answer_token = self._normalize_variable_token(shared_answer_variable)
        if not shared_answer_token:
            return [dict(item) for item in relation_paths]

        branch_variables: list[str] = []
        for constraint in join_structure.get("anchor_constraints") or []:
            if not isinstance(constraint, Mapping):
                continue
            anchor_role = self._normalize_relation_role(constraint.get("anchor_role"))
            if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
                continue
            branch_variable = str(constraint.get("constrains_variable") or "").strip()
            branch_token = self._normalize_variable_token(branch_variable)
            if (
                not branch_variable
                or not branch_token
                or branch_token == shared_answer_token
                or branch_variable in branch_variables
            ):
                continue
            branch_variables.append(branch_variable)
        if len(branch_variables) < 2:
            return [dict(item) for item in relation_paths]

        branch_tokens = {
            self._normalize_variable_token(variable): variable
            for variable in branch_variables
        }
        generic_projection_index: Optional[int] = None
        generic_projection_path: Optional[dict[str, str]] = None
        existing_branch_projection_tokens: set[str] = set()

        for index, relation_path in enumerate(relation_paths):
            path = dict(relation_path)
            from_token = self._normalize_variable_token(path.get("from"))
            to_token = self._normalize_variable_token(path.get("to"))
            from_role = self._normalize_relation_role(path.get("from_role"))
            to_role = self._normalize_relation_role(path.get("to_role"))

            if to_token == shared_answer_token and from_token in branch_tokens:
                existing_branch_projection_tokens.add(from_token)
            elif from_token == shared_answer_token and to_token in branch_tokens:
                existing_branch_projection_tokens.add(to_token)

            if generic_projection_path is not None:
                continue
            if (
                to_token == shared_answer_token
                and from_role in {"candidate_set", "shared_answer", "answer"}
                and from_token
                and from_token not in branch_tokens
                and from_token != shared_answer_token
            ):
                generic_projection_index = index
                generic_projection_path = path
            elif (
                from_token == shared_answer_token
                and to_role in {"candidate_set", "shared_answer", "answer"}
                and to_token
                and to_token not in branch_tokens
                and to_token != shared_answer_token
            ):
                generic_projection_index = index
                generic_projection_path = path

        if generic_projection_path is None or generic_projection_index is None:
            return [dict(item) for item in relation_paths]
        if len(existing_branch_projection_tokens) >= len(branch_variables):
            return [dict(item) for item in relation_paths]

        expanded_paths: list[dict[str, str]] = []
        for index, relation_path in enumerate(relation_paths):
            if index != generic_projection_index:
                expanded_paths.append(dict(relation_path))
                continue
            template = dict(generic_projection_path)
            template_from_token = self._normalize_variable_token(template.get("from"))
            template_to_token = self._normalize_variable_token(template.get("to"))
            for branch_variable in branch_variables:
                branch_token = self._normalize_variable_token(branch_variable)
                if branch_token in existing_branch_projection_tokens:
                    continue
                expanded_path = dict(template)
                if template_to_token == shared_answer_token:
                    expanded_path["from"] = branch_variable
                    expanded_path["from_role"] = "candidate_set"
                    expanded_path["to"] = shared_answer_variable
                    expanded_path["to_role"] = "shared_answer"
                elif template_from_token == shared_answer_token:
                    expanded_path["from"] = shared_answer_variable
                    expanded_path["from_role"] = "shared_answer"
                    expanded_path["to"] = branch_variable
                    expanded_path["to_role"] = "candidate_set"
                expanded_paths.append(expanded_path)
        return expanded_paths

    def _ensure_repair_loop_accepted(
        self,
        *,
        generated_tool_name: str,
        query_plan: Mapping[str, Any] | None = None,
        repair_loop_log: Mapping[str, Any],
    ) -> None:
        final_verdict = str(repair_loop_log.get("final_verdict") or "").strip()
        accepted_attempt = repair_loop_log.get("accepted_attempt")
        if accepted_attempt is not None or final_verdict in {
            "accepted",
            "accepted_best_effort",
        }:
            return
        last_verdict = str(
            repair_loop_log.get("last_verdict") or final_verdict or "no_accepted_candidate"
        ).strip()
        last_reasons = list(repair_loop_log.get("last_reasons") or [])
        self._emit_generated_tools_event(
            {
                "event": "pal_repair_loop_rejected",
                "mode": "pal",
                "tool_name": generated_tool_name,
                "final_verdict": final_verdict or "no_accepted_candidate",
                "last_verdict": last_verdict,
                "last_reasons": last_reasons,
                "total_attempts": repair_loop_log.get("total_attempts"),
                "repair_used": repair_loop_log.get("repair_used"),
            }
        )
        self._maybe_record_family_policy_candidate(
            generated_tool_name=generated_tool_name,
            query_plan=query_plan or {},
            failure_reason=f"pal_query_not_accepted:{last_verdict}",
            repair_loop_log=repair_loop_log,
        )
        raise AgentUnknownException(f"pal_query_not_accepted:{last_verdict}")

    def _build_best_executing_candidate_metadata(
        self,
        *,
        generated_code: str,
        invocation_result: Any,
        query_plan: Mapping[str, Any],
        query_text: str,
        verdict: PlausibilityVerdict,
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> dict[str, Any]:
        result_dict = invocation_result.payload if invocation_result.success else None
        binding_count = (
            self._get_result_binding_count(result_dict)
            if isinstance(result_dict, Mapping)
            else 0
        )
        scalar_count = (
            self._extract_scalar_count_value(result_dict)
            if isinstance(result_dict, Mapping)
            else None
        )
        anchor_results = list(anchor_probe_results or [])
        all_anchors_found = bool(anchor_results) and all(
            bool(result.found) for result in anchor_results
        )
        resolved_anchor_count = sum(
            1 for result in anchor_results if str(result.resolved_entity_id or "").strip()
        )
        query_text_upper = str(query_text or "").upper()
        plan_validation_errors = list(
            self._validate_pal_query_candidate(
                raw_output=generated_code,
                generated_code=generated_code,
                query_text=query_text,
                query_texts=self._extract_sparql_query_texts(generated_code),
                query_plan=query_plan,
            )
        )

        score = 0
        if invocation_result.success and invocation_result.payload is not None:
            score += 10
        if binding_count > 0:
            score += 20
        if scalar_count is not None:
            if scalar_count > 0:
                score += 15
            else:
                score += 5
        if all_anchors_found:
            score += 15
        score += resolved_anchor_count * 5
        if "ORDER BY" in query_text_upper and "LIMIT 1" in query_text_upper:
            score += 10

        soft_bonus = {
            "repairable_bad_count_set": 8,
            "repairable_bad_join": 6,
            "repairable_bad_superlative_structure": 4,
        }
        score += soft_bonus.get(str(verdict.verdict or "").strip(), 0)

        for reason in verdict.reasons:
            reason_text = str(reason or "").strip()
            if reason_text.startswith("anchor_not_found"):
                score -= 100
            elif reason_text == "count_set_path_empty":
                score -= 50
            elif reason_text == "ordering_attribute_path_exploratory":
                score -= 60
            elif reason_text.startswith("ambiguous_anchor_surface_binding"):
                score -= 5
            elif reason_text.startswith("count_anchor_path_low_support"):
                score -= 10
            elif reason_text == "count_query_dynamic_chain_too_weak":
                score -= 10
            elif (
                reason_text == "count_query_unverified_type_constraint"
                or reason_text.startswith("count_query_unverified_type_constraint:")
            ):
                score -= 30
            elif (
                reason_text == "count_query_counts_wrong_variable"
                or reason_text.startswith("count_query_counts_wrong_variable:")
            ):
                score -= 30
            elif reason_text.startswith("count_answer_target_unenforced:"):
                score -= 80
        if plan_validation_errors:
            score -= 120

        return {
            "generated_code": generated_code,
            "invocation_result": invocation_result,
            "query_plan": copy.deepcopy(dict(query_plan)),
            "query_text": query_text,
            "verdict": str(verdict.verdict or "").strip(),
            "verdict_reasons": list(verdict.reasons),
            "binding_count": binding_count,
            "scalar_count": scalar_count,
            "answer_mode": str(query_plan.get("answer_mode") or "").strip().lower(),
            "query_shape": str(query_plan.get("query_shape") or "").strip().lower(),
            "all_anchors_found": all_anchors_found,
            "resolved_anchor_count": resolved_anchor_count,
            "has_order_by_limit": "ORDER BY" in query_text_upper and "LIMIT 1" in query_text_upper,
            "plan_validation_errors": plan_validation_errors,
            "score": score,
        }

    def _should_accept_best_executing_candidate(
        self,
        *,
        candidate_metadata: Mapping[str, Any],
    ) -> bool:
        verdict = str(candidate_metadata.get("verdict") or "").strip()
        if verdict not in {
            "repairable_bad_count_set",
            "repairable_bad_join",
            "repairable_bad_superlative_structure",
        }:
            return False

        binding_count = int(candidate_metadata.get("binding_count") or 0)
        scalar_count = candidate_metadata.get("scalar_count")
        answer_mode = str(candidate_metadata.get("answer_mode") or "").strip().lower()
        query_shape = str(candidate_metadata.get("query_shape") or "").strip().lower()
        reasons = [
            str(reason or "").strip()
            for reason in (candidate_metadata.get("verdict_reasons") or [])
        ]

        if any(reason.startswith("anchor_not_found") for reason in reasons):
            return False
        if "ordering_attribute_path_exploratory" in reasons:
            return False
        if not bool(candidate_metadata.get("all_anchors_found")):
            return False
        if candidate_metadata.get("plan_validation_errors"):
            return False
        if binding_count <= 0 and scalar_count is None:
            return False
        if answer_mode == "count" and scalar_count is None:
            return False
        if any(
            reason == "count_query_counts_wrong_variable"
            or reason.startswith("count_query_counts_wrong_variable:")
            or reason == "count_query_unverified_type_constraint"
            or reason.startswith("count_query_unverified_type_constraint:")
            or reason.startswith("count_answer_target_unenforced:")
            for reason in reasons
        ):
            return False
        if query_shape == "superlative_chain":
            if binding_count <= 0 or not bool(candidate_metadata.get("has_order_by_limit")):
                return False
            return int(candidate_metadata.get("score") or 0) >= 25

        if answer_mode == "count":
            try:
                scalar_count_int = int(scalar_count)
            except Exception:
                return False
            # Do not salvage positive count answers when the validator still says the
            # counted set is weak. Those are the main source of wrong trusted counts.
            if scalar_count_int > 0:
                return False
            if any(
                reason.startswith("ambiguous_anchor_surface_binding:")
                or reason.startswith("count_anchor_path_low_support:")
                or reason == "count_query_dynamic_chain_too_weak"
                or reason == "count_query_zero_with_live_anchor_paths"
                or reason == "count_query_all_relation_paths_exploratory"
                or reason == "count_query_has_unanchored_exploratory_constraint"
                for reason in reasons
            ):
                return False
            return int(candidate_metadata.get("score") or 0) >= 20

        return int(candidate_metadata.get("score") or 0) >= 25

    def _normalize_ordering_attribute(self, raw_value: Any) -> dict[str, str]:
        if not isinstance(raw_value, Mapping):
            return {}
        relation = str(raw_value.get("relation") or "").strip()
        direction = str(raw_value.get("direction") or "").strip().lower()
        source_variable = str(raw_value.get("source_variable") or "").strip()
        attribute_variable = str(raw_value.get("attribute_variable") or "").strip()
        normalized: dict[str, str] = {}
        if relation:
            normalized["relation"] = relation
        if direction in {"forward", "reverse"}:
            normalized["direction"] = direction
        if source_variable:
            normalized["source_variable"] = source_variable
        if attribute_variable:
            normalized["attribute_variable"] = attribute_variable
        return normalized

    def _resolve_superlative_answer_projection_variable(
        self,
        *,
        relation_paths: Sequence[Mapping[str, Any]],
    ) -> str:
        for relation_path in relation_paths:
            if not isinstance(relation_path, Mapping):
                continue
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            if from_role == "answer":
                token = self._normalize_variable_token(relation_path.get("from"))
                if token:
                    return token
            if to_role == "answer":
                token = self._normalize_variable_token(relation_path.get("to"))
                if token:
                    return token
        return ""

    def _normalize_plan_relation_paths(
        self,
        raw_items: Any,
        *,
        query_shape: str,
        answer_mode: str,
        answer_target_phrase: str,
        anchored_entities: Sequence[Mapping[str, Any]],
        shared_answer_variable: str,
        candidate_set_variable: str,
        count_set_variable: str,
        ordering_attribute: Mapping[str, str],
        allow_exploratory: bool,
    ) -> list[dict[str, str]]:
        if not isinstance(raw_items, list):
            return []
        normalized_items: list[dict[str, str]] = []
        for raw_item in raw_items:
            normalized_item = self._normalize_relation_contract_item(
                raw_item,
                query_shape=query_shape,
                answer_mode=answer_mode,
                answer_target_phrase=answer_target_phrase,
                anchored_entities=anchored_entities,
                shared_answer_variable=shared_answer_variable,
                candidate_set_variable=candidate_set_variable,
                count_set_variable=count_set_variable,
                ordering_attribute=ordering_attribute,
                allow_exploratory=allow_exploratory,
            )
            if normalized_item is not None:
                normalized_items.append(normalized_item)
        if query_shape == "single_anchor_lookup":
            normalized_items = self._retag_single_anchor_direct_answer_paths(
                normalized_items,
                answer_mode=answer_mode,
            )
        return normalized_items

    def _normalize_grounded_relation_candidates(
        self,
        *,
        relation_candidates: Sequence[Mapping[str, Any]],
        query_shape: str,
        answer_mode: str,
        answer_target_phrase: str,
        entities: Sequence[str],
    ) -> list[dict[str, str]]:
        anchored_entities: list[dict[str, str]] = []
        if entities:
            total = len(entities)
            for index, entity in enumerate(entities):
                role = "anchor"
                if total > 1:
                    if index == 0:
                        role = "anchor_a"
                    elif index == 1:
                        role = "anchor_b"
                anchored_entities.append(
                    {
                        "surface": entity,
                        "chosen_alias": entity,
                        "role": role,
                    }
                )
        normalized_candidates: list[dict[str, str]] = []
        for raw_candidate in relation_candidates:
            normalized_candidate = self._normalize_relation_contract_item(
                raw_candidate,
                query_shape=query_shape,
                answer_mode=answer_mode,
                answer_target_phrase=answer_target_phrase,
                anchored_entities=anchored_entities,
                shared_answer_variable="",
                candidate_set_variable="",
                count_set_variable="",
                ordering_attribute={},
                allow_exploratory=False,
            )
            if normalized_candidate is None:
                continue
            support = str(raw_candidate.get("support") or "").strip()
            use_when = str(raw_candidate.get("use_when") or "").strip()
            if support:
                normalized_candidate["support"] = support
            if use_when:
                normalized_candidate["use_when"] = use_when
            normalized_candidates.append(normalized_candidate)
        if query_shape == "count_over_direct_relation":
            normalized_candidates = self._retag_direct_count_bridge_candidates(
                normalized_candidates,
                answer_target_phrase=answer_target_phrase,
            )
        return normalized_candidates

    def _retag_direct_count_bridge_candidates(
        self,
        relation_candidates: Sequence[Mapping[str, Any]],
        *,
        answer_target_phrase: str,
    ) -> list[dict[str, str]]:
        answer_tokens = self._semantic_hint_tokens(answer_target_phrase)
        if not answer_tokens:
            return [
                dict(candidate)
                for candidate in relation_candidates
                if isinstance(candidate, Mapping)
            ]

        normalized_candidates = [
            dict(candidate)
            for candidate in relation_candidates
            if isinstance(candidate, Mapping)
        ]
        anchor_roles = {"anchor", "anchor_a", "anchor_b"}

        def _candidate_tokens(candidate: Mapping[str, Any]) -> set[str]:
            return self._semantic_hint_tokens(
                " ".join(
                    str(candidate.get(field) or "").strip()
                    for field in (
                        "relation",
                        "from",
                        "to",
                    )
                )
            )

        for downstream in normalized_candidates:
            downstream_tokens = _candidate_tokens(downstream)
            if not (answer_tokens & downstream_tokens):
                continue

            from_role = self._normalize_relation_role(downstream.get("from_role"))
            to_role = self._normalize_relation_role(downstream.get("to_role"))
            from_matches_answer = self._endpoint_semantically_matches_answer_target(
                endpoint_token=self._normalize_variable_token(downstream.get("from")),
                answer_target_phrase=answer_target_phrase,
            )
            to_matches_answer = self._endpoint_semantically_matches_answer_target(
                endpoint_token=self._normalize_variable_token(downstream.get("to")),
                answer_target_phrase=answer_target_phrase,
            )

            if to_matches_answer and not from_matches_answer:
                answer_side = "to"
            elif from_matches_answer and not to_matches_answer:
                answer_side = "from"
            elif from_role in anchor_roles and to_role not in anchor_roles:
                answer_side = "to"
            elif to_role in anchor_roles and from_role not in anchor_roles:
                answer_side = "from"
            else:
                continue

            pivot_tokens = downstream_tokens - answer_tokens
            if not pivot_tokens:
                continue

            for upstream in normalized_candidates:
                if upstream is downstream:
                    continue
                upstream_from_role = self._normalize_relation_role(upstream.get("from_role"))
                upstream_to_role = self._normalize_relation_role(upstream.get("to_role"))
                if {upstream_from_role, upstream_to_role}.isdisjoint(anchor_roles):
                    continue
                if answer_tokens & _candidate_tokens(upstream):
                    continue
                if not (pivot_tokens & _candidate_tokens(upstream)):
                    continue

                if upstream_from_role in anchor_roles:
                    upstream["to_role"] = "candidate_set"
                elif upstream_to_role in anchor_roles:
                    upstream["from_role"] = "candidate_set"
                else:
                    continue

                if answer_side == "to":
                    downstream["from_role"] = "candidate_set"
                    downstream["to_role"] = "count_set"
                else:
                    downstream["from_role"] = "count_set"
                    downstream["to_role"] = "candidate_set"
                break

        return normalized_candidates

    def _retag_single_anchor_direct_answer_paths(
        self,
        relation_paths: Sequence[Mapping[str, Any]],
        *,
        answer_mode: str,
    ) -> list[dict[str, str]]:
        normalized_paths = [
            dict(path)
            for path in relation_paths
            if isinstance(path, Mapping)
        ]
        if str(answer_mode or "").strip().lower() != "entity" or len(normalized_paths) != 1:
            return normalized_paths

        anchor_roles = {"anchor", "anchor_a", "anchor_b"}
        terminal_roles = {"candidate_set", "shared_answer", "answer"}
        path = normalized_paths[0]
        from_role = self._normalize_relation_role(path.get("from_role"))
        to_role = self._normalize_relation_role(path.get("to_role"))
        if from_role in anchor_roles and to_role in terminal_roles - {"answer"}:
            path["to_role"] = "answer"
            path["to"] = "answer"
        elif to_role in anchor_roles and from_role in terminal_roles - {"answer"}:
            path["from_role"] = "answer"
            path["from"] = "answer"
        return normalized_paths

    def _prune_redundant_grounded_relation_candidates(
        self,
        *,
        relation_candidates: Sequence[Mapping[str, Any]],
        question_text: str,
        answer_target_phrase: str,
    ) -> list[dict[str, str]]:
        if not relation_candidates:
            return []
        lower_text = str(question_text or "").lower()
        answer_target_lower = str(answer_target_phrase or "").strip().lower()
        prefers_biology_breed_temperament = (
            "temperament" in answer_target_lower
            and any(
                token in lower_text
                for token in ("dog", "dogs", "breed", "breeds", "animal", "temperament")
            )
            and "pet" not in lower_text
        )
        if not prefers_biology_breed_temperament:
            return [dict(candidate) for candidate in relation_candidates if isinstance(candidate, Mapping)]

        pruned_candidates: list[dict[str, str]] = []
        saw_biology_breed_temperament = any(
            str(candidate.get("relation") or "").strip() == "biology.animal_breed.temperament"
            for candidate in relation_candidates
            if isinstance(candidate, Mapping)
        )
        for candidate in relation_candidates:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if (
                saw_biology_breed_temperament
                and relation == "pets.pet_breed.temperament"
            ):
                continue
            pruned_candidates.append(dict(candidate))
        return pruned_candidates

    def _normalize_relation_contract_item(
        self,
        raw_item: Any,
        *,
        query_shape: str,
        answer_mode: str,
        answer_target_phrase: str,
        anchored_entities: Sequence[Mapping[str, Any]],
        shared_answer_variable: str,
        candidate_set_variable: str,
        count_set_variable: str,
        ordering_attribute: Mapping[str, str],
        allow_exploratory: bool,
    ) -> Optional[dict[str, str]]:
        if not isinstance(raw_item, Mapping):
            return None
        relation = str(raw_item.get("relation") or "").strip()
        direction = str(raw_item.get("direction") or "").strip().lower()
        from_raw = str(raw_item.get("from") or "").strip()
        to_raw = str(raw_item.get("to") or "").strip()
        if not relation or direction not in {"forward", "reverse"} or not from_raw or not to_raw:
            return None
        normalized_item: dict[str, str] = {
            "relation": relation,
            "direction": direction,
            "from": from_raw,
            "to": to_raw,
        }
        reason = str(raw_item.get("reason") or "").strip()
        if reason:
            normalized_item["reason"] = reason
        grounding_source = self._infer_grounding_source(
            raw_grounding_source=raw_item.get("grounding_source"),
            support=raw_item.get("support"),
            allow_exploratory=allow_exploratory,
        )
        if grounding_source:
            normalized_item["grounding_source"] = grounding_source
        normalized_item["from_role"] = self._infer_relation_endpoint_role(
            explicit_role=raw_item.get("from_role"),
            raw_endpoint=from_raw,
            endpoint_side="from",
            direction=direction,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            anchored_entities=anchored_entities,
            shared_answer_variable=shared_answer_variable,
            candidate_set_variable=candidate_set_variable,
            count_set_variable=count_set_variable,
            ordering_attribute=ordering_attribute,
        )
        normalized_item["to_role"] = self._infer_relation_endpoint_role(
            explicit_role=raw_item.get("to_role"),
            raw_endpoint=to_raw,
            endpoint_side="to",
            direction=direction,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            anchored_entities=anchored_entities,
            shared_answer_variable=shared_answer_variable,
            candidate_set_variable=candidate_set_variable,
            count_set_variable=count_set_variable,
            ordering_attribute=ordering_attribute,
        )
        coerced_item = self._coerce_relation_roles_for_query_shape(
            normalized_item=normalized_item,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            ordering_attribute=ordering_attribute,
            anchored_entities=anchored_entities,
            candidate_set_variable=candidate_set_variable,
        )
        if query_shape == "count_over_direct_relation":
            from_matches_answer = self._token_matches_answer_target(
                self._normalize_variable_token(from_raw),
                answer_target_phrase,
            )
            to_matches_answer = self._token_matches_answer_target(
                self._normalize_variable_token(to_raw),
                answer_target_phrase,
            )
            if from_matches_answer and coerced_item.get("to_role") in {"", "count_set"}:
                coerced_item["from_role"] = "count_set"
                coerced_item["to_role"] = "anchor"
            elif to_matches_answer and coerced_item.get("from_role") in {"", "count_set"}:
                coerced_item["to_role"] = "count_set"
                coerced_item["from_role"] = "anchor"
        return coerced_item

    def _coerce_relation_roles_for_query_shape(
        self,
        *,
        normalized_item: Mapping[str, str],
        query_shape: str,
        answer_mode: str = "",
        answer_target_phrase: str = "",
        ordering_attribute: Mapping[str, str],
        anchored_entities: Sequence[Mapping[str, Any]],
        candidate_set_variable: str,
    ) -> dict[str, str]:
        coerced_item = dict(normalized_item)
        relation = str(coerced_item.get("relation") or "").strip()
        relation_tail = self._normalize_variable_token(relation.split(".")[-1])
        from_token = self._normalize_variable_token(coerced_item.get("from"))
        to_token = self._normalize_variable_token(coerced_item.get("to"))
        ordering_relation = str(ordering_attribute.get("relation") or "").strip()
        candidate_set_token = self._normalize_variable_token(candidate_set_variable)
        from_is_anchor_like = bool(
            self._resolve_anchor_roles_from_label(
                raw_label=coerced_item.get("from"),
                anchored_entities=anchored_entities,
            )
        )
        to_is_anchor_like = bool(
            self._resolve_anchor_roles_from_label(
                raw_label=coerced_item.get("to"),
                anchored_entities=anchored_entities,
            )
        )

        if query_shape == "superlative_chain":
            category_like_tokens = {"category", "categories", "type", "types", "kind", "kinds"}
            from_matches_answer_target = self._endpoint_semantically_matches_answer_target(
                endpoint_token=from_token,
                answer_target_phrase=answer_target_phrase,
            )
            to_matches_answer_target = self._endpoint_semantically_matches_answer_target(
                endpoint_token=to_token,
                answer_target_phrase=answer_target_phrase,
            )
            candidate_like_tokens = {
                token
                for token in {
                    candidate_set_token,
                    "candidate_set",
                    "shared_answer",
                    "answer",
                }
                if token
            }
            if (
                from_token in {"category", "shared_type", "type_set"}
                and to_token in candidate_like_tokens
            ):
                coerced_item["from_role"] = "anchor"
                coerced_item["to_role"] = "candidate_set"
            elif (
                to_token in {"category", "shared_type", "type_set"}
                and from_token in candidate_like_tokens
            ):
                coerced_item["to_role"] = "anchor"
                coerced_item["from_role"] = "candidate_set"
            if (
                self._is_anchor_side(
                    endpoint_side="from",
                    direction=str(coerced_item.get("direction") or "").strip().lower(),
                )
                and (
                    relation_tail in category_like_tokens
                    or to_token in {"category", "shared_type", "type_set"}
                )
                and not to_is_anchor_like
            ):
                coerced_item["from_role"] = "anchor"
                coerced_item["to_role"] = "type_set"
            elif (
                self._is_anchor_side(
                    endpoint_side="to",
                    direction=str(coerced_item.get("direction") or "").strip().lower(),
                )
                and (
                    relation_tail in category_like_tokens
                    or from_token in {"category", "shared_type", "type_set"}
                )
                and not from_is_anchor_like
            ):
                coerced_item["to_role"] = "anchor"
                coerced_item["from_role"] = "type_set"
            elif from_token in {"category", "shared_type", "type_set"} and not to_is_anchor_like:
                coerced_item["from_role"] = "type_set"
                coerced_item["to_role"] = "candidate_set"
            elif to_token in {"category", "shared_type", "type_set"} and not from_is_anchor_like:
                coerced_item["to_role"] = "type_set"
                coerced_item["from_role"] = "candidate_set"
            is_ordering_relation = bool(
                relation
                and (
                    relation == ordering_relation
                    or self._looks_like_ordering_endpoint(relation_tail)
                    or self._looks_like_ordering_endpoint(to_token)
                )
            )
            if is_ordering_relation:
                coerced_item["from_role"] = "candidate_set"
                coerced_item["to_role"] = "ordering_attribute"
            elif (
                coerced_item.get("from_role") == "anchor"
                and coerced_item.get("to_role") == "answer"
            ):
                coerced_item["to_role"] = "candidate_set"
            elif (
                self._is_anchor_side(
                    endpoint_side="from",
                    direction=str(coerced_item.get("direction") or "").strip().lower(),
                )
                and (
                    (candidate_set_token and to_token == candidate_set_token)
                    or (
                        not candidate_set_token
                        and coerced_item.get("from_role") == "candidate_set"
                        and coerced_item.get("to_role") == "candidate_set"
                    )
                )
                and not to_is_anchor_like
                and not from_matches_answer_target
            ):
                coerced_item["from_role"] = "anchor"
                coerced_item["to_role"] = "candidate_set"
            elif (
                self._is_anchor_side(
                    endpoint_side="to",
                    direction=str(coerced_item.get("direction") or "").strip().lower(),
                )
                and (
                    (candidate_set_token and from_token == candidate_set_token)
                    or (
                        not candidate_set_token
                        and coerced_item.get("from_role") == "candidate_set"
                        and coerced_item.get("to_role") == "candidate_set"
                    )
                )
                and not from_is_anchor_like
                and not to_matches_answer_target
            ):
                coerced_item["to_role"] = "anchor"
                coerced_item["from_role"] = "candidate_set"
            elif (
                from_token in {"category", "shared_type", "type_set"}
                and (
                    to_token in candidate_like_tokens
                    or self._normalize_variable_token(coerced_item.get("to_role")) == "candidate_set"
                )
            ):
                coerced_item["from_role"] = "anchor"
                coerced_item["to_role"] = "candidate_set"
            elif (
                to_token in {"category", "shared_type", "type_set"}
                and (
                    from_token in candidate_like_tokens
                    or self._normalize_variable_token(coerced_item.get("from_role")) == "candidate_set"
                )
            ):
                coerced_item["to_role"] = "anchor"
                coerced_item["from_role"] = "candidate_set"
            elif from_token == candidate_set_token and not to_is_anchor_like:
                coerced_item["from_role"] = "candidate_set"
                coerced_item["to_role"] = (
                    "answer" if to_matches_answer_target else "anchor"
                )
            elif to_token == candidate_set_token and not from_is_anchor_like:
                coerced_item["to_role"] = "candidate_set"
                coerced_item["from_role"] = (
                    "answer" if from_matches_answer_target else "anchor"
                )

        return coerced_item

    def _endpoint_semantically_matches_answer_target(
        self,
        *,
        endpoint_token: str,
        answer_target_phrase: str,
    ) -> bool:
        normalized_endpoint = self._normalize_variable_token(endpoint_token)
        if not normalized_endpoint:
            return False
        if self._token_matches_answer_target(normalized_endpoint, answer_target_phrase):
            return True
        normalized_answer = self._normalize_variable_token(answer_target_phrase)
        if normalized_answer and self._variable_token_matches_endpoint_token(
            variable_token=normalized_answer,
            endpoint_token=normalized_endpoint,
        ):
            return True
        for answer_part in re.split(r"[\s_/.-]+", answer_target_phrase):
            normalized_part = self._normalize_variable_token(answer_part)
            if not normalized_part:
                continue
            if self._variable_token_matches_endpoint_token(
                variable_token=normalized_part,
                endpoint_token=normalized_endpoint,
            ):
                return True
        return False

    def _infer_grounding_source(
        self,
        *,
        raw_grounding_source: Any,
        support: Any,
        allow_exploratory: bool,
    ) -> str:
        grounding_source = str(raw_grounding_source or "").strip().lower()
        if grounding_source in {"curated", "dynamic_probe", "exploratory"}:
            return grounding_source
        support_text = str(support or "").strip().lower()
        if support_text.startswith("dynamic_probe_"):
            return "dynamic_probe"
        if support_text:
            return "curated"
        return "exploratory" if allow_exploratory else "curated"

    def _infer_relation_endpoint_role(
        self,
        *,
        explicit_role: Any,
        raw_endpoint: str,
        endpoint_side: str,
        direction: str,
        query_shape: str,
        answer_mode: str,
        answer_target_phrase: str,
        anchored_entities: Sequence[Mapping[str, Any]],
        shared_answer_variable: str,
        candidate_set_variable: str,
        count_set_variable: str,
        ordering_attribute: Mapping[str, str],
    ) -> str:
        endpoint_token = self._normalize_variable_token(raw_endpoint)
        if not endpoint_token:
            return ""

        for anchored_entity in anchored_entities:
            anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
            surface_tokens = {
                self._normalize_variable_token(anchored_entity.get("surface")),
                self._normalize_variable_token(anchored_entity.get("chosen_alias")),
                anchor_role,
            }
            if endpoint_token in surface_tokens:
                return anchor_role or "anchor"

        normalized_explicit_role = self._normalize_relation_role(explicit_role)
        if normalized_explicit_role in self._STABLE_RELATION_ROLES:
            if query_shape == "count_over_joined_set" and answer_mode == "count":
                candidate_set_token = self._normalize_variable_token(candidate_set_variable)
                if (
                    normalized_explicit_role in {"candidate_set", "shared_answer"}
                    and self._token_matches_answer_target(
                        endpoint_token,
                        answer_target_phrase,
                    )
                    and endpoint_token != candidate_set_token
                ):
                    return "count_set"
            return normalized_explicit_role

        if answer_mode == "count":
            exact_role_candidates = [
                (
                    self._normalize_variable_token(count_set_variable),
                    "count_set",
                ),
                (
                    self._normalize_variable_token(candidate_set_variable),
                    "candidate_set",
                ),
                (
                    self._normalize_variable_token(shared_answer_variable),
                    "shared_answer",
                ),
                (
                    self._normalize_variable_token(ordering_attribute.get("source_variable")),
                    "candidate_set",
                ),
                (
                    self._normalize_variable_token(ordering_attribute.get("attribute_variable")),
                    "ordering_attribute",
                ),
            ]
            fuzzy_role_candidates = [
                (
                    self._normalize_variable_token(candidate_set_variable),
                    "candidate_set",
                ),
                (
                    self._normalize_variable_token(count_set_variable),
                    "count_set",
                ),
                (
                    self._normalize_variable_token(shared_answer_variable),
                    "shared_answer",
                ),
                (
                    self._normalize_variable_token(ordering_attribute.get("source_variable")),
                    "candidate_set",
                ),
                (
                    self._normalize_variable_token(ordering_attribute.get("attribute_variable")),
                    "ordering_attribute",
                ),
            ]
        else:
            exact_role_candidates = [
                (
                    self._normalize_variable_token(shared_answer_variable),
                    "shared_answer",
                ),
                (
                    self._normalize_variable_token(candidate_set_variable),
                    "candidate_set",
                ),
                (
                    self._normalize_variable_token(count_set_variable),
                    "count_set",
                ),
                (
                    self._normalize_variable_token(ordering_attribute.get("source_variable")),
                    "candidate_set",
                ),
                (
                    self._normalize_variable_token(ordering_attribute.get("attribute_variable")),
                    "ordering_attribute",
                ),
            ]
            fuzzy_role_candidates = exact_role_candidates
        for variable_token, variable_role in exact_role_candidates:
            if endpoint_token == variable_token and variable_role:
                return variable_role
        for variable_token, variable_role in fuzzy_role_candidates:
            if not variable_token or not variable_role:
                continue
            if self._variable_token_matches_endpoint_token(
                variable_token=variable_token,
                endpoint_token=endpoint_token,
            ):
                return variable_role

        if query_shape == "superlative_chain":
            if self._looks_like_ordering_endpoint(endpoint_token):
                return "ordering_attribute"
            if self._looks_like_value_endpoint(endpoint_token):
                return "anchor_value"
            return "candidate_set"

        if query_shape == "shared_type_intersection":
            if "type" in endpoint_token:
                return "shared_type" if endpoint_side == "to" else "type_set"
            if self._token_matches_answer_target(endpoint_token, answer_target_phrase):
                return "shared_answer"
            if self._looks_like_value_endpoint(endpoint_token):
                return "constraint_value"
            return "shared_answer"

        if query_shape == "count_over_direct_relation":
            if self._token_matches_answer_target(endpoint_token, answer_target_phrase):
                return "count_set"
            if self._looks_like_value_endpoint(endpoint_token):
                return "constraint_value"
            if self._is_anchor_side(endpoint_side=endpoint_side, direction=direction):
                return "anchor"
            return "count_set"

        if query_shape == "count_over_joined_set":
            if self._token_matches_answer_target(endpoint_token, answer_target_phrase):
                return "shared_answer" if answer_mode != "count" else "count_set"
            if self._looks_like_value_endpoint(endpoint_token):
                return "constraint_value"
            return "shared_answer" if answer_mode != "count" else "candidate_set"

        if query_shape == "multi_anchor_intersection":
            if self._token_matches_answer_target(endpoint_token, answer_target_phrase):
                return "shared_answer" if answer_mode != "count" else "count_set"
            if self._looks_like_value_endpoint(endpoint_token):
                return "constraint_value"
            return "shared_answer" if answer_mode != "count" else "count_set"

        if query_shape in {
            "single_anchor_lookup",
            "single_anchor_chain_lookup",
            "containment_or_ownership_lookup",
        }:
            if self._is_anchor_side(endpoint_side=endpoint_side, direction=direction):
                return "anchor"
            return (
                "candidate_set"
                if query_shape == "single_anchor_chain_lookup"
                else "answer"
            )

        if self._token_matches_answer_target(endpoint_token, answer_target_phrase):
            return "answer"
        if self._looks_like_value_endpoint(endpoint_token):
            return "constraint_value"
        return endpoint_token

    def _variable_token_matches_endpoint_token(
        self,
        *,
        variable_token: str,
        endpoint_token: str,
    ) -> bool:
        normalized_variable = self._normalize_variable_token(variable_token)
        normalized_endpoint = self._normalize_variable_token(endpoint_token)
        if not normalized_variable or not normalized_endpoint:
            return False
        if normalized_variable == normalized_endpoint:
            return True

        variable_parts = {
            part
            for part in normalized_variable.split("_")
            if part and part not in {"candidate", "shared", "count", "set", "answer", "ordering", "attribute", "source"}
        }
        endpoint_parts = {
            part
            for part in normalized_endpoint.split("_")
            if part and part not in {"candidate", "shared", "count", "set", "answer", "ordering", "attribute", "source"}
        }
        if not variable_parts or not endpoint_parts:
            return False
        return variable_parts <= endpoint_parts or endpoint_parts <= variable_parts

    def _is_anchor_side(self, *, endpoint_side: str, direction: str) -> bool:
        if direction == "forward":
            return endpoint_side == "from"
        return endpoint_side == "to"

    def _anchor_position_for_relation_endpoint(
        self,
        *,
        endpoint_side: str,
        direction: str,
        endpoint_role: str = "",
        counterpart_role: str = "",
    ) -> str:
        normalized_endpoint_role = self._normalize_relation_role(endpoint_role)
        normalized_counterpart_role = self._normalize_relation_role(counterpart_role)
        if (
            direction == "reverse"
            and normalized_endpoint_role in {"anchor", "anchor_a", "anchor_b"}
            and normalized_counterpart_role
            in {"candidate_set", "shared_answer", "answer", "count_set"}
        ):
            return "object"
        return (
            "subject"
            if self._is_anchor_side(endpoint_side=endpoint_side, direction=direction)
            else "object"
        )

    def _token_matches_answer_target(self, token: str, answer_target_phrase: str) -> bool:
        if not token or not answer_target_phrase:
            return False
        full_answer_token = self._normalize_variable_token(answer_target_phrase)
        answer_tokens = {
            self._normalize_variable_token(part)
            for part in re.split(r"[\s_/.-]+", answer_target_phrase)
            if part.strip()
        }
        if full_answer_token:
            answer_tokens.add(full_answer_token)
        answer_tokens.discard("")
        if {"film_character", "fictional_character"} & answer_tokens:
            answer_tokens |= {"film_character", "fictional_character"}
        answer_tokens |= {
            self._normalize_variable_token(self._singularize_surface_token(part))
            for part in tuple(answer_tokens)
            if part
        }
        answer_tokens.discard("")
        token_variants = {self._normalize_variable_token(token)}
        if {"film_character", "fictional_character"} & token_variants:
            token_variants |= {"film_character", "fictional_character"}
        if token_variants & answer_tokens:
            return True
        singular_token = self._normalize_variable_token(
            self._singularize_surface_token(token.replace("_", " "))
        )
        if singular_token:
            token_variants.add(singular_token)
            if singular_token in {"film_character", "fictional_character"}:
                token_variants |= {"film_character", "fictional_character"}
        return bool(token_variants & answer_tokens)

    def _looks_like_value_endpoint(self, token: str) -> bool:
        return any(hint in token for hint in self._VALUE_ROLE_HINTS)

    def _looks_like_ordering_endpoint(self, token: str) -> bool:
        normalized_token = self._normalize_variable_token(token)
        if not normalized_token:
            return False
        parts = [part for part in normalized_token.split("_") if part]
        if normalized_token in self._ORDERING_ROLE_HINTS:
            return True
        return any(part in self._ORDERING_ROLE_HINTS for part in parts)

    def _normalize_relation_role(self, raw_value: Any) -> str:
        token = self._normalize_variable_token(raw_value)
        if not token:
            return ""
        return self._CANONICAL_ROLE_SYNONYMS.get(token, token)

    def _normalize_variable_token(self, raw_value: Any) -> str:
        value = str(raw_value or "").strip()
        if not value:
            return ""
        value = value.lstrip("?")
        value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
        return value

    def _relation_role_aliases(
        self,
        *,
        role: Any,
        raw_label: Any,
        anchor_count: int,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        aliases: set[str] = set()
        normalized_role = self._normalize_relation_role(role)
        normalized_raw = self._normalize_relation_role(raw_label)
        if normalized_role:
            aliases.add(normalized_role)
        if normalized_raw:
            aliases.add(normalized_raw)
        expanded = set(aliases)
        resolved_anchor_roles = self._resolve_anchor_roles_from_label(
            raw_label=raw_label,
            anchored_entities=anchored_entities,
        )
        for alias in list(aliases):
            if alias == "anchor":
                if anchor_count > 1:
                    if resolved_anchor_roles:
                        expanded.update(resolved_anchor_roles)
                    else:
                        expanded.add("anchor")
                else:
                    expanded.add("anchor")
            elif alias in {"anchor_a", "anchor_b"}:
                if anchor_count <= 1:
                    expanded.add("anchor")
            elif alias in {"answer", "shared_answer", "candidate_set", "count_set"}:
                expanded.update({"answer", "shared_answer", "candidate_set", "count_set"})
            elif alias in {"constraint_value", "anchor_value"}:
                expanded.update({"constraint_value", "anchor_value"})
            elif alias in {"ordering_attribute"}:
                expanded.add("constraint_value")
            elif alias in {"shared_type", "type_set"}:
                expanded.update({"shared_type", "type_set"})
        if anchor_count > 1 and resolved_anchor_roles:
            expanded.update(resolved_anchor_roles)
        return expanded

    def _resolve_anchor_roles_from_label(
        self,
        *,
        raw_label: Any,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        raw_token = self._normalize_variable_token(raw_label)
        if not raw_token:
            return set()
        raw_compact = raw_token.replace("_", "")
        resolved_roles: set[str] = set()
        for anchored_entity in anchored_entities:
            if not isinstance(anchored_entity, Mapping):
                continue
            anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
            if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
                continue
            candidate_tokens = {
                self._normalize_variable_token(anchored_entity.get("surface")),
                self._normalize_variable_token(anchored_entity.get("chosen_alias")),
                anchor_role,
            }
            candidate_tokens.discard("")
            for candidate_token in candidate_tokens:
                candidate_compact = candidate_token.replace("_", "")
                if (
                    raw_token == candidate_token
                    or raw_compact == candidate_compact
                    or raw_compact.startswith(candidate_compact)
                    or candidate_compact.startswith(raw_compact)
                ):
                    resolved_roles.add(anchor_role)
                    break
        return resolved_roles

    def _label_is_clean_anchor_reference(
        self,
        *,
        raw_label: Any,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> bool:
        raw_token = self._normalize_variable_token(raw_label)
        if not raw_token:
            return False
        raw_compact = raw_token.replace("_", "")
        for anchored_entity in anchored_entities:
            if not isinstance(anchored_entity, Mapping):
                continue
            candidate_tokens = {
                self._normalize_variable_token(anchored_entity.get("surface")),
                self._normalize_variable_token(anchored_entity.get("chosen_alias")),
                self._normalize_variable_token(anchored_entity.get("role")),
            }
            candidate_tokens.discard("")
            for candidate_token in candidate_tokens:
                candidate_compact = candidate_token.replace("_", "")
                if raw_token == candidate_token or raw_compact == candidate_compact:
                    return True
        return False

    def _relation_contract_match_details(
        self,
        *,
        planned_path: Mapping[str, Any],
        grounded_candidate: Mapping[str, Any],
        query_shape: str,
        anchor_count: int,
        anchored_entities: Sequence[Mapping[str, Any]],
        query_plan: Mapping[str, Any] | None = None,
    ) -> tuple[bool, str]:
        plan_from_aliases = self._relation_role_aliases(
            role=planned_path.get("from_role"),
            raw_label=planned_path.get("from"),
            anchor_count=anchor_count,
            anchored_entities=anchored_entities,
        )
        plan_to_aliases = self._relation_role_aliases(
            role=planned_path.get("to_role"),
            raw_label=planned_path.get("to"),
            anchor_count=anchor_count,
            anchored_entities=anchored_entities,
        )
        candidate_from_aliases = self._relation_role_aliases(
            role=grounded_candidate.get("from_role"),
            raw_label=grounded_candidate.get("from"),
            anchor_count=anchor_count,
            anchored_entities=anchored_entities,
        )
        candidate_to_aliases = self._relation_role_aliases(
            role=grounded_candidate.get("to_role"),
            raw_label=grounded_candidate.get("to"),
            anchor_count=anchor_count,
            anchored_entities=anchored_entities,
        )
        planned_from_role = self._normalize_relation_role(planned_path.get("from_role"))
        planned_to_role = self._normalize_relation_role(planned_path.get("to_role"))
        candidate_from_role = self._normalize_relation_role(
            grounded_candidate.get("from_role")
        )
        candidate_to_role = self._normalize_relation_role(
            grounded_candidate.get("to_role")
        )
        same_direction = str(planned_path.get("direction") or "") == str(
            grounded_candidate.get("direction") or ""
        )
        from_matches = bool(plan_from_aliases & candidate_from_aliases)
        to_matches = bool(plan_to_aliases & candidate_to_aliases)
        specific_anchor_aliases = {"anchor_a", "anchor_b"}
        if not from_matches and (
            (
                "constraint_value" in candidate_from_aliases
                and plan_from_aliases & specific_anchor_aliases
            )
            or (
                "constraint_value" in plan_from_aliases
                and candidate_from_aliases & specific_anchor_aliases
            )
        ):
            from_matches = True
        if not to_matches and (
            (
                "constraint_value" in candidate_to_aliases
                and plan_to_aliases & specific_anchor_aliases
            )
            or (
                "constraint_value" in plan_to_aliases
                and candidate_to_aliases & specific_anchor_aliases
            )
        ):
            to_matches = True
        answer_like_aliases = {"answer", "shared_answer", "candidate_set", "count_set"}
        if query_shape == "count_over_direct_relation":
            if not from_matches and (
                plan_from_aliases & answer_like_aliases
                and candidate_from_aliases & answer_like_aliases
            ):
                from_matches = True
            if not to_matches and (
                plan_to_aliases & answer_like_aliases
                and candidate_to_aliases & answer_like_aliases
            ):
                to_matches = True
            if (
                not from_matches
                and "candidate_set" in plan_from_aliases
                and "anchor" in candidate_from_aliases
                and self._count_plan_preserves_anchor_to_candidate_pivot(
                    query_plan=query_plan,
                    planned_path=planned_path,
                )
            ):
                from_matches = True
            if (
                not to_matches
                and "candidate_set" in plan_to_aliases
                and "anchor" in candidate_to_aliases
                and self._count_plan_preserves_anchor_to_candidate_pivot(
                    query_plan=query_plan,
                    planned_path=planned_path,
                )
            ):
                to_matches = True
        if query_shape == "superlative_chain":
            if not from_matches and (
                "candidate_set" in plan_from_aliases
                and candidate_from_aliases & {"anchor", "answer", "candidate_set"}
            ):
                from_matches = True
            if not to_matches and (
                plan_to_aliases & {"answer", "constraint_value"}
                and candidate_to_aliases & {"answer", "constraint_value"}
            ):
                to_matches = True
            if not to_matches and (
                "ordering_attribute" in plan_to_aliases
                and candidate_to_aliases & {"answer", "constraint_value"}
            ):
                to_matches = True
        if query_shape in {
            "multi_anchor_intersection",
            "count_over_joined_set",
            "shared_type_intersection",
        }:
            if not from_matches and (
                query_shape == "multi_anchor_intersection"
                and plan_from_aliases & specific_anchor_aliases
                and "anchor" in candidate_from_aliases
            ):
                from_matches = True
            if not to_matches and (
                query_shape == "multi_anchor_intersection"
                and plan_to_aliases & specific_anchor_aliases
                and "anchor" in candidate_to_aliases
            ):
                to_matches = True
            if not from_matches and (
                query_shape == "count_over_joined_set"
                and plan_from_aliases & specific_anchor_aliases
                and "anchor" in candidate_from_aliases
                and not (candidate_from_aliases & specific_anchor_aliases)
            ):
                from_matches = True
            if not to_matches and (
                query_shape == "count_over_joined_set"
                and plan_to_aliases & specific_anchor_aliases
                and "anchor" in candidate_to_aliases
                and not (candidate_to_aliases & specific_anchor_aliases)
            ):
                to_matches = True
            if not from_matches and (
                query_shape == "count_over_joined_set"
                and plan_from_aliases & specific_anchor_aliases
                and candidate_from_aliases & specific_anchor_aliases
                and planned_to_role in {"candidate_set", "shared_answer", "answer"}
                and candidate_to_role in {"candidate_set", "shared_answer", "answer"}
                and self._label_is_clean_anchor_reference(
                    raw_label=planned_path.get("from"),
                    anchored_entities=anchored_entities,
                )
                and self._label_is_clean_anchor_reference(
                    raw_label=grounded_candidate.get("from"),
                    anchored_entities=anchored_entities,
                )
            ):
                from_matches = True
            if not to_matches and (
                query_shape == "count_over_joined_set"
                and plan_to_aliases & specific_anchor_aliases
                and candidate_to_aliases & specific_anchor_aliases
                and planned_from_role in {"candidate_set", "shared_answer", "answer"}
                and candidate_from_role in {"candidate_set", "shared_answer", "answer"}
                and self._label_is_clean_anchor_reference(
                    raw_label=planned_path.get("to"),
                    anchored_entities=anchored_entities,
                )
                and self._label_is_clean_anchor_reference(
                    raw_label=grounded_candidate.get("to"),
                    anchored_entities=anchored_entities,
                )
            ):
                to_matches = True
            if not from_matches and (
                plan_from_aliases & specific_anchor_aliases
                and candidate_from_aliases & answer_like_aliases
            ):
                from_matches = True
            if not to_matches and (
                plan_to_aliases & specific_anchor_aliases
                and candidate_to_aliases & answer_like_aliases
            ):
                to_matches = True
            if not from_matches and (
                query_shape == "count_over_joined_set"
                and "candidate_set" in plan_from_aliases
                and "anchor" in candidate_from_aliases
                and self._joined_count_plan_preserves_anchor_to_candidate_pivot(
                    query_plan=query_plan,
                    planned_path=planned_path,
                )
            ):
                from_matches = True
            if not to_matches and (
                query_shape == "count_over_joined_set"
                and "candidate_set" in plan_to_aliases
                and "anchor" in candidate_to_aliases
                and self._joined_count_plan_preserves_anchor_to_candidate_pivot(
                    query_plan=query_plan,
                    planned_path=planned_path,
                )
            ):
                to_matches = True
            if not from_matches and (
                ("count_set" in plan_from_aliases and "constraint_value" in candidate_from_aliases)
                or ("constraint_value" in plan_from_aliases and "count_set" in candidate_from_aliases)
            ):
                from_matches = True
            if not to_matches and (
                ("count_set" in plan_to_aliases and "constraint_value" in candidate_to_aliases)
                or ("constraint_value" in plan_to_aliases and "count_set" in candidate_to_aliases)
            ):
                to_matches = True
        if same_direction and from_matches and to_matches:
            return True, "accepted_role_match"
        swapped_from_matches = bool(plan_from_aliases & candidate_to_aliases)
        swapped_to_matches = bool(plan_to_aliases & candidate_from_aliases)
        if not same_direction and swapped_from_matches and swapped_to_matches:
            return True, "accepted_role_swap_match"
        mismatch_parts: list[str] = []
        if not same_direction:
            mismatch_parts.append("direction")
        if not from_matches:
            mismatch_parts.append("from_role")
        if not to_matches:
            mismatch_parts.append("to_role")
        return False, "mismatch:" + ",".join(mismatch_parts)

    def _count_plan_preserves_anchor_to_candidate_pivot(
        self,
        *,
        query_plan: Mapping[str, Any] | None,
        planned_path: Mapping[str, Any],
    ) -> bool:
        if not isinstance(query_plan, Mapping):
            return False
        planned_roles = {
            self._normalize_variable_token(planned_path.get("from_role")),
            self._normalize_variable_token(planned_path.get("to_role")),
        }
        if "candidate_set" not in planned_roles:
            return False
        if "count_set" not in planned_roles:
            return False
        relation_paths = query_plan.get("relation_paths") or []
        for relation_path in relation_paths:
            if not isinstance(relation_path, Mapping):
                continue
            if relation_path is planned_path:
                continue
            roles = {
                self._normalize_variable_token(relation_path.get("from_role")),
                self._normalize_variable_token(relation_path.get("to_role")),
            }
            if "anchor" in roles and "candidate_set" in roles:
                return True
        return False

    def _joined_count_plan_preserves_anchor_to_candidate_pivot(
        self,
        *,
        query_plan: Mapping[str, Any] | None,
        planned_path: Mapping[str, Any],
    ) -> bool:
        if not isinstance(query_plan, Mapping):
            return False
        planned_roles = {
            self._normalize_variable_token(planned_path.get("from_role")),
            self._normalize_variable_token(planned_path.get("to_role")),
        }
        if "candidate_set" not in planned_roles or "constraint_value" not in planned_roles:
            return False
        relation_paths = query_plan.get("relation_paths") or []
        for relation_path in relation_paths:
            if not isinstance(relation_path, Mapping):
                continue
            if relation_path is planned_path:
                continue
            roles = {
                self._normalize_variable_token(relation_path.get("from_role")),
                self._normalize_variable_token(relation_path.get("to_role")),
            }
            if "candidate_set" in roles and (
                roles & {"anchor", "anchor_a", "anchor_b"}
            ):
                return True
        return False

    def _anchor_role_identities_for_path(
        self,
        *,
        relation_path: Mapping[str, Any],
        anchor_count: int,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        identities: set[str] = set()
        for endpoint_key, role_key in (("from", "from_role"), ("to", "to_role")):
            aliases = self._relation_role_aliases(
                role=relation_path.get(role_key),
                raw_label=relation_path.get(endpoint_key),
                anchor_count=anchor_count,
                anchored_entities=anchored_entities,
            )
            identities.update(
                alias for alias in aliases if alias in {"anchor_a", "anchor_b"}
            )
        return identities

    def _normalize_plan_items(
        self,
        raw_items: Any,
        *,
        required_keys: Sequence[str],
    ) -> list[dict[str, str]]:
        if not isinstance(raw_items, list):
            return []
        normalized_items: list[dict[str, str]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            normalized_item: dict[str, str] = {}
            missing_key = False
            for key in required_keys:
                value = str(raw_item.get(key) or "").strip()
                if not value:
                    missing_key = True
                    break
                normalized_item[key] = value
            if not missing_key:
                normalized_items.append(normalized_item)
        return normalized_items

    def _normalize_string_list(self, raw_value: Any) -> list[str]:
        if not isinstance(raw_value, list):
            return []
        return [str(item).strip() for item in raw_value if str(item).strip()]

    def _validate_query_plan_grounding(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
    ) -> list[str]:
        allow_exploratory = bool(query_plan.get("allow_exploratory_predicates"))
        relation_paths = [
            rp for rp in query_plan.get("relation_paths", [])
            if isinstance(rp, Mapping)
        ]
        anchored_entities = [
            item
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        anchor_count = len(anchored_entities)

        # Fail-closed: no candidates and not exploratory mode → reject
        if not relation_grounding:
            if not allow_exploratory and relation_paths:
                self._emit_generated_tools_event(
                    {
                        "event": "pal_plan_rejected_no_grounding",
                        "mode": "pal",
                        "relation_path_count": len(relation_paths),
                        "reason": "no_grounded_candidates_and_exploratory_not_set",
                    }
                )
                return [
                    "no_grounded_candidates:plan used relation paths without any "
                    "grounded_relation_candidates — set allow_exploratory_predicates=true "
                    "or ensure the grounding card contains candidates"
                ]
            return []  # Exploratory mode acknowledged, or no relation paths

        grounded_by_relation: dict[str, list[Mapping[str, str]]] = {}
        for candidate in relation_grounding:
            relation = str(candidate.get("relation") or "").strip()
            if relation:
                grounded_by_relation.setdefault(relation, []).append(candidate)

        errors: list[str] = []
        validation_decisions: list[dict[str, Any]] = []
        active_family_bundle = get_reusable_family_policy_bundle(
            str(query_plan.get("query_shape") or "").strip()
        )
        if active_family_bundle is not None:
            scaffold_signature = self._build_scaffold_signature(query_plan)
            if (
                scaffold_signature
                and scaffold_signature
                in set(active_family_bundle.blocked_scaffold_signatures)
            ):
                errors.append(
                    "family_policy_blocked_scaffold_signature:" + scaffold_signature
                )
            forbidden_relation_families = set(
                active_family_bundle.forbidden_relation_families
            )
            if forbidden_relation_families:
                used_relation_names = set(self._relation_names_from_plan(query_plan))
                for relation_name in sorted(
                    used_relation_names & forbidden_relation_families
                ):
                    errors.append(
                        "family_policy_forbidden_relation_family:" + relation_name
                    )
        for relation_path in relation_paths:
            relation = str(relation_path.get("relation") or "")
            relation_candidates = grounded_by_relation.get(relation, [])
            grounding_source = str(relation_path.get("grounding_source") or "").strip()
            relation_decision: dict[str, Any] = {
                "relation": relation,
                "direction": relation_path.get("direction"),
                "from": relation_path.get("from"),
                "to": relation_path.get("to"),
                "from_role": relation_path.get("from_role"),
                "to_role": relation_path.get("to_role"),
                "grounding_source": grounding_source,
            }
            if not relation_candidates:
                if grounding_source == "exploratory" or allow_exploratory:
                    relation_decision["decision"] = "accepted_exploratory_no_grounding"
                    validation_decisions.append(relation_decision)
                    continue
                errors.append(f"relation_not_grounded:{relation}")
                relation_decision["decision"] = "rejected_relation_not_grounded"
                validation_decisions.append(relation_decision)
                continue
            matched = False
            candidate_decisions: list[dict[str, Any]] = []
            for candidate in relation_candidates:
                candidate_match, reason = self._relation_contract_match_details(
                    planned_path=relation_path,
                    grounded_candidate=candidate,
                    query_shape=str(query_plan.get("query_shape") or ""),
                    anchor_count=anchor_count,
                    anchored_entities=anchored_entities,
                    query_plan=query_plan,
                )
                candidate_decisions.append(
                    {
                        "grounding_source": candidate.get("grounding_source"),
                        "direction": candidate.get("direction"),
                        "from": candidate.get("from"),
                        "to": candidate.get("to"),
                        "from_role": candidate.get("from_role"),
                        "to_role": candidate.get("to_role"),
                        "reason": reason,
                    }
                )
                if candidate_match:
                    relation_decision["decision"] = reason
                    matched = True
                    break
            relation_decision["grounded_candidates"] = candidate_decisions
            if not matched:
                dynamic_only_candidates = relation_candidates and all(
                    str(candidate.get("grounding_source") or "").strip().lower()
                    == "dynamic_probe"
                    for candidate in relation_candidates
                )
                if dynamic_only_candidates:
                    planned_anchor_identities = self._anchor_role_identities_for_path(
                        relation_path=relation_path,
                        anchor_count=anchor_count,
                        anchored_entities=anchored_entities,
                    )
                    if anchor_count > 1 and planned_anchor_identities:
                        candidate_anchor_identities: set[str] = set()
                        for candidate in relation_candidates:
                            candidate_anchor_identities.update(
                                self._anchor_role_identities_for_path(
                                    relation_path=candidate,
                                    anchor_count=anchor_count,
                                    anchored_entities=anchored_entities,
                                )
                            )
                        if not (planned_anchor_identities & candidate_anchor_identities):
                            from_role = str(
                                relation_path.get("from_role")
                                or relation_path.get("from")
                                or ""
                            )
                            to_role = str(
                                relation_path.get("to_role")
                                or relation_path.get("to")
                                or ""
                            )
                            errors.append(
                                f"relation_shape_not_grounded:{relation}:{from_role}:{to_role}"
                            )
                            relation_decision["decision"] = (
                                "rejected_dynamic_relation_family_anchor_mismatch"
                            )
                            validation_decisions.append(relation_decision)
                            continue
                    relation_decision["decision"] = (
                        "accepted_dynamic_relation_family_match"
                    )
                    validation_decisions.append(relation_decision)
                    continue
                from_role = str(
                    relation_path.get("from_role") or relation_path.get("from") or ""
                )
                to_role = str(
                    relation_path.get("to_role") or relation_path.get("to") or ""
                )
                errors.append(
                    f"relation_shape_not_grounded:{relation}:{from_role}:{to_role}"
                )
                relation_decision["decision"] = "rejected_relation_shape_not_grounded"
            validation_decisions.append(relation_decision)
        self._emit_generated_tools_event(
            {
                "event": "pal_grounding_validation",
                "mode": "pal",
                "query_shape": query_plan.get("query_shape"),
                "allow_exploratory_predicates": allow_exploratory,
                "grounded_relation_candidates": [
                    {
                        "relation": candidate.get("relation"),
                        "direction": candidate.get("direction"),
                        "from": candidate.get("from"),
                        "to": candidate.get("to"),
                        "from_role": candidate.get("from_role"),
                        "to_role": candidate.get("to_role"),
                        "grounding_source": candidate.get("grounding_source"),
                        "support": candidate.get("support"),
                    }
                    for candidate in relation_grounding
                ],
                "planned_relation_paths": [
                    {
                        "relation": relation_path.get("relation"),
                        "direction": relation_path.get("direction"),
                        "from": relation_path.get("from"),
                        "to": relation_path.get("to"),
                        "from_role": relation_path.get("from_role"),
                        "to_role": relation_path.get("to_role"),
                        "grounding_source": relation_path.get("grounding_source"),
                    }
                    for relation_path in relation_paths
                ],
                "decisions": validation_decisions,
                "errors": errors,
            }
        )
        return errors

    def _build_grounded_relation_candidates(
        self,
        task_question: str,
    ) -> list[dict[str, str]]:
        question_text, _entities = self._split_task_question(task_question)
        lower_text = str(question_text or "").lower()
        candidates: list[dict[str, str]] = []
        if "cheese" in lower_text:
            candidates.extend(
                [
                    {
                        "relation": "food.cheese.texture",
                        "direction": "forward",
                        "from": "cheese",
                        "to": "texture_value",
                        "support": "observed_live_food_predicate",
                        "use_when": "apply a texture constraint to a cheese entity",
                    },
                    {
                        "relation": "food.cheese.source_of_milk",
                        "direction": "forward",
                        "from": "cheese",
                        "to": "milk_source",
                        "support": "observed_live_food_predicate",
                        "use_when": "apply a milk-source constraint to a cheese entity",
                    },
                    {
                        "relation": "food.cheese_texture.cheeses",
                        "direction": "reverse",
                        "from": "texture_value",
                        "to": "cheese",
                        "support": "observed_live_food_predicate",
                        "use_when": "start from a bound texture value and retrieve cheeses",
                    },
                    {
                        "relation": "food.cheese_milk_source.cheeses",
                        "direction": "reverse",
                        "from": "milk_source",
                        "to": "cheese",
                        "support": "observed_live_food_predicate",
                        "use_when": "start from a bound milk-source node and retrieve cheeses",
                    },
                ]
            )
        if "dosage form" in lower_text or (
            "drug" in lower_text and "formulat" in lower_text
        ):
            candidates.extend(
                [
                    {
                        "relation": "medicine.drug.active_moieties",
                        "direction": "forward",
                        "from": "drug",
                        "to": "active_moiety",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "anchor a drug by an active moiety",
                    },
                    {
                        "relation": "medicine.drug.marketed_formulations",
                        "direction": "forward",
                        "from": "drug",
                        "to": "formulation",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "move from a drug to its marketed formulations",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                        "direction": "forward",
                        "from": "active_ingredient",
                        "to": "formulation",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "move from an active ingredient directly to formulations that contain it",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_moiety_of_formulation",
                        "direction": "forward",
                        "from": "active_moiety",
                        "to": "formulation",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "move from an active moiety directly to formulations that contain it",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_moiety_of_drug",
                        "direction": "forward",
                        "from": "active_moiety",
                        "to": "drug",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "move from an active moiety to drugs that contain it",
                    },
                    {
                        "relation": "medicine.drug_formulation.formulation_of",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "drug_or_ingredient",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "anchor a formulation to the drug or ingredient it formulates",
                    },
                    {
                        "relation": "medicine.drug_formulation.active_ingredients",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "active_ingredient",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "bind a formulation by active ingredient",
                    },
                    {
                        "relation": "medicine.drug_formulation.active_ingredient_moieties",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "active_moiety",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "bind a formulation by active ingredient moiety",
                    },
                    {
                        "relation": "medicine.drug_formulation.dosage_form",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "dosage_form",
                        "support": "observed_live_medicine_predicate",
                        "use_when": "project the dosage form of a formulation",
                    },
                ]
            )
        # --- ROYALTY / MONARCHY / NOBILITY ---
        if any(kw in lower_text for kw in ("monarch", "kingdom", "ruler", "royalt", "queen", "king")):
            candidates.extend(
                [
                    {
                        "relation": "royalty.kingdom.rulers",
                        "direction": "forward",
                        "from": "kingdom",
                        "to": "ruler",
                        "support": "curated_royalty_predicate",
                        "use_when": "find rulers/monarchs of a kingdom entity",
                    },
                    {
                        "relation": "royalty.kingdom.monarchs",
                        "direction": "forward",
                        "from": "kingdom",
                        "to": "monarch",
                        "support": "curated_royalty_predicate",
                        "use_when": "find monarchs of a kingdom entity",
                    },
                    {
                        "relation": "royalty.monarch.kingdoms",
                        "direction": "forward",
                        "from": "monarch",
                        "to": "kingdom",
                        "support": "curated_royalty_predicate",
                        "use_when": "find kingdoms ruled by a monarch",
                    },
                ]
            )
        if any(
            kw in lower_text
            for kw in ("baronet", "nobility", "noble rank", "system of nobility", "rank first")
        ):
            candidates.extend(
                [
                    {
                        "relation": "royalty.noble_rank.used_in",
                        "direction": "reverse",
                        "from": "rank",
                        "to": "rank_relationship",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_royalty_predicate",
                        "use_when": "start from a noble rank and retrieve the system-rank relationship records that use it",
                    },
                    {
                        "relation": "royalty.system_rank_relationship.system",
                        "direction": "reverse",
                        "from": "rank_relationship",
                        "to": "system_of_nobility",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_royalty_predicate",
                        "use_when": "project a system-rank relationship to the system of nobility it belongs to",
                    },
                    {
                        "relation": "royalty.system_of_nobility.used_from_date",
                        "direction": "forward",
                        "from": "system_of_nobility",
                        "to": "start_date",
                        "from_role": "answer",
                        "to_role": "ordering_attribute",
                        "support": "curated_royalty_predicate",
                        "use_when": "order systems of nobility by the date they first came into use",
                    },
                ]
            )
        fictional_character_occupation_context = (
            any(kw in lower_text for kw in ("occupation", "profession"))
            and any(
                kw in lower_text
                for kw in (
                    "fictional character",
                    "film character",
                    "species",
                    "fictional universe",
                    "fictional world",
                )
            )
        )
        # --- FICTIONAL UNIVERSE / SPECIES ---
        if any(kw in lower_text for kw in ("fictional", "species", "inhabitant", "universe", "world")):
            candidates.extend(
                [
                    {
                        "relation": "fictional_universe.fictional_setting.universe",
                        "direction": "forward",
                        "from": "setting",
                        "to": "universe",
                        "support": "curated_fictional_universe_predicate",
                        "use_when": "pivot from a fictional setting or world entity to its universe before retrieving inhabitants, races, or species",
                    },
                    {
                        "relation": "fictional_universe.fictional_universe.species",
                        "direction": "forward",
                        "from": "fictional_world",
                        "to": "species",
                        "support": "curated_fictional_universe_predicate",
                        "use_when": "find species in a fictional universe/world",
                    },
                    {
                        "relation": "fictional_universe.fictional_universe.races",
                        "direction": "forward",
                        "from": "fictional_world",
                        "to": "race",
                        "support": "curated_fictional_universe_predicate",
                        "use_when": "find races in a fictional universe/world",
                    },
                    {
                        "relation": "fictional_universe.fictional_universe.characters",
                        "direction": "forward",
                        "from": "fictional_world",
                        "to": "character",
                        "support": "curated_fictional_universe_predicate",
                        "use_when": "find characters in a fictional universe/world",
                    },
                    {
                        "relation": "fictional_universe.fictional_universe.works_set_here",
                        "direction": "forward",
                        "from": "fictional_universe",
                        "to": "work",
                        "support": "curated_fictional_universe_predicate",
                        "use_when": "retrieve works that are set in a fictional universe before filtering or ordering them",
                    },
                    *(
                        [
                            {
                                "relation": "fictional_universe.fictional_character.occupation",
                                "direction": "forward",
                                "from": "fictional_character",
                                "to": "occupation",
                                "support": "curated_fictional_universe_predicate",
                                "use_when": "filter fictional characters by occupation when the question asks for characters of a given job or profession",
                            }
                        ]
                        if fictional_character_occupation_context
                        else []
                    ),
                ]
            )
        # --- BROADCAST / CONTENT / PRODUCER ---
        if any(
            kw in lower_text
            for kw in ("producer of", "produced by", "broadcast", "content", "higher education", "knowledge")
        ):
            candidates.extend(
                [
                    {
                        "relation": "broadcast.genre.content",
                        "direction": "reverse",
                        "from": "genre",
                        "to": "content",
                        "from_role": "anchor",
                        "to_role": "count_set",
                        "support": "curated_broadcast_predicate",
                        "use_when": "start from a broadcast genre/topic entity and retrieve content in that genre",
                    },
                    {
                        "relation": "broadcast.content.producer",
                        "direction": "forward",
                        "from": "content",
                        "to": "producer",
                        "from_role": "count_set",
                        "to_role": "candidate_set",
                        "support": "curated_broadcast_predicate",
                        "use_when": "retrieve the producer of a content/program entity",
                    },
                    {
                        "relation": "broadcast.producer.produces",
                        "direction": "forward",
                        "from": "producer",
                        "to": "content",
                        "from_role": "candidate_set",
                        "to_role": "count_set",
                        "support": "curated_broadcast_predicate",
                        "use_when": "retrieve content produced by a known producer",
                    },
                ]
            )
        if any(
            kw in lower_text
            for kw in (
                "short story",
                "novel",
                "book",
                "written work",
                "copyright date",
                "publication date",
                "date of first publication",
            )
        ):
            candidates.extend(
                [
                    {
                        "relation": "book.written_work.copyright_date",
                        "direction": "forward",
                        "from": "work",
                        "to": "copyright_date",
                        "support": "curated_book_predicate",
                        "use_when": "order a written work candidate set by copyright date",
                    },
                    {
                        "relation": "book.written_work.date_of_first_publication",
                        "direction": "forward",
                        "from": "work",
                        "to": "publication_date",
                        "support": "curated_book_predicate",
                        "use_when": "order a written work candidate set by first publication date",
                    },
                ]
            )
        # --- CALENDARS / DAYS OF WEEK ---
        if any(
            kw in lower_text
            for kw in ("gregorian calendar", "day of the week", "days of the week")
        ):
            candidates.extend(
                [
                    {
                        "relation": "time.calendar.days_of_week",
                        "direction": "forward",
                        "from": "calendar",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_time_predicate",
                        "use_when": "retrieve the day-of-week entities defined by a calendar system",
                    },
                    {
                        "relation": "time.day_of_week.sequence_number",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "ordering_attribute",
                        "from_role": "candidate_set",
                        "to_role": "ordering_attribute",
                        "support": "curated_time_predicate",
                        "use_when": "order day-of-week entities by their sequence number to find the first or last day",
                    },
                ]
            )
        # --- ASTRONOMY / STARS ---
        if any(
            kw in lower_text
            for kw in (
                "flare star",
                "celestial",
                "star category",
                "temperature",
                "temperature k",
                "nebula",
                "farthest away",
                "furthest away",
                "cosmological distance",
            )
        ):
            candidates.extend(
                [
                    {
                        "relation": "astronomy.celestial_object_category.objects",
                        "direction": "reverse",
                        "from": "category",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_astronomy_predicate",
                        "use_when": "retrieve celestial objects that belong to a known category before projecting or ordering them",
                    },
                    {
                        "relation": "astronomy.star.temperature_k",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "answer",
                        "from_role": "candidate_set",
                        "to_role": "ordering_attribute",
                        "support": "curated_astronomy_predicate",
                        "use_when": "retrieve the temperature of a star as a literal scalar",
                    },
                    {
                        "relation": "astronomy.celestial_object.category",
                        "direction": "reverse",
                        "from": "category",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_astronomy_predicate",
                        "use_when": "retrieve celestial objects that belong to a known category such as Nebula",
                    },
                    {
                        "relation": "astronomy.celestial_object.cosmological_distance",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "ordering_attribute",
                        "from_role": "candidate_set",
                        "to_role": "ordering_attribute",
                        "support": "curated_astronomy_predicate",
                        "use_when": "order celestial objects by cosmological distance to find the farthest or nearest one",
                    },
                ]
            )
        if any(
            kw in lower_text
            for kw in ("cyclone", "hurricane", "typhoon", "tropical storm")
        ):
            candidates.extend(
                [
                    {
                        "relation": "meteorology.tropical_cyclone.category",
                        "direction": "forward",
                        "from": "cyclone",
                        "to": "category",
                        "support": "curated_meteorology_predicate",
                        "use_when": "bind the category of a cyclone before retrieving other cyclones in the same category",
                    },
                    {
                        "relation": "meteorology.tropical_cyclone_category.tropical_cyclones",
                        "direction": "forward",
                        "from": "category",
                        "to": "cyclone",
                        "support": "curated_meteorology_predicate",
                        "use_when": "retrieve cyclones that belong to a known tropical cyclone category",
                    },
                    {
                        "relation": "meteorology.tropical_cyclone.formed",
                        "direction": "forward",
                        "from": "cyclone",
                        "to": "formed_date",
                        "support": "curated_meteorology_predicate",
                        "use_when": "order tropical cyclones by formation date in superlative questions",
                    },
                ]
            )
        # --- MEDICINE SYMPTOMS / SIDE EFFECTS ---
        if any(kw in lower_text for kw in ("side effect", "symptom", "medical treatment")):
            candidates.extend(
                [
                    {
                        "relation": "medicine.symptom.side_effect_of",
                        "direction": "forward",
                        "from": "symptom",
                        "to": "treatment",
                        "support": "curated_medicine_symptom_predicate",
                        "use_when": "find treatments that a symptom is a side effect of (symptom → treatment direction)",
                    },
                    {
                        "relation": "medicine.drug.side_effects",
                        "direction": "forward",
                        "from": "drug",
                        "to": "symptom",
                        "support": "curated_medicine_symptom_predicate",
                        "use_when": "find side effects of a drug",
                    },
                    {
                        "relation": "medicine.disease.symptom_of_disease",
                        "direction": "forward",
                        "from": "disease",
                        "to": "symptom",
                        "support": "curated_medicine_symptom_predicate",
                        "use_when": "find symptoms of a disease",
                    },
                ]
            )
        # --- TELEVISION / SONG SEGMENTS ---
        if any(
            kw in lower_text
            for kw in ("episode segment", "tv segment", "tv song", "performer", "performed")
        ):
            candidates.extend(
                [
                    {
                        "relation": "tv.tv_song.episode_segments",
                        "direction": "reverse",
                        "from": "song",
                        "to": "segment_relationship",
                        "from_role": "candidate_set",
                        "to_role": "candidate_set",
                        "support": "curated_tv_predicate",
                        "use_when": "move from a song/composition to the episode-segment song relationships where it appears",
                    },
                    {
                        "relation": "tv.tv_segment_song_relationship.performers",
                        "direction": "reverse",
                        "from": "segment_relationship",
                        "to": "performer",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_tv_predicate",
                        "use_when": "project an episode-segment song relationship to its performers",
                    },
                ]
            )
        # --- MEDIA GENRE ---
        if "genre" in lower_text:
            candidates.extend(
                [
                    {
                        "relation": "media_common.media_genre.child_genres",
                        "direction": "forward",
                        "from": "parent_genre",
                        "to": "child_genre",
                        "support": "curated_media_genre_predicate",
                        "use_when": "find child/derived genres of a parent genre (parent → child direction)",
                    },
                    {
                        "relation": "media_common.media_genre.parent_genre",
                        "direction": "forward",
                        "from": "child_genre",
                        "to": "parent_genre",
                        "support": "curated_media_genre_predicate",
                        "use_when": "find parent genre of a child genre",
                    },
                ]
            )
        # --- MUSIC ---
        if any(kw in lower_text for kw in ("song", "music", "recording", "track", "release", "album", "musical")):
            candidates.extend(
                [
                    {
                        "relation": "music.artist.track",
                        "direction": "forward",
                        "from": "artist",
                        "to": "track",
                        "support": "curated_music_predicate",
                        "use_when": "find tracks performed/recorded by a music artist",
                    },
                    {
                        "relation": "music.recording.releases",
                        "direction": "forward",
                        "from": "recording",
                        "to": "release",
                        "support": "curated_music_predicate",
                        "use_when": "find releases that contain a recording",
                    },
                    {
                        "relation": "music.release.tracks",
                        "direction": "forward",
                        "from": "release",
                        "to": "track",
                        "support": "curated_music_predicate",
                        "use_when": "find tracks on a release",
                    },
                    {
                        "relation": "music.release_component.recordings",
                        "direction": "forward",
                        "from": "release_component",
                        "to": "recording",
                        "support": "curated_music_predicate",
                        "use_when": "find recordings on a release component",
                    },
                    {
                        "relation": "music.recording.length",
                        "direction": "forward",
                        "from": "recording",
                        "to": "length",
                        "support": "curated_music_predicate",
                        "use_when": "get the length/duration of a recording (for argmax/longest queries)",
                    },
                    {
                        "relation": "music.artist.album",
                        "direction": "forward",
                        "from": "artist",
                        "to": "album",
                        "support": "curated_music_predicate",
                        "use_when": "find albums by an artist",
                    },
                    {
                        "relation": "music.composer.compositions",
                        "direction": "reverse",
                        "from": "composer",
                        "to": "composition",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_music_predicate",
                        "use_when": "retrieve compositions written by a composer before chaining to downstream media relations",
                    },
                ]
            )
        # --- VIDEO GAMES / DISTRIBUTION SYSTEMS ---
        if any(
            kw in lower_text
            for kw in (
                "video game",
                "game version",
                "games distributed",
                "distribution system",
                "platform",
                "developer",
                "developed by",
                "region",
                "released where",
                "virtual console",
            )
        ):
            candidates.extend(
                [
                    {
                        "relation": "cvg.computer_game_distribution_system.games_distributed",
                        "direction": "reverse",
                        "from": "distribution_system",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_cvg_predicate",
                        "use_when": "retrieve game versions distributed by a known distribution system",
                    },
                    {
                        "relation": "cvg.cvg_developer.game_versions_developed",
                        "direction": "reverse",
                        "from": "developer",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_cvg_predicate",
                        "use_when": "retrieve game versions developed by a known developer",
                    },
                    {
                        "relation": "cvg.game_version.regions",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "answer",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_cvg_predicate",
                        "use_when": "project a game version to its release regions",
                    },
                    {
                        "relation": "cvg.computer_game_distribution_system.platforms_supported",
                        "direction": "forward",
                        "from": "distribution_system",
                        "to": "answer",
                        "from_role": "anchor",
                        "to_role": "answer",
                        "support": "curated_cvg_predicate",
                        "use_when": "project a game distribution system to the platforms it supports",
                    },
                ]
            )
        # --- PEOPLE / PROFESSION ---
        if (
            any(kw in lower_text for kw in ("profession", "occupation", "songwriter", "percussionist", "musician"))
            and not fictional_character_occupation_context
        ):
            candidates.extend(
                [
                    {
                        "relation": "people.profession.people_with_this_profession",
                        "direction": "forward",
                        "from": "profession",
                        "to": "person",
                        "support": "curated_people_predicate",
                        "use_when": "find people who have the given profession",
                    },
                    {
                        "relation": "people.person.profession",
                        "direction": "forward",
                        "from": "person",
                        "to": "profession",
                        "support": "curated_people_predicate",
                        "use_when": "find the profession(s) of a person",
                    },
                    {
                        "relation": "music.artist.profession",
                        "direction": "forward",
                        "from": "artist",
                        "to": "profession",
                        "support": "curated_people_predicate",
                        "use_when": "find the profession(s) of a music artist",
                    },
                ]
            )
        # --- SPORTS / TEAM MEMBERSHIP ---
        if any(
            kw in lower_text
            for kw in ("basketball", "played for", "team roster", "teams that", "sports team")
        ):
            candidates.extend(
                [
                    {
                        "relation": "sports.pro_athlete.teams",
                        "direction": "forward",
                        "from": "athlete",
                        "to": "team_roster",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_sports_predicate",
                        "use_when": "retrieve the team-roster stints associated with a professional athlete",
                    },
                    {
                        "relation": "sports.sports_team_roster.team",
                        "direction": "forward",
                        "from": "team_roster",
                        "to": "team",
                        "from_role": "candidate_set",
                        "to_role": "count_set",
                        "support": "curated_sports_predicate",
                        "use_when": "project a roster stint to the sports team for counting or lookup questions",
                    },
                ]
            )
        # --- AEROSPACE / SPACECRAFT ---
        if any(kw in lower_text for kw in ("spacecraft", "satellite", "rocket", "aerospace", "space")):
            candidates.extend(
                [
                    {
                        "relation": "aerospace.spacecraft.manufacturer",
                        "direction": "forward",
                        "from": "spacecraft",
                        "to": "manufacturer",
                        "support": "curated_aerospace_predicate",
                        "use_when": "find the manufacturer of a spacecraft",
                    },
                    {
                        "relation": "spaceflight.spacecraft.manufacturer",
                        "direction": "forward",
                        "from": "spacecraft",
                        "to": "manufacturer",
                        "support": "curated_aerospace_predicate",
                        "use_when": "find the manufacturer of a spacecraft (spaceflight namespace)",
                    },
                ]
            )
        # --- PRODUCT / COMPANY / BRAND ---
        if any(kw in lower_text for kw in ("product", "candy", "manufacturer", "brand", "introduced", "manufactured")):
            candidates.extend(
                [
                    {
                        "relation": "business.brand.products",
                        "direction": "forward",
                        "from": "brand",
                        "to": "product",
                        "support": "curated_business_predicate",
                        "use_when": "find products of a brand/company",
                    },
                    {
                        "relation": "business.product_line.products",
                        "direction": "forward",
                        "from": "product_line",
                        "to": "product",
                        "support": "curated_business_predicate",
                        "use_when": "find products in a product line",
                    },
                    {
                        "relation": "product.product.introduction_date",
                        "direction": "forward",
                        "from": "product",
                        "to": "introduction_date",
                        "support": "curated_business_predicate",
                        "use_when": "get the introduction/launch date of a product (for argmax/latest queries)",
                    },
                    {
                        "relation": "product.product.manufacturer",
                        "direction": "forward",
                        "from": "product",
                        "to": "manufacturer",
                        "support": "curated_business_predicate",
                        "use_when": "find the manufacturer of a product",
                    },
                ]
            )
        # --- DIGITAL CAMERAS / SENSOR TYPE ---
        if any(
            kw in lower_text
            for kw in (
                "camera sensor",
                "sensor type",
                "digital camera",
                "lumix",
                "color filter array",
                "iso setting",
                "iso settings",
            )
        ):
            candidates.extend(
                [
                    {
                        "relation": "digicams.digital_camera.sensor_type",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "answer",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_digicams_predicate",
                        "use_when": "project a digital camera product to its sensor type",
                    },
                    {
                        "relation": "product.camera.sensor_type",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "answer",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_digicams_predicate",
                        "use_when": "fallback camera sensor-type projection when the product is modeled under a broader camera namespace",
                    },
                    {
                        "relation": "digicams.digital_camera.color_filter_array_type",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "constraint_value",
                        "from_role": "candidate_set",
                        "to_role": "constraint_value",
                        "support": "curated_digicams_predicate",
                        "use_when": "constrain a digital camera product by its color filter array type such as Bayer filter",
                    },
                    {
                        "relation": "digicams.digital_camera.iso_setting",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "constraint_value",
                        "from_role": "candidate_set",
                        "to_role": "constraint_value",
                        "support": "curated_digicams_predicate",
                        "use_when": "constrain a digital camera product by its ISO setting",
                    },
                ]
            )
        # --- ANIMAL / DOG BREEDS / BIOLOGY ORGANISM ---
        if any(kw in lower_text for kw in ("dog", "breed", "animal", "temperament")):
            candidates.extend(
                [
                    {
                        "relation": "biology.animal_breed.country_of_origin",
                        "direction": "forward",
                        "from": "breed",
                        "to": "country",
                        "support": "curated_biology_predicate",
                        "use_when": "find the country of origin of an animal breed",
                    },
                    {
                        "relation": "biology.breed_origin.breeds_originating_here",
                        "direction": "reverse",
                        "from": "country",
                        "to": "breed",
                        "support": "curated_biology_predicate",
                        "use_when": "start from a country or region and retrieve animal breeds originating there",
                    },
                    {
                        "relation": "biology.animal_breed.temperament",
                        "direction": "forward",
                        "from": "breed",
                        "to": "temperament",
                        "support": "curated_biology_predicate",
                        "use_when": "find the temperament of an animal breed",
                    },
                    {
                        "relation": "pets.pet_breed.temperament",
                        "direction": "forward",
                        "from": "breed",
                        "to": "temperament",
                        "support": "curated_biology_predicate",
                        "use_when": "find the temperament of a pet breed",
                    },
                ]
            )
        if any(
            kw in lower_text
            for kw in (
                "royal line",
                "royal lines",
                "dynasty",
                "dynasties",
                "preceded by",
                "succeeded by",
            )
        ):
            candidates.extend(
                [
                    {
                        "relation": "royalty.royal_line.preceded_by",
                        "direction": "forward",
                        "from": "successor_line",
                        "to": "predecessor_line",
                        "from_role": "anchor",
                        "to_role": "shared_answer",
                        "support": "curated_royalty_lineage_predicate",
                        "use_when": "find the predecessor royal line or dynasty of a given royal line",
                    },
                    {
                        "relation": "royalty.royal_line.preceded_by",
                        "direction": "reverse",
                        "from": "predecessor_line",
                        "to": "successor_line",
                        "from_role": "shared_answer",
                        "to_role": "anchor",
                        "support": "curated_royalty_lineage_predicate",
                        "use_when": "find royal lines or dynasties that are preceded by a given line",
                    },
                ]
            )
        # --- DISEASE / TRANSMISSION ---
        if any(kw in lower_text for kw in ("disease", "transmit", "infectious", "virus", "bacteria")):
            candidates.extend(
                [
                    {
                        "relation": "biology.organism.diseases_transmitted",
                        "direction": "forward",
                        "from": "organism",
                        "to": "disease",
                        "support": "curated_biology_predicate",
                        "use_when": "find diseases transmitted by an organism",
                    },
                    {
                        "relation": "medicine.disease.transmitted_by",
                        "direction": "forward",
                        "from": "disease",
                        "to": "transmitter",
                        "support": "curated_medicine_disease_predicate",
                        "use_when": "find organisms/vectors that transmit a disease",
                    },
                ]
            )
        if any(
            kw in lower_text
            for kw in ("prevention factor", "prevention factors", "parent disease")
        ):
            candidates.extend(
                [
                    {
                        "relation": "medicine.disease.parent_disease",
                        "direction": "reverse",
                        "from": "parent_disease",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_medicine_disease_predicate",
                        "use_when": "retrieve diseases that are children of a known parent disease",
                    },
                    {
                        "relation": "medicine.disease.prevention_factors",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "answer",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_medicine_disease_predicate",
                        "use_when": "retrieve prevention factors associated with a disease",
                    },
                ]
            )
        # --- EXHIBITIONS ---
        if "exhibition" in lower_text:
            candidates.extend(
                [
                    {
                        "relation": "exhibitions.exhibition.exhibition_types",
                        "direction": "forward",
                        "from": "exhibition",
                        "to": "exhibition_type",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_exhibitions_predicate",
                        "use_when": "retrieve the type/category of an exhibition",
                    },
                    {
                        "relation": "exhibitions.type_of_exhibition.exhibitions_of_this_type",
                        "direction": "forward",
                        "from": "exhibition_type",
                        "to": "candidate_set",
                        "from_role": "candidate_set",
                        "to_role": "count_set",
                        "support": "curated_exhibitions_predicate",
                        "use_when": "retrieve exhibitions that belong to a known exhibition type",
                    },
                    {
                        "relation": "exhibitions.exhibition.subjects",
                        "direction": "forward",
                        "from": "candidate_set",
                        "to": "answer",
                        "from_role": "count_set",
                        "to_role": "answer",
                        "support": "curated_exhibitions_predicate",
                        "use_when": "project an exhibition to its subjects",
                    },
                ]
            )
        # --- ORGANIZATION / INSTITUTION ---
        if any(kw in lower_text for kw in ("institution", "organization", "university", "owner", "parent organization")):
            candidates.extend(
                [
                    {
                        "relation": "organization.organization.parent_organization",
                        "direction": "forward",
                        "from": "organization",
                        "to": "parent",
                        "support": "curated_organization_predicate",
                        "use_when": "find the parent organization of an organization",
                    },
                    {
                        "relation": "organization.organization.child_organizations",
                        "direction": "forward",
                        "from": "parent",
                        "to": "child_organization",
                        "support": "curated_organization_predicate",
                        "use_when": "find child organizations",
                    },
                    {
                        "relation": "organization.organization_member.organization",
                        "direction": "forward",
                        "from": "member",
                        "to": "organization",
                        "support": "curated_organization_predicate",
                        "use_when": "find the organization a member entity belongs to",
                    },
                ]
            )
        if any(
            kw in lower_text
            for kw in ("leader", "leaders of", "led by", "board member")
        ):
            candidates.extend(
                [
                    {
                        "relation": "business.board_member.leader_of",
                        "direction": "forward",
                        "from": "leader",
                        "to": "candidate_set",
                        "from_role": "anchor",
                        "to_role": "candidate_set",
                        "support": "curated_organization_predicate",
                        "use_when": "retrieve organizations led by a specific leader or board member",
                    },
                    {
                        "relation": "organization.leadership.organization",
                        "direction": "reverse",
                        "from": "leadership_record",
                        "to": "organization",
                        "from_role": "candidate_set",
                        "to_role": "answer",
                        "support": "curated_organization_predicate",
                        "use_when": "project a leadership record to its organization when a direct leader-of path binds through an intermediate leadership node",
                    },
                ]
            )
        return candidates

    def _augment_generic_type_relation_candidates(
        self,
        *,
        relation_candidates: Sequence[Mapping[str, Any]],
        answer_target_phrase: str,
        question_interpretation: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = [
            dict(candidate)
            for candidate in relation_candidates
            if isinstance(candidate, Mapping)
        ]
        interpreted_inputs = [
            item
            for item in ((question_interpretation or {}).get("question_inputs") or [])
            if isinstance(item, Mapping)
        ]
        has_type_like_input = any(
            str(item.get("kind") or "").strip() in {"class_phrase", "type_constraint"}
            or str(item.get("role_hint") or "").strip() == "type_set"
            or str(item.get("kind") or "").strip() == "shared_attribute"
            and str(item.get("surface") or "").strip().lower() in {"type", "category"}
            for item in interpreted_inputs
        ) or any(
            token in str(answer_target_phrase or "").lower()
            for token in ("type", "types", "kind", "kinds", "category", "categories")
        )
        if has_type_like_input and self._answer_target_is_relation_encoded_type_projection(
            answer_target_phrase=answer_target_phrase,
            relation_candidates=candidates,
        ):
            has_type_like_input = any(
                str(item.get("kind") or "").strip() in {"class_phrase", "type_constraint"}
                or str(item.get("role_hint") or "").strip() == "type_set"
                or (
                    str(item.get("kind") or "").strip() == "shared_attribute"
                    and str(item.get("surface") or "").strip().lower() in {"type", "category"}
                )
                for item in interpreted_inputs
            )
        if not has_type_like_input:
            return candidates

        existing_signatures = {
            (
                str(candidate.get("relation") or "").strip(),
                str(candidate.get("from_role") or "").strip(),
                str(candidate.get("to_role") or "").strip(),
            )
            for candidate in candidates
            if str(candidate.get("relation") or "").strip()
        }
        has_specific_type_family = any(
            isinstance(candidate, Mapping)
            and str(candidate.get("support") or "").strip().lower()
            != "generic_type_relation"
            and (
                (
                    str(candidate.get("from_role") or "").strip()
                    in {"anchor", "shared_answer"}
                    and str(candidate.get("to_role") or "").strip()
                    in {"type_set", "shared_type"}
                )
                or (
                    str(candidate.get("from_role") or "").strip()
                    in {"type_set", "shared_type"}
                    and str(candidate.get("to_role") or "").strip()
                    in {"candidate_set", "shared_answer"}
                )
            )
            for candidate in candidates
        )
        if has_specific_type_family:
            return candidates
        generic_type_candidates = [
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "entity",
                "to": "type",
                "from_role": "anchor",
                "to_role": "type_set",
                "support": "generic_type_relation",
                "use_when": "bind or filter the type/category of an entity when the question includes a class/category phrase",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "type",
                "from_role": "candidate_set",
                "to_role": "type_set",
                "support": "generic_type_relation",
                "use_when": "filter a candidate or shared answer set by an explicit type/category node",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "shared_answer",
                "to": "shared_type",
                "from_role": "shared_answer",
                "to_role": "shared_type",
                "support": "generic_type_relation",
                "use_when": "extract the shared type/category of an answer set before projecting instances or the type itself",
            },
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "type",
                "to": "instance",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "support": "generic_type_relation",
                "use_when": "retrieve instances that belong to a type/category or connect a shared type node to answer entities",
            },
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "shared_type",
                "to": "instance",
                "from_role": "shared_type",
                "to_role": "shared_answer",
                "support": "generic_type_relation",
                "use_when": "retrieve instances that belong to a type/category or connect a shared type node to answer entities",
            },
        ]
        for candidate in generic_type_candidates:
            signature = (
                str(candidate.get("relation") or "").strip(),
                str(candidate.get("from_role") or "").strip(),
                str(candidate.get("to_role") or "").strip(),
            )
            if signature[0] and signature not in existing_signatures:
                candidates.append(candidate)
        return candidates

    def _extract_json_object(self, raw_output: str) -> str:
        stripped_output = str(raw_output or "").strip()
        if stripped_output.startswith("{") and stripped_output.endswith("}"):
            return stripped_output
        json_match = re.search(r"\{.*\}", stripped_output, flags=re.DOTALL)
        if json_match is None:
            raise ValueError("json_object_not_found")
        return json_match.group(0)

    def _capture_pal_query_plan_artifact(
        self,
        *,
        generated_tool_name: str,
        query_plan: Mapping[str, Any],
    ) -> Path:
        artifact_dir = self._get_pal_query_artifact_dir()
        plan_path = artifact_dir / f"{generated_tool_name}.plan.json"
        plan_path.write_text(
            json.dumps(query_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return plan_path

    def _generate_validated_pal_candidate(
        self,
        *,
        task_question: str,
        grounding_card: str,
        query_plan: Mapping[str, Any],
        generated_tool_name: str,
        extra_repair_feedback: Sequence[str] = (),
    ) -> tuple[str, dict[str, str]]:
        # Pre-populate validation_errors with any plausibility repair feedback
        # so that it appears in the very first generator prompt for this attempt.
        validation_errors: list[str] = list(extra_repair_feedback)
        last_error = ""
        reusable_selection = select_reusable_tool(query_plan)
        if reusable_selection is not None:
            self._emit_generated_tools_event(
                {
                    "event": "pal_reusable_tool_selected",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "family_name": reusable_selection.family_name,
                    "family_bundle_version": reusable_selection.policy_bundle.version,
                    "renderer_name": reusable_selection.renderer_name,
                    "fit_score": reusable_selection.fit_score,
                    "selection_reasons": list(reusable_selection.reasons),
                    "family_policy_bundle": reusable_selection.policy_bundle.as_dict(),
                    "query_shape": query_plan.get("query_shape"),
                    "answer_mode": query_plan.get("answer_mode"),
                }
            )
            try:
                generated_output = render_reusable_tool(
                    query_plan=query_plan,
                    selection=reusable_selection,
                )
                self._emit_generated_tools_event(
                    {
                        "event": "toolgen_candidate",
                        "mode": "pal",
                        "round": 0,
                        "tool_name": generated_tool_name,
                        "code_len": len(generated_output),
                        "code_sha1": self._sha1_text(generated_output),
                        "patch_round": False,
                        "patch_ops": len(validation_errors),
                        "query_plan_answer_mode": query_plan.get("answer_mode"),
                        "query_plan_strategy": query_plan.get("strategy"),
                        "generation_source": "reusable_tool",
                        "reusable_family_name": reusable_selection.family_name,
                    }
                )
                generated_code = extract_and_validate_code(generated_output)
                query_texts = self._extract_sparql_query_texts(generated_code)
                generated_code, query_texts = self._rewrite_generated_code_anchor_bindings(
                    generated_code=generated_code,
                    query_texts=query_texts,
                    query_plan=query_plan,
                )
                query_text = query_texts[0] if query_texts else ""
                reusable_errors = self._validate_pal_query_candidate(
                    raw_output=generated_output,
                    generated_code=generated_code,
                    query_text=query_text,
                    query_texts=query_texts,
                    query_plan=query_plan,
                )
                if reusable_errors:
                    last_error = ",".join(reusable_errors)
                    validation_errors = [
                        *validation_errors,
                        f"reusable_tool_validation_failed:{reusable_selection.family_name}",
                        *reusable_errors,
                    ]
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_query_candidate_rejected",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "attempt": 0,
                            "rejection_reasons": list(reusable_errors),
                            "query_text_summary": self._summarize_text(
                                query_text,
                                max_len=240,
                            ),
                            "generation_source": "reusable_tool",
                            "reusable_family_name": reusable_selection.family_name,
                        }
                    )
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_reusable_tool_fallback",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "family_name": reusable_selection.family_name,
                            "reason": last_error,
                        }
                    )
                else:
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_query_candidate_accepted",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "attempt": 0,
                            "projected_variables": self._extract_projected_variables(
                                query_text
                            ),
                            "query_text_summary": self._summarize_text(
                                query_text,
                                max_len=240,
                            ),
                            "generation_source": "reusable_tool",
                            "reusable_family_name": reusable_selection.family_name,
                        }
                    )
                    return generated_code, {
                        "query_text": query_text,
                        "validation_feedback": self._build_validation_feedback([]),
                        "generation_source": "reusable_tool",
                        "reusable_selection": reusable_selection,
                    }
            except Exception as exc:
                last_error = str(exc)
                validation_errors = [
                    *validation_errors,
                    f"reusable_tool_failed:{reusable_selection.family_name}",
                    last_error,
                ]
                self._emit_generated_tools_event(
                    {
                        "event": "pal_reusable_tool_fallback",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "family_name": reusable_selection.family_name,
                        "reason": last_error,
                    }
                )
        for attempt in range(1, self._PAL_QUERY_CODE_MAX_ATTEMPTS + 1):
            generator_prompt = self._format_prompt_template(
                template=GENERATOR_CODE_SYSTEM_PROMPT,
                ontology_card=grounding_card,
                task_question=task_question,
                query_plan_json=json.dumps(query_plan, ensure_ascii=False, indent=2),
                validation_feedback=self._build_validation_feedback(validation_errors),
            )
            generated_output = self._run_text_prompt(
                system_prompt=generator_prompt,
                user_prompt=task_question,
            )
            self._emit_generated_tools_event(
                {
                    "event": "toolgen_candidate",
                    "mode": "pal",
                    "round": attempt,
                    "tool_name": generated_tool_name,
                    "code_len": len(generated_output),
                    "code_sha1": self._sha1_text(generated_output),
                    "patch_round": attempt > 1,
                    "patch_ops": len(validation_errors),
                    "query_plan_answer_mode": query_plan.get("answer_mode"),
                    "query_plan_strategy": query_plan.get("strategy"),
                    "generation_source": "llm_generator",
                }
            )
            try:
                generated_code = extract_and_validate_code(generated_output)
            except Exception as exc:
                last_error = str(exc)
                validation_errors = [f"parser_error:{last_error}"]
                self._emit_generated_tools_event(
                    {
                        "event": "pal_query_candidate_rejected",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "rejection_reasons": list(validation_errors),
                        "raw_output_summary": self._summarize_text(
                            generated_output,
                            max_len=240,
                        ),
                    }
                )
                continue

            query_texts = self._extract_sparql_query_texts(generated_code)
            generated_code, query_texts = self._rewrite_generated_code_anchor_bindings(
                generated_code=generated_code,
                query_texts=query_texts,
                query_plan=query_plan,
            )
            query_text = query_texts[0] if query_texts else ""
            validation_errors = self._validate_pal_query_candidate(
                raw_output=generated_output,
                generated_code=generated_code,
                query_text=query_text,
                query_texts=query_texts,
                query_plan=query_plan,
            )
            if validation_errors:
                last_error = ",".join(validation_errors)
                self._emit_generated_tools_event(
                    {
                        "event": "pal_query_candidate_rejected",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "rejection_reasons": list(validation_errors),
                        "query_text_summary": self._summarize_text(
                            query_text,
                            max_len=240,
                        ),
                    }
                )
                continue

            self._emit_generated_tools_event(
                {
                    "event": "pal_query_candidate_accepted",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "attempt": attempt,
                    "projected_variables": self._extract_projected_variables(query_text),
                    "query_text_summary": self._summarize_text(query_text, max_len=240),
                }
            )
            return generated_code, {
                "query_text": query_text,
                "validation_feedback": self._build_validation_feedback([]),
                "generation_source": "llm_generator",
                "reusable_selection": None,
            }
        raise ValueError(
            f"pal_query_candidate_invalid:{last_error or 'validation_failed'}"
        )

    def _rewrite_generated_code_anchor_bindings(
        self,
        *,
        generated_code: str,
        query_texts: Sequence[str],
        query_plan: Mapping[str, Any],
    ) -> tuple[str, list[str]]:
        rewritten_code = str(generated_code or "")
        rewritten_queries: list[str] = []
        for query_text in query_texts or ():
            rewritten_query = self._rewrite_query_text_with_resolved_anchor_bindings(
                query_text=query_text,
                query_plan=query_plan,
            )
            rewritten_query = self._rewrite_query_text_with_direct_anchor_entity_ids(
                query_text=rewritten_query,
                query_plan=query_plan,
            )
            rewritten_query = self._rewrite_query_text_with_alias_aware_anchor_bindings(
                query_text=rewritten_query,
                query_plan=query_plan,
            )
            if rewritten_query != query_text:
                rewritten_code = rewritten_code.replace(query_text, rewritten_query, 1)
            rewritten_queries.append(rewritten_query)
        return rewritten_code, rewritten_queries

    def _rewrite_query_text_with_resolved_anchor_bindings(
        self,
        *,
        query_text: str,
        query_plan: Mapping[str, Any],
    ) -> str:
        rewritten_query = str(query_text or "")
        anchored_entities = [
            item
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if not rewritten_query or not anchored_entities:
            return rewritten_query

        for anchored_entity in anchored_entities:
            resolved_entity_id = str(
                anchored_entity.get("resolved_entity_id") or ""
            ).strip()
            if not resolved_entity_id:
                continue

            anchor_literals = [
                str(anchored_entity.get("chosen_alias") or "").strip(),
                str(anchored_entity.get("surface") or "").strip(),
            ]
            anchor_literals = [literal for literal in anchor_literals if literal]
            seen_literals: set[str] = set()
            ordered_literals: list[str] = []
            for literal in anchor_literals:
                lowered = literal.lower()
                if lowered in seen_literals:
                    continue
                seen_literals.add(lowered)
                ordered_literals.append(literal)

            for anchor_literal in ordered_literals:
                escaped_literal = re.escape(anchor_literal)
                lowered_literal = json.dumps(anchor_literal.lower())

                direct_pattern = re.compile(
                    rf'(?P<indent>[ \t]*)(?P<var>\?[A-Za-z_][A-Za-z0-9_]*)\s+fb:type\.object\.name\s+"{escaped_literal}"@en\s*\.',
                    flags=re.IGNORECASE,
                )

                def _replace_direct(match: re.Match[str]) -> str:
                    anchor_var = match.group("var")
                    indent = match.group("indent") or ""
                    return f"{indent}VALUES {anchor_var} {{ fb:{resolved_entity_id} }}"

                rewritten_query = direct_pattern.sub(_replace_direct, rewritten_query)

                filtered_pattern = re.compile(
                    rf'(?P<indent>[ \t]*)(?P<var>\?[A-Za-z_][A-Za-z0-9_]*)\s+fb:type\.object\.name\s+(?P<label>\?[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*\n(?P=indent)[ \t]*FILTER\(LCASE\((?:STR\()?(?P=label)\)?\)\s*=\s*{lowered_literal}\s*\)',
                    flags=re.IGNORECASE,
                )

                def _replace_filtered(match: re.Match[str]) -> str:
                    anchor_var = match.group("var")
                    indent = match.group("indent") or ""
                    return f"{indent}VALUES {anchor_var} {{ fb:{resolved_entity_id} }}"

                rewritten_query = filtered_pattern.sub(
                    _replace_filtered, rewritten_query
                )
                rewritten_query = self._collapse_resolved_anchor_union_binding_blocks(
                    query_text=rewritten_query,
                    resolved_entity_id=resolved_entity_id,
                    anchor_literal=anchor_literal,
                )

        return rewritten_query

    def _rewrite_query_text_with_direct_anchor_entity_ids(
        self,
        *,
        query_text: str,
        query_plan: Mapping[str, Any],
    ) -> str:
        rewritten_query = str(query_text or "")
        anchored_entities = [
            item
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        relation_paths = [
            item
            for item in (query_plan.get("relation_paths") or [])
            if isinstance(item, Mapping)
        ]
        if not rewritten_query or len(anchored_entities) != 1 or not relation_paths:
            return rewritten_query

        anchored_entity = anchored_entities[0]
        resolved_entity_id = str(
            anchored_entity.get("resolved_entity_id") or ""
        ).strip()
        anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
        if not resolved_entity_id or not anchor_role:
            return rewritten_query

        for relation_path in relation_paths:
            relation = str(relation_path.get("relation") or "").strip()
            if not relation:
                continue
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            if anchor_role not in {from_role, to_role}:
                continue

            other_tokens = {
                token
                for token in {
                    self._normalize_variable_token(relation_path.get("from")),
                    self._normalize_variable_token(relation_path.get("to")),
                    self._normalize_variable_token(relation_path.get("from_role")),
                    self._normalize_variable_token(relation_path.get("to_role")),
                }
                if token and token != anchor_role
            }
            escaped_relation = re.escape(relation)

            subject_pattern = re.compile(
                rf"fb:(?P<entity>[a-z]\.[A-Za-z0-9_]+)\s+fb:{escaped_relation}\s+(?P<var>\?[A-Za-z_][A-Za-z0-9_]*)",
                flags=re.IGNORECASE,
            )

            def _replace_subject(match: re.Match[str]) -> str:
                variable = match.group("var")
                variable_token = self._normalize_variable_token(variable)
                entity_id = str(match.group("entity") or "").strip()
                if entity_id == resolved_entity_id:
                    return match.group(0)
                if other_tokens and variable_token not in other_tokens:
                    return match.group(0)
                return f"fb:{resolved_entity_id} fb:{relation} {variable}"

            rewritten_query = subject_pattern.sub(_replace_subject, rewritten_query)

            object_pattern = re.compile(
                rf"(?P<var>\?[A-Za-z_][A-Za-z0-9_]*)\s+fb:{escaped_relation}\s+fb:(?P<entity>[a-z]\.[A-Za-z0-9_]+)",
                flags=re.IGNORECASE,
            )

            def _replace_object(match: re.Match[str]) -> str:
                variable = match.group("var")
                variable_token = self._normalize_variable_token(variable)
                entity_id = str(match.group("entity") or "").strip()
                if entity_id == resolved_entity_id:
                    return match.group(0)
                if other_tokens and variable_token not in other_tokens:
                    return match.group(0)
                return f"{variable} fb:{relation} fb:{resolved_entity_id}"

            rewritten_query = object_pattern.sub(_replace_object, rewritten_query)

        return rewritten_query

    def _collapse_resolved_anchor_union_binding_blocks(
        self,
        *,
        query_text: str,
        resolved_entity_id: str,
        anchor_literal: str,
    ) -> str:
        rewritten_query = str(query_text or "")
        entity_id = str(resolved_entity_id or "").strip()
        literal = str(anchor_literal or "").strip()
        if not rewritten_query or not entity_id or not literal:
            return rewritten_query

        lowered_literal = json.dumps(literal.lower())
        escaped_entity_id = re.escape(entity_id)
        union_pattern = re.compile(
            rf'(?P<indent>[ \t]*)\{{\s*'
            rf'VALUES (?P<var>\?[A-Za-z_][A-Za-z0-9_]*) \{{ fb:{escaped_entity_id} \}}\s*'
            rf'\}}\s*UNION\s*\{{\s*'
            rf'(?P=var)\s+fb:(?:type\.object\.name|common\.topic\.alias)\s+(?P<label>\?[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*'
            rf'FILTER\(LCASE\((?:STR\()?(?P=label)\)?\)\s*=\s*{lowered_literal}\s*\)\s*'
            rf'\}}',
            flags=re.IGNORECASE | re.DOTALL,
        )

        def _replace_union(match: re.Match[str]) -> str:
            indent = match.group("indent") or ""
            anchor_var = match.group("var")
            return f"{indent}VALUES {anchor_var} {{ fb:{entity_id} }}"

        while True:
            collapsed_query = union_pattern.sub(_replace_union, rewritten_query)
            if collapsed_query == rewritten_query:
                return rewritten_query
            rewritten_query = collapsed_query

    def _rewrite_query_text_with_alias_aware_anchor_bindings(
        self,
        *,
        query_text: str,
        query_plan: Mapping[str, Any],
    ) -> str:
        rewritten_query = str(query_text or "")
        anchored_entities = [
            item
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if not rewritten_query or not anchored_entities:
            return rewritten_query

        for index, anchored_entity in enumerate(anchored_entities):
            anchor_literals = [
                str(anchored_entity.get("chosen_alias") or "").strip(),
                str(anchored_entity.get("surface") or "").strip(),
            ]
            anchor_literals = [literal for literal in anchor_literals if literal]
            seen_literals: set[str] = set()
            ordered_literals: list[str] = []
            for literal in anchor_literals:
                lowered = literal.lower()
                if lowered in seen_literals:
                    continue
                seen_literals.add(lowered)
                ordered_literals.append(literal)

            for anchor_literal in ordered_literals:
                escaped_literal = re.escape(anchor_literal)
                lowered_literal = json.dumps(anchor_literal.lower())

                direct_pattern = re.compile(
                    rf'(?P<indent>[ \t]*)(?P<var>\?[A-Za-z_][A-Za-z0-9_]*)\s+fb:type\.object\.name\s+"{escaped_literal}"@en\s*\.',
                    flags=re.IGNORECASE,
                )

                def _replace_direct(match: re.Match[str]) -> str:
                    anchor_var = match.group("var")
                    indent = match.group("indent") or ""
                    label_var = f"{anchor_var}_label_{index + 1}"
                    block = self._build_probe_anchor_match_block(
                        anchor_var=anchor_var,
                        label_var=label_var,
                        anchor_name=anchor_literal,
                    )
                    return "\n".join(
                        f"{indent}{line}" if line else line
                        for line in block.splitlines()
                    )

                rewritten_query = direct_pattern.sub(_replace_direct, rewritten_query)

                filtered_pattern = re.compile(
                    rf'(?P<indent>[ \t]*)(?P<var>\?[A-Za-z_][A-Za-z0-9_]*)\s+fb:type\.object\.name\s+(?P<label>\?[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*\n(?P=indent)[ \t]*FILTER\(LCASE\((?:STR\()?(?P=label)\)?\)\s*=\s*{lowered_literal}\s*\)',
                    flags=re.IGNORECASE,
                )

                def _replace_filtered(match: re.Match[str]) -> str:
                    anchor_var = match.group("var")
                    label_var = match.group("label")
                    indent = match.group("indent") or ""
                    block = self._build_probe_anchor_match_block(
                        anchor_var=anchor_var,
                        label_var=label_var,
                        anchor_name=anchor_literal,
                    )
                    return "\n".join(
                        f"{indent}{line}" if line else line
                        for line in block.splitlines()
                    )

                rewritten_query = filtered_pattern.sub(
                    _replace_filtered, rewritten_query
                )
        return rewritten_query

    def _retry_execution_with_resolved_anchor_ids(
        self,
        *,
        generated_code: str,
        invocation_result: Any,
        query_plan: Mapping[str, Any],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> tuple[str, Any, dict[str, Any], bool]:
        if not getattr(invocation_result, "success", False):
            return generated_code, invocation_result, dict(query_plan), False
        if not anchor_probe_results:
            return generated_code, invocation_result, dict(query_plan), False

        result_dict = (
            invocation_result.payload if isinstance(invocation_result.payload, Mapping) else None
        )
        binding_count = self._get_result_binding_count(result_dict) if result_dict else 0
        scalar_count = self._extract_scalar_count_value(result_dict)
        answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()

        has_live_resolved_probe = any(
            str(getattr(result, "resolved_entity_id", "") or "").strip()
            and (getattr(result, "path_count", None) or 0) > 0
            for result in anchor_probe_results
        )
        has_ambiguous_resolved_probe = any(
            str(getattr(result, "resolved_entity_id", "") or "").strip()
            and int(getattr(result, "entity_count", 0) or 0) > 1
            for result in anchor_probe_results
        )
        if not has_live_resolved_probe:
            return generated_code, invocation_result, dict(query_plan), False

        should_retry = binding_count == 0 or (
            answer_mode == "count" and scalar_count == 0
        ) or has_ambiguous_resolved_probe
        if not should_retry:
            return generated_code, invocation_result, dict(query_plan), False

        repaired_query_plan, feedback = self._apply_probe_guided_anchor_entity_repairs(
            query_plan=query_plan,
            anchor_probe_results=anchor_probe_results,
        )
        if not feedback:
            return generated_code, invocation_result, dict(query_plan), False

        query_texts = self._extract_sparql_query_texts(generated_code)
        rewritten_code, rewritten_queries = self._rewrite_generated_code_anchor_bindings(
            generated_code=generated_code,
            query_texts=query_texts,
            query_plan=repaired_query_plan,
        )
        if rewritten_code == generated_code:
            return generated_code, invocation_result, repaired_query_plan, False

        retried_result = execute_pal_code_with_result(rewritten_code)
        if not retried_result.success:
            return generated_code, invocation_result, repaired_query_plan, False

        retried_payload = (
            retried_result.payload if isinstance(retried_result.payload, Mapping) else None
        )
        retried_binding_count = (
            self._get_result_binding_count(retried_payload) if retried_payload else 0
        )
        retried_scalar_count = self._extract_scalar_count_value(retried_payload)
        improved = retried_binding_count > binding_count or (
            answer_mode == "count"
            and retried_scalar_count is not None
            and scalar_count is not None
            and retried_scalar_count > scalar_count
        )
        if has_ambiguous_resolved_probe and rewritten_code != generated_code:
            improved = True
        if not improved:
            return generated_code, invocation_result, repaired_query_plan, False

        self._emit_generated_tools_event(
            {
                "event": "pal_same_attempt_anchor_entity_retry",
                "mode": "pal",
                "before_binding_count": binding_count,
                "after_binding_count": retried_binding_count,
                "before_scalar_count": scalar_count,
                "after_scalar_count": retried_scalar_count,
                "query_text_summary": self._summarize_text(
                    rewritten_queries[0] if rewritten_queries else "", max_len=240
                ),
            }
        )
        return rewritten_code, retried_result, repaired_query_plan, True

    def _build_validation_feedback(self, validation_errors: Sequence[str]) -> str:
        feedback_items = self._canonicalize_feedback_items(validation_errors)
        if not feedback_items:
            return "none"
        return "\n".join(f"- {item}" for item in feedback_items)

    # ------------------------------------------------------------------
    # Bounded generate → execute → plausibility-check → repair loop
    # ------------------------------------------------------------------

    def _run_pal_repair_loop(
        self,
        *,
        task_question: str,
        grounding_card: str,
        query_plan: dict[str, Any],
        generated_tool_name: str,
        question_entities: list[str],
        relation_grounding: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> tuple[str, Any, dict[str, Any]]:
        """Bounded PAL repair loop: generate → execute → validate → repair.

        Produces at most ``2 + _PAL_REPAIR_MAX_ATTEMPTS`` full attempts.
        The extra slot is reserved for a last structural scaffold switch
        after same-family repairs have already been exhausted.

        Returns:
            (best_code, best_invocation_result, loop_log)

        Fallback behaviour:
        - If a candidate passes the plausibility validator it is accepted
          immediately.
        - If all attempts are exhausted without an accepted candidate the
          best *successfully-executing* candidate is returned anyway (the
          caller's existing failure path handles the rest).
        - If the first code-generation attempt raises, the exception
          propagates (no repair can help a hard static-validation failure).
        - If a *repair* attempt's code generation raises, we break to the
          fallback path rather than propagating.

        Logs one ``pal_repair_loop_attempt`` event per iteration so the loop
        is fully auditable from the generated_tools log.
        """
        max_total: int = self._PAL_REPAIR_MAX_ATTEMPTS + 2  # initial + repairs + one structural rewrite retry

        best_executing: dict[str, Any] | None = None
        plausibility_feedback: list[str] = []
        cumulative_plan_feedback: list[str] = []
        last_code: str = ""
        last_result: Any = None
        working_query_plan: dict[str, Any] = copy.deepcopy(query_plan)
        working_grounding_card: str = grounding_card
        working_relation_grounding: list[dict[str, str]] = [
            dict(candidate)
            for candidate in (relation_grounding or [])
            if isinstance(candidate, Mapping)
        ]
        seen_attempt_signatures: dict[str, int] = {}
        loop_log: dict[str, Any] = {
            "total_attempts": 0,
            "accepted_attempt": None,
            "repair_used": False,
            "final_verdict": None,
            "last_verdict": None,
            "last_reasons": [],
            "attempt_decisions": [],
        }

        def _sync_return_query_plan(plan_to_sync: Mapping[str, Any] | None) -> None:
            if not isinstance(plan_to_sync, Mapping):
                return
            query_plan.clear()
            query_plan.update(copy.deepcopy(dict(plan_to_sync)))

        if str(working_query_plan.get("query_shape") or "").strip().lower() == "multi_anchor_intersection":
            if self._has_asymmetric_anchor_clues(
                task_question=task_question,
                anchored_entities=working_query_plan.get("anchored_entities") or [],
            ):
                synthesized_bridge_candidates = (
                    self._synthesize_anchor_specific_bridge_candidates(
                        task_question=task_question,
                        query_plan=working_query_plan,
                        relation_grounding=working_relation_grounding,
                    )
                )
                if synthesized_bridge_candidates:
                    working_relation_grounding = self._merge_relation_grounding_candidates(
                        working_relation_grounding,
                        synthesized_bridge_candidates,
                        prefer_extra=True,
                    )

        for attempt in range(1, max_total + 1):
            is_repair: bool = attempt > 1
            loop_log["total_attempts"] = attempt

            # ---- Code generation ----------------------------------------
            try:
                generated_code, generation_metadata = self._generate_validated_pal_candidate(
                    task_question=task_question,
                    grounding_card=working_grounding_card,
                    query_plan=working_query_plan,
                    generated_tool_name=generated_tool_name,
                    extra_repair_feedback=(
                        plausibility_feedback if is_repair else ()
                    ),
                )
            except Exception as exc:
                if not is_repair:
                    # First attempt: propagate so the outer except can handle it
                    raise
                # Repair attempt: code gen failed — fall back to best candidate
                self._emit_generated_tools_event(
                    {
                        "event": "pal_repair_loop_codegen_failed",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "is_repair": is_repair,
                        "error": str(exc),
                    }
                )
                break

            last_code = generated_code
            reusable_selection = generation_metadata.get("reusable_selection")
            generation_source = str(
                generation_metadata.get("generation_source") or "llm_generator"
            ).strip()

            # ---- Execution ----------------------------------------------
            invocation_result = execute_pal_code_with_result(generated_code)
            last_result = invocation_result

            query_texts = self._extract_sparql_query_texts(generated_code)
            query_text = query_texts[0] if query_texts else ""
            result_dict: Any = (
                invocation_result.payload if invocation_result.success else None
            )

            # ---- Anchor existence probes --------------------------------
            # Probe when execution succeeded but results look suspicious:
            #   • result is empty for a non-exploratory plan, OR
            #   • answer_mode is "count" (verify the counted set path)
            # This adds ~2-5 s of latency but only when needed.
            anchor_probe_results: list[AnchorProbeResult] | None = None
            relation_paths_now = [
                relation_path
                for relation_path in (working_query_plan.get("relation_paths") or [])
                if isinstance(relation_path, Mapping)
            ]
            has_grounded_paths = any(
                str(relation_path.get("grounding_source") or "").strip().lower()
                != "exploratory"
                for relation_path in relation_paths_now
            )
            if invocation_result.success:
                _binding_count_now = (
                    self._get_result_binding_count(result_dict)
                    if isinstance(result_dict, Mapping)
                    else 0
                )
                _answer_mode_now = str(
                    working_query_plan.get("answer_mode") or "entity"
                ).lower()
                _query_shape_now = str(
                    working_query_plan.get("query_shape") or ""
                ).lower()
                _result_is_empty = _binding_count_now == 0 and (
                    not isinstance(result_dict, Mapping)
                    or "boolean" not in result_dict
                )
                _is_count_mode = _answer_mode_now == "count"
                _has_resolved_anchor_id = any(
                    str(anchored_entity.get("resolved_entity_id") or "").strip()
                    for anchored_entity in (working_query_plan.get("anchored_entities") or [])
                    if isinstance(anchored_entity, Mapping)
                )
                _has_dynamic_grounded_paths = any(
                    str(relation_path.get("grounding_source") or "").strip().lower()
                    == "dynamic_probe"
                    for relation_path in relation_paths_now
                )
                _has_only_dynamic_grounded_paths = (
                    bool(relation_paths_now)
                    and _has_dynamic_grounded_paths
                    and all(
                        str(relation_path.get("grounding_source") or "").strip().lower()
                        in {"dynamic_probe", "exploratory"}
                        for relation_path in relation_paths_now
                    )
                )
                _dynamic_single_anchor_needs_disambiguation_probe = (
                    _binding_count_now > 0
                    and _answer_mode_now in {"entity", "literal"}
                    and _query_shape_now
                    in {
                        "single_anchor_lookup",
                        "single_anchor_chain_lookup",
                        "containment_or_ownership_lookup",
                    }
                    and _has_only_dynamic_grounded_paths
                    and not _has_resolved_anchor_id
                )
                _should_probe_paths = _is_count_mode or (
                    _result_is_empty
                    and _query_shape_now in {
                        "single_anchor_lookup",
                        "single_anchor_chain_lookup",
                        "multi_anchor_intersection",
                        "shared_type_intersection",
                        "containment_or_ownership_lookup",
                        "superlative_chain",
                    }
                ) or _dynamic_single_anchor_needs_disambiguation_probe

                if (
                    (_result_is_empty and (has_grounded_paths or relation_paths_now))
                    or (_is_count_mode and has_grounded_paths)
                    or _dynamic_single_anchor_needs_disambiguation_probe
                ):
                    # Probe empty-result plans even when all paths are exploratory so
                    # alias repair and dynamic grounding augmentation have evidence.
                    anchor_probe_results = self._run_anchor_existence_probes(
                        query_plan=working_query_plan,
                        probe_paths=_should_probe_paths,
                        timeout_s=2.5,
                    )

            (
                generated_code,
                invocation_result,
                resolved_query_plan,
                same_attempt_anchor_retry_used,
            ) = self._retry_execution_with_resolved_anchor_ids(
                generated_code=generated_code,
                invocation_result=invocation_result,
                query_plan=working_query_plan,
                anchor_probe_results=anchor_probe_results,
            )
            if same_attempt_anchor_retry_used:
                working_query_plan = resolved_query_plan
                _sync_return_query_plan(working_query_plan)
                last_code = generated_code
                last_result = invocation_result
                query_texts = self._extract_sparql_query_texts(generated_code)
                query_text = query_texts[0] if query_texts else ""
                result_dict = (
                    invocation_result.payload if invocation_result.success else None
                )

            # ---- Plausibility check -------------------------------------
            verdict: PlausibilityVerdict = validate_pal_execution(
                query_plan=working_query_plan,
                query_text=query_text,
                result_dict=result_dict,
                entities=question_entities,
                anchor_probe_results=anchor_probe_results,
                execution_success=bool(invocation_result.success),
                execution_failure_kind=(
                    str(getattr(invocation_result, "failure_kind", "") or "").strip()
                    or None
                ),
            )

            self._emit_generated_tools_event(
                {
                    "event": "pal_repair_loop_attempt",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "attempt": attempt,
                    "is_repair": is_repair,
                        "grounding_mode": (
                            "exploratory"
                            if working_query_plan.get("allow_exploratory_predicates")
                            else "grounded"
                        ),
                    "strategy": working_query_plan.get("strategy"),
                    "plan": {
                        "answer_mode": working_query_plan.get("answer_mode"),
                        "query_shape": working_query_plan.get("query_shape"),
                        "shared_answer_variable": working_query_plan.get(
                            "shared_answer_variable"
                        ),
                        "candidate_set_variable": working_query_plan.get(
                            "candidate_set_variable"
                        ),
                        "count_set_variable": working_query_plan.get("count_set_variable"),
                        "ordering_attribute": working_query_plan.get("ordering_attribute"),
                        "ordering_direction": working_query_plan.get("ordering_direction"),
                        "join_structure": working_query_plan.get("join_structure"),
                        "anchored_entity_count": len(
                            working_query_plan.get("anchored_entities") or []
                        ),
                        "relation_path_count": len(
                            working_query_plan.get("relation_paths") or []
                        ),
                        "relation_paths": working_query_plan.get("relation_paths") or [],
                    },
                    "query_text_summary": self._summarize_text(
                        query_text, max_len=240
                    ),
                    "execution_success": invocation_result.success,
                    "execution_binding_count": (
                        self._get_result_binding_count(result_dict)
                        if isinstance(result_dict, Mapping)
                        else None
                    ),
                    "anchor_probes": (
                        [
                            {
                                "anchor": r.anchor_name,
                                "entity_count": r.entity_count,
                                "found": r.found,
                                "path_count": r.path_count,
                                "relation_probed": r.relation_probed,
                                "anchor_position": r.anchor_position,
                                "resolved_entity_id": r.resolved_entity_id,
                            }
                            for r in anchor_probe_results
                        ]
                        if anchor_probe_results is not None
                        else None
                    ),
                    "verdict": verdict.verdict,
                    "verdict_reasons": verdict.reasons,
                    "repair_feedback_used": (
                        list(plausibility_feedback) if is_repair else []
                    ),
                    "accepted": verdict.is_accepted,
                }
            )
            attempt_decision = self._build_attempt_decision_record(
                query_plan=working_query_plan,
                generation_source=generation_source,
                reusable_selection=reusable_selection,
                verdict=verdict,
            )
            loop_log["attempt_decisions"].append(attempt_decision.as_dict())
            loop_log["final_attempt_query_plan"] = copy.deepcopy(dict(working_query_plan))
            self._emit_generated_tools_event(
                {
                    "event": "pal_attempt_decision",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "attempt": attempt,
                    **attempt_decision.as_dict(),
                }
            )
            loop_log["last_verdict"] = verdict.verdict
            loop_log["last_reasons"] = list(verdict.reasons)

            if invocation_result.success and invocation_result.payload is not None:
                candidate_metadata = self._build_best_executing_candidate_metadata(
                    generated_code=generated_code,
                    invocation_result=invocation_result,
                    query_plan=working_query_plan,
                    query_text=query_text,
                    verdict=verdict,
                    anchor_probe_results=anchor_probe_results,
                )
                if (
                    best_executing is None
                    or int(candidate_metadata.get("score") or 0)
                    > int(best_executing.get("score") or 0)
                ):
                    best_executing = candidate_metadata

            stagnation_signature = self._build_repair_loop_stagnation_signature(
                query_plan=working_query_plan,
                query_text=query_text,
                verdict=verdict,
                anchor_probe_results=anchor_probe_results,
                invocation_result=invocation_result,
            )
            signature_repeat_count = seen_attempt_signatures.get(stagnation_signature, 0) + 1
            seen_attempt_signatures[stagnation_signature] = signature_repeat_count

            if verdict.is_accepted:
                _sync_return_query_plan(working_query_plan)
                loop_log["accepted_attempt"] = attempt
                loop_log["repair_used"] = is_repair
                loop_log["final_verdict"] = verdict.verdict
                self._emit_generated_tools_event(
                    {
                        "event": "pal_repair_loop_accepted",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "repair_used": is_repair,
                        "verdict": verdict.verdict,
                    }
                )
                return generated_code, invocation_result, loop_log

            if signature_repeat_count >= self._PAL_REPAIR_STAGNATION_REPEAT_LIMIT:
                loop_log["final_verdict"] = verdict.verdict
                loop_log["last_reasons"] = list(verdict.reasons) + [
                    "repair_loop_stalled"
                ]
                self._emit_generated_tools_event(
                    {
                        "event": "pal_repair_loop_stalled",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "verdict": verdict.verdict,
                        "signature_repeat_count": signature_repeat_count,
                    }
                )
                break

            # ---- Prepare repair feedback for next round -----------------
            if attempt < max_total:
                plausibility_feedback = build_repair_feedback(verdict)
                (
                    working_query_plan,
                    working_grounding_card,
                    alias_repair_feedback,
                ) = self._apply_probe_guided_alias_repairs(
                    task_question=task_question,
                    query_plan=working_query_plan,
                    grounding_card=working_grounding_card,
                    relation_grounding=working_relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                )
                (
                    working_query_plan,
                    anchor_entity_repair_feedback,
                ) = self._apply_probe_guided_anchor_entity_repairs(
                    query_plan=working_query_plan,
                    anchor_probe_results=anchor_probe_results,
                )
                (
                    working_grounding_card,
                    working_relation_grounding,
                    dynamic_grounding_feedback,
                ) = self._augment_grounding_with_dynamic_probe_on_path_failure(
                    task_question=task_question,
                    query_plan=working_query_plan,
                    relation_grounding=working_relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                )
                (
                    working_grounding_card,
                    working_relation_grounding,
                    structural_repair_feedback,
                ) = self._augment_grounding_for_structural_repair(
                    task_question=task_question,
                    query_plan=working_query_plan,
                    relation_grounding=working_relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                    verdict=verdict,
                )
                (
                    working_grounding_card,
                    working_relation_grounding,
                    suppressed_relation_feedback,
                ) = self._suppress_dead_grounded_relations(
                    task_question=task_question,
                    query_plan=working_query_plan,
                    relation_grounding=working_relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                )
                _sync_return_query_plan(working_query_plan)
                if alias_repair_feedback:
                    plausibility_feedback = alias_repair_feedback + plausibility_feedback
                if anchor_entity_repair_feedback:
                    plausibility_feedback = (
                        anchor_entity_repair_feedback + plausibility_feedback
                    )
                if dynamic_grounding_feedback:
                    plausibility_feedback = (
                        dynamic_grounding_feedback + plausibility_feedback
                    )
                if structural_repair_feedback:
                    plausibility_feedback = (
                        structural_repair_feedback + plausibility_feedback
                    )
                if suppressed_relation_feedback:
                    plausibility_feedback = (
                        suppressed_relation_feedback + plausibility_feedback
                    )
                cumulative_plan_feedback = self._merge_feedback_items(
                    cumulative_plan_feedback,
                    plausibility_feedback,
                )
                repair_action = self._decide_family_repair_action(
                    verdict=verdict,
                    alias_repair_feedback=(
                        list(alias_repair_feedback) + list(anchor_entity_repair_feedback)
                    ),
                    repair_feedback=plausibility_feedback,
                )
                if repair_action == "stay_in_family":
                    loop_log["repair_used"] = True
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_repair_loop_retry_same_plan_after_alias_repair",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "query_shape": working_query_plan.get("query_shape"),
                            "answer_mode": working_query_plan.get("answer_mode"),
                            "relation_paths": working_query_plan.get("relation_paths") or [],
                        }
                    )
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_repair_loop_repair_scheduled",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "verdict": verdict.verdict,
                            "repair_feedback": cumulative_plan_feedback,
                        }
                    )
                    continue
                rewritten_query_plan, rewrite_family = self._build_structural_repair_query_plan(
                    task_question=task_question,
                    query_plan=working_query_plan,
                    relation_grounding=working_relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                )
                if rewritten_query_plan is not None:
                    working_query_plan = rewritten_query_plan
                    (
                        working_query_plan,
                        post_rewrite_anchor_entity_feedback,
                    ) = self._apply_probe_guided_anchor_entity_repairs(
                        query_plan=working_query_plan,
                        anchor_probe_results=anchor_probe_results,
                    )
                    query_plan.clear()
                    query_plan.update(copy.deepcopy(working_query_plan))
                    if post_rewrite_anchor_entity_feedback:
                        cumulative_plan_feedback = self._merge_feedback_items(
                            cumulative_plan_feedback,
                            post_rewrite_anchor_entity_feedback,
                        )
                    self._emit_generated_tools_event(
                        {
                            "event": "pal_repair_loop_plan_rewritten",
                            "mode": "pal",
                            "tool_name": generated_tool_name,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "rewrite_family": rewrite_family,
                            "query_shape": working_query_plan.get("query_shape"),
                            "shared_answer_variable": working_query_plan.get(
                                "shared_answer_variable"
                            ),
                            "relation_paths": working_query_plan.get("relation_paths") or [],
                        }
                    )
                else:
                    previous_alias_assignments = (
                        self._extract_anchor_alias_assignments(working_query_plan)
                    )
                    try:
                        refreshed_query_plan = self._generate_pal_query_plan(
                            task_question=task_question,
                            grounding_card=working_grounding_card,
                            relation_grounding=working_relation_grounding,
                            generated_tool_name=generated_tool_name,
                            extra_plan_feedback=cumulative_plan_feedback,
                        )
                    except Exception as exc:
                        self._emit_generated_tools_event(
                            {
                                "event": "pal_repair_loop_plan_refresh_failed",
                                "mode": "pal",
                                "tool_name": generated_tool_name,
                                "attempt": attempt,
                                "next_attempt": attempt + 1,
                                "error": str(exc),
                            }
                        )
                    else:
                        working_query_plan = refreshed_query_plan
                        (
                            working_query_plan,
                            post_refresh_anchor_entity_feedback,
                        ) = self._apply_probe_guided_anchor_entity_repairs(
                            query_plan=working_query_plan,
                            anchor_probe_results=anchor_probe_results,
                        )
                        _sync_return_query_plan(working_query_plan)
                        (
                            working_grounding_card,
                            working_relation_grounding,
                            alias_refresh_feedback,
                        ) = self._refresh_dynamic_grounding_after_anchor_alias_change(
                            task_question=task_question,
                            previous_alias_assignments=previous_alias_assignments,
                            query_plan=working_query_plan,
                            relation_grounding=working_relation_grounding,
                        )
                        rewritten_after_alias_refresh, _ = self._build_structural_repair_query_plan(
                            task_question=task_question,
                            query_plan=working_query_plan,
                            relation_grounding=working_relation_grounding,
                            anchor_probe_results=anchor_probe_results,
                        )
                        if rewritten_after_alias_refresh is not None:
                            working_query_plan = rewritten_after_alias_refresh
                            _sync_return_query_plan(working_query_plan)
                        if alias_refresh_feedback:
                            cumulative_plan_feedback = self._merge_feedback_items(
                                cumulative_plan_feedback,
                                alias_refresh_feedback,
                            )
                        if post_refresh_anchor_entity_feedback:
                            cumulative_plan_feedback = self._merge_feedback_items(
                                cumulative_plan_feedback,
                                post_refresh_anchor_entity_feedback,
                            )
                        self._emit_generated_tools_event(
                            {
                                "event": "pal_repair_loop_plan_refreshed",
                                "mode": "pal",
                                "tool_name": generated_tool_name,
                                "attempt": attempt,
                                "next_attempt": attempt + 1,
                                "query_shape": working_query_plan.get("query_shape"),
                                "answer_mode": working_query_plan.get("answer_mode"),
                                "relation_paths": working_query_plan.get("relation_paths") or [],
                            }
                        )
                loop_log["repair_used"] = True
                self._emit_generated_tools_event(
                    {
                        "event": "pal_repair_loop_repair_scheduled",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "verdict": verdict.verdict,
                        "repair_feedback": cumulative_plan_feedback,
                    }
                )

        # ---- All attempts exhausted without acceptance ------------------
        if not loop_log.get("final_verdict"):
            loop_log["final_verdict"] = "no_accepted_candidate"

        if best_executing is not None:
            loop_log["best_executing_candidate"] = {
                "verdict": best_executing.get("verdict"),
                "verdict_reasons": best_executing.get("verdict_reasons") or [],
                "binding_count": best_executing.get("binding_count"),
                "scalar_count": best_executing.get("scalar_count"),
                "answer_mode": best_executing.get("answer_mode"),
                "query_shape": best_executing.get("query_shape"),
                "all_anchors_found": best_executing.get("all_anchors_found"),
                "resolved_anchor_count": best_executing.get("resolved_anchor_count"),
                "score": best_executing.get("score"),
            }
            if self._should_accept_best_executing_candidate(
                candidate_metadata=best_executing
            ):
                loop_log["accepted_attempt"] = "best_executing"
                loop_log["final_verdict"] = "accepted_best_effort"
                self._emit_generated_tools_event(
                    {
                        "event": "pal_repair_loop_salvaged",
                        "mode": "pal",
                        "tool_name": generated_tool_name,
                        "salvage_source": "best_executing_candidate",
                        "original_verdict": best_executing.get("verdict"),
                        "score": best_executing.get("score"),
                        "binding_count": best_executing.get("binding_count"),
                        "scalar_count": best_executing.get("scalar_count"),
                    }
                )
                _sync_return_query_plan(best_executing.get("query_plan"))
                return (
                    str(best_executing.get("generated_code") or ""),
                    best_executing.get("invocation_result"),
                    loop_log,
                )
            self._emit_generated_tools_event(
                {
                    "event": "pal_repair_loop_fallback",
                    "mode": "pal",
                    "tool_name": generated_tool_name,
                    "reason": "no_accepted_candidate_using_best_executing",
                    "total_attempts": loop_log["total_attempts"],
                    "repair_used": loop_log["repair_used"],
                    "score": best_executing.get("score"),
                    "best_verdict": best_executing.get("verdict"),
                }
            )
            _sync_return_query_plan(best_executing.get("query_plan"))
            return (
                str(best_executing.get("generated_code") or ""),
                best_executing.get("invocation_result"),
                loop_log,
            )

        # Nothing executed successfully — return last result to trigger the
        # caller's existing execution-failure path.
        self._emit_generated_tools_event(
            {
                "event": "pal_repair_loop_all_failed",
                "mode": "pal",
                "tool_name": generated_tool_name,
                "total_attempts": loop_log["total_attempts"],
                "repair_used": loop_log["repair_used"],
            }
        )
        _sync_return_query_plan(working_query_plan)
        return last_code, last_result, loop_log

    def _should_retry_same_plan_after_alias_repair(
        self,
        *,
        verdict: PlausibilityVerdict,
        alias_repair_feedback: Sequence[str],
        repair_feedback: Sequence[str] = (),
    ) -> bool:
        if verdict.verdict not in {
            "repairable_bad_count_set",
            "repairable_anchor_not_found",
            "repairable_anchor_path_empty",
            "repairable_bad_join",
        }:
            return False
        if not any(
            str(item).startswith("anchor_alias_override:")
            or str(item).startswith("anchor_entity_override:")
            for item in (alias_repair_feedback or [])
        ):
            return False
        if verdict.verdict != "repairable_anchor_not_found":
            structural_retry_blockers = (
                "plausibility_feedback:dynamic_grounding_augmented",
                "plausibility_feedback:dead_relation_suppressed",
                "repair_hint:consider_dynamic_candidates",
                "repair_hint:do_not_reuse_dead_relation_instance",
                "dynamic_candidate_priority:",
                "dead_relation_suppressed:",
            )
            if any(
                str(item or "").startswith(structural_retry_blockers)
                for item in (repair_feedback or [])
            ):
                return False
        return True

    def _decide_family_repair_action(
        self,
        *,
        verdict: PlausibilityVerdict,
        alias_repair_feedback: Sequence[str],
        repair_feedback: Sequence[str] = (),
    ) -> str:
        if str(verdict.verdict or "").strip() in {
            "rejected_dangerous_overreach",
            "rejected_unsupported_predicate",
            "rejected_unbounded_exploration",
        }:
            return "switch_family"
        if self._should_retry_same_plan_after_alias_repair(
            verdict=verdict,
            alias_repair_feedback=alias_repair_feedback,
            repair_feedback=repair_feedback,
        ):
            return "stay_in_family"
        if verdict.is_accepted:
            return "accept"
        return "switch_family"

    def _build_structural_repair_query_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> tuple[Optional[dict[str, Any]], str]:
        rewrite_builders: tuple[
            tuple[str, Callable[[], Optional[dict[str, Any]]]],
            ...,
        ] = (
            (
                "joined_count_boundary",
                lambda: self._build_joined_count_boundary_repair_plan(
                    query_plan=query_plan,
                ),
            ),
            (
                "projected_answer_intersection",
                lambda: self._build_projected_answer_intersection_repair_plan(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                ),
            ),
            (
                "shared_type_pivot_bridge",
                lambda: self._build_shared_type_pivot_bridge_repair_plan(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                ),
            ),
            (
                "single_anchor_dynamic_lookup",
                lambda: self._build_single_anchor_dynamic_lookup_repair_plan(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                ),
            ),
            (
                "superlative_anchor_alternative",
                lambda: self._build_superlative_anchor_alternative_repair_plan(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                ),
            ),
            (
                "superlative_dynamic_anchor",
                lambda: self._build_superlative_dynamic_anchor_repair_plan(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                ),
            ),
            (
                "class_filtered_count",
                lambda: self._build_class_filtered_count_repair_plan(
                    task_question=task_question,
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                ),
            ),
            (
                "pivot_preserving_count_family",
                lambda: self._build_pivot_preserving_count_repair_plan(
                    query_plan=query_plan,
                    relation_grounding=relation_grounding,
                    anchor_probe_results=anchor_probe_results,
                ),
            ),
        )

        for rewrite_family, builder in rewrite_builders:
            rewritten_query_plan = builder()
            if rewritten_query_plan is not None:
                return rewritten_query_plan, rewrite_family
        return None, ""

    def _build_joined_count_boundary_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return None
        if (
            str(query_plan.get("query_shape") or "").strip().lower()
            != "count_over_direct_relation"
        ):
            return None

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 2:
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        anchor_roles = {
            self._normalize_relation_role(item.get("role"))
            for item in anchored_entities
            if self._normalize_relation_role(item.get("role"))
            in {"anchor", "anchor_a", "anchor_b"}
        }
        variable_roles = {"candidate_set", "count_set", "shared_answer", "answer"}
        constraint_roles = {
            "constraint_value",
            "type_set",
            "shared_type",
            "anchor_value",
        }
        has_non_anchor_downstream_path = False
        has_candidate_constraint_filter = False
        observed_variable_roles: set[str] = set()
        for path in relation_paths:
            from_role = self._normalize_relation_role(path.get("from_role"))
            to_role = self._normalize_relation_role(path.get("to_role"))
            roles = {from_role, to_role}
            if roles & variable_roles and roles & constraint_roles:
                has_candidate_constraint_filter = True
            observed_variable_roles.update(role for role in roles if role in variable_roles)
            touches_anchor = any(
                self._path_touches_anchor_role(path, anchor_role=anchor_role)
                for anchor_role in anchor_roles
            )
            if not touches_anchor and roles & variable_roles:
                has_non_anchor_downstream_path = True
        has_mixed_count_projection_roles = (
            "candidate_set" in observed_variable_roles
            and "count_set" in observed_variable_roles
        )

        if not (
            has_candidate_constraint_filter
            or has_non_anchor_downstream_path
            or has_mixed_count_projection_roles
        ):
            return None

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["query_shape"] = "count_over_joined_set"
        strategy = str(query_plan.get("strategy") or "").strip()
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by treating the downstream count filters/projections "
            "as a joined counted set instead of a direct anchor count."
        ).strip()
        plan_rationale = [
            str(item).strip()
            for item in (query_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "A downstream count filter or projection is present beyond the anchor-side relation, so the failed direct-count scaffold should be reclassified as joined-count before retrying."
        )
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _apply_probe_guided_anchor_entity_repairs(
        self,
        *,
        query_plan: Mapping[str, Any],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> tuple[dict[str, Any], list[str]]:
        if not anchor_probe_results:
            return dict(query_plan), []

        anchored_entities: list[dict[str, Any]] = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if not anchored_entities:
            return dict(query_plan), []

        applied_repairs: list[tuple[str, str]] = []
        for anchored_entity, probe_result in zip(anchored_entities, anchor_probe_results):
            resolved_entity_id = str(
                getattr(probe_result, "resolved_entity_id", "") or ""
            ).strip()
            if not resolved_entity_id:
                continue
            current_entity_id = str(
                anchored_entity.get("resolved_entity_id") or ""
            ).strip()
            if current_entity_id == resolved_entity_id:
                continue
            anchored_entity["resolved_entity_id"] = resolved_entity_id
            surface = str(
                anchored_entity.get("surface")
                or anchored_entity.get("chosen_alias")
                or ""
            ).strip()
            if surface:
                applied_repairs.append((surface, resolved_entity_id))

        if not applied_repairs:
            return dict(query_plan), []

        repaired_query_plan = copy.deepcopy(dict(query_plan))
        repaired_query_plan["anchored_entities"] = anchored_entities
        feedback = [
            "plausibility_feedback:anchor_entity_resolved — a live KG probe resolved the anchor to a specific Freebase entity id",
            "repair_hint:bind_anchor_to_resolved_entity_id — reuse the resolved Freebase entity id in the next query instead of rebinding only by surface string",
        ]
        for surface, resolved_entity_id in applied_repairs:
            feedback.append(f"anchor_entity_override:{surface}=>{resolved_entity_id}")
        return repaired_query_plan, feedback

    def _apply_probe_guided_alias_repairs(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        grounding_card: str,
        relation_grounding: Optional[Sequence[Mapping[str, str]]],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> tuple[dict[str, Any], str, list[str]]:
        if not anchor_probe_results:
            return dict(query_plan), grounding_card, []

        question_text, _ = self._split_task_question(task_question)
        anchored_entities: list[Mapping[str, Any]] = [
            item
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if not anchored_entities:
            return dict(query_plan), grounding_card, []

        alias_overrides = self._extract_anchor_alias_overrides(query_plan)
        selected_repairs: dict[str, str] = {}
        repair_events: list[dict[str, Any]] = []

        for anchored_entity, probe_result in zip(anchored_entities, anchor_probe_results):
            entity_count = getattr(probe_result, "entity_count", -1)
            path_count = getattr(probe_result, "path_count", None)
            relation_probed = getattr(probe_result, "relation_probed", None)
            anchor_position = str(
                getattr(probe_result, "anchor_position", None) or "subject"
            ).strip().lower()
            probe_target_inferred = False
            if not relation_probed:
                inferred_probe = self._infer_anchor_relation_probe_from_query_plan(
                    query_plan=query_plan,
                    anchored_entity=anchored_entity,
                )
                if inferred_probe is not None:
                    relation_probed = inferred_probe[0]
                    anchor_position = inferred_probe[1]
                    probe_target_inferred = True
            should_repair_alias = entity_count == 0 or (
                entity_count > 0
                and path_count == 0
                and bool(relation_probed)
            )
            if not should_repair_alias:
                continue
            surface = str(
                anchored_entity.get("surface")
                or anchored_entity.get("chosen_alias")
                or ""
            ).strip()
            if not surface:
                continue
            clue = self._infer_entity_clue(question_text, surface)
            current_alias = str(
                anchored_entity.get("chosen_alias") or surface
            ).strip()
            anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
            if (
                anchor_role == "constraint_value"
                and clue.startswith("attribute_value.")
            ):
                alias_candidates = self._build_entity_alias_candidates(
                    surface,
                    entity_clue=clue,
                )
            elif anchor_role in {"type_set", "shared_type", "anchor_value"} or clue.startswith("type_constraint"):
                alias_candidates = self._build_type_constraint_alias_candidates(surface)
            else:
                alias_candidates = self._build_entity_alias_candidates(
                    surface,
                    entity_clue=clue,
                )
            candidate_counts: list[dict[str, Any]] = []
            replacement_alias = ""
            replacement_path_count: int | None = None

            def _probe_candidate_relation_support(candidate_alias: str) -> int | None:
                if not relation_probed:
                    return None
                candidate_path_count = self._probe_anchor_path_count(
                    candidate_alias,
                    relation_probed,
                    anchor_position=anchor_position,
                    timeout_s=1.5,
                )
                if (
                    probe_target_inferred
                    and int(candidate_path_count or 0) <= 0
                    and anchor_position in {"subject", "object"}
                ):
                    fallback_anchor_position = (
                        "object" if anchor_position == "subject" else "subject"
                    )
                    fallback_path_count = self._probe_anchor_path_count(
                        candidate_alias,
                        relation_probed,
                        anchor_position=fallback_anchor_position,
                        timeout_s=1.5,
                )
                    if int(fallback_path_count or 0) > int(candidate_path_count or 0):
                        return fallback_path_count
                return candidate_path_count

            for candidate_alias in alias_candidates:
                if candidate_alias == current_alias:
                    continue
                candidate_count = self._probe_entity_name_count(
                    candidate_alias,
                    timeout_s=1.5,
                )
                candidate_path_count: int | None = None
                if candidate_count > 0 and relation_probed:
                    candidate_path_count = _probe_candidate_relation_support(
                        candidate_alias
                    )
                candidate_counts.append(
                    {
                        "candidate_alias": candidate_alias,
                        "entity_count": candidate_count,
                        "path_count": candidate_path_count,
                    }
                )
                if candidate_path_count is not None and candidate_path_count > 0:
                    replacement_alias = candidate_alias
                    replacement_path_count = candidate_path_count
                    break
                if candidate_count > 0 and entity_count == 0 and not relation_probed:
                    replacement_alias = candidate_alias
                    break
            if not replacement_alias:
                searched_aliases = self._probe_entity_name_candidates_by_token_search(
                    surface,
                )
                for candidate_alias in searched_aliases:
                    if candidate_alias in alias_candidates or candidate_alias == current_alias:
                        continue
                    candidate_count = self._probe_entity_name_count(
                        candidate_alias,
                        timeout_s=1.5,
                    )
                    candidate_path_count: int | None = None
                    if candidate_count > 0 and relation_probed:
                        candidate_path_count = _probe_candidate_relation_support(
                            candidate_alias
                        )
                    candidate_counts.append(
                        {
                            "candidate_alias": candidate_alias,
                            "entity_count": candidate_count,
                            "path_count": candidate_path_count,
                            "candidate_source": "token_search",
                        }
                    )
                    if (
                        candidate_path_count is not None
                        and candidate_path_count > 0
                        and self._token_search_alias_preserves_surface_semantics(
                            surface=surface,
                            candidate_alias=candidate_alias,
                        )
                    ):
                        replacement_alias = candidate_alias
                        replacement_path_count = candidate_path_count
                        break
                if (
                    not replacement_alias
                    and entity_count == 0
                    and searched_aliases
                ):
                    lexical_fallback_alias = self._select_high_confidence_token_search_alias(
                        surface=surface,
                        candidate_counts=candidate_counts,
                    )
                    if lexical_fallback_alias:
                        replacement_alias = lexical_fallback_alias
                    elif len(searched_aliases) == 1 and candidate_counts:
                        last_candidate = candidate_counts[-1]
                        if int(last_candidate.get("entity_count") or 0) > 0:
                            replacement_alias = str(
                                last_candidate.get("candidate_alias") or ""
                            ).strip()
            if not replacement_alias:
                repair_events.append(
                    {
                        "surface": surface,
                        "current_alias": current_alias,
                        "selected_alias": "",
                        "candidate_counts": candidate_counts,
                    }
                )
                continue
            selected_repairs[surface] = replacement_alias
            repair_events.append(
                {
                        "surface": surface,
                        "current_alias": current_alias,
                        "selected_alias": replacement_alias,
                        "selected_path_count": replacement_path_count,
                        "candidate_counts": candidate_counts,
                    }
            )

        if not selected_repairs:
            if repair_events:
                self._emit_generated_tools_event(
                    {
                        "event": "pal_alias_repair_probe_attempt",
                        "mode": "pal",
                        "applied": False,
                        "repairs": repair_events,
                    }
                )
            return dict(query_plan), grounding_card, []

        alias_overrides.update(selected_repairs)
        repaired_query_plan = self._apply_anchor_alias_overrides(
            query_plan=query_plan,
            alias_overrides=alias_overrides,
        )
        repaired_grounding_card = self._build_pal_grounding_card(
            task_question,
            relation_grounding=relation_grounding,
            alias_overrides=alias_overrides,
        )
        self._emit_generated_tools_event(
            {
                "event": "pal_alias_repair_probe_attempt",
                "mode": "pal",
                "applied": True,
                "repairs": repair_events,
            }
        )
        feedback = [
            "plausibility_feedback:anchor_alias_repaired — a failed anchor alias was replaced using a live KG existence probe",
        ]
        for surface, alias in selected_repairs.items():
            feedback.append(f"anchor_alias_override:{surface}=>{alias}")
        feedback.append(
            "repair_hint:use_selected_aliases — use the selected anchor_alias_override values from the grounding card and query plan exactly"
        )
        return repaired_query_plan, repaired_grounding_card, feedback

    def _infer_anchor_relation_probe_from_query_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        anchored_entity: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        anchor_role = self._normalize_relation_role(anchored_entity.get("role")) or "anchor"
        relation_paths = query_plan.get("relation_paths") or []
        for relation_path in relation_paths:
            if not isinstance(relation_path, Mapping):
                continue
            relation = str(relation_path.get("relation") or "").strip()
            if not relation:
                continue
            direction = str(relation_path.get("direction") or "forward").strip().lower()
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            if anchor_role == from_role:
                return relation, self._anchor_position_for_relation_endpoint(
                    endpoint_side="from",
                    direction=direction,
                    endpoint_role=from_role,
                    counterpart_role=to_role,
                )
            if anchor_role == to_role:
                return relation, self._anchor_position_for_relation_endpoint(
                    endpoint_side="to",
                    direction=direction,
                    endpoint_role=to_role,
                    counterpart_role=from_role,
                )
        return None

    def _probe_entity_name_candidates_by_token_search(
        self,
        entity: str,
        *,
        timeout_s: float = 2.0,
        limit: int = 8,
    ) -> list[str]:
        if not _SPARQL_PROBE_AVAILABLE:
            return []
        raw_entity = str(entity or "").strip()
        if not raw_entity:
            return []
        informative_tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9&'.-]+", raw_entity.lower())
            if len(token) > 2 and token not in {"the", "and", "of", "for", "inc", "ltd"}
        ]
        if not informative_tokens:
            return []
        ranked_tokens = sorted(
            informative_tokens,
            key=lambda token: (-len(token), informative_tokens.index(token)),
        )
        endpoint = self._get_runtime_sparql_endpoint()
        candidates: list[str] = []
        seen_lower: set[str] = set()
        for token in ranked_tokens[:2]:
            sparql = (
                "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
                "SELECT DISTINCT ?name WHERE {\n"
                "  ?entity fb:type.object.name ?name .\n"
                f'  FILTER(CONTAINS(LCASE(STR(?name)), "{token}"))\n'
                f"}} LIMIT {max(limit * 2, 10)}"
            )
            values = self._run_probe_sparql_query(
                endpoint=endpoint,
                sparql=sparql,
                timeout_s=timeout_s,
            )
            ranked_names = sorted(
                {
                    str(value or "").strip()
                    for value in values
                    if str(value or "").strip()
                },
                key=lambda name: (
                    0 if name.lower() == token else 1,
                    0 if name.lower().startswith(token + " ") or name.lower() == token else 1,
                    abs(len(name) - len(token)),
                    name.lower(),
                ),
            )
            for name in ranked_names:
                lowered = name.lower()
                if lowered in seen_lower:
                    continue
                seen_lower.add(lowered)
                candidates.append(name)
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def _token_search_alias_preserves_surface_semantics(
        self,
        *,
        surface: str,
        candidate_alias: str,
    ) -> bool:
        normalized_surface = self._normalize_alias_surface_text(surface)
        normalized_candidate = self._normalize_alias_surface_text(candidate_alias)
        if not normalized_surface or not normalized_candidate:
            return False
        if (
            normalized_surface == normalized_candidate
            or normalized_surface in normalized_candidate
            or normalized_candidate in normalized_surface
        ):
            return True

        surface_tokens = self._tokenize_alias_surface(surface)
        candidate_tokens = self._tokenize_alias_surface(candidate_alias)
        if not surface_tokens or not candidate_tokens:
            return False
        overlap_count = len(surface_tokens & candidate_tokens)
        if len(surface_tokens) <= 1:
            return overlap_count >= 1
        return overlap_count >= 2

    def _select_high_confidence_token_search_alias(
        self,
        *,
        surface: str,
        candidate_counts: Sequence[Mapping[str, Any]],
    ) -> str:
        surface_text = str(surface or "").strip()
        if not surface_text:
            return ""
        surface_tokens = self._tokenize_alias_surface(surface_text)
        if len(surface_tokens) < 2:
            return ""
        normalized_surface = self._normalize_alias_surface_text(surface_text)
        lexical_candidates: list[tuple[int, int, float, int, int, str]] = []
        prefix_candidates: list[tuple[int, int, float, int, int, str]] = []
        for candidate_payload in candidate_counts:
            if str(candidate_payload.get("candidate_source") or "").strip() != "token_search":
                continue
            candidate_alias = str(candidate_payload.get("candidate_alias") or "").strip()
            if not candidate_alias:
                continue
            entity_count = int(candidate_payload.get("entity_count") or 0)
            if entity_count <= 0 or entity_count > 50:
                continue
            candidate_tokens = self._tokenize_alias_surface(candidate_alias)
            if not candidate_tokens:
                continue
            overlap_count = len(surface_tokens & candidate_tokens)
            if overlap_count < 2:
                continue
            candidate_coverage = overlap_count / max(len(candidate_tokens), 1)
            surface_coverage = overlap_count / max(len(surface_tokens), 1)
            if candidate_coverage < 0.6:
                continue
            normalized_candidate = self._normalize_alias_surface_text(candidate_alias)
            prefix_match = int(
                bool(normalized_candidate)
                and (
                    normalized_surface.startswith(normalized_candidate)
                    or normalized_candidate.startswith(normalized_surface)
                )
            )
            score_tuple = (
                prefix_match,
                overlap_count,
                surface_coverage,
                -entity_count,
                -abs(len(normalized_candidate) - len(normalized_surface)),
                candidate_alias,
            )
            lexical_candidates.append(score_tuple)
            if prefix_match:
                prefix_candidates.append(score_tuple)
        if prefix_candidates:
            prefix_candidates.sort(reverse=True)
            return prefix_candidates[0][-1]
        if len(lexical_candidates) == 1:
            return lexical_candidates[0][-1]
        return ""

    def _tokenize_alias_surface(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9&'.-]+", str(text or "").lower())
            if len(token) > 2 and token not in {"the", "and", "of", "for", "inc", "ltd"}
        }

    def _normalize_alias_surface_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    def _merge_feedback_items(
        self,
        existing_feedback: Sequence[str],
        new_feedback: Sequence[str],
    ) -> list[str]:
        return self._canonicalize_feedback_items([*existing_feedback, *new_feedback])

    def _feedback_override_key(self, item: str) -> tuple[str, str] | None:
        normalized_item = str(item or "").strip()
        if not normalized_item:
            return None
        for prefix in ("anchor_alias_override:", "anchor_entity_override:"):
            if not normalized_item.startswith(prefix):
                continue
            remainder = normalized_item[len(prefix) :]
            surface, _, _ = remainder.partition("=>")
            normalized_surface = str(surface or "").strip()
            if normalized_surface:
                return prefix[:-1], normalized_surface
        return None

    def _canonicalize_feedback_items(self, items: Sequence[str]) -> list[str]:
        merged: list[str] = []
        seen_items: set[str] = set()
        override_indices: dict[tuple[str, str], int] = {}
        for item in items:
            normalized_item = str(item).strip()
            if not normalized_item:
                continue
            override_key = self._feedback_override_key(normalized_item)
            if override_key is not None:
                existing_index = override_indices.get(override_key)
                if existing_index is not None:
                    merged[existing_index] = normalized_item
                else:
                    override_indices[override_key] = len(merged)
                    merged.append(normalized_item)
                continue
            if normalized_item in seen_items:
                continue
            seen_items.add(normalized_item)
            merged.append(normalized_item)
        return merged

    def _extract_anchor_alias_overrides(
        self,
        query_plan: Mapping[str, Any],
    ) -> dict[str, str]:
        alias_overrides: dict[str, str] = {}
        for anchored_entity in query_plan.get("anchored_entities") or []:
            if not isinstance(anchored_entity, Mapping):
                continue
            surface = str(anchored_entity.get("surface") or "").strip()
            chosen_alias = str(anchored_entity.get("chosen_alias") or "").strip()
            if surface and chosen_alias and chosen_alias != surface:
                alias_overrides[surface] = chosen_alias
        return alias_overrides

    def _extract_anchor_alias_assignments(
        self,
        query_plan: Mapping[str, Any],
    ) -> dict[str, str]:
        alias_assignments: dict[str, str] = {}
        for anchored_entity in query_plan.get("anchored_entities") or []:
            if not isinstance(anchored_entity, Mapping):
                continue
            surface = str(anchored_entity.get("surface") or "").strip()
            chosen_alias = str(
                anchored_entity.get("chosen_alias") or surface
            ).strip()
            if surface and chosen_alias:
                alias_assignments[surface] = chosen_alias
        return alias_assignments

    def _apply_anchor_alias_overrides(
        self,
        *,
        query_plan: Mapping[str, Any],
        alias_overrides: Mapping[str, str],
    ) -> dict[str, Any]:
        repaired_query_plan: dict[str, Any] = copy.deepcopy(dict(query_plan))
        normalized_aliases: list[dict[str, Any]] = [
            dict(item)
            for item in repaired_query_plan.get("normalized_aliases") or []
            if isinstance(item, Mapping)
        ]
        seen_surfaces = {
            str(item.get("surface") or "").strip()
            for item in normalized_aliases
            if str(item.get("surface") or "").strip()
        }

        for anchored_entity in repaired_query_plan.get("anchored_entities") or []:
            if not isinstance(anchored_entity, dict):
                continue
            surface = str(anchored_entity.get("surface") or "").strip()
            replacement_alias = str(alias_overrides.get(surface) or "").strip()
            if not replacement_alias:
                continue
            anchored_entity["chosen_alias"] = replacement_alias

        for normalized_alias in normalized_aliases:
            surface = str(normalized_alias.get("surface") or "").strip()
            replacement_alias = str(alias_overrides.get(surface) or "").strip()
            if not replacement_alias:
                continue
            normalized_alias["chosen_alias"] = replacement_alias
            normalized_alias["reason"] = (
                f"live probe repair selected alias {replacement_alias!r}"
            )

        for surface, replacement_alias in alias_overrides.items():
            if surface in seen_surfaces:
                continue
            normalized_aliases.append(
                {
                    "surface": surface,
                    "chosen_alias": replacement_alias,
                    "reason": f"live probe repair selected alias {replacement_alias!r}",
                }
            )

        repaired_query_plan["normalized_aliases"] = normalized_aliases
        return repaired_query_plan

    def _augment_grounding_with_dynamic_probe_on_path_failure(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> tuple[str, list[dict[str, str]], list[str]]:
        if not anchor_probe_results:
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=self._extract_anchor_alias_overrides(query_plan),
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        has_path_failure = any(
            getattr(result, "entity_count", -1) > 0
            and getattr(result, "path_count", None) == 0
            for result in anchor_probe_results
        )
        if not has_path_failure:
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=self._extract_anchor_alias_overrides(query_plan),
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        answer_target_phrase = self._extract_answer_target_phrase(
            self._split_task_question(task_question)[0]
        )
        domain_hints = self._infer_domain_hints(task_question)
        probe_entities = [
            str(anchor.get("chosen_alias") or anchor.get("surface") or "").strip()
            for anchor in (query_plan.get("anchored_entities") or [])
            if isinstance(anchor, Mapping)
            and str(anchor.get("chosen_alias") or anchor.get("surface") or "").strip()
        ]
        anchored_entities = [
            anchor
            for anchor in (query_plan.get("anchored_entities") or [])
            if isinstance(anchor, Mapping)
        ]
        if anchored_entities:
            dynamic_candidates = self._probe_dynamic_relation_candidates_for_anchors(
                anchored_entities=anchored_entities,
                answer_target_phrase=answer_target_phrase,
                domain_hints=domain_hints,
                question_text=self._split_task_question(task_question)[0],
            )
        else:
            dynamic_candidates = self._probe_dynamic_relation_candidates(
                entities=probe_entities,
                answer_target_phrase=answer_target_phrase,
                domain_hints=domain_hints,
                question_text=self._split_task_question(task_question)[0],
            )
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()
        answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()
        dynamic_candidates = self._normalize_grounded_relation_candidates(
            relation_candidates=dynamic_candidates,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            entities=probe_entities,
        )
        merged_grounding = self._merge_relation_grounding_candidates(
            relation_grounding,
            dynamic_candidates,
            prefer_extra=True,
        )
        if len(merged_grounding) == len(list(relation_grounding)):
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=self._extract_anchor_alias_overrides(query_plan),
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )
        grounding_card = self._build_pal_grounding_card(
            task_question,
            relation_grounding=merged_grounding,
            alias_overrides=self._extract_anchor_alias_overrides(query_plan),
        )
        self._emit_generated_tools_event(
            {
                "event": "pal_dynamic_grounding_repair_augmented",
                "mode": "pal",
                "added_relations": [
                    candidate.get("relation")
                    for candidate in dynamic_candidates
                    if isinstance(candidate, Mapping)
                ],
                "total_relation_candidates": len(merged_grounding),
            }
        )
        feedback = [
            "plausibility_feedback:dynamic_grounding_augmented — the prior grounded path was empty, so live dynamic predicate candidates were added for the next repair attempt",
            "repair_hint:consider_dynamic_candidates — if the curated path stays empty, prefer a dynamic_probe relation from the augmented grounding card that actually links out from the resolved anchor entity",
        ]
        if dynamic_candidates:
            feedback.append(
                "repair_hint:prefer_single_dynamic_relation — when switching to dynamic_probe grounding, pick one best-supported relation path rather than adding UNION branches that are not in the query plan"
            )
            prioritized_relations = [
                str(candidate.get("relation") or "")
                for candidate in dynamic_candidates[:4]
                if str(candidate.get("relation") or "")
            ]
            if prioritized_relations:
                feedback.append(
                    "dynamic_candidate_priority:" + ", ".join(prioritized_relations)
                )
        return grounding_card, merged_grounding, feedback

    def _refresh_dynamic_grounding_after_anchor_alias_change(
        self,
        *,
        task_question: str,
        previous_alias_assignments: Mapping[str, str],
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
    ) -> tuple[str, list[dict[str, str]], list[str]]:
        current_alias_assignments = self._extract_anchor_alias_assignments(query_plan)
        alias_overrides = self._extract_anchor_alias_overrides(query_plan)
        if current_alias_assignments == dict(previous_alias_assignments):
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=alias_overrides,
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        if any(
            str(candidate.get("grounding_source") or "").strip().lower() == "curated"
            for candidate in relation_grounding
            if isinstance(candidate, Mapping)
        ):
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=alias_overrides,
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        anchored_entities = [
            anchor
            for anchor in (query_plan.get("anchored_entities") or [])
            if isinstance(anchor, Mapping)
        ]
        if not anchored_entities:
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=alias_overrides,
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        question_text = self._split_task_question(task_question)[0]
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        domain_hints = self._infer_domain_hints(task_question)
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()
        answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()

        dynamic_candidates = self._probe_dynamic_relation_candidates_for_anchors(
            anchored_entities=anchored_entities,
            answer_target_phrase=answer_target_phrase,
            domain_hints=domain_hints,
            question_text=question_text,
        )
        dynamic_candidates = self._normalize_grounded_relation_candidates(
            relation_candidates=dynamic_candidates,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            entities=[
                str(anchor.get("chosen_alias") or anchor.get("surface") or "").strip()
                for anchor in anchored_entities
                if str(anchor.get("chosen_alias") or anchor.get("surface") or "").strip()
            ],
        )
        merged_grounding = self._merge_relation_grounding_candidates(
            relation_grounding,
            dynamic_candidates,
            prefer_extra=True,
        )
        grounding_card = self._build_pal_grounding_card(
            task_question,
            relation_grounding=merged_grounding,
            alias_overrides=alias_overrides,
        )
        if len(merged_grounding) == len(list(relation_grounding)):
            return grounding_card, merged_grounding, []

        alias_changes = [
            f"{surface}=>{alias}"
            for surface, alias in current_alias_assignments.items()
            if alias != str(previous_alias_assignments.get(surface) or alias)
        ]
        self._emit_generated_tools_event(
            {
                "event": "pal_dynamic_grounding_alias_refresh",
                "mode": "pal",
                "alias_changes": alias_changes,
                "added_relations": [
                    candidate.get("relation")
                    for candidate in dynamic_candidates
                    if isinstance(candidate, Mapping)
                ],
                "total_relation_candidates": len(merged_grounding),
            }
        )
        feedback = [
            "plausibility_feedback:dynamic_grounding_refreshed_after_alias_change — a refreshed plan switched to a new anchor alias, so dynamic grounding was reprobed for that alias before the next attempt",
            "repair_hint:prefer_candidates_from_current_alias — when an anchor alias changes, prefer dynamic candidates that were probed from the current chosen_alias instead of stale candidates from the old surface form",
        ]
        if alias_changes:
            feedback.append("alias_change_reprobe:" + ", ".join(alias_changes))
        return grounding_card, merged_grounding, feedback

    def _augment_grounding_for_structural_repair(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
        verdict: PlausibilityVerdict,
    ) -> tuple[str, list[dict[str, str]], list[str]]:
        grounding_candidates = [
            dict(candidate)
            for candidate in relation_grounding
            if isinstance(candidate, Mapping)
        ]
        feedback: list[str] = []
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()

        if (
            verdict.verdict == "repairable_grounded_empty_result"
            and self._anchor_paths_live_but_join_is_empty(anchor_probe_results)
        ):
            used_relations = self._relation_names_from_plan(query_plan)
            unused_relations = self._unused_grounded_relation_names(
                query_plan=query_plan,
                relation_grounding=grounding_candidates,
            )
            scaffold_signature = self._build_scaffold_signature(query_plan)
            shared_answer_variable = str(
                query_plan.get("shared_answer_variable")
                or query_plan.get("candidate_set_variable")
                or query_plan.get("count_set_variable")
                or ""
            ).strip()
            feedback.extend(
                [
                    "plausibility_feedback:join_overlap_empty — each anchor path is live, but the combined scaffold still returns no bindings",
                    "repair_hint:do_not_reuse_same_scaffold_family — keep the anchor entities but choose a different grounded scaffold family or insert a grounded bridge to a different shared answer set",
                    "repair_hint:change_scaffold_family_not_query_wording — do not merely rewrite FILTERs or alias bindings if the same relation family already failed",
                ]
            )
            if scaffold_signature:
                feedback.append(f"dead_scaffold_signature:{scaffold_signature}")
            if used_relations:
                feedback.append("failed_relation_family:" + ", ".join(used_relations))
            if unused_relations:
                feedback.append(
                    "unused_grounded_relations:" + ", ".join(unused_relations[:6])
                )
            anchor_clue_feedback = self._build_anchor_clue_feedback(
                task_question,
                query_plan,
                relation_grounding=grounding_candidates,
            )
            if anchor_clue_feedback:
                feedback.extend(anchor_clue_feedback)
            projection_answer_variable = self._extract_primary_projection_variable(
                query_plan
            )
            if (
                projection_answer_variable
                and self._normalize_variable_token(projection_answer_variable)
                != self._normalize_variable_token(shared_answer_variable)
            ):
                feedback.extend(
                    [
                        "plausibility_feedback:projected_answer_intersection_available — the failed join happened on an upstream bridge variable, but the plan already has a grounded projection to the requested answer variable",
                        "repair_hint:intersect_on_projected_answer — if anchor paths are live but the upstream bridge variable has empty overlap, keep separate branch-local candidate variables, project each branch to the requested answer variable via the grounded projection relation, and intersect on that projected answer variable instead",
                    ]
                )
            synthesized_bridge_candidates = self._synthesize_anchor_specific_bridge_candidates(
                task_question=task_question,
                query_plan=query_plan,
                relation_grounding=grounding_candidates,
            )
            if synthesized_bridge_candidates:
                grounding_candidates = self._merge_relation_grounding_candidates(
                    grounding_candidates,
                    synthesized_bridge_candidates,
                    prefer_extra=True,
                )
                feedback.extend(
                    [
                        "plausibility_feedback:anchor_bridge_candidates_added — generic grounded bridge relations were specialized to the current anchors and shared bridge variable",
                        "repair_hint:prefer_anchor_specific_bridge_candidates — when a curated relation can bind an anchor directly to the bridge/shared variable, prefer that anchor-specific bridge over a generic shared-answer variant",
                    ]
                )
            anchor_dynamic_candidates = self._probe_dynamic_relation_candidates_for_anchors(
                anchored_entities=query_plan.get("anchored_entities") or [],
                answer_target_phrase=self._extract_answer_target_phrase(
                    self._split_task_question(task_question)[0]
                ),
                domain_hints=self._infer_domain_hints(task_question),
                question_text=self._split_task_question(task_question)[0],
            )
            if anchor_dynamic_candidates:
                asymmetric_anchor_clues = self._has_asymmetric_anchor_clues(
                    task_question=task_question,
                    anchored_entities=query_plan.get("anchored_entities") or [],
                )
                anchor_dynamic_candidates = self._prioritize_anchor_bridge_candidates(
                    anchor_dynamic_candidates,
                    bridge_variable=shared_answer_variable,
                )
                shared_anchor_relation_families: list[str] = []
                if not asymmetric_anchor_clues:
                    (
                        anchor_dynamic_candidates,
                        shared_anchor_relation_families,
                    ) = self._prioritize_shared_anchor_relation_families(
                        anchor_dynamic_candidates
                    )
                grounding_candidates = self._merge_relation_grounding_candidates(
                    grounding_candidates,
                    anchor_dynamic_candidates,
                    prefer_extra=True,
                )
                self._emit_generated_tools_event(
                    {
                        "event": "pal_structural_repair_anchor_dynamic_candidates_added",
                        "mode": "pal",
                        "query_shape": query_shape,
                        "added_relation_candidates": anchor_dynamic_candidates,
                    }
                )
                feedback.extend(
                    [
                        "plausibility_feedback:anchor_dynamic_grounding_augmented — each anchor path is live but the join is empty, so live anchor-side relations were added for scaffold switching",
                        "repair_hint:consider_upstream_anchor_relations — if the current shared-answer intersection is empty, consider live anchor-side relations that reach an upstream shared entity set",
                        "repair_hint:prefer_live_anchor_relations_over_inferred_clues — when live anchor-side probe evidence conflicts with a weaker question-side clue, trust the live relation evidence first",
                    ]
                )
                if shared_answer_variable:
                    feedback.append(
                        "repair_hint:prefer_direct_bridge_to_failed_shared_variable — if a live grounded relation lands directly on the same shared bridge variable, prefer that direct bridge over a longer detour through another upstream set"
                    )
                if asymmetric_anchor_clues:
                    feedback.append(
                        "repair_hint:prefer_asymmetric_bridge_scaffold — if anchors play different semantic roles in the question, they may use different relation families as long as they converge on the same bridge/shared variable"
                    )
                if shared_anchor_relation_families:
                    feedback.extend(
                        [
                            "repair_hint:prefer_shared_anchor_relation_family — if multiple anchors have live relations in the same family, use that shared family to reach the shared answer set before mixing asymmetric anchor paths",
                            "shared_anchor_relation_family:"
                            + ", ".join(shared_anchor_relation_families[:4]),
                        ]
                    )
                grouped_dynamic_priorities = self._group_dynamic_anchor_candidates(
                    anchor_dynamic_candidates,
                    anchored_entities=query_plan.get("anchored_entities") or [],
                )
                for surface, relations in grouped_dynamic_priorities.items():
                    if relations:
                        feedback.append(
                            f"anchor_dynamic_priority:{surface}="
                            + ", ".join(relations[:4])
                        )
            pivot_dynamic_candidates = self._probe_dynamic_relation_candidates_for_live_pivots(
                query_plan=query_plan,
                anchor_probe_results=anchor_probe_results,
                answer_target_phrase=self._extract_answer_target_phrase(
                    self._split_task_question(task_question)[0]
                ),
                domain_hints=self._infer_domain_hints(task_question),
                question_text=self._split_task_question(task_question)[0],
            )
            if pivot_dynamic_candidates:
                grounding_candidates = self._merge_relation_grounding_candidates(
                    grounding_candidates,
                    pivot_dynamic_candidates,
                    prefer_extra=True,
                )
                self._emit_generated_tools_event(
                    {
                        "event": "pal_structural_repair_pivot_dynamic_candidates_added",
                        "mode": "pal",
                        "query_shape": query_shape,
                        "added_relation_candidates": pivot_dynamic_candidates,
                    }
                )
                feedback.extend(
                    [
                        "plausibility_feedback:pivot_dynamic_grounding_augmented — a live anchor-side path reached an intermediate pivot node, so second-hop bridge relations from that pivot were added",
                        "repair_hint:prefer_pivot_bridge_completion — when a live anchor-side path ends on a pivot node, prefer a grounded second hop from that pivot to the shared answer set before abandoning the anchor branch",
                    ]
                )

        if verdict.verdict == "rejected_dangerous_overreach":
            overreach_reasons = {
                str(reason or "").strip()
                for reason in (verdict.reasons or [])
                if str(reason or "").strip().startswith("dangerous_overreach:")
            }
            if overreach_reasons & {
                "dangerous_overreach:broad_type_expansion",
                "dangerous_overreach:ontology_dump",
                "dangerous_overreach:clipped_subset",
            }:
                used_relations = self._relation_names_from_plan(query_plan)
                unused_relations = self._unused_grounded_relation_names(
                    query_plan=query_plan,
                    relation_grounding=grounding_candidates,
                )
                scaffold_signature = self._build_scaffold_signature(query_plan)
                feedback.extend(
                    [
                        "plausibility_feedback:scaffold_overreaches_answer_semantics — the current scaffold returns a broader or clipped answer set than the question allows",
                        "repair_hint:do_not_reuse_same_scaffold_family — keep the anchor entities but switch to a different grounded scaffold family or add an explicit grounded filter before retrying",
                        "repair_hint:change_scaffold_family_not_query_wording — do not keep the same relation family and only rewrite aliases or projection names",
                    ]
                )
                if scaffold_signature:
                    feedback.append(f"dead_scaffold_signature:{scaffold_signature}")
                if used_relations:
                    feedback.append("failed_relation_family:" + ", ".join(used_relations))
                if unused_relations:
                    feedback.append(
                        "unused_grounded_relations:" + ", ".join(unused_relations[:6])
                    )

        if verdict.verdict == "repairable_bad_count_set":
            pivot_candidates = self._synthesize_pivot_relation_candidates(
                query_shape=query_shape,
                relation_grounding=grounding_candidates,
                anchor_probe_results=anchor_probe_results,
            )
            if pivot_candidates:
                grounding_candidates = self._merge_relation_grounding_candidates(
                    grounding_candidates,
                    pivot_candidates,
                    prefer_extra=True,
                )
                self._emit_generated_tools_event(
                    {
                        "event": "pal_structural_repair_candidates_added",
                        "mode": "pal",
                        "query_shape": query_shape,
                        "added_relation_candidates": pivot_candidates,
                    }
                )
                feedback.extend(
                    [
                        "plausibility_feedback:anchor_role_reinterpretation_candidates_added — the anchor exists but the direct anchored family is empty, so pivot-friendly grounded candidates were added",
                        "repair_hint:insert_pivot_before_reusing_curated_family — choose one strong dynamic_probe relation from the resolved anchor to a pivot entity, then apply a reinterpreted curated relation from that pivot to the answer/count set",
                        "repair_hint:do_not_repeat_dead_direct_anchor_family — if a direct anchored count/lookup relation was proven empty, do not use that same direct anchor-to-set relation again without a pivot",
                        "repair_hint:preserve_count_target_family_after_pivot — when a direct counted relation family is dead but a pivot-friendly version of that same family exists, preserve that counted family after the pivot instead of switching to a sibling relation",
                    ]
                )
                dynamic_priorities = [
                    str(candidate.get("relation") or "")
                    for candidate in grounding_candidates
                    if str(candidate.get("grounding_source") or "").strip().lower()
                    == "dynamic_probe"
                    and str(candidate.get("relation") or "").strip()
                ]
                if dynamic_priorities:
                    feedback.append(
                        "pivot_candidate_priority:" + ", ".join(dynamic_priorities[:4])
                    )

        grounding_card = self._build_pal_grounding_card(
            task_question,
            relation_grounding=grounding_candidates,
            alias_overrides=self._extract_anchor_alias_overrides(query_plan),
        )
        return grounding_card, grounding_candidates, feedback

    def _group_dynamic_anchor_candidates(
        self,
        dynamic_candidates: Sequence[Mapping[str, str]],
        *,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[str]]:
        surface_by_role: dict[str, str] = {}
        for anchored_entity in anchored_entities:
            if not isinstance(anchored_entity, Mapping):
                continue
            role = self._normalize_relation_role(anchored_entity.get("role"))
            surface = str(
                anchored_entity.get("surface")
                or anchored_entity.get("chosen_alias")
                or ""
            ).strip()
            if role and surface:
                surface_by_role[role] = surface

        grouped: dict[str, list[str]] = {}
        for candidate in dynamic_candidates:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation:
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            anchor_role = ""
            if from_role in {"anchor", "anchor_a", "anchor_b"}:
                anchor_role = from_role
            elif to_role in {"anchor", "anchor_a", "anchor_b"}:
                anchor_role = to_role
            surface = surface_by_role.get(anchor_role)
            if not surface:
                continue
            relations = grouped.setdefault(surface, [])
            if relation not in relations:
                relations.append(relation)
        return grouped

    def _has_asymmetric_anchor_clues(
        self,
        *,
        task_question: str,
        anchored_entities: Sequence[Mapping[str, Any]],
    ) -> bool:
        question_text, _ = self._split_task_question(task_question)
        clues: list[str] = []
        for anchored_entity in anchored_entities[:2]:
            if not isinstance(anchored_entity, Mapping):
                continue
            surface = str(
                anchored_entity.get("surface")
                or anchored_entity.get("chosen_alias")
                or ""
            ).strip()
            if not surface:
                continue
            clue = self._infer_entity_clue(question_text, surface)
            if clue and clue != "surface_constraint" and clue not in clues:
                clues.append(clue)
        return len(clues) >= 2

    def _candidate_non_anchor_endpoint_token(
        self,
        candidate: Mapping[str, Any],
    ) -> str:
        from_role = self._normalize_relation_role(candidate.get("from_role"))
        to_role = self._normalize_relation_role(candidate.get("to_role"))
        if from_role in {"anchor", "anchor_a", "anchor_b"}:
            return self._normalize_variable_token(candidate.get("to"))
        if to_role in {"anchor", "anchor_a", "anchor_b"}:
            return self._normalize_variable_token(candidate.get("from"))
        return ""

    def _anchor_clue_matches_endpoint_label(
        self,
        *,
        clue: str,
        endpoint_label: str,
    ) -> bool:
        token = self._normalize_variable_token(endpoint_label)
        if not token:
            return False
        clue_matches: dict[str, set[str]] = {
            "active_ingredient": {
                "active_ingredient",
                "active_moiety",
                "ingredient",
                "drug_ingredient",
            },
            "formulation_input": {
                "drug",
                "drug_or_ingredient",
                "formulation",
                "drug_formulation",
            },
            "source_constraint": {
                "source",
                "milk_source",
                "parent",
                "owner",
                "producer",
            },
            "origin_constraint": {
                "breed",
                "country",
                "location",
                "origin",
                "region",
            },
        }
        return token in clue_matches.get(clue, set())

    def _synthesize_anchor_specific_bridge_candidates(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        question_text, _ = self._split_task_question(task_question)
        bridge_variable = str(
            query_plan.get("shared_answer_variable")
            or query_plan.get("candidate_set_variable")
            or ""
        ).strip()
        bridge_token = self._normalize_variable_token(bridge_variable)
        if not bridge_token:
            return []

        synthesized: list[dict[str, str]] = []
        for anchored_entity in query_plan.get("anchored_entities") or []:
            if not isinstance(anchored_entity, Mapping):
                continue
            anchor_alias = str(
                anchored_entity.get("chosen_alias")
                or anchored_entity.get("surface")
                or ""
            ).strip()
            anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
            if not anchor_alias or anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
                continue
            clue = self._infer_entity_clue(question_text, anchor_alias)
            if clue == "surface_constraint":
                continue

            for candidate in relation_grounding:
                if not isinstance(candidate, Mapping):
                    continue
                if str(candidate.get("grounding_source") or "").strip().lower() != "curated":
                    continue
                relation = str(candidate.get("relation") or "").strip()
                direction = str(candidate.get("direction") or "").strip().lower()
                from_label = str(candidate.get("from") or "").strip()
                to_label = str(candidate.get("to") or "").strip()
                if not relation or direction not in {"forward", "reverse"}:
                    continue

                if (
                    direction == "forward"
                    and self._normalize_variable_token(to_label) == bridge_token
                    and self._anchor_clue_matches_endpoint_label(
                        clue=clue,
                        endpoint_label=from_label,
                    )
                ):
                    synthesized.append(
                        {
                            "relation": relation,
                            "direction": direction,
                            "from": anchor_alias,
                            "to": bridge_variable,
                            "from_role": anchor_role,
                            "to_role": "candidate_set",
                            "grounding_source": "curated",
                            "support": "curated_anchor_bridge_synthesized",
                            "use_when": str(candidate.get("use_when") or ""),
                        }
                    )
                elif (
                    direction == "reverse"
                    and self._normalize_variable_token(from_label) == bridge_token
                    and self._anchor_clue_matches_endpoint_label(
                        clue=clue,
                        endpoint_label=to_label,
                    )
                ):
                    synthesized.append(
                        {
                            "relation": relation,
                            "direction": direction,
                            "from": bridge_variable,
                            "to": anchor_alias,
                            "from_role": "candidate_set",
                            "to_role": anchor_role,
                            "grounding_source": "curated",
                            "support": "curated_anchor_bridge_synthesized",
                            "use_when": str(candidate.get("use_when") or ""),
                        }
                    )
        return synthesized

    def _extract_primary_projection_variable(
        self,
        query_plan: Mapping[str, Any],
    ) -> str:
        for item in query_plan.get("projection") or []:
            token = str(item or "").strip()
            if not token:
                continue
            normalized = self._normalize_variable_token(token)
            if "name" in normalized or "label" in normalized:
                continue
            return token
        return ""

    def _build_projected_answer_intersection_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
        anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "multi_anchor_intersection":
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        anchor_roles = [
            self._normalize_relation_role(item.get("role"))
            for item in anchored_entities
            if self._normalize_relation_role(item.get("role")) in {"anchor", "anchor_a", "anchor_b"}
        ]
        if len(anchor_roles) < 2:
            return None

        bridge_variable = str(
            query_plan.get("shared_answer_variable")
            or query_plan.get("candidate_set_variable")
            or ""
        ).strip()
        answer_variable = self._extract_primary_projection_variable(query_plan)
        bridge_token = self._normalize_variable_token(bridge_variable)
        answer_token = self._normalize_variable_token(answer_variable)
        if not bridge_token or not answer_token or bridge_token == answer_token:
            return None

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if len(relation_paths) < 3:
            return None

        projection_template: Optional[dict[str, Any]] = None
        anchor_paths_by_role: dict[str, dict[str, Any]] = {}
        needed_anchor_roles = anchor_roles[:2]
        used_relations = set(self._relation_names_from_plan(query_plan))
        probe_by_role: dict[str, AnchorProbeResult] = {}
        for anchored_entity, probe_result in zip(anchored_entities, anchor_probe_results or []):
            role = self._normalize_relation_role(anchored_entity.get("role"))
            if role in {"anchor", "anchor_a", "anchor_b"}:
                probe_by_role[role] = probe_result
        for relation_path in relation_paths:
            from_token = self._normalize_variable_token(relation_path.get("from"))
            to_token = self._normalize_variable_token(relation_path.get("to"))
            path_anchor_roles = self._anchor_role_identities_for_path(
                relation_path=relation_path,
                anchor_count=len(anchor_roles),
                anchored_entities=anchored_entities,
            )
            if path_anchor_roles:
                for anchor_role in path_anchor_roles:
                    if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
                        continue
                    if bridge_token in {from_token, to_token} and anchor_role not in anchor_paths_by_role:
                        anchor_paths_by_role[anchor_role] = relation_path
            if (
                projection_template is None
                and bridge_token in {from_token, to_token}
                and answer_token in {from_token, to_token}
            ):
                projection_template = relation_path

        if projection_template is None:
            for candidate in relation_grounding:
                if not isinstance(candidate, Mapping):
                    continue
                path = dict(candidate)
                from_token = self._normalize_variable_token(path.get("from"))
                to_token = self._normalize_variable_token(path.get("to"))
                if bridge_token in {from_token, to_token} and answer_token in {from_token, to_token}:
                    projection_template = path
                    break

        if relation_grounding:
            for anchor_role in needed_anchor_roles:
                preferred_relation = str(
                    getattr(probe_by_role.get(anchor_role), "relation_probed", "") or ""
                ).strip()
                preferred_anchor_position = str(
                    getattr(probe_by_role.get(anchor_role), "anchor_position", "") or ""
                ).strip()
                current_selected = anchor_paths_by_role.get(anchor_role)
                selected_path: Optional[dict[str, Any]] = None
                for allow_used_relations in (False, True):
                    matching_candidates: list[dict[str, Any]] = []
                    for candidate in relation_grounding:
                        if not isinstance(candidate, Mapping):
                            continue
                        relation_name = str(candidate.get("relation") or "").strip()
                        if (
                            not allow_used_relations
                            and relation_name
                            and relation_name in used_relations
                            and relation_name != preferred_relation
                        ):
                            continue
                        path = dict(candidate)
                        from_role = self._normalize_relation_role(path.get("from_role"))
                        to_role = self._normalize_relation_role(path.get("to_role"))
                        from_token = self._normalize_variable_token(path.get("from"))
                        to_token = self._normalize_variable_token(path.get("to"))
                        matches_anchor = (
                            from_role == anchor_role and to_token == bridge_token
                        ) or (
                            to_role == anchor_role and from_token == bridge_token
                        )
                        if matches_anchor:
                            matching_candidates.append(path)
                    if matching_candidates:
                        selected_path = max(
                            matching_candidates,
                            key=lambda path: self._score_projected_answer_anchor_path(
                                relation_path=path,
                                anchor_role=anchor_role,
                                bridge_token=bridge_token,
                                preferred_relation=preferred_relation,
                                preferred_anchor_position=preferred_anchor_position,
                            ),
                        )
                    if selected_path is not None:
                        break
                if selected_path is not None:
                    if current_selected is None:
                        anchor_paths_by_role[anchor_role] = selected_path
                    else:
                        current_score = self._score_projected_answer_anchor_path(
                            relation_path=current_selected,
                            anchor_role=anchor_role,
                            bridge_token=bridge_token,
                            preferred_relation=preferred_relation,
                            preferred_anchor_position=preferred_anchor_position,
                        )
                        selected_score = self._score_projected_answer_anchor_path(
                            relation_path=selected_path,
                            anchor_role=anchor_role,
                            bridge_token=bridge_token,
                            preferred_relation=preferred_relation,
                            preferred_anchor_position=preferred_anchor_position,
                        )
                        required_improvement = 1 if preferred_relation else 3
                        if selected_score >= current_score + required_improvement:
                            anchor_paths_by_role[anchor_role] = selected_path

        if projection_template is None or not all(
            role in anchor_paths_by_role for role in needed_anchor_roles
        ):
            return None

        bridge_to_answer_forward = (
            self._normalize_variable_token(projection_template.get("from")) == bridge_token
            and self._normalize_variable_token(projection_template.get("to")) == answer_token
        )
        bridge_to_answer_reverse = (
            self._normalize_variable_token(projection_template.get("to")) == bridge_token
            and self._normalize_variable_token(projection_template.get("from")) == answer_token
        )
        if not bridge_to_answer_forward and not bridge_to_answer_reverse:
            return None

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_paths: list[dict[str, Any]] = []
        rewritten_constraints: list[dict[str, str]] = []

        for anchor_role in needed_anchor_roles:
            anchor_path = copy.deepcopy(anchor_paths_by_role[anchor_role])
            branch_variable = f"candidate_set_{anchor_role}"
            anchor_from_role = self._normalize_relation_role(anchor_path.get("from_role"))
            anchor_to_role = self._normalize_relation_role(anchor_path.get("to_role"))
            anchor_from_token = self._normalize_variable_token(anchor_path.get("from"))
            anchor_to_token = self._normalize_variable_token(anchor_path.get("to"))

            if anchor_from_token == bridge_token and anchor_to_role in {"anchor", "anchor_a", "anchor_b"}:
                anchor_path["from"] = branch_variable
                anchor_path["from_role"] = "candidate_set"
                anchor_path["to_role"] = anchor_role
            elif anchor_to_token == bridge_token and anchor_from_role in {"anchor", "anchor_a", "anchor_b"}:
                anchor_path["to"] = branch_variable
                anchor_path["to_role"] = "candidate_set"
                anchor_path["from_role"] = anchor_role
            else:
                return None

            projection_path = copy.deepcopy(projection_template)
            if bridge_to_answer_forward:
                projection_path["from"] = branch_variable
                projection_path["to"] = answer_variable
                projection_path["from_role"] = "candidate_set"
                projection_path["to_role"] = "shared_answer"
            else:
                projection_path["from"] = answer_variable
                projection_path["to"] = branch_variable
                projection_path["from_role"] = "shared_answer"
                projection_path["to_role"] = "candidate_set"

            rewritten_paths.append(anchor_path)
            rewritten_paths.append(projection_path)
            rewritten_constraints.append(
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": branch_variable,
                    "notes": (
                        f"{anchor_role} constrains a branch-local candidate set; "
                        f"project that branch to {answer_variable} before intersecting answers."
                    ),
                }
            )

        strategy = str(rewritten_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The previous scaffold proved each anchor path was live but the shared bridge variable had empty overlap, so switch to intersecting on the projected answer variable instead."
        )
        plan_rationale.append(
            f"Each anchor now builds its own branch-local {bridge_variable} set and projects it through the grounded relation to {answer_variable}; the intersection happens on {answer_variable}, not on {bridge_variable}."
        )

        rewritten_plan["shared_answer_variable"] = answer_variable
        rewritten_plan["candidate_set_variable"] = "candidate_set"
        rewritten_plan["join_structure"] = {
            "type": "intersection",
            "anchor_constraints": rewritten_constraints,
        }
        rewritten_plan["relation_paths"] = rewritten_paths
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by intersecting on the projected answer variable "
            f"{answer_variable} instead of the failed bridge variable {bridge_variable}."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _score_projected_answer_anchor_path(
        self,
        *,
        relation_path: Mapping[str, Any],
        anchor_role: str,
        bridge_token: str,
        preferred_relation: str = "",
        preferred_anchor_position: str = "",
    ) -> int:
        score = 0
        relation_name = str(relation_path.get("relation") or "").strip()
        from_role = self._normalize_relation_role(relation_path.get("from_role"))
        to_role = self._normalize_relation_role(relation_path.get("to_role"))
        from_token = self._normalize_variable_token(relation_path.get("from"))
        to_token = self._normalize_variable_token(relation_path.get("to"))
        grounding_source = str(
            relation_path.get("grounding_source") or ""
        ).strip().lower()
        support = str(relation_path.get("support") or "").strip().lower()

        if from_role == anchor_role and to_token == bridge_token:
            score += 6
        if to_role == anchor_role and from_token == bridge_token:
            score += 2
        if from_role == anchor_role and to_role in {"candidate_set", "shared_answer"}:
            score += 4
        if to_role == anchor_role and from_role in {"candidate_set", "shared_answer"}:
            score -= 1
        if grounding_source == "curated":
            score += 2
        elif grounding_source == "dynamic_probe":
            score += 1
        if "dynamic_probe_outgoing" in support or "curated_anchor_bridge" in support:
            score += 2
        if str(relation_path.get("direction") or "").strip().lower() == "forward":
            score += 1
        if preferred_relation and relation_name == preferred_relation:
            score += 8
        if preferred_anchor_position == "subject" and from_role == anchor_role:
            score += 3
        elif preferred_anchor_position == "object" and to_role == anchor_role:
            score += 3
        return score

    def _build_direct_dynamic_count_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
        anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_direct_relation":
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) != 1:
            return None
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role"))
        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        dead_probe = next(
            (
                result
                for result in (anchor_probe_results or [])
                if getattr(result, "entity_count", -1) > 0
                and getattr(result, "path_count", None) == 0
                and str(getattr(result, "relation_probed", "") or "").strip()
            ),
            None,
        )
        if dead_probe is None:
            return None
        dead_relation = str(getattr(dead_probe, "relation_probed", "") or "").strip()
        if not dead_relation:
            return None
        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        dead_path = next(
            (
                path
                for path in relation_paths
                if str(path.get("relation") or "").strip() == dead_relation
                and self._path_touches_anchor_role(path, anchor_role=anchor_role)
            ),
            None,
        )
        if dead_path is None:
            return None

        candidate = self._select_direct_dynamic_count_candidate(
            relation_grounding=relation_grounding,
            anchor_role=anchor_role,
            dead_relation=dead_relation,
            dead_target_token=self._count_set_token_for_path(
                relation_path=dead_path,
                query_plan=query_plan,
            ),
        )
        if candidate is None:
            return None

        count_variable = self._candidate_non_anchor_endpoint_token(candidate) or "count_set"
        rewritten_path = copy.deepcopy(candidate)
        from_role = self._normalize_relation_role(rewritten_path.get("from_role"))
        to_role = self._normalize_relation_role(rewritten_path.get("to_role"))
        if from_role == anchor_role and to_role == "candidate_set":
            rewritten_path["to"] = count_variable
            rewritten_path["to_role"] = "count_set"
        elif to_role == anchor_role and from_role == "candidate_set":
            rewritten_path["from"] = count_variable
            rewritten_path["from_role"] = "count_set"
        elif from_role == "count_set" and to_role == anchor_role:
            count_variable = (
                self._normalize_variable_token(rewritten_path.get("from"))
                or count_variable
            )
        elif to_role == "count_set" and from_role == anchor_role:
            count_variable = (
                self._normalize_variable_token(rewritten_path.get("to"))
                or count_variable
            )
        else:
            return None

        strategy = str(query_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (query_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The direct anchored count relation was proven empty, so switch to a live direct dynamic relation that already reaches the counted set from the resolved anchor."
        )

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["shared_answer_variable"] = count_variable
        rewritten_plan["candidate_set_variable"] = count_variable
        rewritten_plan["count_set_variable"] = count_variable
        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": count_variable,
                    "notes": (
                        f"{anchor_role} directly constrains the counted set via a live "
                        "dynamic relation from the anchor."
                    ),
                }
            ],
        }
        rewritten_plan["relation_paths"] = [rewritten_path]
        rewritten_plan["projection"] = ["count"]
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by replacing the dead direct count relation {dead_relation} "
            "with a live direct dynamic counted relation."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _build_pivot_preserving_count_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
        anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_direct_relation":
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) != 1:
            return None
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role"))
        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        dead_probe = next(
            (
                result
                for result in (anchor_probe_results or [])
                if getattr(result, "entity_count", -1) > 0
                and getattr(result, "path_count", None) == 0
                and str(getattr(result, "relation_probed", "") or "").strip()
            ),
            None,
        )
        if dead_probe is None:
            return None
        dead_relation = str(getattr(dead_probe, "relation_probed", "") or "").strip()
        if not dead_relation:
            return None

        direct_dynamic_plan = self._build_direct_dynamic_count_repair_plan(
            query_plan=query_plan,
            relation_grounding=relation_grounding,
            anchor_probe_results=anchor_probe_results,
        )
        if direct_dynamic_plan is not None:
            return direct_dynamic_plan

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        dead_path = next(
            (
                path
                for path in relation_paths
                if str(path.get("relation") or "").strip() == dead_relation
                and self._path_touches_anchor_role(path, anchor_role=anchor_role)
            ),
            None,
        )
        if dead_path is None:
            return None

        pivot_candidate = self._select_count_pivot_candidate(
            relation_grounding=relation_grounding,
            anchor_role=anchor_role,
            dead_relation=dead_relation,
        )
        preserved_target_candidate = self._select_preserved_count_target_candidate(
            query_plan=query_plan,
            relation_grounding=relation_grounding,
            dead_path=dead_path,
        )
        if pivot_candidate is None or preserved_target_candidate is None:
            return None

        pivot_variable = self._candidate_non_anchor_endpoint_token(pivot_candidate)
        if not pivot_variable:
            pivot_variable = self._normalize_variable_token(
                pivot_candidate.get("to") or pivot_candidate.get("from")
            )
        count_variable = self._candidate_count_endpoint_token(
            preserved_target_candidate,
            query_plan=query_plan,
        )
        if not pivot_variable or not count_variable:
            return None

        rewritten_pivot = copy.deepcopy(pivot_candidate)
        rewritten_target = copy.deepcopy(preserved_target_candidate)
        rewritten_pivot = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_pivot,
            target_role="candidate_set",
            replacement_variable=pivot_variable,
        )
        rewritten_target = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_target,
            target_role="candidate_set",
            replacement_variable=pivot_variable,
        )
        rewritten_target = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_target,
            target_role="count_set",
            replacement_variable=count_variable,
        )

        strategy = str(query_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (query_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The direct anchor-to-count-set family was proven empty by a live path probe, so insert a single pivot from the anchor before retrying that counted family."
        )
        plan_rationale.append(
            f"Preserve the counted target family {dead_relation} after the pivot instead of switching to a sibling relation family."
        )

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["shared_answer_variable"] = count_variable
        rewritten_plan["candidate_set_variable"] = pivot_variable
        rewritten_plan["count_set_variable"] = count_variable
        rewritten_plan["join_structure"] = {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": pivot_variable,
                    "notes": (
                        f"{anchor_role} binds to a pivot variable first; count the preserved "
                        f"{count_variable} set reached from that pivot."
                    ),
                }
            ],
        }
        rewritten_plan["relation_paths"] = [rewritten_pivot, rewritten_target]
        rewritten_plan["projection"] = ["count"]
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by inserting one pivot before reusing the counted "
            f"relation family {dead_relation}."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _select_shared_type_pivot_bridge_candidate(
        self,
        *,
        relation_grounding: Sequence[Mapping[str, str]],
    ) -> Optional[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if (
                from_role == "candidate_set"
                and to_role in {"shared_answer", "answer"}
            ):
                score = 12
            elif (
                to_role == "candidate_set"
                and from_role in {"shared_answer", "answer"}
            ):
                score = 9
            else:
                continue
            relation = str(candidate.get("relation") or "").strip()
            support = str(candidate.get("support") or "").strip().lower()
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if grounding_source == "dynamic_probe":
                score += 10
            elif grounding_source == "curated":
                score += 4
            if "pivot" in support:
                score += 8
            if relation.endswith(".organization"):
                score += 4
            if relation.endswith(".member"):
                score -= 4
            scored.append((score, dict(candidate)))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], str(item[1].get("relation") or "")))
        return scored[0][1]

    def _select_shared_answer_type_candidate(
        self,
        *,
        relation_grounding: Sequence[Mapping[str, str]],
        preferred_relation: str | None = None,
    ) -> Optional[dict[str, Any]]:
        preferred_relation_norm = str(preferred_relation or "").strip()
        scored: list[tuple[int, dict[str, Any]]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if from_role in {"shared_answer", "candidate_set"} and to_role in {"shared_type", "type_set"}:
                score = 8
            elif to_role in {"shared_answer", "candidate_set"} and from_role in {"shared_type", "type_set"}:
                score = 4
            else:
                continue
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if relation == preferred_relation_norm and preferred_relation_norm:
                score += 18
            elif preferred_relation_norm and relation == "type.object.type":
                score -= 4
            if relation == "type.object.type":
                score += 10
            elif relation == "organization.organization.organization_type":
                score += 6
            if grounding_source == "curated":
                score += 6
            elif grounding_source == "dynamic_probe":
                score += 2
            scored.append((score, dict(candidate)))
        synthesized: list[tuple[int, dict[str, Any]]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            direction = str(candidate.get("direction") or "").strip().lower()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if direction != "forward":
                continue
            if relation not in {
                "type.object.type",
                "organization.organization.organization_type",
            }:
                continue
            if from_role not in {"anchor", "anchor_a", "anchor_b"}:
                continue
            if to_role not in {"candidate_set", "shared_type", "type_set"}:
                continue
            synthesized_candidate = {
                "relation": relation,
                "direction": "forward",
                "from": "shared_answer",
                "to": "shared_type",
                "from_role": "shared_answer",
                "to_role": "shared_type",
                "grounding_source": str(candidate.get("grounding_source") or ""),
                "support": "anchor_type_family_reinterpreted",
            }
            score = 10 if relation == "organization.organization.organization_type" else 8
            if relation == preferred_relation_norm and preferred_relation_norm:
                score += 18
            elif preferred_relation_norm and relation == "type.object.type":
                score -= 4
            if str(candidate.get("grounding_source") or "").strip().lower() == "dynamic_probe":
                score += 4
            synthesized.append((score, synthesized_candidate))
        scored.extend(synthesized)
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], str(item[1].get("relation") or "")))
        return scored[0][1]

    def _select_anchor_shared_type_candidate(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_role: str,
    ) -> Optional[dict[str, Any]]:
        existing_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        for path in existing_paths:
            roles = {
                self._normalize_relation_role(path.get("from_role")),
                self._normalize_relation_role(path.get("to_role")),
            }
            if anchor_role in roles and roles & {"shared_type", "type_set"}:
                return path

        scored: list[tuple[int, dict[str, Any]]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            roles = {from_role, to_role}
            if anchor_role not in roles or not roles & {"shared_type", "type_set"}:
                continue
            score = 6
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if relation == "type.object.type":
                score += 10
            elif relation == "organization.organization.organization_type":
                score += 6
            elif relation == "type.type.instance":
                score += 2
            if grounding_source == "curated":
                score += 6
            elif grounding_source == "dynamic_probe":
                score += 2
            scored.append((score, dict(candidate)))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], str(item[1].get("relation") or "")))
        return scored[0][1]

    def _build_shared_type_pivot_bridge_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
        anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("query_shape") or "").strip().lower() not in {
            "shared_type_intersection",
            "multi_anchor_intersection",
        }:
            return None
        if not self._anchor_paths_live_but_join_is_empty(anchor_probe_results):
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) < 2:
            return None

        anchor_a_role = self._normalize_relation_role(anchored_entities[0].get("role"))
        anchor_b_role = self._normalize_relation_role(anchored_entities[1].get("role"))
        if anchor_a_role not in {"anchor", "anchor_a", "anchor_b"} or anchor_b_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        existing_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        anchor_a_paths = [
            dict(path)
            for path in existing_paths
            if self._path_touches_anchor_role(path, anchor_role=anchor_a_role)
            and {
                self._normalize_relation_role(path.get("from_role")),
                self._normalize_relation_role(path.get("to_role")),
            }
            & {"candidate_set", "shared_answer"}
        ]
        if not anchor_a_paths:
            return None

        anchor_b_type_candidate = self._select_anchor_shared_type_candidate(
            query_plan=query_plan,
            relation_grounding=relation_grounding,
            anchor_role=anchor_b_role,
        )
        bridge_candidate = self._select_shared_type_pivot_bridge_candidate(
            relation_grounding=relation_grounding,
        )
        shared_answer_type_candidate = self._select_shared_answer_type_candidate(
            relation_grounding=relation_grounding,
            preferred_relation=(
                str(anchor_b_type_candidate.get("relation") or "").strip()
                if isinstance(anchor_b_type_candidate, Mapping)
                else None
            ),
        )
        if (
            bridge_candidate is None
            or shared_answer_type_candidate is None
            or anchor_b_type_candidate is None
        ):
            return None

        candidate_set_variable = str(
            query_plan.get("candidate_set_variable") or "candidate_set"
        ).strip()
        shared_answer_variable = str(
            query_plan.get("shared_answer_variable") or "shared_answer"
        ).strip()
        shared_type_variable = "shared_type"

        rewritten_bridge = self._rewrite_path_endpoint_to_variable(
            relation_path=copy.deepcopy(bridge_candidate),
            target_role="candidate_set",
            replacement_variable=candidate_set_variable,
        )
        rewritten_bridge = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_bridge,
            target_role="shared_answer",
            replacement_variable=shared_answer_variable,
        )

        rewritten_answer_type = self._rewrite_path_endpoint_to_variable(
            relation_path=copy.deepcopy(shared_answer_type_candidate),
            target_role="candidate_set",
            replacement_variable=shared_answer_variable,
        )
        rewritten_answer_type = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_answer_type,
            target_role="shared_answer",
            replacement_variable=shared_answer_variable,
        )
        rewritten_answer_type = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_answer_type,
            target_role="shared_type",
            replacement_variable=shared_type_variable,
        )
        rewritten_answer_type = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_answer_type,
            target_role="type_set",
            replacement_variable=shared_type_variable,
        )

        rewritten_anchor_b_type = self._rewrite_path_endpoint_to_variable(
            relation_path=copy.deepcopy(anchor_b_type_candidate),
            target_role="shared_type",
            replacement_variable=shared_type_variable,
        )
        rewritten_anchor_b_type = self._rewrite_path_endpoint_to_variable(
            relation_path=rewritten_anchor_b_type,
            target_role="type_set",
            replacement_variable=shared_type_variable,
        )

        strategy = str(query_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (query_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The anchor-side path is live up to a pivot entity, so complete that branch with a second hop from the pivot to the shared answer set before applying the shared-type filter."
        )

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["shared_answer_variable"] = shared_answer_variable
        rewritten_plan["candidate_set_variable"] = candidate_set_variable
        rewritten_anchor_a_paths = [
            self._rewrite_path_endpoint_to_variable(
                relation_path=copy.deepcopy(path),
                target_role="candidate_set",
                replacement_variable=candidate_set_variable,
            )
            for path in anchor_a_paths
        ]
        rewritten_plan["relation_paths"] = [
            *rewritten_anchor_a_paths,
            rewritten_bridge,
            rewritten_answer_type,
            rewritten_anchor_b_type,
        ]
        rewritten_plan["projection"] = [shared_answer_variable, f"{shared_answer_variable}_name"]
        rewritten_plan["join_structure"] = {
            "type": "shared_type",
            "anchor_constraints": [
                {
                    "anchor_role": anchor_a_role,
                    "constrains_variable": candidate_set_variable,
                    "notes": "Anchor A constrains a live pivot-derived candidate set that is then bridged to the shared answer set.",
                },
                {
                    "anchor_role": anchor_b_role,
                    "constrains_variable": shared_type_variable,
                    "notes": "Anchor B constrains the shared type/category used to filter the bridged answer set.",
                },
            ],
        }
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by keeping the live anchor-side pivot branch, adding a grounded bridge from that pivot to the shared answer set, then filtering by the shared type."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _select_class_filtered_count_candidate(
        self,
        *,
        relation_grounding: Sequence[Mapping[str, str]],
        preferred_relation_root: str = "",
    ) -> Optional[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        normalized_preferred_root = str(preferred_relation_root or "").strip().lower()
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if {from_role, to_role}.isdisjoint({"type_set", "shared_type", "constraint_value"}):
                continue
            if {from_role, to_role}.isdisjoint(
                {"candidate_set", "shared_answer", "count_set", "anchor"}
            ):
                continue
            score = 0
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            relation = str(candidate.get("relation") or "").strip()
            if grounding_source == "dynamic_probe":
                score += 10
            elif grounding_source == "curated":
                score += 6
            else:
                score += 3
            if relation == "people.profession.people_with_this_profession":
                score += 6
            elif relation == "type.type.instance":
                score += 4
            elif relation == "type.object.type":
                score += 2
            relation_root = relation.split(".", 1)[0].strip().lower() if relation else ""
            if normalized_preferred_root and relation_root == normalized_preferred_root:
                score += 4
            if relation.startswith("people.person."):
                score += 3
            if from_role in {"type_set", "shared_type"}:
                score += 2
            if to_role in {"candidate_set", "shared_answer", "count_set"}:
                score += 2
            elif to_role == "anchor":
                score += 1
            scored.append((score, dict(candidate)))
        if not scored:
            return None
        scored.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("relation") or ""),
            )
        )
        return scored[0][1]

    def _build_class_filtered_count_repair_plan(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return None
        if str(query_plan.get("query_shape") or "").strip().lower() not in {
            "count_over_direct_relation",
            "count_over_joined_set",
        }:
            return None

        question_text, explicit_entities = self._split_task_question(task_question)
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        if not self._should_treat_answer_target_as_count_class(
            question_text=question_text,
            answer_target_phrase=answer_target_phrase,
        ):
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
            and self._normalize_relation_role(item.get("role")) in {"anchor", "anchor_a", "anchor_b"}
        ]
        explicit_constraint_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
            and self._normalize_relation_role(item.get("role")) == "constraint_value"
        ]
        answer_target_token = self._normalize_variable_token(
            self._singularize_phrase(answer_target_phrase)
        )
        if any(
            self._normalize_variable_token(
                self._singularize_phrase(
                    str(item.get("surface") or item.get("chosen_alias") or "").strip()
                )
            )
            not in {"", answer_target_token}
            for item in explicit_constraint_entities
        ):
            return None
        if len(anchored_entities) != 1:
            return None

        relation_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if self._count_answer_target_head_is_encoded_in_relation(
            answer_target_phrase=answer_target_phrase,
            relation_candidates=relation_paths,
        ):
            return None
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role")) or "anchor"
        anchor_path = next(
            (
                path
                for path in relation_paths
                if self._path_touches_anchor_role(path, anchor_role=anchor_role)
                and {
                    self._normalize_relation_role(path.get("from_role")),
                    self._normalize_relation_role(path.get("to_role")),
                }
                & {"candidate_set", "shared_answer", "count_set"}
            ),
            None,
        )
        if anchor_path is None:
            return None

        preferred_relation_root = str(anchor_path.get("relation") or "").split(".", 1)[0]
        class_candidate = self._select_class_filtered_count_candidate(
            relation_grounding=relation_grounding,
            preferred_relation_root=preferred_relation_root,
        )
        if class_candidate is None:
            return None

        shared_variable = "shared_answer"
        rewritten_anchor_path = copy.deepcopy(anchor_path)
        if self._normalize_relation_role(rewritten_anchor_path.get("from_role")) in {"candidate_set", "shared_answer", "count_set"}:
            rewritten_anchor_path["from"] = shared_variable
            rewritten_anchor_path["from_role"] = "candidate_set"
        if self._normalize_relation_role(rewritten_anchor_path.get("to_role")) in {"candidate_set", "shared_answer", "count_set"}:
            rewritten_anchor_path["to"] = shared_variable
            rewritten_anchor_path["to_role"] = "candidate_set"

        rewritten_class_path = copy.deepcopy(class_candidate)
        filter_role = next(
            (
                role
                for role in (
                    self._normalize_relation_role(rewritten_class_path.get("from_role")),
                    self._normalize_relation_role(rewritten_class_path.get("to_role")),
                )
                if role in {"type_set", "shared_type", "constraint_value"}
            ),
            "type_set",
        )
        if self._normalize_relation_role(rewritten_class_path.get("from_role")) in {
            "candidate_set",
            "shared_answer",
            "count_set",
            "anchor",
        }:
            rewritten_class_path["from"] = shared_variable
            rewritten_class_path["from_role"] = "candidate_set"
        elif self._normalize_relation_role(rewritten_class_path.get("from_role")) in {"type_set", "shared_type", "constraint_value"}:
            rewritten_class_path["from"] = filter_role
            rewritten_class_path["from_role"] = filter_role
        if self._normalize_relation_role(rewritten_class_path.get("to_role")) in {
            "candidate_set",
            "shared_answer",
            "count_set",
            "anchor",
        }:
            rewritten_class_path["to"] = shared_variable
            rewritten_class_path["to_role"] = "candidate_set"
        elif self._normalize_relation_role(rewritten_class_path.get("to_role")) in {"type_set", "shared_type", "constraint_value"}:
            rewritten_class_path["to"] = filter_role
            rewritten_class_path["to_role"] = filter_role

        class_surface = self._singularize_phrase(answer_target_phrase) or answer_target_phrase
        strategy = str(query_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (query_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "Treat the answer target as an explicit answer-class filter and intersect it with the anchor-derived candidate set before counting."
        )

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_plan["query_shape"] = "count_over_joined_set"
        rewritten_plan["shared_answer_variable"] = shared_variable
        rewritten_plan["candidate_set_variable"] = shared_variable
        rewritten_plan["count_set_variable"] = shared_variable
        rewritten_plan["anchored_entities"] = [
            {
                "surface": str(
                    anchored_entities[0].get("surface")
                    or anchored_entities[0].get("chosen_alias")
                    or ""
                ).strip(),
                "chosen_alias": str(
                    anchored_entities[0].get("chosen_alias")
                    or anchored_entities[0].get("surface")
                    or ""
                ).strip(),
                "role": anchor_role,
                **(
                    {
                        "resolved_entity_id": str(
                            anchored_entities[0].get("resolved_entity_id") or ""
                        ).strip()
                    }
                    if str(anchored_entities[0].get("resolved_entity_id") or "").strip()
                    else {}
                ),
            },
            {
                "surface": answer_target_phrase,
                "chosen_alias": class_surface,
                "role": filter_role,
            },
        ]
        rewritten_plan["join_structure"] = {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": shared_variable,
                    "notes": "The primary anchor constrains the shared counted entity set.",
                },
                {
                    "anchor_role": filter_role,
                    "constrains_variable": shared_variable,
                    "notes": "The answer target phrase is treated as an explicit answer-class or answer-constraint filter on the same counted set.",
                },
            ],
        }
        rewritten_plan["relation_paths"] = [rewritten_anchor_path, rewritten_class_path]
        rewritten_plan["projection"] = ["count"]
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by separating the answer class phrase from the anchor and counting the intersection."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _build_single_anchor_dynamic_lookup_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "single_anchor_lookup":
            return None
        if str(query_plan.get("answer_mode") or "entity").strip().lower() != "entity":
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) != 1:
            return None
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role"))
        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        existing_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if not existing_paths:
            return None
        live_relation_names = {
            str(candidate.get("relation") or "").strip()
            for candidate in relation_grounding
            if isinstance(candidate, Mapping) and str(candidate.get("relation") or "").strip()
        }
        existing_relations = {
            str(path.get("relation") or "").strip()
            for path in existing_paths
            if str(path.get("relation") or "").strip()
        }
        stale_grounded_paths = bool(existing_relations) and existing_relations.isdisjoint(
            live_relation_names
        )
        if not (
            all(
                str(path.get("grounding_source") or "").strip().lower() == "exploratory"
                for path in existing_paths
            )
            or stale_grounded_paths
        ):
            return None

        replacement_candidate: Optional[dict[str, Any]] = None
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            if str(candidate.get("grounding_source") or "").strip().lower() != "dynamic_probe":
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if from_role == anchor_role and to_role in {"candidate_set", "answer", "shared_answer"}:
                replacement_candidate = dict(candidate)
                replacement_candidate["from"] = "anchor"
                replacement_candidate["to"] = "answer"
                replacement_candidate["from_role"] = anchor_role
                replacement_candidate["to_role"] = "answer"
                break
            if to_role == anchor_role and from_role in {"candidate_set", "answer", "shared_answer"}:
                replacement_candidate = dict(candidate)
                replacement_candidate["from"] = "answer"
                replacement_candidate["to"] = "anchor"
                replacement_candidate["from_role"] = "answer"
                replacement_candidate["to_role"] = anchor_role
                break

        if replacement_candidate is None:
            return None

        rewritten_plan = copy.deepcopy(dict(query_plan))
        strategy = str(rewritten_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "A refreshed anchor alias produced live dynamic anchor-direct predicates, so replace the stale exploratory relation with the strongest anchor-direct dynamic candidate."
        )
        rewritten_plan["shared_answer_variable"] = "answer"
        rewritten_plan["candidate_set_variable"] = ""
        rewritten_plan["count_set_variable"] = ""
        rewritten_plan["join_structure"] = {
            "type": "single_path",
            "anchor_constraints": [
                {
                    "anchor_role": anchor_role,
                    "constrains_variable": "answer",
                    "notes": "Direct single-hop lookup from the repaired anchor alias to the answer entity.",
                }
            ],
        }
        rewritten_plan["relation_paths"] = [replacement_candidate]
        rewritten_plan["projection"] = ["answer", "answer_name"]
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by switching to the best live dynamic anchor-direct relation "
            f"{replacement_candidate.get('relation') or ''} after alias refresh."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _build_superlative_dynamic_anchor_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
        anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
    ) -> Optional[dict[str, Any]]:
        alternative_repair = self._build_superlative_anchor_alternative_repair_plan(
            query_plan=query_plan,
            relation_grounding=relation_grounding,
            anchor_probe_results=anchor_probe_results,
        )
        if alternative_repair is not None:
            return alternative_repair
        if str(query_plan.get("query_shape") or "").strip().lower() != "superlative_chain":
            return None
        if str(query_plan.get("answer_mode") or "entity").strip().lower() != "entity":
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) != 1:
            return None
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role"))
        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        existing_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if not existing_paths:
            return None

        anchor_path_index: Optional[int] = None
        anchor_path: Optional[dict[str, Any]] = None
        for index, relation_path in enumerate(existing_paths):
            from_role = self._normalize_relation_role(relation_path.get("from_role"))
            to_role = self._normalize_relation_role(relation_path.get("to_role"))
            if {from_role, to_role} >= {anchor_role, "candidate_set"}:
                anchor_path_index = index
                anchor_path = dict(relation_path)
                break
        if anchor_path_index is None or anchor_path is None:
            return None

        current_relation = str(anchor_path.get("relation") or "").strip()
        current_relation_tokens = {
            self._normalize_variable_token(self._singularize_surface_token(token))
            for token in re.split(r"[._]+", current_relation)
            if token
        }
        if "." in current_relation:
            current_relation_tokens.discard(
                self._normalize_variable_token(current_relation.split(".", 1)[0])
            )
        live_relation_names = {
            str(candidate.get("relation") or "").strip()
            for candidate in relation_grounding
            if isinstance(candidate, Mapping) and str(candidate.get("relation") or "").strip()
        }
        dead_anchor_path = any(
            str(getattr(result, "relation_probed", "") or "").strip() == current_relation
            and int(getattr(result, "entity_count", 0) or 0) > 0
            and (getattr(result, "path_count", None) or 0) == 0
            for result in (anchor_probe_results or [])
        )
        stale_grounded_path = bool(current_relation) and current_relation not in live_relation_names
        if not (dead_anchor_path or stale_grounded_path):
            return None

        replacement_candidate: Optional[dict[str, Any]] = None
        original_direction = str(anchor_path.get("direction") or "").strip().lower()
        anchor_path_from_role = self._normalize_relation_role(anchor_path.get("from_role"))
        anchor_path_to_role = self._normalize_relation_role(anchor_path.get("to_role"))
        anchor_path_from = str(anchor_path.get("from") or "").strip()
        anchor_path_to = str(anchor_path.get("to") or "").strip()
        original_anchor_endpoint = ""
        original_candidate_endpoint = ""
        original_anchor_endpoint = original_anchor_endpoint or anchor_role
        next_path: Optional[dict[str, Any]] = None
        next_path_index: Optional[int] = None
        if anchor_path_index + 1 < len(existing_paths):
            candidate_path = dict(existing_paths[anchor_path_index + 1])
            candidate_path_from = str(candidate_path.get("from") or "").strip()
            if (
                self._normalize_variable_token(candidate_path_from)
                in {
                    self._normalize_variable_token(anchor_path_from),
                    self._normalize_variable_token(anchor_path_to),
                }
                and self._normalize_relation_role(candidate_path.get("to_role"))
                in {"candidate_set", "shared_answer", "answer"}
            ):
                original_candidate_endpoint = candidate_path_from
                if self._normalize_variable_token(candidate_path_from) == self._normalize_variable_token(anchor_path_from):
                    original_anchor_endpoint = anchor_path_to
                else:
                    original_anchor_endpoint = anchor_path_from
                next_path = candidate_path
                next_path_index = anchor_path_index + 1
        if not original_candidate_endpoint:
            if anchor_path_from_role == anchor_role:
                original_anchor_endpoint = anchor_path_from
                original_candidate_endpoint = anchor_path_to
            elif anchor_path_to_role == anchor_role:
                original_anchor_endpoint = anchor_path_to
                original_candidate_endpoint = anchor_path_from
            else:
                original_anchor_endpoint = (
                    anchor_path_from
                    if self._is_anchor_side(
                        endpoint_side="from",
                        direction=original_direction,
                    )
                    else anchor_path_to
                )
                original_candidate_endpoint = (
                    anchor_path_to
                    if self._is_anchor_side(
                        endpoint_side="from",
                        direction=original_direction,
                    )
                    else anchor_path_from
                )
        original_anchor_endpoint = original_anchor_endpoint or anchor_role
        original_candidate_endpoint = (
            original_candidate_endpoint
            or str(query_plan.get("candidate_set_variable") or "").strip()
            or "candidate_set"
        )
        replacement_target_hint = ""
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            if str(candidate.get("grounding_source") or "").strip().lower() != "dynamic_probe":
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation or relation == current_relation:
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if from_role == anchor_role and to_role in {"candidate_set", "answer", "shared_answer"}:
                replacement_candidate = dict(candidate)
                replacement_target_hint = str(candidate.get("to") or "").strip()
                replacement_candidate["from"] = original_anchor_endpoint
                replacement_candidate["to"] = original_candidate_endpoint
                replacement_candidate["from_role"] = anchor_role
                replacement_candidate["to_role"] = "candidate_set"
                break
            if to_role == anchor_role and from_role in {"candidate_set", "answer", "shared_answer"}:
                replacement_candidate = dict(candidate)
                replacement_target_hint = str(candidate.get("from") or "").strip()
                replacement_candidate["from"] = original_candidate_endpoint
                replacement_candidate["to"] = original_anchor_endpoint
                replacement_candidate["from_role"] = "candidate_set"
                replacement_candidate["to_role"] = anchor_role
                break

        if replacement_candidate is None:
            return None

        replacement_target_endpoint = (
            str(replacement_candidate.get("to") or "").strip()
            if self._normalize_relation_role(replacement_candidate.get("from_role"))
            == anchor_role
            else str(replacement_candidate.get("from") or "").strip()
        )
        if next_path is not None:
            next_path_target = str(next_path.get("to") or "").strip()
            next_path_relation = str(next_path.get("relation") or "").strip()
            next_path_relation_tail = next_path_relation.rsplit(".", 1)[-1].replace("_", " ")
            next_path_relation_tail_token = self._normalize_variable_token(
                self._singularize_surface_token(next_path_relation_tail)
            )
            if (
                self._normalize_variable_token(replacement_target_hint)
                and self._normalize_variable_token(replacement_target_hint)
                in {
                    self._normalize_variable_token(next_path_target),
                    next_path_relation_tail_token,
                }
            ):
                if (
                    self._normalize_relation_role(replacement_candidate.get("from_role"))
                    == anchor_role
                ):
                    replacement_candidate["from"] = anchor_role
                    replacement_candidate["to"] = next_path_target or original_candidate_endpoint
                    replacement_candidate["from_role"] = anchor_role
                    replacement_candidate["to_role"] = (
                        self._normalize_relation_role(next_path.get("to_role"))
                        or "candidate_set"
                    )
                else:
                    replacement_candidate["from"] = next_path_target or original_candidate_endpoint
                    replacement_candidate["to"] = anchor_role
                    replacement_candidate["from_role"] = (
                        self._normalize_relation_role(next_path.get("to_role"))
                        or "candidate_set"
                    )
                    replacement_candidate["to_role"] = anchor_role

        rewritten_plan = copy.deepcopy(dict(query_plan))
        strategy = str(rewritten_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The original superlative anchor path produced no live results, so replace it with the strongest live dynamic anchor-to-candidate relation while preserving the ordering path."
        )
        rewritten_paths = list(existing_paths)
        rewritten_paths[anchor_path_index] = replacement_candidate
        if next_path_index is not None:
            replacement_target_token = (
                self._normalize_variable_token(replacement_target_hint)
                or self._normalize_variable_token(replacement_target_endpoint)
            )
            next_path_target_token = self._normalize_variable_token(next_path.get("to"))
            next_path_relation = str(next_path.get("relation") or "").strip()
            next_path_relation_tail = next_path_relation.rsplit(".", 1)[-1].replace("_", " ")
            next_path_relation_tail_token = self._normalize_variable_token(
                self._singularize_surface_token(next_path_relation_tail)
            )
            if replacement_target_token and replacement_target_token in {
                next_path_target_token,
                next_path_relation_tail_token,
            }:
                del rewritten_paths[next_path_index]
        rewritten_plan["relation_paths"] = rewritten_paths
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by switching the dead anchor relation to the live dynamic relation "
            f"{replacement_candidate.get('relation') or ''} before ordering the candidate set."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _build_superlative_anchor_alternative_repair_plan(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]] = (),
        anchor_probe_results: Sequence[AnchorProbeResult] | None = None,
    ) -> Optional[dict[str, Any]]:
        if str(query_plan.get("query_shape") or "").strip().lower() != "superlative_chain":
            return None
        if str(query_plan.get("answer_mode") or "entity").strip().lower() != "entity":
            return None

        anchored_entities = [
            dict(item)
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        if len(anchored_entities) != 1:
            return None
        anchor_role = self._normalize_relation_role(anchored_entities[0].get("role"))
        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        existing_paths = [
            dict(path)
            for path in (query_plan.get("relation_paths") or [])
            if isinstance(path, Mapping)
        ]
        if not existing_paths:
            return None

        anchor_path_index: Optional[int] = None
        anchor_path: Optional[dict[str, Any]] = None
        for index, relation_path in enumerate(existing_paths):
            roles = {
                self._normalize_relation_role(relation_path.get("from_role")),
                self._normalize_relation_role(relation_path.get("to_role")),
            }
            if anchor_role in roles and "candidate_set" in roles:
                anchor_path_index = index
                anchor_path = dict(relation_path)
                break
        if anchor_path_index is None or anchor_path is None:
            return None

        current_relation = str(anchor_path.get("relation") or "").strip()
        current_relation_tokens = {
            self._normalize_variable_token(self._singularize_surface_token(token))
            for token in re.split(r"[._]+", current_relation)
            if token
        }
        if "." in current_relation:
            current_relation_tokens.discard(
                self._normalize_variable_token(current_relation.split(".", 1)[0])
            )
        live_relation_names = {
            str(candidate.get("relation") or "").strip()
            for candidate in relation_grounding
            if isinstance(candidate, Mapping) and str(candidate.get("relation") or "").strip()
        }
        dead_anchor_path = any(
            str(getattr(result, "relation_probed", "") or "").strip() == current_relation
            and int(getattr(result, "entity_count", 0) or 0) > 0
            and (getattr(result, "path_count", None) or 0) == 0
            for result in (anchor_probe_results or [])
        )
        stale_grounded_path = bool(current_relation) and current_relation not in live_relation_names
        current_relation_tail = self._normalize_variable_token(current_relation.rsplit(".", 1)[-1])
        stronger_grounded_alternative_exists = False
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation or relation == current_relation:
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if not (
                (from_role == anchor_role and to_role in {"candidate_set", "answer", "shared_answer"})
                or (to_role == anchor_role and from_role in {"candidate_set", "answer", "shared_answer"})
            ):
                continue
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if grounding_source not in {"curated", "dynamic_probe"}:
                continue
            candidate_relation_tail = self._normalize_variable_token(relation.rsplit(".", 1)[-1])
            if candidate_relation_tail not in {"object", "objects"} or current_relation_tail in {
                "object",
                "objects",
            }:
                continue
            candidate_relation_tokens = {
                self._normalize_variable_token(self._singularize_surface_token(token))
                for token in re.split(r"[._]+", relation)
                if token
            }
            if "." in relation:
                candidate_relation_tokens.discard(
                    self._normalize_variable_token(relation.split(".", 1)[0])
                )
            shared_relation_tokens = {
                token
                for token in (current_relation_tokens & candidate_relation_tokens)
                if token and token not in {"object", "objects", "topic", "type"}
            }
            if relation.split(".", 1)[0] == current_relation.split(".", 1)[0] or shared_relation_tokens:
                stronger_grounded_alternative_exists = True
                break
        if not (dead_anchor_path or stale_grounded_path or stronger_grounded_alternative_exists):
            return None

        original_direction = str(anchor_path.get("direction") or "").strip().lower()
        anchor_path_from_role = self._normalize_relation_role(anchor_path.get("from_role"))
        anchor_path_to_role = self._normalize_relation_role(anchor_path.get("to_role"))
        anchor_path_from = str(anchor_path.get("from") or "").strip()
        anchor_path_to = str(anchor_path.get("to") or "").strip()
        original_anchor_endpoint = anchor_role
        original_candidate_endpoint = ""
        next_path: Optional[dict[str, Any]] = None
        next_path_index: Optional[int] = None
        if anchor_path_index + 1 < len(existing_paths):
            candidate_path = dict(existing_paths[anchor_path_index + 1])
            candidate_path_from = str(candidate_path.get("from") or "").strip()
            if (
                self._normalize_variable_token(candidate_path_from)
                in {
                    self._normalize_variable_token(anchor_path_from),
                    self._normalize_variable_token(anchor_path_to),
                }
                and self._normalize_relation_role(candidate_path.get("to_role"))
                in {"candidate_set", "shared_answer", "answer"}
            ):
                original_candidate_endpoint = candidate_path_from
                if (
                    self._normalize_variable_token(candidate_path_from)
                    == self._normalize_variable_token(anchor_path_from)
                ):
                    original_anchor_endpoint = anchor_path_to
                else:
                    original_anchor_endpoint = anchor_path_from
                next_path = candidate_path
                next_path_index = anchor_path_index + 1
        if not original_candidate_endpoint:
            if anchor_path_from_role == anchor_role:
                original_anchor_endpoint = anchor_path_from
                original_candidate_endpoint = anchor_path_to
            elif anchor_path_to_role == anchor_role:
                original_anchor_endpoint = anchor_path_to
                original_candidate_endpoint = anchor_path_from
            else:
                original_anchor_endpoint = (
                    anchor_path_from
                    if self._is_anchor_side(
                        endpoint_side="from",
                        direction=original_direction,
                    )
                    else anchor_path_to
                )
                original_candidate_endpoint = (
                    anchor_path_to
                    if self._is_anchor_side(
                        endpoint_side="from",
                        direction=original_direction,
                    )
                    else anchor_path_from
                )
        original_candidate_endpoint = (
            original_candidate_endpoint
            or str(query_plan.get("candidate_set_variable") or "").strip()
            or "candidate_set"
        )

        scored_candidates: list[tuple[int, dict[str, Any], str]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation or relation == current_relation:
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if not (
                (from_role == anchor_role and to_role in {"candidate_set", "answer", "shared_answer"})
                or (to_role == anchor_role and from_role in {"candidate_set", "answer", "shared_answer"})
            ):
                continue
            score = 0
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if grounding_source == "curated":
                score += 30
            elif grounding_source == "dynamic_probe":
                score += 20
            relation_tail = self._normalize_variable_token(relation.rsplit(".", 1)[-1])
            if relation_tail in {"object", "objects"}:
                score += 20
            if relation.split(".", 1)[0] == current_relation.split(".", 1)[0]:
                score += 8
            candidate_relation_tokens = {
                self._normalize_variable_token(self._singularize_surface_token(token))
                for token in re.split(r"[._]+", relation)
                if token
            }
            if "." in relation:
                candidate_relation_tokens.discard(
                    self._normalize_variable_token(relation.split(".", 1)[0])
                )
            shared_relation_tokens = {
                token
                for token in (current_relation_tokens & candidate_relation_tokens)
                if token and token not in {"object", "objects", "topic", "type"}
            }
            score += 12 * len(shared_relation_tokens)
            if "category" in relation:
                score += 6
            candidate_copy = dict(candidate)
            replacement_target_hint = ""
            if from_role == anchor_role:
                replacement_target_hint = str(candidate.get("to") or "").strip()
                candidate_copy["from"] = original_anchor_endpoint
                candidate_copy["to"] = original_candidate_endpoint
                candidate_copy["from_role"] = anchor_role
                candidate_copy["to_role"] = "candidate_set"
            else:
                replacement_target_hint = str(candidate.get("from") or "").strip()
                candidate_copy["from"] = original_candidate_endpoint
                candidate_copy["to"] = original_anchor_endpoint
                candidate_copy["from_role"] = "candidate_set"
                candidate_copy["to_role"] = anchor_role
            if next_path is not None:
                next_path_target = str(next_path.get("to") or "").strip()
                next_path_relation = str(next_path.get("relation") or "").strip()
                next_path_relation_tail = next_path_relation.rsplit(".", 1)[-1].replace("_", " ")
                next_path_relation_tail_token = self._normalize_variable_token(
                    self._singularize_surface_token(next_path_relation_tail)
                )
                if (
                    self._normalize_variable_token(replacement_target_hint)
                    and self._normalize_variable_token(replacement_target_hint)
                    in {
                        self._normalize_variable_token(next_path_target),
                        next_path_relation_tail_token,
                    }
                ):
                    if (
                        self._normalize_relation_role(candidate_copy.get("from_role"))
                        == anchor_role
                    ):
                        candidate_copy["from"] = anchor_role
                        candidate_copy["to"] = next_path_target or original_candidate_endpoint
                        candidate_copy["from_role"] = anchor_role
                        candidate_copy["to_role"] = (
                            self._normalize_relation_role(next_path.get("to_role"))
                            or "candidate_set"
                        )
                    else:
                        candidate_copy["from"] = next_path_target or original_candidate_endpoint
                        candidate_copy["to"] = anchor_role
                        candidate_copy["from_role"] = (
                            self._normalize_relation_role(next_path.get("to_role"))
                            or "candidate_set"
                        )
                        candidate_copy["to_role"] = anchor_role
            scored_candidates.append((score, candidate_copy, replacement_target_hint))

        if not scored_candidates:
            return None

        scored_candidates.sort(
            key=lambda item: (-item[0], str(item[1].get("relation") or ""))
        )
        replacement_candidate = scored_candidates[0][1]
        replacement_target_hint = scored_candidates[0][2]

        rewritten_plan = copy.deepcopy(dict(query_plan))
        rewritten_paths = list(existing_paths)
        rewritten_paths[anchor_path_index] = replacement_candidate
        replacement_target_endpoint = (
            str(replacement_candidate.get("to") or "").strip()
            if self._normalize_relation_role(replacement_candidate.get("from_role"))
            == anchor_role
            else str(replacement_candidate.get("from") or "").strip()
        )
        if next_path_index is not None and next_path is not None:
            replacement_target_token = (
                self._normalize_variable_token(replacement_target_hint)
                or self._normalize_variable_token(replacement_target_endpoint)
            )
            next_path_target_token = self._normalize_variable_token(next_path.get("to"))
            next_path_relation = str(next_path.get("relation") or "").strip()
            next_path_relation_tail = next_path_relation.rsplit(".", 1)[-1].replace("_", " ")
            next_path_relation_tail_token = self._normalize_variable_token(
                self._singularize_surface_token(next_path_relation_tail)
            )
            if replacement_target_token and replacement_target_token in {
                next_path_target_token,
                next_path_relation_tail_token,
            }:
                del rewritten_paths[next_path_index]
        rewritten_plan["relation_paths"] = rewritten_paths
        strategy = str(rewritten_plan.get("strategy") or "").strip()
        plan_rationale = [
            str(item).strip()
            for item in (rewritten_plan.get("plan_rationale") or [])
            if str(item).strip()
        ]
        plan_rationale.append(
            "The original anchor path stayed weak for the superlative repair, so replace it with the strongest grounded anchor-to-candidate alternative before ordering."
        )
        rewritten_plan["strategy"] = (
            f"{strategy} Repair by switching the anchor relation to the strongest grounded alternative "
            f"{replacement_candidate.get('relation') or ''} before ordering the candidate set."
        ).strip()
        rewritten_plan["plan_rationale"] = plan_rationale
        return self._normalize_pal_query_plan(rewritten_plan)

    def _path_touches_anchor_role(
        self,
        relation_path: Mapping[str, Any],
        *,
        anchor_role: str,
    ) -> bool:
        return anchor_role in {
            self._normalize_relation_role(relation_path.get("from_role")),
            self._normalize_relation_role(relation_path.get("to_role")),
        }

    def _select_count_pivot_candidate(
        self,
        *,
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_role: str,
        dead_relation: str,
    ) -> Optional[dict[str, str]]:
        prioritized: list[dict[str, str]] = []
        deferred: list[dict[str, str]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if not relation or relation == dead_relation:
                continue
            matches_anchor = (
                from_role == anchor_role and to_role == "candidate_set"
            ) or (
                to_role == anchor_role and from_role == "candidate_set"
            )
            if not matches_anchor:
                continue
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            target_bucket = prioritized if grounding_source == "dynamic_probe" else deferred
            target_bucket.append(dict(candidate))
        ordered_candidates = prioritized + deferred
        return ordered_candidates[0] if ordered_candidates else None

    def _select_direct_dynamic_count_candidate(
        self,
        *,
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_role: str,
        dead_relation: str,
        dead_target_token: str,
    ) -> Optional[dict[str, str]]:
        candidates: list[tuple[int, dict[str, str]]] = []
        generic_tokens = {
            "type",
            "topic",
            "image",
            "webpage",
            "relationship",
            "classification",
            "rank",
            "gallery",
            "date",
        }
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation or relation == dead_relation:
                continue
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if grounding_source not in {"curated", "dynamic_probe"}:
                continue
            support = str(candidate.get("support") or "").strip().lower()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if from_role == anchor_role and to_role in {"candidate_set", "count_set"}:
                endpoint_token = self._normalize_variable_token(candidate.get("to"))
                score = 9
            elif to_role == anchor_role and from_role in {"candidate_set", "count_set"}:
                endpoint_token = self._normalize_variable_token(candidate.get("from"))
                score = 8
            else:
                continue
            if grounding_source == "dynamic_probe":
                score += 8
            elif grounding_source == "curated":
                score += 2
            if dead_target_token and endpoint_token != dead_target_token:
                if dead_target_token not in relation:
                    continue
            if relation.startswith("type.") or relation.startswith("common."):
                score -= 6
            if endpoint_token in generic_tokens:
                score -= 4
            if "dynamic_probe_incoming" in support:
                score += 6
            if "dynamic_probe_outgoing" in support:
                score += 2
            if "curated_anchor_role_reinterpretation" in support:
                score -= 3
            if "disease" in relation or endpoint_token == "disease":
                score += 2
            candidates.append((score, dict(candidate)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _select_preserved_count_target_candidate(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
        dead_path: Mapping[str, Any],
    ) -> Optional[dict[str, str]]:
        dead_relation = str(dead_path.get("relation") or "").strip()
        dead_target_token = self._count_set_token_for_path(
            relation_path=dead_path,
            query_plan=query_plan,
        )
        exact_matches: list[dict[str, str]] = []
        token_matches: list[dict[str, str]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            if self._normalize_relation_role(candidate.get("from_role")) != "candidate_set":
                continue
            if self._normalize_relation_role(candidate.get("to_role")) != "count_set":
                continue
            normalized_candidate = dict(candidate)
            if str(candidate.get("relation") or "").strip() == dead_relation:
                exact_matches.append(normalized_candidate)
                continue
            if dead_target_token and self._candidate_count_endpoint_token(
                normalized_candidate,
                query_plan=query_plan,
            ) == dead_target_token:
                token_matches.append(normalized_candidate)
        if exact_matches:
            return exact_matches[0]
        if token_matches:
            return token_matches[0]
        return None

    def _count_set_token_for_path(
        self,
        *,
        relation_path: Mapping[str, Any],
        query_plan: Mapping[str, Any],
    ) -> str:
        from_role = self._normalize_relation_role(relation_path.get("from_role"))
        to_role = self._normalize_relation_role(relation_path.get("to_role"))
        if to_role == "count_set":
            return self._normalize_variable_token(relation_path.get("to"))
        if from_role == "count_set":
            return self._normalize_variable_token(relation_path.get("from"))
        return self._normalize_variable_token(query_plan.get("count_set_variable"))

    def _candidate_count_endpoint_token(
        self,
        candidate: Mapping[str, Any],
        *,
        query_plan: Mapping[str, Any],
    ) -> str:
        from_role = self._normalize_relation_role(candidate.get("from_role"))
        to_role = self._normalize_relation_role(candidate.get("to_role"))
        if to_role == "count_set":
            return self._normalize_variable_token(candidate.get("to"))
        if from_role == "count_set":
            return self._normalize_variable_token(candidate.get("from"))
        return self._normalize_variable_token(query_plan.get("count_set_variable"))

    def _rewrite_path_endpoint_to_variable(
        self,
        *,
        relation_path: Mapping[str, Any],
        target_role: str,
        replacement_variable: str,
    ) -> dict[str, Any]:
        rewritten_path = dict(relation_path)
        if self._normalize_relation_role(rewritten_path.get("from_role")) == target_role:
            rewritten_path["from"] = replacement_variable
        if self._normalize_relation_role(rewritten_path.get("to_role")) == target_role:
            rewritten_path["to"] = replacement_variable
        return rewritten_path

    def _prioritize_anchor_bridge_candidates(
        self,
        dynamic_candidates: Sequence[Mapping[str, str]],
        *,
        bridge_variable: str,
    ) -> list[dict[str, str]]:
        bridge_token = self._normalize_variable_token(bridge_variable)
        if not bridge_token:
            return [dict(candidate) for candidate in dynamic_candidates]

        prioritized: list[dict[str, str]] = []
        deferred: list[dict[str, str]] = []
        for candidate in dynamic_candidates:
            if not isinstance(candidate, Mapping):
                continue
            endpoint_token = self._candidate_non_anchor_endpoint_token(candidate)
            if endpoint_token == bridge_token:
                prioritized.append(dict(candidate))
            else:
                deferred.append(dict(candidate))
        return prioritized + deferred

    def _prioritize_shared_anchor_relation_families(
        self,
        dynamic_candidates: Sequence[Mapping[str, str]],
    ) -> tuple[list[dict[str, str]], list[str]]:
        relation_roles: dict[str, set[str]] = {}
        for candidate in dynamic_candidates:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation:
                continue
            roles = relation_roles.setdefault(relation, set())
            for raw_role in (candidate.get("from_role"), candidate.get("to_role")):
                role = self._normalize_relation_role(raw_role)
                if role in {"anchor", "anchor_a", "anchor_b"}:
                    roles.add(role)

        shared_relations = [
            relation
            for relation, roles in relation_roles.items()
            if len({role for role in roles if role in {"anchor_a", "anchor_b"}}) >= 2
        ]
        if not shared_relations:
            return [dict(candidate) for candidate in dynamic_candidates], []

        prioritized: list[dict[str, str]] = []
        deferred: list[dict[str, str]] = []
        shared_relation_set = set(shared_relations)
        for candidate in dynamic_candidates:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            target_bucket = prioritized if relation in shared_relation_set else deferred
            target_bucket.append(dict(candidate))
        return prioritized + deferred, shared_relations

    def _extract_freebase_entity_id_from_probe_value(self, value: Any) -> str:
        raw_value = str(value or "").strip()
        if not raw_value:
            return ""
        if "/ns/" in raw_value:
            raw_value = raw_value.rsplit("/ns/", 1)[-1]
        return raw_value if re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", raw_value) else ""

    def _sample_live_pivot_entity_ids(
        self,
        *,
        query_plan: Mapping[str, Any],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
        max_samples: int = 1,
    ) -> list[dict[str, str]]:
        if not anchor_probe_results or not _SPARQL_PROBE_AVAILABLE:
            return []

        anchored_entities = [
            item
            for item in (query_plan.get("anchored_entities") or [])
            if isinstance(item, Mapping)
        ]
        endpoint = self._get_runtime_sparql_endpoint()
        samples: list[dict[str, str]] = []
        seen_ids: set[tuple[str, str]] = set()

        for anchored_entity, probe_result in zip(anchored_entities, anchor_probe_results):
            if (
                getattr(probe_result, "entity_count", -1) <= 0
                or (getattr(probe_result, "path_count", 0) or 0) <= 0
            ):
                continue
            resolved_entity_id = str(
                getattr(probe_result, "resolved_entity_id", "") or ""
            ).strip()
            relation = str(getattr(probe_result, "relation_probed", "") or "").strip()
            anchor_position = str(
                getattr(probe_result, "anchor_position", "") or ""
            ).strip().lower()
            anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
            if (
                not resolved_entity_id
                or not relation
                or anchor_position not in {"subject", "object"}
                or anchor_role not in {"anchor", "anchor_a", "anchor_b"}
            ):
                continue

            if anchor_position == "subject":
                sparql = (
                    "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
                    "SELECT DISTINCT ?pivot WHERE {\n"
                    f"  fb:{resolved_entity_id} fb:{relation} ?pivot .\n"
                    f"}} LIMIT {max_samples}"
                )
            else:
                sparql = (
                    "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
                    "SELECT DISTINCT ?pivot WHERE {\n"
                    f"  ?pivot fb:{relation} fb:{resolved_entity_id} .\n"
                    f"}} LIMIT {max_samples}"
                )
            raw_values = self._run_probe_sparql_query(
                endpoint=endpoint,
                sparql=sparql,
                timeout_s=2.5,
            )
            for raw_value in raw_values[:max_samples]:
                pivot_entity_id = self._extract_freebase_entity_id_from_probe_value(
                    raw_value
                )
                if not pivot_entity_id:
                    continue
                key = (anchor_role, pivot_entity_id)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                samples.append(
                    {
                        "anchor_role": anchor_role,
                        "pivot_entity_id": pivot_entity_id,
                        "pivot_relation": relation,
                    }
                )
        return samples

    def _determine_pivot_probe_target_role(
        self,
        *,
        query_plan: Mapping[str, Any],
    ) -> str:
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()
        answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()
        if answer_mode == "count" or query_shape == "count_over_joined_set":
            return "count_set"
        return "shared_answer"

    def _probe_dynamic_relation_candidates_for_entity_id(
        self,
        *,
        entity_id: str,
        pivot_role: str,
        target_role: str,
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str = "",
        probe_timeout_s: float = 5.0,
    ) -> list[dict[str, str]]:
        if not entity_id or not _SPARQL_PROBE_AVAILABLE:
            return []

        endpoint = self._get_runtime_sparql_endpoint()
        outgoing_sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT ?p WHERE {\n"
            f"  fb:{entity_id} ?p ?o .\n"
            "} LIMIT 40"
        )
        incoming_sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT ?p WHERE {\n"
            f"  ?s ?p fb:{entity_id} .\n"
            "} LIMIT 40"
        )
        outgoing_iris = self._run_probe_sparql_query(
            endpoint=endpoint,
            sparql=outgoing_sparql,
            timeout_s=probe_timeout_s,
        )
        incoming_iris = self._run_probe_sparql_query(
            endpoint=endpoint,
            sparql=incoming_sparql,
            timeout_s=probe_timeout_s,
        )
        if not outgoing_iris and not incoming_iris:
            return []

        seen: set[str] = set()

        def _clean(iris: Sequence[str]) -> list[str]:
            cleaned: list[str] = []
            for iri in iris:
                if self._is_noise_predicate(iri):
                    continue
                short = self._normalize_predicate_iri(iri)
                if short and short not in seen:
                    seen.add(short)
                    cleaned.append(short)
            return cleaned

        def _rank(names: Sequence[str]) -> list[str]:
            return sorted(
                names,
                key=lambda s: self._score_probe_predicate(
                    s,
                    answer_target_phrase,
                    domain_hints,
                    question_text,
                ),
                reverse=True,
            )[:8]

        candidates: list[dict[str, str]] = []
        for short_name in _rank(_clean(outgoing_iris)):
            _, target_label = self._infer_probe_relation_endpoint_labels(short_name)
            candidates.append(
                {
                    "relation": short_name,
                    "direction": "forward",
                    "from": "pivot",
                    "to": target_label or "?target",
                    "from_role": pivot_role,
                    "to_role": target_role,
                    "grounding_source": "dynamic_probe",
                    "support": "dynamic_probe_pivot_outgoing",
                    "use_when": (
                        f"traverse outgoing {short_name} from a live pivot entity"
                        + (f" to reach {target_label}" if target_label else "")
                    ),
                }
            )
        for short_name in _rank(_clean(incoming_iris)):
            source_label, _ = self._infer_probe_relation_endpoint_labels(short_name)
            candidates.append(
                {
                    "relation": short_name,
                    "direction": "reverse",
                    "from": source_label or "?source",
                    "to": "pivot",
                    "from_role": target_role,
                    "to_role": pivot_role,
                    "grounding_source": "dynamic_probe",
                    "support": "dynamic_probe_pivot_incoming",
                    "use_when": (
                        f"find entities that link to a live pivot via {short_name}"
                        + (f" from {source_label}" if source_label else "")
                    ),
                }
            )
        return candidates[:12]

    def _probe_dynamic_relation_candidates_for_live_pivots(
        self,
        *,
        query_plan: Mapping[str, Any],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str = "",
    ) -> list[dict[str, str]]:
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()
        if query_shape not in {
            "multi_anchor_intersection",
            "shared_type_intersection",
            "count_over_joined_set",
        }:
            return []

        target_role = self._determine_pivot_probe_target_role(query_plan=query_plan)
        merged_candidates: list[dict[str, str]] = []
        for pivot_sample in self._sample_live_pivot_entity_ids(
            query_plan=query_plan,
            anchor_probe_results=anchor_probe_results,
        ):
            pivot_candidates = self._probe_dynamic_relation_candidates_for_entity_id(
                entity_id=str(pivot_sample.get("pivot_entity_id") or "").strip(),
                pivot_role="candidate_set",
                target_role=target_role,
                answer_target_phrase=answer_target_phrase,
                domain_hints=domain_hints,
                question_text=question_text,
                probe_timeout_s=2.5,
            )
            if not pivot_candidates:
                continue
            merged_candidates = self._merge_relation_grounding_candidates(
                merged_candidates,
                pivot_candidates,
                prefer_extra=False,
            )
        return merged_candidates

    def _build_anchor_clue_feedback(
        self,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> list[str]:
        question_text, _ = self._split_task_question(task_question)
        feedback: list[str] = []
        for anchored_entity in query_plan.get("anchored_entities") or []:
            if not isinstance(anchored_entity, Mapping):
                continue
            surface = str(
                anchored_entity.get("surface")
                or anchored_entity.get("chosen_alias")
                or ""
            ).strip()
            if not surface:
                continue
            clue = self._infer_entity_clue(question_text, surface)
            if not clue or clue == "surface_constraint":
                continue
            feedback.append(f"anchor_clue:{surface}={clue}")
            anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
            preferred_relations = self._preferred_relations_for_anchor_clue(
                clue=clue,
                relation_grounding=relation_grounding or (),
                anchor_role=anchor_role,
            )
            if preferred_relations:
                feedback.append(
                    f"anchor_preferred_relations:{surface}="
                    + ", ".join(preferred_relations[:4])
                )
        if feedback:
            feedback.append(
                "repair_hint:preserve_anchor_semantics — when changing scaffold families, preserve each anchor's original clue from the question and do not swap anchor meanings"
            )
        return feedback

    def _preferred_relations_for_anchor_clue(
        self,
        *,
        clue: str,
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_role: str = "",
    ) -> list[str]:
        clue_keywords: dict[str, tuple[str, ...]] = {
            "active_ingredient": (
                "active_ingredient",
                "active_ingredients",
                "ingredient",
                "moiety",
            ),
            "formulation_input": (
                "formulation_of",
                "formulate",
                "formulation",
                "marketed_formulations",
            ),
            "source_constraint": (
                "source",
                "milk_source",
                "parent",
                "owner",
                "producer",
            ),
        }
        keywords = clue_keywords.get(clue, ())
        if not keywords:
            return []
        scored: list[tuple[int, str]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation:
                continue
            haystack = " ".join(
                [
                    relation.lower(),
                    str(candidate.get("use_when") or "").lower(),
                    str(candidate.get("from") or "").lower(),
                    str(candidate.get("to") or "").lower(),
                ]
            )
            score = 0
            for keyword in keywords:
                if keyword in haystack:
                    score += 5 if keyword in relation.lower() else 2
            candidate_anchor_roles = {
                self._normalize_relation_role(candidate.get("from_role")),
                self._normalize_relation_role(candidate.get("to_role")),
            }
            if anchor_role and anchor_role in candidate_anchor_roles:
                score += 12
            support = str(candidate.get("support") or "").lower()
            if "anchor_bridge_synthesized" in support:
                score += 8
            if score > 0:
                scored.append((score, relation))
        scored.sort(key=lambda item: (-item[0], item[1]))
        preferred: list[str] = []
        for _, relation in scored:
            if relation not in preferred:
                preferred.append(relation)
        return preferred

    def _synthesize_pivot_relation_candidates(
        self,
        *,
        query_shape: str,
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> list[dict[str, str]]:
        if query_shape not in {
            "count_over_direct_relation",
            "single_anchor_lookup",
            "single_anchor_chain_lookup",
        }:
            return []
        if not anchor_probe_results:
            return []
        if not any(
            getattr(result, "entity_count", -1) > 0
            and getattr(result, "path_count", None) == 0
            for result in anchor_probe_results
        ):
            return []

        synthesized: list[dict[str, str]] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            grounding_source = str(candidate.get("grounding_source") or "").strip().lower()
            if grounding_source != "curated":
                continue
            relation = str(candidate.get("relation") or "").strip()
            direction = str(candidate.get("direction") or "").strip().lower()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if not relation or direction not in {"forward", "reverse"}:
                continue

            adapted_candidate = dict(candidate)
            adapted_candidate["support"] = "curated_anchor_role_reinterpretation"
            adapted_candidate["use_when"] = (
                str(candidate.get("use_when") or "").strip()
                or "reuse this curated relation after a single pivot from the resolved anchor"
            )

            if from_role == "anchor" and to_role in {
                "answer",
                "shared_answer",
                "candidate_set",
                "count_set",
            }:
                adapted_candidate["from"] = "pivot"
                adapted_candidate["from_role"] = "candidate_set"
            elif to_role == "anchor" and from_role in {
                "answer",
                "shared_answer",
                "candidate_set",
                "count_set",
            }:
                adapted_candidate["to"] = "pivot"
                adapted_candidate["to_role"] = "candidate_set"
            else:
                continue
            synthesized.append(adapted_candidate)

        return synthesized

    def _anchor_paths_live_but_join_is_empty(
        self,
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> bool:
        if not anchor_probe_results or len(anchor_probe_results) < 2:
            return False
        live_paths = [
            result
            for result in anchor_probe_results
            if getattr(result, "entity_count", -1) > 0
            and getattr(result, "path_count", None) is not None
        ]
        return len(live_paths) >= 2 and all(
            (result.path_count or 0) > 0 for result in live_paths[:2]
        )

    def _relation_names_from_plan(
        self,
        query_plan: Mapping[str, Any],
    ) -> list[str]:
        relation_names: list[str] = []
        for relation_path in query_plan.get("relation_paths") or []:
            if not isinstance(relation_path, Mapping):
                continue
            relation = str(relation_path.get("relation") or "").strip()
            if relation and relation not in relation_names:
                relation_names.append(relation)
        return relation_names

    def _unused_grounded_relation_names(
        self,
        *,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
    ) -> list[str]:
        used_relations = set(self._relation_names_from_plan(query_plan))
        unused_relations: list[str] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            if not relation or relation in used_relations or relation in unused_relations:
                continue
            unused_relations.append(relation)
        return unused_relations

    def _build_scaffold_signature(
        self,
        query_plan: Mapping[str, Any],
    ) -> str:
        query_shape = str(query_plan.get("query_shape") or "").strip().lower()
        shared_answer_variable = str(
            query_plan.get("shared_answer_variable")
            or query_plan.get("candidate_set_variable")
            or query_plan.get("count_set_variable")
            or ""
        ).strip()
        relation_names = self._relation_names_from_plan(query_plan)
        if not relation_names:
            return ""
        signature_parts = [query_shape or "other", shared_answer_variable or "unknown"]
        signature_parts.extend(sorted(relation_names))
        return "|".join(signature_parts)

    def _suppress_dead_grounded_relations(
        self,
        *,
        task_question: str,
        query_plan: Mapping[str, Any],
        relation_grounding: Sequence[Mapping[str, str]],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> tuple[str, list[dict[str, str]], list[str]]:
        if not anchor_probe_results:
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=self._extract_anchor_alias_overrides(query_plan),
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        dead_relations = {
            str(result.relation_probed or "").strip()
            for result in anchor_probe_results
            if getattr(result, "entity_count", -1) > 0
            and getattr(result, "path_count", None) == 0
            and str(getattr(result, "relation_probed", "") or "").strip()
        }
        if not dead_relations:
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=self._extract_anchor_alias_overrides(query_plan),
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        filtered_grounding: list[dict[str, str]] = []
        removed_relations: list[str] = []
        removed_relation_descriptors: list[str] = []
        for candidate in relation_grounding:
            if not isinstance(candidate, Mapping):
                continue
            matching_dead_probe = self._matching_dead_anchor_probe(
                candidate=candidate,
                anchor_probe_results=anchor_probe_results,
            )
            if matching_dead_probe is not None:
                relation = str(candidate.get("relation") or "").strip()
                removed_relations.append(relation)
                anchor_name = str(getattr(matching_dead_probe, "anchor_name", "") or "").strip()
                descriptor_parts = [relation]
                if anchor_name:
                    descriptor_parts.append(f"anchor={anchor_name}")
                direction = str(candidate.get("direction") or "").strip().lower()
                from_role = self._normalize_relation_role(candidate.get("from_role"))
                to_role = self._normalize_relation_role(candidate.get("to_role"))
                if direction:
                    descriptor_parts.append(f"direction={direction}")
                if from_role:
                    descriptor_parts.append(f"from_role={from_role}")
                if to_role:
                    descriptor_parts.append(f"to_role={to_role}")
                removed_relation_descriptors.append("[".join([descriptor_parts[0], ", ".join(descriptor_parts[1:]) + "]"]) if len(descriptor_parts) > 1 else descriptor_parts[0])
                continue
            filtered_grounding.append(dict(candidate))

        if len(filtered_grounding) == len(list(relation_grounding)):
            return (
                self._build_pal_grounding_card(
                    task_question,
                    relation_grounding=relation_grounding,
                    alias_overrides=self._extract_anchor_alias_overrides(query_plan),
                ),
                [dict(candidate) for candidate in relation_grounding],
                [],
            )

        grounding_card = self._build_pal_grounding_card(
            task_question,
            relation_grounding=filtered_grounding,
            alias_overrides=self._extract_anchor_alias_overrides(query_plan),
        )
        deduped_removed_relations: list[str] = []
        for relation in removed_relations:
            if relation not in deduped_removed_relations:
                deduped_removed_relations.append(relation)
        deduped_removed_descriptors: list[str] = []
        for descriptor in removed_relation_descriptors:
            if descriptor not in deduped_removed_descriptors:
                deduped_removed_descriptors.append(descriptor)
        surviving_relation_names = {
            str(candidate.get("relation") or "").strip()
            for candidate in filtered_grounding
            if isinstance(candidate, Mapping) and str(candidate.get("relation") or "").strip()
        }
        surviving_removed_relations = [
            relation for relation in deduped_removed_relations if relation in surviving_relation_names
        ]
        self._emit_generated_tools_event(
            {
                "event": "pal_dead_grounding_relation_suppressed",
                "mode": "pal",
                "removed_relations": deduped_removed_descriptors or deduped_removed_relations,
                "remaining_relation_candidates": len(filtered_grounding),
            }
        )
        feedback = [
            "plausibility_feedback:dead_relation_suppressed — a previously grounded anchor-scoped relation path was removed after a live anchor-path probe proved that specific path empty",
            "repair_hint:do_not_reuse_dead_relation_instance — do not reuse the same anchor-scoped relation instance named in dead_relation_suppressed feedback on the next repair attempt; other role or direction variants of the same relation may still be valid",
        ]
        if deduped_removed_descriptors:
            feedback.append(
                "dead_relation_suppressed:" + ", ".join(deduped_removed_descriptors)
            )
        if surviving_removed_relations:
            feedback.append(
                "relation_still_available_in_other_roles:" + ", ".join(surviving_removed_relations)
            )
        return grounding_card, filtered_grounding, feedback

    def _matching_dead_anchor_probe(
        self,
        *,
        candidate: Mapping[str, Any],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> AnchorProbeResult | None:
        if not anchor_probe_results:
            return None
        relation = str(candidate.get("relation") or "").strip()
        direction = str(candidate.get("direction") or "").strip().lower()
        from_role = self._normalize_relation_role(candidate.get("from_role"))
        to_role = self._normalize_relation_role(candidate.get("to_role"))
        if not relation or direction not in {"forward", "reverse"}:
            return None
        expected_anchor_positions: set[str] = set()
        if from_role in {"anchor", "anchor_a", "anchor_b"}:
            expected_anchor_positions.add(
                self._anchor_position_for_relation_endpoint(
                    endpoint_side="from",
                    direction=direction,
                    endpoint_role=from_role,
                    counterpart_role=to_role,
                )
            )
        if to_role in {"anchor", "anchor_a", "anchor_b"}:
            expected_anchor_positions.add(
                self._anchor_position_for_relation_endpoint(
                    endpoint_side="to",
                    direction=direction,
                    endpoint_role=to_role,
                    counterpart_role=from_role,
                )
            )
        if not expected_anchor_positions:
            return None
        for probe_result in anchor_probe_results:
            if (
                getattr(probe_result, "entity_count", -1) <= 0
                or getattr(probe_result, "path_count", None) != 0
                or str(getattr(probe_result, "relation_probed", "") or "").strip()
                != relation
            ):
                continue
            anchor_position = str(
                getattr(probe_result, "anchor_position", "") or ""
            ).strip().lower()
            if anchor_position in expected_anchor_positions:
                return probe_result
        return None

    def _candidate_matches_dead_anchor_probe(
        self,
        *,
        candidate: Mapping[str, Any],
        anchor_probe_results: Sequence[AnchorProbeResult] | None,
    ) -> bool:
        return (
            self._matching_dead_anchor_probe(
                candidate=candidate,
                anchor_probe_results=anchor_probe_results,
            )
            is not None
        )

    def _merge_relation_grounding_candidates(
        self,
        base_candidates: Sequence[Mapping[str, str]],
        extra_candidates: Sequence[Mapping[str, str]],
        *,
        prefer_extra: bool = False,
    ) -> list[dict[str, str]]:
        merged: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str, str, str]] = set()
        candidate_stream = (
            list(extra_candidates) + list(base_candidates)
            if prefer_extra
            else list(base_candidates) + list(extra_candidates)
        )
        for candidate in candidate_stream:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            direction = str(candidate.get("direction") or "").strip()
            from_role = str(candidate.get("from_role") or "").strip()
            to_role = str(candidate.get("to_role") or "").strip()
            from_value = str(candidate.get("from") or "").strip()
            to_value = str(candidate.get("to") or "").strip()
            key = (
                relation,
                direction,
                from_role,
                to_role,
                from_value,
                to_value,
            )
            if not relation or key in seen:
                continue
            seen.add(key)
            merged.append(dict(candidate))
        return merged

    def _validate_pal_query_candidate(
        self,
        *,
        raw_output: str,
        generated_code: str,
        query_text: str,
        query_texts: Sequence[str] | None = None,
        query_plan: Mapping[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        normalized_query_texts: list[str] = []
        for candidate_query in query_texts or ():
            cleaned_query = str(candidate_query or "").strip()
            if cleaned_query and cleaned_query not in normalized_query_texts:
                normalized_query_texts.append(cleaned_query)
        cleaned_primary_query = str(query_text or "").strip()
        if cleaned_primary_query and cleaned_primary_query not in normalized_query_texts:
            normalized_query_texts.insert(0, cleaned_primary_query)

        validation_query_text = "\n\n".join(normalized_query_texts)
        primary_query_text = normalized_query_texts[0] if normalized_query_texts else ""
        code_and_query_text = "\n".join(
            (raw_output or "", generated_code or "", validation_query_text)
        )
        if "http://localhost:9999/blazegraph/namespace/kb/sparql" in code_and_query_text:
            errors.append("stale_endpoint_literal")
        if re.search(r"https?://[^\s\"']*sparql", generated_code):
            errors.append("hardcoded_endpoint_literal")
        if not normalized_query_texts:
            errors.append("missing_query_text")
        if self._has_top_level_text_leakage(generated_code):
            errors.append("top_level_text_leakage")
        allow_exploratory = bool(query_plan.get("allow_exploratory_predicates"))
        if re.search(
            r"\?[A-Za-z_][A-Za-z0-9_]*\s+\?[A-Za-z_][A-Za-z0-9_]*\s+\?[A-Za-z_][A-Za-z0-9_]*\s*\.",
            validation_query_text,
        ):
            if not allow_exploratory:
                errors.append("unconstrained_variable_triple_strategy")
        if re.search(
            r"\?[A-Za-z_][A-Za-z0-9_]*\s+\?p[A-Za-z0-9_]*\s+\?[A-Za-z_][A-Za-z0-9_]*\s*\.",
            validation_query_text,
        ):
            if not allow_exploratory:
                errors.append("predicate_variable_strategy")
        union_count = max(
            (
                len(re.findall(r"\bUNION\b", candidate_query, flags=re.IGNORECASE))
                for candidate_query in normalized_query_texts
            ),
            default=0,
        )
        relation_count = len(query_plan.get("relation_paths", []))
        anchor_binding_count = len(
            [
                anchored_entity
                for anchored_entity in (query_plan.get("anchored_entities") or [])
                if isinstance(anchored_entity, Mapping)
                and str(anchored_entity.get("surface") or anchored_entity.get("chosen_alias") or "").strip()
            ]
        )
        if union_count > 2 and union_count > max(1, relation_count + anchor_binding_count):
            errors.append(f"excessive_union_branches:{union_count}")
        for candidate_query in normalized_query_texts:
            errors.extend(
                self._validate_query_predicates_against_plan(
                    query_text=candidate_query,
                    query_plan=query_plan,
                )
            )
            errors.extend(
                self._validate_query_projection_bindings_against_plan(
                    query_text=candidate_query,
                    query_plan=query_plan,
                )
            )
            errors.extend(
                self._validate_query_constraint_bindings_against_plan(
                    query_text=candidate_query,
                )
            )

        answer_mode = str(query_plan.get("answer_mode") or "entity")
        projected_variables = self._extract_projected_variables(primary_query_text)
        if answer_mode == "count":
            if not any(
                re.search(r"\bCOUNT\s*\(", candidate_query, flags=re.IGNORECASE)
                for candidate_query in normalized_query_texts
            ):
                errors.append("count_plan_without_count_projection")
        elif answer_mode == "entity":
            if not projected_variables:
                errors.append("missing_entity_projection")
            else:
                first_projection = projected_variables[0].lower()
                if "name" in first_projection or "label" in first_projection:
                    errors.append("entity_plan_must_project_entity_first")
        elif answer_mode == "boolean":
            if not any(
                re.search(r"\bASK\b", candidate_query, flags=re.IGNORECASE)
                for candidate_query in normalized_query_texts
            ):
                errors.append("boolean_plan_without_ask")

        deduped_errors: list[str] = []
        for error in errors:
            if error not in deduped_errors:
                deduped_errors.append(error)
        return deduped_errors

    def _has_top_level_text_leakage(self, generated_code: str) -> bool:
        try:
            parsed = ast.parse(generated_code)
        except SyntaxError:
            return False
        for node in parsed.body:
            if not isinstance(node, ast.Expr):
                continue
            if not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            leaked_text = node.value.value.strip()
            if leaked_text:
                return True
        return False

    def _validate_query_predicates_against_plan(
        self,
        *,
        query_text: str,
        query_plan: Mapping[str, Any],
    ) -> list[str]:
        plan_relations = {
            str(relation_path.get("relation") or "").strip()
            for relation_path in (query_plan.get("relation_paths") or [])
            if isinstance(relation_path, Mapping)
            and str(relation_path.get("relation") or "").strip()
        }
        ordering_relation = str(
            (
                query_plan.get("ordering_attribute")
                if isinstance(query_plan.get("ordering_attribute"), Mapping)
                else {}
            ).get("relation")
            or ""
        ).strip()
        if ordering_relation:
            plan_relations.add(ordering_relation)
        if not plan_relations:
            return []

        auxiliary_relations = {
            "type.object.name",
            "type.object.type",
            "common.topic.alias",
        }
        used_relations = self._extract_query_predicates(query_text)
        unexpected = sorted(
            relation
            for relation in used_relations
            if relation not in plan_relations and relation not in auxiliary_relations
        )
        return [f"query_uses_unplanned_predicate:{relation}" for relation in unexpected]

    def _validate_query_projection_bindings_against_plan(
        self,
        *,
        query_text: str,
        query_plan: Mapping[str, Any],
    ) -> list[str]:
        answer_mode = str(query_plan.get("answer_mode") or "entity").strip().lower()
        if answer_mode != "entity":
            return []
        projected_variables = self._extract_projected_variables(query_text)
        if not projected_variables:
            return []
        entity_variable = projected_variables[0]
        if self._query_variable_appears_in_non_optional_triples(
            query_text=query_text,
            variable_name=entity_variable,
        ):
            return []
        return [f"projected_entity_not_structurally_bound:{entity_variable}"]

    def _validate_query_constraint_bindings_against_plan(
        self,
        *,
        query_text: str,
    ) -> list[str]:
        errors: list[str] = []
        for variable_name in self._extract_ungrounded_type_instance_subjects(query_text):
            errors.append(f"ungrounded_type_set_variable:{variable_name}")
        for variable_name in self._extract_single_use_helper_variables(query_text):
            errors.append(f"single_use_helper_variable:{variable_name}")
        deduped_errors: list[str] = []
        for error in errors:
            if error not in deduped_errors:
                deduped_errors.append(error)
        return deduped_errors

    def _query_variable_appears_in_non_optional_triples(
        self,
        *,
        query_text: str,
        variable_name: str,
    ) -> bool:
        pattern = re.compile(
            rf"\?{re.escape(variable_name)}\b",
            flags=re.IGNORECASE,
        )
        for line in self._extract_structural_triple_lines(query_text):
            if pattern.search(line):
                return True
        return False

    def _extract_structural_triple_lines(self, query_text: str) -> list[str]:
        triple_lines: list[str] = []
        normalized_query_text = str(query_text or "")
        normalized_query_text = normalized_query_text.replace("{", "\n{\n").replace(
            "}", "\n}\n"
        )
        normalized_query_text = re.sub(
            r"\s+\.\s+(?=(?:[?A-Za-z_]|fb:))",
            " .\n",
            normalized_query_text,
            flags=re.IGNORECASE,
        )
        optional_depth = 0
        pending_optional_block = False
        for raw_line in normalized_query_text.splitlines():
            line = raw_line.strip()
            upper_line = line.upper()
            if not line:
                continue
            if upper_line.startswith("OPTIONAL"):
                pending_optional_block = True
                continue
            if line == "{":
                if pending_optional_block:
                    optional_depth += 1
                    pending_optional_block = False
                elif optional_depth > 0:
                    optional_depth += 1
                continue
            if line == "}":
                if optional_depth > 0:
                    optional_depth -= 1
                pending_optional_block = False
                continue
            if optional_depth > 0:
                continue
            if " fb:" not in line or not line.endswith("."):
                continue
            triple_lines.append(line)
        return triple_lines

    def _extract_ungrounded_type_instance_subjects(self, query_text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(
            r"(\?[A-Za-z_][A-Za-z0-9_]*)\s+fb:type\.type\.instance\s+\?[A-Za-z_][A-Za-z0-9_]*\s*\.",
            query_text or "",
            flags=re.IGNORECASE,
        ):
            variable_name = str(match.group(1) or "").lstrip("?").strip()
            if not variable_name or variable_name in seen:
                continue
            if self._query_variable_has_anchor_binding_clause(
                query_text=query_text,
                variable_name=variable_name,
            ):
                continue
            triple_degree = self._count_variable_occurrences_in_structural_triples(
                query_text=query_text,
                variable_name=variable_name,
            )
            if triple_degree >= 2:
                continue
            seen.add(variable_name)
            candidates.append(variable_name)
        return candidates

    def _extract_single_use_helper_variables(self, query_text: str) -> list[str]:
        candidates: list[str] = []
        projected_variables = {
            variable.lower()
            for variable in self._extract_projected_variables(query_text)
        }
        seen: set[str] = set()
        for line in self._extract_structural_triple_lines(query_text):
            for match in re.finditer(
                r"\?([A-Za-z_][A-Za-z0-9_]*)\b",
                line,
                flags=re.IGNORECASE,
            ):
                variable_name = str(match.group(1) or "").strip()
                variable_key = variable_name.lower()
                if (
                    not variable_name
                    or variable_key in seen
                    or variable_key in projected_variables
                    or self._query_variable_has_anchor_binding_clause(
                        query_text=query_text,
                        variable_name=variable_name,
                    )
                ):
                    continue
                structural_count = self._count_variable_occurrences_in_structural_triples(
                    query_text=query_text,
                    variable_name=variable_name,
                )
                total_count = len(
                    re.findall(
                        rf"\?{re.escape(variable_name)}\b",
                        query_text or "",
                        flags=re.IGNORECASE,
                    )
                )
                if structural_count == 1 and total_count <= 1:
                    seen.add(variable_key)
                    candidates.append(variable_name)
        return candidates

    def _query_variable_has_anchor_binding_clause(
        self,
        *,
        query_text: str,
        variable_name: str,
    ) -> bool:
        patterns = (
            rf"VALUES\s+\?{re.escape(variable_name)}\b",
            rf"\?{re.escape(variable_name)}\b\s+fb:type\.object\.name\b",
            rf"\?{re.escape(variable_name)}\b\s+fb:common\.topic\.alias\b",
        )
        return any(
            re.search(pattern, query_text or "", flags=re.IGNORECASE)
            for pattern in patterns
        )

    def _count_variable_occurrences_in_structural_triples(
        self,
        *,
        query_text: str,
        variable_name: str,
    ) -> int:
        pattern = re.compile(
            rf"\?{re.escape(variable_name)}\b",
            flags=re.IGNORECASE,
        )
        count = 0
        for line in self._extract_structural_triple_lines(query_text):
            if pattern.search(line):
                count += 1
        return count

    def _extract_query_predicates(self, query_text: str) -> set[str]:
        predicates: set[str] = set()
        for match in re.finditer(r"\bfb:([A-Za-z0-9_.]+)\b", query_text or ""):
            token = match.group(1)
            if re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", token):
                continue
            predicates.add(token)
        return predicates

    def _split_task_question(self, task_question: str) -> tuple[str, list[str]]:
        raw_text = str(task_question or "").strip()
        if ", Entities:" not in raw_text:
            return raw_text, []
        question_text, raw_entities = raw_text.split(", Entities:", 1)
        return question_text.strip(), self._parse_entities_payload(raw_entities.strip())

    def _parse_entities_payload(self, raw_entities: str) -> list[str]:
        try:
            parsed = ast.literal_eval(raw_entities)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        return [
            str(entity).strip()
            for entity in parsed
            if str(entity).strip()
        ]

    def _build_question_interpretation(
        self,
        *,
        question_text: str,
        explicit_entities: Sequence[str],
        answer_target_phrase: str,
    ) -> dict[str, Any]:
        lower_text = str(question_text or "").lower()
        is_count_question = self._is_count_question_text(question_text)
        inputs: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        answer_target_clean = str(answer_target_phrase or "").strip()
        answer_target_class_phrase, _ = self._split_answer_target_compound_phrase(
            answer_target_clean
        )

        def _add_input(
            surface: str,
            *,
            kind: str,
            role_hint: str,
            reason: str,
        ) -> None:
            cleaned_surface = self._clean_question_input_surface(surface)
            if not cleaned_surface:
                return
            normalized_kind = self._normalize_question_input_kind(kind)
            normalized_role = self._normalize_question_input_role(role_hint)
            signature = (
                cleaned_surface.lower(),
                normalized_kind,
                normalized_role,
            )
            if signature in seen:
                return
            seen.add(signature)
            inputs.append(
                {
                    "surface": cleaned_surface,
                    "kind": normalized_kind,
                    "role_hint": normalized_role,
                    "reason": str(reason or "").strip() or "question_semantics",
                }
            )

        explicit_anchor_role = "anchor_a" if len(explicit_entities) > 1 else "anchor"
        normalized_explicit_entities = [
            self._clean_question_input_surface(entity).lower()
            for entity in explicit_entities
            if self._clean_question_input_surface(entity)
        ]
        for index, entity in enumerate(explicit_entities):
            clue = self._infer_entity_clue(question_text, entity)
            if (
                clue.startswith("attribute_value.")
                and len(explicit_entities) == 1
                and is_count_question
            ):
                kind = "named_entity"
                role_hint = "anchor"
            elif clue.startswith("attribute_value."):
                kind = "attribute_value"
                role_hint = "constraint_value"
            elif (
                len(explicit_entities) == 1
                and
                clue == "surface_constraint"
                and answer_target_class_phrase
                and str(entity or "").strip().lower()
                == answer_target_class_phrase.lower()
            ):
                kind = "class_phrase"
                role_hint = "type_set"
            else:
                kind = "named_entity"
                if len(explicit_entities) == 1:
                    role_hint = "anchor"
                else:
                    role_hint = "anchor_a" if index == 0 else "anchor_b"
            _add_input(
                entity,
                kind=kind,
                role_hint=role_hint,
                reason=f"explicit_entity:{clue}",
            )

        shared_attribute_patterns = (
            r"\bsame ([a-z][a-z _-]+?) as\b",
            r"\bsame ([a-z][a-z _-]+?) with\b",
            r"\bhave the same ([a-z][a-z _-]+?) as\b",
            r"\b(?:by|from|through) the ([a-z][a-z _-]+?) of\b",
            r"\bhave in common\b",
        )
        for pattern in shared_attribute_patterns:
            match = re.search(pattern, lower_text)
            if match is None:
                continue
            if match.groups():
                attribute_surface = match.group(1).strip()
            else:
                attribute_surface = str(answer_target_phrase or "").strip() or "shared_attribute"
            normalized_attribute_surface = self._clean_question_input_surface(attribute_surface).lower()
            if normalized_attribute_surface and any(
                explicit_entity.startswith(normalized_attribute_surface + " of ")
                for explicit_entity in normalized_explicit_entities
            ):
                continue
            if attribute_surface:
                _add_input(
                    attribute_surface,
                    kind="shared_attribute",
                    role_hint="shared_attribute",
                    reason="shared/common cue in question text",
                )

        if any(
            phrase in lower_text
            for phrase in (
                "same type as",
                "share a type with",
                "share their type with",
                "of the same type as",
                "same category as",
                "same category",
            )
        ):
            type_surface = (
                "category"
                if "category" in lower_text
                else "type"
            )
            _add_input(
                type_surface,
                kind="shared_attribute",
                role_hint="shared_attribute",
                reason="shared type/category cue in question text",
            )

        for keyword in (
            "longest",
            "latest",
            "most recently",
            "most recent",
            "earliest",
            "first",
            "greatest",
            "highest",
            "lowest",
            "farthest",
            "furthest",
            "closest",
            "nearest",
        ):
            if keyword in lower_text:
                _add_input(
                    keyword,
                    kind="ordering_cue",
                    role_hint="ordering_attribute",
                    reason="ordering cue in question text",
                )
                break

        if answer_target_clean:
            _add_input(
                answer_target_clean,
                kind="answer_target",
                role_hint="answer_target",
                reason="derived answer target phrase",
            )

        if not explicit_entities:
            cue_patterns: list[tuple[str, str, str]] = [
                (
                    r"^\s*(?:Question:\s*)?([^,?]+?),\s+which\s+(?:is|was|were)\b",
                    "named_entity",
                    "anchor_a",
                ),
                (
                    r"^\s*(?:Question:\s*)?(?:the\s+)?([^?]+?)\s+has\s+how many\b",
                    "named_entity",
                    "constraint_value",
                ),
                (r"\bmade by ([^?]+?)(?: and |\?|$)", "named_entity", explicit_anchor_role),
                (r"\bfeatured ([^?]+?)(?: and |\?|$)", "named_entity", "anchor_b"),
                (r"\bfeaturing ([^?]+?)(?: and |\?|$)", "named_entity", "anchor_b"),
                (r"\bcreated by ([^?]+?)(?: and |\?|$)", "named_entity", explicit_anchor_role),
                (r"\bcomposed by ([^?]+?)(?: and |\?|$)", "named_entity", explicit_anchor_role),
                (r"\bproduced by ([^?]+?)(?: and |\?|$)", "named_entity", explicit_anchor_role),
                (r"\bdeveloped by ([^?,]+?)(?:,| and |\?|$)", "named_entity", explicit_anchor_role),
                (r"\bdistributed through ([^?]+?)(?: and |\?|$)", "named_entity", "constraint_value"),
                (r"\bsame [^?]*? as ([^?]+?)(?:\?|$)", "named_entity", "anchor_b"),
                (r"\bof the same type as ([^?]+?)(?:\?|$)", "named_entity", "anchor_b"),
                (r"\bshare (?:their )?type with ([^?]+?)(?:\?|$)", "named_entity", "anchor_b"),
                (r"\bfrom ([^?]+?)(?: have| has| that| who| which| and |\?|$)", "named_entity", "constraint_value"),
                (r"\bon ([^?]+?)(?:\?|$)", "named_entity", "anchor"),
                (r"\bof ([^?]+?)(?: is| are| was| were|\?|$)", "named_entity", "anchor"),
                (r"\bpart of ([^?]+?)(?: and |\?|$)", "named_entity", "anchor_a"),
            ]
            for pattern, kind, role_hint in cue_patterns:
                match = re.search(pattern, question_text, flags=re.IGNORECASE)
                if match is None:
                    continue
                captured = match.group(1).strip()
                if (
                    pattern == r"\bof ([^?]+?)(?: is| are| was| were|\?|$)"
                    and any(
                        token in captured.lower()
                        for token in (" have ", " has ", " who ", " which ", " that ", " as ")
                    )
                ):
                    continue
                if "'s " in captured:
                    captured = captured.split("'s", 1)[0].strip()
                for candidate in self._split_question_input_candidates(captured):
                    _add_input(
                        candidate,
                        kind=kind,
                        role_hint=role_hint,
                        reason=f"derived_from_pattern:{pattern}",
                    )

            possessive_match = re.search(
                r"\b([A-Za-z0-9][A-Za-z0-9.&+/_-]{0,80})'s\b",
                question_text,
                flags=re.IGNORECASE,
            )
            if possessive_match is not None:
                _add_input(
                    possessive_match.group(1),
                    kind="named_entity",
                    role_hint="anchor",
                    reason="possessive anchor phrase",
                )

            two_anchor_match = re.search(
                r"\b([A-Za-z0-9][A-Za-z0-9 .&+/_-]{0,60}?)\s+and\s+([A-Za-z0-9][A-Za-z0-9 .&+/_-]{0,60}?)\s+(?:have|has|did|do|made|make)\b",
                question_text,
                flags=re.IGNORECASE,
            )
            if two_anchor_match is not None:
                _add_input(
                    two_anchor_match.group(1),
                    kind="named_entity",
                    role_hint="anchor_a",
                    reason="two-anchor conjunction before governing verb",
                )
                _add_input(
                    two_anchor_match.group(2),
                    kind="named_entity",
                    role_hint="anchor_b",
                    reason="two-anchor conjunction before governing verb",
                )

        if not is_count_question:
            class_phrase, target_head = self._split_answer_target_compound_phrase(
                answer_target_clean
            )
            if target_head and target_head.lower() != answer_target_clean.lower():
                _add_input(
                    target_head,
                    kind="answer_target",
                    role_hint="answer_target",
                    reason="head noun phrase extracted from answer target",
                )
            if class_phrase:
                _add_input(
                    class_phrase,
                    kind="class_phrase",
                    role_hint="type_set",
                    reason="class/category qualifier extracted from answer target",
                )

        preferred_scaffolds = self._build_preferred_scaffold_candidates(
            question_text=question_text,
            answer_target_phrase=answer_target_phrase,
            question_inputs=inputs,
        )
        return {
            "question_inputs": inputs,
            "preferred_scaffolds": preferred_scaffolds,
        }

    def _normalize_question_input_kind(self, raw_kind: Any) -> str:
        normalized_kind = str(raw_kind or "").strip().lower().replace("-", "_")
        if normalized_kind in {
            "named_entity",
            "class_phrase",
            "attribute_value",
            "type_constraint",
            "shared_attribute",
            "answer_target",
            "ordering_cue",
        }:
            return normalized_kind
        return "named_entity"

    def _normalize_question_input_role(self, raw_role: Any) -> str:
        normalized_role = self._normalize_relation_role(raw_role)
        if normalized_role in {
            "anchor",
            "anchor_a",
            "anchor_b",
            "constraint_value",
            "type_set",
            "shared_attribute",
            "ordering_attribute",
            "answer_target",
        }:
            return normalized_role
        return "anchor"

    def _clean_question_input_surface(self, surface: Any) -> str:
        cleaned = str(surface or "").strip()
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" ,.;:?")

    def _split_question_input_candidates(self, surface: str) -> list[str]:
        raw_surface = str(surface or "").strip()
        if not raw_surface:
            return []
        pieces = re.split(r"\s+and\s+|,\s*", raw_surface)
        candidates: list[str] = []
        for piece in pieces:
            cleaned = self._clean_question_input_surface(piece)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        return candidates

    def _split_answer_target_compound_phrase(
        self,
        answer_target_phrase: str,
    ) -> tuple[str, str]:
        phrase = str(answer_target_phrase or "").strip()
        lower_phrase = phrase.lower()
        if any(
            marker in lower_phrase
            for marker in (
                " that ",
                " which ",
                " who ",
                " can ",
                " with ",
                " from ",
            )
        ):
            return "", phrase
        tokens = [token for token in phrase.split() if token]
        if len(tokens) < 3:
            return "", phrase
        tail = tokens[-2:]
        head_phrase = " ".join(tail)
        qualifier_phrase = " ".join(tokens[:-2]).strip()
        if not qualifier_phrase:
            return "", phrase
        return qualifier_phrase, head_phrase

    def _semantic_hint_tokens(self, text: Any) -> set[str]:
        tokens: set[str] = set()
        for raw_token in re.split(r"[\s_/.\-]+", str(text or "").lower()):
            cleaned = self._normalize_variable_token(
                self._singularize_surface_token(raw_token)
            )
            if cleaned in {
                "",
                "a",
                "an",
                "the",
                "and",
                "or",
                "of",
                "in",
                "on",
                "for",
                "to",
                "from",
                "with",
                "by",
                "that",
                "which",
                "who",
                "what",
                "answer",
                "candidate",
                "set",
                "shared",
                "value",
            }:
                continue
            tokens.add(cleaned)
        return tokens

    def _answer_target_qualifier_is_encoded_in_answer_relation(
        self,
        *,
        answer_target_phrase: str,
        relation_candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        qualifier_phrase, target_head = self._split_answer_target_compound_phrase(
            answer_target_phrase
        )
        if not qualifier_phrase or not target_head:
            return False
        qualifier_tokens = self._semantic_hint_tokens(qualifier_phrase)
        head_tokens = self._semantic_hint_tokens(target_head)
        if not qualifier_tokens or not head_tokens:
            return False

        for candidate in relation_candidates:
            if not isinstance(candidate, Mapping):
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if "answer" not in {from_role, to_role}:
                continue
            relation_text = " ".join(
                str(candidate.get(field) or "").strip()
                for field in (
                    "relation",
                    "from",
                    "to",
                    "from_role",
                    "to_role",
                    "support",
                    "use_when",
                )
            )
            relation_tokens = self._semantic_hint_tokens(relation_text)
            if qualifier_tokens & relation_tokens and head_tokens & relation_tokens:
                return True
        return False

    def _answer_target_is_relation_encoded_type_projection(
        self,
        *,
        answer_target_phrase: str,
        relation_candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        phrase = str(answer_target_phrase or "").strip()
        lower_phrase = phrase.lower()
        if not phrase:
            return False
        if not any(
            token in lower_phrase
            for token in ("type", "types", "kind", "kinds", "category", "categories")
        ):
            return False
        if re.match(
            r"^(?:other\s+)?(?:type|types|kind|kinds|category|categories)\s+of\b",
            lower_phrase,
        ):
            return False
        generic_tokens = {"type", "types", "kind", "kinds", "category", "categories"}
        phrase_tokens = self._semantic_hint_tokens(phrase)
        specific_tokens = {token for token in phrase_tokens if token not in generic_tokens}
        if not specific_tokens:
            return False

        for candidate in relation_candidates:
            if not isinstance(candidate, Mapping):
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if "answer" not in {from_role, to_role}:
                continue
            relation_text = " ".join(
                str(candidate.get(field) or "").strip()
                for field in (
                    "relation",
                    "from",
                    "to",
                    "from_role",
                    "to_role",
                    "support",
                    "use_when",
                )
            )
            relation_tokens = self._semantic_hint_tokens(relation_text)
            if specific_tokens.issubset(relation_tokens) and phrase_tokens & relation_tokens:
                return True
        return False

    def _count_answer_target_head_is_encoded_in_relation(
        self,
        *,
        answer_target_phrase: str,
        relation_candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        qualifier_phrase, target_head = self._split_answer_target_compound_phrase(
            answer_target_phrase
        )
        del qualifier_phrase
        target_phrase = target_head or answer_target_phrase
        head_tokens = self._semantic_hint_tokens(target_phrase)
        if not head_tokens:
            return False

        for candidate in relation_candidates:
            if not isinstance(candidate, Mapping):
                continue
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if {from_role, to_role}.isdisjoint(
                {"candidate_set", "count_set", "shared_answer", "answer"}
            ):
                continue
            relation_text = " ".join(
                str(candidate.get(field) or "").strip()
                for field in (
                    "relation",
                    "from",
                    "to",
                    "from_role",
                    "to_role",
                    "support",
                    "use_when",
                )
            )
            relation_tokens = self._semantic_hint_tokens(relation_text)
            if head_tokens & relation_tokens:
                return True
        return False

    def _refine_question_interpretation_with_grounding(
        self,
        *,
        question_text: str,
        answer_target_phrase: str,
        question_interpretation: Mapping[str, Any],
        grounded_relation_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        interpretation = dict(question_interpretation or {})
        question_inputs = [
            dict(item)
            for item in (interpretation.get("question_inputs") or [])
            if isinstance(item, Mapping)
        ]
        if not question_inputs:
            return interpretation
        qualifier_encoded = self._answer_target_qualifier_is_encoded_in_answer_relation(
            answer_target_phrase=answer_target_phrase,
            relation_candidates=grounded_relation_candidates,
        )
        relation_encoded_type_projection = (
            self._answer_target_is_relation_encoded_type_projection(
                answer_target_phrase=answer_target_phrase,
                relation_candidates=grounded_relation_candidates,
            )
        )
        if not qualifier_encoded and not relation_encoded_type_projection:
            return interpretation

        qualifier_phrase, _ = self._split_answer_target_compound_phrase(
            answer_target_phrase
        )
        qualifier_surface = self._clean_question_input_surface(qualifier_phrase).lower()
        filtered_inputs: list[dict[str, Any]] = []
        removed = False
        for item in question_inputs:
            kind = str(item.get("kind") or "").strip()
            reason = str(item.get("reason") or "").lower()
            surface = self._clean_question_input_surface(item.get("surface")).lower()
            if (
                qualifier_encoded
                and kind in {"class_phrase", "type_constraint"}
                and surface == qualifier_surface
                and (
                    "answer target" in reason
                    or "class/category qualifier" in reason
                    or "head noun phrase" in reason
                )
            ):
                removed = True
                continue
            filtered_inputs.append(item)
        if not removed:
            filtered_inputs = question_inputs

        refined = dict(interpretation)
        if removed:
            refined["question_inputs"] = filtered_inputs
        preferred_scaffolds = self._build_preferred_scaffold_candidates(
            question_text=question_text,
            answer_target_phrase=answer_target_phrase,
            question_inputs=filtered_inputs,
        )
        if qualifier_encoded or relation_encoded_type_projection:
            preferred_scaffolds = [
                item
                for item in preferred_scaffolds
                if str(item.get("name") or "").strip() != "shared_type_lookup"
            ]
        refined["preferred_scaffolds"] = preferred_scaffolds
        return refined

    def _build_preferred_scaffold_candidates(
        self,
        *,
        question_text: str,
        answer_target_phrase: str,
        question_inputs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        lower_text = re.sub(
            r"^\s*question:\s*",
            "",
            str(question_text or "").lower(),
            flags=re.IGNORECASE,
        )
        answer_target_lower = str(answer_target_phrase or "").lower()
        is_count = (
            lower_text.startswith("how many")
            or re.search(r"\bhow many\b", lower_text) is not None
            or re.search(r"\bhow much\b", lower_text) is not None
            or "number of" in lower_text
            or "total number of" in lower_text
            or "amount of" in lower_text
        )
        is_superlative = any(
            phrase in lower_text
            for phrase in (
                "longest",
                "shortest",
                "largest",
                "smallest",
                "latest",
                "most recently",
                "most recent",
                "earliest",
                "first",
                "greatest",
                "highest",
                "lowest",
                "farthest",
                "furthest",
                "closest",
                "nearest",
                "last day",
                "maximum",
                "minimum",
            )
        )
        has_shared_attribute = any(
            str(item.get("kind") or "") == "shared_attribute"
            for item in question_inputs
        )
        has_class_phrase = any(
            str(item.get("kind") or "") in {"class_phrase", "type_constraint"}
            for item in question_inputs
        )
        anchor_like_count = sum(
            1
            for item in question_inputs
            if str(item.get("role_hint") or "") in {"anchor", "anchor_a", "anchor_b"}
        )
        surface_constraint_count = sum(
            1
            for item in question_inputs
            if str(item.get("role_hint") or "") in {"anchor", "anchor_a", "anchor_b", "constraint_value"}
            or str(item.get("kind") or "") == "attribute_value"
        )
        candidates: list[dict[str, Any]] = []

        def _add(name: str, priority: int, reason: str) -> None:
            if any(existing.get("name") == name for existing in candidates):
                return
            candidates.append(
                {
                    "name": name,
                    "priority": priority,
                    "reason": str(reason or "").strip(),
                }
            )

        if is_count and has_shared_attribute:
            _add(
                "count_shared_attribute",
                1,
                "count question with an explicit shared/common attribute cue",
            )
        if has_shared_attribute and not is_count:
            _add(
                "shared_attribute_intersection",
                1,
                "lookup question where anchors meet on a shared attribute or category",
            )
        if has_class_phrase and is_count:
            _add(
                "class_filtered_count",
                1 if not has_shared_attribute else 2,
                "count question with a derived class/category phrase",
            )
        if has_class_phrase and not is_count:
            _add(
                "type_instance_lookup",
                2 if has_shared_attribute else 1,
                "lookup requires separating a class/category phrase from the returned instances",
            )
        if is_superlative:
            _add(
                "superlative_over_candidate_set",
                1,
                "question contains an explicit ordering cue",
            )
        if surface_constraint_count >= 2 and is_count and not has_shared_attribute:
            _add(
                "count_over_joined_set",
                2,
                "count question with multiple surface constraints",
            )
        if surface_constraint_count >= 2 and not is_count and not is_superlative:
            _add(
                "projected_answer_intersection",
                2,
                "multiple surface anchors constrain a shared answer set",
            )
        if any(
            token in answer_target_lower
            for token in ("type", "types", "kind", "kinds", "category", "categories")
        ):
            _add(
                "shared_type_lookup",
                1 if has_shared_attribute else 2,
                "answer target explicitly asks for a type/category",
            )
        if any(token in lower_text for token in ("creator", "creators", "created by")):
            _add(
                "pivoted_chain_lookup",
                3,
                "question likely requires a pivot from a superlative-selected entity to a creator relation",
            )
        if not candidates:
            _add(
                "direct_lookup" if not is_count else "direct_count",
                1,
                "fallback scaffold inferred from the question form",
            )
        return sorted(candidates, key=lambda item: int(item.get("priority") or 999))

    def _extract_grounding_entities_from_question_interpretation(
        self,
        question_interpretation: Mapping[str, Any],
    ) -> list[str]:
        question_inputs = [
            item
            for item in (question_interpretation.get("question_inputs") or [])
            if isinstance(item, Mapping)
        ]
        named_entity_items = [
            item
            for item in question_inputs
            if str(item.get("kind") or "").strip() == "named_entity"
        ]
        preserve_answer_target_named_entities = len(named_entity_items) >= 2
        non_entity_surfaces = {
            self._clean_question_input_surface(item.get("surface")).lower()
            for item in question_inputs
            if str(item.get("kind") or "") != "named_entity"
            and (
                not preserve_answer_target_named_entities
                or str(item.get("kind") or "").strip() != "answer_target"
            )
            and self._clean_question_input_surface(item.get("surface"))
        }
        extracted_entities: list[str] = []
        for item in question_inputs:
            if str(item.get("kind") or "") != "named_entity":
                continue
            surface = self._clean_question_input_surface(item.get("surface"))
            if surface.lower() in non_entity_surfaces:
                continue
            if surface and surface not in extracted_entities:
                extracted_entities.append(surface)
        return extracted_entities

    def _build_pal_grounding_card(
        self,
        task_question: str,
        *,
        relation_grounding: Optional[Sequence[Mapping[str, str]]] = None,
        alias_overrides: Optional[Mapping[str, str]] = None,
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> str:
        question_text, entities = self._split_task_question(task_question)
        normalized_entities = [
            self._normalize_grounding_entity(entity)
            for entity in entities
        ]
        answer_target_phrase = self._extract_answer_target_phrase(question_text)
        domain_hints = self._infer_domain_hints(question_text)
        normalized_question_interpretation = (
            question_interpretation
            if isinstance(question_interpretation, Mapping)
            else self._build_question_interpretation(
                question_text=question_text,
                explicit_entities=entities,
                answer_target_phrase=answer_target_phrase,
            )
        )
        grounded_relation_candidates = list(
            relation_grounding
            if relation_grounding is not None
            else self._build_grounded_relation_candidates(task_question)
        )
        normalized_question_interpretation = self._refine_question_interpretation_with_grounding(
            question_text=question_text,
            answer_target_phrase=answer_target_phrase,
            question_interpretation=normalized_question_interpretation,
            grounded_relation_candidates=grounded_relation_candidates,
        )
        interpreted_inputs = [
            item
            for item in (normalized_question_interpretation.get("question_inputs") or [])
            if isinstance(item, Mapping)
        ]
        preferred_scaffolds = [
            item
            for item in (
                normalized_question_interpretation.get("preferred_scaffolds") or []
            )
            if isinstance(item, Mapping)
        ]
        derived_entities = (
            entities
            if entities
            else self._extract_grounding_entities_from_question_interpretation(
                normalized_question_interpretation
            )
        )
        query_shape = self._infer_query_shape(
            question_text=question_text,
            entities=derived_entities,
            answer_target_phrase=answer_target_phrase,
            question_inputs=interpreted_inputs,
            grounded_relation_candidates=grounded_relation_candidates,
        )
        family_success_patterns = self._format_family_success_archetypes(
            family_name=query_shape,
        )
        lines = [
            "PAL grounding hints:",
            f"- question_text: {question_text}",
            f"- answer_target_phrase: {answer_target_phrase or 'unknown'}",
            f"- domain_hints: {', '.join(domain_hints) if domain_hints else 'none'}",
            f"- query_shape: {query_shape}",
        ]
        relation_hints = [
            str(candidate.get("relation") or "")
            for candidate in grounded_relation_candidates
            if str(candidate.get("relation") or "")
        ]
        if relation_hints:
            lines.append(f"- relation_hints: {', '.join(relation_hints)}")
        if grounded_relation_candidates:
            lines.append("- grounded_relation_candidates:")
            for candidate in grounded_relation_candidates:
                lines.append(
                    "  - relation="
                    + repr(str(candidate.get("relation") or ""))
                    + "; direction="
                    + str(candidate.get("direction") or "")
                    + "; from_role="
                    + str(candidate.get("from_role") or "")
                    + "; to_role="
                    + str(candidate.get("to_role") or "")
                    + "; from="
                    + str(candidate.get("from") or "")
                    + "; to="
                    + str(candidate.get("to") or "")
                    + "; grounding_source="
                    + str(candidate.get("grounding_source") or "")
                    + "; support="
                    + str(candidate.get("support") or "")
                    + "; use_when="
                    + str(candidate.get("use_when") or "")
                )
        if interpreted_inputs:
            lines.append("- question_inputs:")
            for item in interpreted_inputs:
                surface = self._clean_question_input_surface(item.get("surface"))
                kind = self._normalize_question_input_kind(item.get("kind"))
                role_hint = self._normalize_question_input_role(item.get("role_hint"))
                if kind in {"class_phrase", "type_constraint", "answer_target"}:
                    alias_candidates = self._build_type_constraint_alias_candidates(surface)
                elif kind == "attribute_value":
                    alias_candidates = self._build_entity_alias_candidates(
                        surface,
                        entity_clue="attribute_value",
                    )
                else:
                    alias_candidates = self._build_entity_alias_candidates(
                        surface,
                        entity_clue="surface_constraint",
                    )
                lines.append(
                    "  - surface="
                    + repr(surface)
                    + "; kind="
                    + kind
                    + "; role_hint="
                    + role_hint
                    + "; alias_candidates="
                    + repr(alias_candidates)
                    + "; recommended_alias="
                    + repr(alias_candidates[0] if alias_candidates else surface)
                    + "; reason="
                    + repr(str(item.get("reason") or "question_semantics"))
                )
        if preferred_scaffolds:
            lines.append("- scaffold_candidates:")
            for scaffold in preferred_scaffolds:
                lines.append(
                    "  - name="
                    + str(scaffold.get("name") or "")
                    + "; priority="
                    + str(scaffold.get("priority") or "")
                    + "; reason="
                    + repr(str(scaffold.get("reason") or ""))
                )
        if entities:
            lines.append("- question_entities:")
            for raw_entity, normalized_entity in zip(entities, normalized_entities):
                clue = self._infer_entity_clue(question_text, raw_entity)
                alias_candidates = self._build_entity_alias_candidates(
                    raw_entity,
                    entity_clue=clue,
                )
                override_alias = str(
                    (alias_overrides or {}).get(raw_entity) or ""
                ).strip()
                if override_alias:
                    alias_candidates = [
                        candidate
                        for candidate in alias_candidates
                        if candidate != override_alias
                    ]
                    alias_candidates.insert(0, override_alias)
                lines.append(
                    "  - raw="
                    + repr(raw_entity)
                    + "; normalized="
                    + repr(normalized_entity)
                    + "; clue="
                    + clue
                    + "; alias_candidates="
                    + repr(alias_candidates)
                    + "; recommended_alias="
                    + repr(alias_candidates[0] if alias_candidates else raw_entity)
                )
            if self._should_surface_answer_target_as_type_constraint(
                answer_target_phrase=answer_target_phrase,
                grounded_relation_candidates=grounded_relation_candidates,
            ):
                type_alias_candidates = self._build_type_constraint_alias_candidates(
                    answer_target_phrase
                )
                if type_alias_candidates:
                    lines.append(
                        "  - raw="
                        + repr(answer_target_phrase)
                        + "; normalized="
                        + repr(type_alias_candidates[0])
                        + "; clue=type_constraint; alias_candidates="
                        + repr(type_alias_candidates)
                        + "; recommended_alias="
                        + repr(type_alias_candidates[0])
                    )
        shape_guidance = self._build_query_shape_guidance(query_shape)
        if shape_guidance:
            lines.append("- shape_guidance:")
            for guidance_line in shape_guidance:
                lines.append(f"  - {guidance_line}")
        if family_success_patterns:
            lines.append("- family_success_patterns:")
            for pattern in family_success_patterns:
                lines.append(f"  - {pattern}")
        lines.extend(
            [
                "- guidance:",
                "  - Apply all surface constraints to the same answer entity unless the question explicitly asks for separate outputs.",
                "  - Preserve benchmark entity surface forms unless the grounding card explicitly gives a better canonical alias.",
                "  - Prefer binding surface entities and attribute values by English names first, then join through relations.",
                "  - Preserve the question_inputs role hints unless grounded evidence forces a different interpretation.",
                "  - If scaffold_candidates are listed, start with the highest-priority scaffold family before inventing an alternative.",
                "  - Choose relation_paths only from grounded_relation_candidates unless there is no grounded option and exploratory mode is explicitly justified.",
                "  - Prefer relation names from the grounding card over semantically similar guesses.",
                "  - Avoid guessing `fb:en.*` identifiers for question terms unless the exact identifier is already known.",
            ]
        )
        return "\n".join(lines)

    def _infer_query_shape(
        self,
        *,
        question_text: str,
        entities: Sequence[str],
        answer_target_phrase: str,
        question_inputs: Optional[Sequence[Mapping[str, Any]]] = None,
        grounded_relation_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        lower_text = re.sub(
            r"^\s*question:\s*",
            "",
            str(question_text or "").lower(),
            flags=re.IGNORECASE,
        )
        answer_target_lower = str(answer_target_phrase or "").lower()
        normalized_question_inputs = [
            item
            for item in (question_inputs or [])
            if isinstance(item, Mapping)
        ]
        anchor_like_count = max(
            len(entities),
            sum(
                1
                for item in normalized_question_inputs
                if str(item.get("role_hint") or "").strip()
                in {"anchor", "anchor_a", "anchor_b"}
            ),
        )
        has_shared_attribute = any(
            str(item.get("kind") or "").strip() == "shared_attribute"
            or str(item.get("role_hint") or "").strip() == "shared_attribute"
            for item in normalized_question_inputs
        )
        has_constraint_value = any(
            str(item.get("kind") or "").strip() == "attribute_value"
            or str(item.get("role_hint") or "").strip() == "constraint_value"
            for item in normalized_question_inputs
        )
        is_type_target = any(
            token in answer_target_lower
            for token in ("type", "types", "kind", "kinds", "category", "categories")
        )
        if is_type_target and grounded_relation_candidates:
            if self._answer_target_is_relation_encoded_type_projection(
                answer_target_phrase=answer_target_phrase,
                relation_candidates=grounded_relation_candidates,
            ):
                is_type_target = False
        is_count = (
            lower_text.startswith("how many")
            or "number of" in lower_text
            or "amount of" in lower_text
            or lower_text.startswith("how much")
        )
        if any(
            phrase in lower_text
            for phrase in (
                "introduced first",
                "introduced earliest",
                "introduced latest",
                "came first",
                "earliest",
                "oldest",
                "youngest",
                "latest",
                "farthest",
                "furthest",
                "closest",
                "nearest",
                "last day",
                "first?",
            )
        ):
            return "superlative_chain"
        if is_count:
            if has_shared_attribute or has_constraint_value or anchor_like_count > 1:
                return "count_over_joined_set"
            return "count_over_direct_relation"
        if is_type_target and (has_shared_attribute or anchor_like_count > 1):
            return "shared_type_intersection"
        if anchor_like_count > 1:
            return "multi_anchor_intersection"
        if any(
            phrase in lower_text
            for phrase in (
                "influenced by",
                "developed by",
                "designed by",
                "made by",
                "produced by",
                "leader of",
                "leaders of",
            )
        ):
            return "single_anchor_chain_lookup"
        if answer_target_lower:
            return "single_anchor_lookup"
        return "single_anchor_lookup"

    def _build_query_shape_guidance(self, query_shape: str) -> list[str]:
        if query_shape == "count_over_direct_relation":
            return [
                "Define one answer-set variable reached from the anchor and count that variable.",
                "Prefer one grounded relation family from the anchor before adding helper type or class constraints.",
                "If a helper type/category constraint is needed, surface it as a separate type_set or constraint_value role instead of folding it into the main anchor.",
            ]
        if query_shape == "count_over_joined_set":
            return [
                "Define one candidate answer variable first, then treat the other anchors as filters on that same answer set.",
                "Prefer conjunctive filtering on the answer variable over serial narrative chains between surface anchors.",
                "Do not invent intermediate producer/creator/owner hops unless those hops are grounded or explicitly exploratory.",
            ]
        if query_shape == "multi_anchor_intersection":
            return [
                "Use one shared answer entity variable and apply each anchor as a constraint on that same variable.",
                "Prefer intersection-style filtering over chaining one anchor through another unless the chain is grounded.",
            ]
        if query_shape == "shared_type_intersection":
            return [
                "Extract the shared type/category/value set explicitly and decide whether that shared set is the final answer or a filter on answer instances.",
                "If the question asks for other entities that share a type/category, keep the shared type as an intermediate variable and project the entity set separately.",
            ]
        if query_shape == "superlative_chain":
            return [
                "First build the candidate answer set, then bind the ordering attribute or date, then order and limit.",
                "Do not collapse candidate selection and ordering into one ungrounded relation guess.",
            ]
        if query_shape == "single_anchor_chain_lookup":
            return [
                "Keep the chain short and ensure each hop is grounded or explicitly exploratory.",
                "Bind the surface anchor first, then follow the minimal relation chain needed for the answer.",
            ]
        if query_shape == "single_anchor_lookup":
            return [
                "Bind the anchor by English name and prefer one direct grounded relation to the answer variable.",
                "Use semantic roles such as anchor, answer, candidate_set, or count_set instead of raw SPARQL variable names.",
                "Avoid adding extra helper joins unless the grounding card explicitly supports them.",
            ]
        return []

    def _build_grounded_relation_candidates_with_dynamic_fallback(
        self,
        *,
        task_question: str,
        entities: Sequence[str],
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_interpretation: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, str]]:
        """Return curated candidates when available; fall back to live predicate probe otherwise."""
        interpreted_inputs = [
            item
            for item in ((question_interpretation or {}).get("question_inputs") or [])
            if isinstance(item, Mapping)
        ]
        question_text = self._split_task_question(task_question)[0]
        base_relation_grounding = self._build_grounded_relation_candidates(task_question)
        query_shape = self._infer_query_shape(
            question_text=question_text,
            entities=entities,
            answer_target_phrase=answer_target_phrase,
            question_inputs=interpreted_inputs,
            grounded_relation_candidates=base_relation_grounding,
        )
        answer_mode = "count" if self._is_count_question_text(question_text) else "entity"
        type_dynamic = self._probe_answer_class_dynamic_candidates(
            task_question=task_question,
            entities=entities,
            answer_target_phrase=answer_target_phrase,
            domain_hints=domain_hints,
            interpreted_inputs=interpreted_inputs,
            query_shape=query_shape,
            answer_mode=answer_mode,
        )
        direct_count_dynamic: list[dict[str, str]] = []
        if query_shape == "count_over_direct_relation" and len(entities) == 1:
            direct_count_dynamic = self._normalize_grounded_relation_candidates(
                relation_candidates=self._probe_dynamic_relation_candidates(
                    entities=entities,
                    answer_target_phrase=answer_target_phrase,
                    domain_hints=domain_hints,
                    question_text=question_text,
                ),
                query_shape=query_shape,
                answer_mode=answer_mode,
                answer_target_phrase=answer_target_phrase,
                entities=entities,
            )
        curated = self._normalize_grounded_relation_candidates(
            relation_candidates=self._augment_generic_type_relation_candidates(
                relation_candidates=base_relation_grounding,
                answer_target_phrase=answer_target_phrase,
                question_interpretation=question_interpretation,
            ),
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            entities=entities,
        )
        if direct_count_dynamic:
            curated = self._merge_relation_grounding_candidates(
                curated,
                direct_count_dynamic,
                prefer_extra=False,
            )
        if type_dynamic:
            curated = self._merge_relation_grounding_candidates(
                curated,
                type_dynamic,
                prefer_extra=True,
            )
        curated = self._prune_redundant_grounded_relation_candidates(
            relation_candidates=curated,
            question_text=question_text,
            answer_target_phrase=answer_target_phrase,
        )
        if curated:
            self._emit_generated_tools_event(
                {
                    "event": "pal_grounding_curated_hit",
                    "mode": "pal",
                    "curated_count": len(curated),
                    "relations": [c.get("relation") for c in curated],
                    "normalized_candidates": curated,
                }
            )
            return curated
        self._emit_generated_tools_event(
            {
                "event": "pal_grounding_curated_miss",
                "mode": "pal",
                "question_preview": (answer_target_phrase or "")[:80],
            }
        )
        if direct_count_dynamic and query_shape == "count_over_direct_relation" and len(entities) == 1:
            dynamic = direct_count_dynamic
        elif len(entities) > 1 and query_shape in {
            "multi_anchor_intersection",
            "count_over_joined_set",
            "shared_type_intersection",
        }:
            normalized_anchored_entities = self._normalize_anchored_entities(
                [
                    {"surface": entity, "chosen_alias": entity, "role": "anchor"}
                    for entity in entities
                ],
                query_shape=query_shape,
            )
            dynamic = self._probe_dynamic_relation_candidates_for_anchors(
                anchored_entities=normalized_anchored_entities,
                answer_target_phrase=answer_target_phrase,
                domain_hints=domain_hints,
                question_text=question_text,
            )
        else:
            dynamic = self._probe_dynamic_relation_candidates(
                entities=entities,
                answer_target_phrase=answer_target_phrase,
                domain_hints=domain_hints,
                question_text=question_text,
            )
        dynamic = self._normalize_grounded_relation_candidates(
            relation_candidates=dynamic,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            entities=entities,
        )
        if dynamic:
            self._emit_generated_tools_event(
                {
                    "event": "pal_grounding_dynamic_hit",
                    "mode": "pal",
                    "dynamic_count": len(dynamic),
                    "relations": [c.get("relation") for c in dynamic],
                    "normalized_candidates": dynamic,
                }
            )
        else:
            self._emit_generated_tools_event(
                {
                    "event": "pal_grounding_dynamic_miss",
                    "mode": "pal",
                    "reason": "probe_returned_no_usable_predicates",
                }
            )
        if type_dynamic:
            dynamic = self._merge_relation_grounding_candidates(
                dynamic,
                type_dynamic,
                prefer_extra=True,
            )
        return self._prune_redundant_grounded_relation_candidates(
            relation_candidates=dynamic,
            question_text=question_text,
            answer_target_phrase=answer_target_phrase,
        )

    def _probe_answer_class_dynamic_candidates(
        self,
        *,
        task_question: str,
        entities: Sequence[str],
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        interpreted_inputs: Sequence[Mapping[str, Any]],
        query_shape: str,
        answer_mode: str,
    ) -> list[dict[str, str]]:
        if (
            answer_mode != "count"
            or len(entities) > 1
            or not any(
                self._normalize_question_input_role(item.get("role_hint")) == "type_set"
                for item in interpreted_inputs
            )
        ):
            return []
        type_surfaces: list[str] = []
        for item in interpreted_inputs:
            if not isinstance(item, Mapping):
                continue
            if self._normalize_question_input_role(item.get("role_hint")) != "type_set":
                continue
            surface = self._clean_question_input_surface(item.get("surface"))
            if not surface:
                continue
            normalized_surface = self._singularize_phrase(surface)
            if normalized_surface and normalized_surface not in type_surfaces:
                type_surfaces.append(normalized_surface)
        if not type_surfaces:
            return []
        type_dynamic = self._probe_dynamic_relation_candidates_for_anchors(
            anchored_entities=[
                {
                    "surface": surface,
                    "chosen_alias": surface,
                    "role": "type_set",
                }
                for surface in type_surfaces[:1]
            ],
            answer_target_phrase=answer_target_phrase,
            domain_hints=domain_hints,
            question_text=self._split_task_question(task_question)[0],
            max_anchors=1,
        )
        type_dynamic = self._normalize_grounded_relation_candidates(
            relation_candidates=type_dynamic,
            query_shape=query_shape,
            answer_mode=answer_mode,
            answer_target_phrase=answer_target_phrase,
            entities=list(entities) + type_surfaces[:1],
        )
        if type_dynamic:
            self._emit_generated_tools_event(
                {
                    "event": "pal_grounding_answer_class_dynamic_hit",
                    "mode": "pal",
                    "type_surface": type_surfaces[0],
                    "dynamic_count": len(type_dynamic),
                    "relations": [c.get("relation") for c in type_dynamic],
                }
            )
        return type_dynamic

    # ------------------------------------------------------------------
    # Anchor existence probes (used by the repair loop)
    # ------------------------------------------------------------------

    def _probe_entity_name_count(
        self,
        anchor_name: str,
        *,
        timeout_s: float = 2.5,
    ) -> int:
        """Return the number of KG entities whose English name matches *anchor_name*.

        Returns:
            count ≥ 1  — entity found
            0          — entity not found
            -1         — probe timed out or endpoint unavailable
        """
        if not _SPARQL_PROBE_AVAILABLE or not anchor_name.strip():
            return -1
        anchor_match = self._build_probe_anchor_match_block(
            anchor_var="?x",
            label_var="?anchor_label",
            anchor_name=anchor_name,
        )
        sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(?x) AS ?count) WHERE {\n"
            f"{anchor_match}\n"
            "} LIMIT 1"
        )
        values = self._run_probe_sparql_query(
            endpoint=self._get_runtime_sparql_endpoint(),
            sparql=sparql,
            timeout_s=timeout_s,
        )
        for raw in values:
            try:
                return int(float(raw))
            except (ValueError, TypeError):
                pass
        # _run_probe_sparql_query returns [] on timeout/error
        return -1 if not values else 0

    def _build_probe_anchor_match_block(
        self,
        *,
        anchor_var: str,
        label_var: str,
        anchor_name: str,
    ) -> str:
        normalized_name = str(anchor_name or "").strip()
        if not normalized_name:
            return ""
        if re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", normalized_name):
            return f"  VALUES {anchor_var} {{ fb:{normalized_name} }}"
        normalized_literal = json.dumps(normalized_name.lower())
        return (
            "  {\n"
            f"    {anchor_var} fb:type.object.name {label_var} .\n"
            f"    FILTER(LCASE(STR({label_var})) = {normalized_literal})\n"
            "  }\n"
            "  UNION\n"
            "  {\n"
            f"    {anchor_var} fb:common.topic.alias {label_var} .\n"
            f"    FILTER(LCASE(STR({label_var})) = {normalized_literal})\n"
            "  }"
        )

    def _anchor_role_matches_probe_endpoint(
        self,
        *,
        anchor_role: str,
        endpoint_role: str,
    ) -> bool:
        if not anchor_role or not endpoint_role:
            return False
        if anchor_role == endpoint_role:
            return True
        if anchor_role in {"anchor_a", "anchor_b"} and endpoint_role == "anchor":
            return True
        if anchor_role == "anchor" and endpoint_role in {"anchor_a", "anchor_b"}:
            return True
        return False

    def _is_probe_answer_role(self, raw_role: Any) -> bool:
        return self._normalize_relation_role(raw_role) in {
            "answer",
            "shared_answer",
            "candidate_set",
            "count_set",
        }

    def _is_probe_constraint_role(self, raw_role: Any) -> bool:
        return self._normalize_relation_role(raw_role) in {
            "constraint_value",
            "anchor_value",
        }

    def _resolve_anchor_probe_target(
        self,
        *,
        anchored_entity: Mapping[str, Any],
        query_plan: Mapping[str, Any],
        relation_paths: Sequence[Mapping[str, Any]],
        used_path_indexes: set[int],
    ) -> tuple[str | None, str | None]:
        anchor_alias = str(
            anchored_entity.get("chosen_alias") or anchored_entity.get("surface") or ""
        ).strip()
        anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
        anchor_alias_token = self._normalize_variable_token(anchor_alias)

        for path_index, relation_path in enumerate(relation_paths):
            relation = str(relation_path.get("relation") or "").strip()
            if not relation or path_index in used_path_indexes:
                continue
            direction = str(relation_path.get("direction") or "forward").strip().lower()
            from_role = self._normalize_relation_role(
                relation_path.get("from_role") or relation_path.get("from")
            )
            to_role = self._normalize_relation_role(
                relation_path.get("to_role") or relation_path.get("to")
            )
            from_token = self._normalize_variable_token(relation_path.get("from"))
            to_token = self._normalize_variable_token(relation_path.get("to"))

            if self._anchor_role_matches_probe_endpoint(
                anchor_role=anchor_role,
                endpoint_role=from_role,
            ) or (anchor_alias_token and anchor_alias_token == from_token):
                used_path_indexes.add(path_index)
                return relation, self._anchor_position_for_relation_endpoint(
                    endpoint_side="from",
                    direction=direction,
                    endpoint_role=from_role,
                    counterpart_role=to_role,
                )
            if self._anchor_role_matches_probe_endpoint(
                anchor_role=anchor_role,
                endpoint_role=to_role,
            ) or (anchor_alias_token and anchor_alias_token == to_token):
                used_path_indexes.add(path_index)
                return relation, self._anchor_position_for_relation_endpoint(
                    endpoint_side="to",
                    direction=direction,
                    endpoint_role=to_role,
                    counterpart_role=from_role,
                )

        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None, None

        note_text = ""
        for constraint in ((query_plan.get("join_structure") or {}).get("anchor_constraints") or []):
            if not isinstance(constraint, Mapping):
                continue
            if self._normalize_relation_role(constraint.get("anchor_role")) != anchor_role:
                continue
            note_text = str(constraint.get("notes") or "").strip().lower()
            if note_text:
                break

        constraint_bridge_paths: list[tuple[int, Mapping[str, Any], str]] = []
        for path_index, relation_path in enumerate(relation_paths):
            relation = str(relation_path.get("relation") or "").strip()
            if not relation or (not note_text and path_index in used_path_indexes):
                continue
            from_role = self._normalize_relation_role(
                relation_path.get("from_role") or relation_path.get("from")
            )
            to_role = self._normalize_relation_role(
                relation_path.get("to_role") or relation_path.get("to")
            )
            if self._is_probe_constraint_role(from_role) and self._is_probe_answer_role(
                to_role
            ):
                constraint_bridge_paths.append((path_index, relation_path, "subject"))
            elif self._is_probe_answer_role(from_role) and self._is_probe_constraint_role(
                to_role
            ):
                constraint_bridge_paths.append((path_index, relation_path, "object"))

        if not constraint_bridge_paths:
            return None, None

        preferred_index = 0
        if note_text:
            scored_paths: list[tuple[int, int]] = []
            for ranked_index, (_, relation_path, _) in enumerate(constraint_bridge_paths):
                relation = str(relation_path.get("relation") or "").strip().lower()
                relation_leaf = relation.split(".")[-1] if relation else ""
                from_token = self._normalize_variable_token(relation_path.get("from"))
                to_token = self._normalize_variable_token(relation_path.get("to"))
                score = 0
                for cue in (relation_leaf, from_token, to_token):
                    if cue and cue in note_text:
                        score += 1
                scored_paths.append((score, ranked_index))
            scored_paths.sort(reverse=True)
            if scored_paths and scored_paths[0][0] > 0:
                preferred_index = scored_paths[0][1]

        if preferred_index >= len(constraint_bridge_paths):
            preferred_index = 0

        path_index, relation_path, anchor_position = constraint_bridge_paths[
            preferred_index
        ]
        if not note_text:
            used_path_indexes.add(path_index)
        relation = str(relation_path.get("relation") or "").strip()
        if not relation:
            return None, None
        return relation, anchor_position

    def _probe_anchor_path_count(
        self,
        anchor_name: str,
        relation: str,
        *,
        anchor_position: str = "subject",
        timeout_s: float = 2.5,
    ) -> int:
        """Return the number of answers reachable from *anchor_name* via *relation*.

        Returns:
            count ≥ 1  — path produces results
            0          — path is empty
            -1         — probe failed / timed out
        """
        if not _SPARQL_PROBE_AVAILABLE or not anchor_name.strip() or not relation.strip():
            return -1
        normalized_anchor_position = str(anchor_position or "subject").strip().lower()
        if normalized_anchor_position == "object":
            triple_pattern = f"  ?answer fb:{relation} ?anchor .\n"
        else:
            triple_pattern = f"  ?anchor fb:{relation} ?answer .\n"
        anchor_match = self._build_probe_anchor_match_block(
            anchor_var="?anchor",
            label_var="?anchor_label",
            anchor_name=anchor_name,
        )
        sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?answer) AS ?count) WHERE {\n"
            f"{anchor_match}\n"
            f"{triple_pattern}"
            "} LIMIT 1"
        )
        values = self._run_probe_sparql_query(
            endpoint=self._get_runtime_sparql_endpoint(),
            sparql=sparql,
            timeout_s=timeout_s,
        )
        for raw in values:
            try:
                return int(float(raw))
            except (ValueError, TypeError):
                pass
        return -1 if not values else 0

    def _normalize_probe_entity_id(self, raw_value: Any) -> str:
        text = str(raw_value or "").strip()
        if not text:
            return ""
        if text.startswith(self._FB_NS):
            text = text[len(self._FB_NS):]
        if text.startswith("fb:"):
            text = text[3:]
        if text.startswith("/m/"):
            text = "m." + text[3:]
        elif text.startswith("/g/"):
            text = "g." + text[3:]
        return text if re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", text) else ""

    def _probe_anchor_entity_ids(
        self,
        *,
        anchor_name: str,
        relation: str | None = None,
        anchor_position: str = "subject",
        timeout_s: float = 2.5,
    ) -> list[str]:
        if not _SPARQL_PROBE_AVAILABLE or not anchor_name.strip():
            return []
        anchor_match = self._build_probe_anchor_match_block(
            anchor_var="?anchor",
            label_var="?anchor_label",
            anchor_name=anchor_name,
        )
        triple_pattern = ""
        normalized_anchor_position = str(anchor_position or "subject").strip().lower()
        if relation:
            if normalized_anchor_position == "object":
                triple_pattern = f"  ?answer fb:{relation} ?anchor .\n"
            else:
                triple_pattern = f"  ?anchor fb:{relation} ?answer .\n"
        sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT ?anchor WHERE {\n"
            f"{anchor_match}\n"
            f"{triple_pattern}"
            "} LIMIT 5"
        )
        values = self._run_probe_sparql_query(
            endpoint=self._get_runtime_sparql_endpoint(),
            sparql=sparql,
            timeout_s=timeout_s,
        )
        normalized_ids: list[str] = []
        for raw in values:
            entity_id = self._normalize_probe_entity_id(raw)
            if entity_id and entity_id not in normalized_ids:
                normalized_ids.append(entity_id)
        return normalized_ids

    def _resolve_anchor_count_chain_probe(
        self,
        *,
        query_plan: Mapping[str, Any],
        anchored_entity: Mapping[str, Any],
        relation_paths: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str, tuple[dict[str, str], dict[str, str]]] | None:
        if str(query_plan.get("answer_mode") or "").strip().lower() != "count":
            return None
        if str(query_plan.get("query_shape") or "").strip().lower() != "count_over_direct_relation":
            return None
        if len(relation_paths) < 2:
            return None

        anchor_role = self._normalize_relation_role(anchored_entity.get("role"))
        count_set_token = self._normalize_variable_token(
            query_plan.get("count_set_variable")
        )
        if anchor_role not in {"anchor", "anchor_a", "anchor_b"}:
            return None

        first_path_candidates: list[tuple[dict[str, str], str, str]] = []
        for relation_path in relation_paths:
            if not isinstance(relation_path, Mapping):
                continue
            normalized_path = dict(relation_path)
            from_role = self._normalize_relation_role(normalized_path.get("from_role"))
            to_role = self._normalize_relation_role(normalized_path.get("to_role"))
            from_token = self._normalize_variable_token(normalized_path.get("from"))
            to_token = self._normalize_variable_token(normalized_path.get("to"))
            if from_role == anchor_role:
                first_path_candidates.append((normalized_path, "subject", to_token))
            elif to_role == anchor_role:
                first_path_candidates.append((normalized_path, "object", from_token))

        for first_path, anchor_position, intermediate_token in first_path_candidates:
            if not intermediate_token:
                continue
            for second_path in relation_paths:
                if not isinstance(second_path, Mapping):
                    continue
                if dict(second_path) == first_path:
                    continue
                normalized_second = dict(second_path)
                second_from_token = self._normalize_variable_token(
                    normalized_second.get("from")
                )
                second_to_token = self._normalize_variable_token(
                    normalized_second.get("to")
                )
                second_from_role = self._normalize_relation_role(
                    normalized_second.get("from_role")
                )
                second_to_role = self._normalize_relation_role(
                    normalized_second.get("to_role")
                )
                if (
                    second_from_token == intermediate_token
                    and (
                        second_to_role == "count_set"
                        or (count_set_token and second_to_token == count_set_token)
                    )
                ):
                    relation_label = (
                        f"{first_path.get('relation')} -> {normalized_second.get('relation')}"
                    )
                    return relation_label, anchor_position, (first_path, normalized_second)
                if (
                    second_to_token == intermediate_token
                    and (
                        second_from_role == "count_set"
                        or (count_set_token and second_from_token == count_set_token)
                    )
                ):
                    relation_label = (
                        f"{first_path.get('relation')} -> {normalized_second.get('relation')}"
                    )
                    return relation_label, anchor_position, (first_path, normalized_second)
        return None

    def _probe_anchor_relation_chain_count(
        self,
        *,
        anchor_name: str,
        relation_paths: tuple[Mapping[str, Any], Mapping[str, Any]],
        timeout_s: float = 2.5,
    ) -> int:
        if not _SPARQL_PROBE_AVAILABLE or not anchor_name.strip():
            return -1
        first_path, second_path = relation_paths
        first_relation = str(first_path.get("relation") or "").strip()
        second_relation = str(second_path.get("relation") or "").strip()
        if not first_relation or not second_relation:
            return -1

        first_from_role = self._normalize_relation_role(first_path.get("from_role"))
        first_to_role = self._normalize_relation_role(first_path.get("to_role"))
        first_from_token = self._normalize_variable_token(first_path.get("from"))
        first_to_token = self._normalize_variable_token(first_path.get("to"))
        if first_from_role in {"anchor", "anchor_a", "anchor_b"}:
            first_pattern = f"  ?anchor fb:{first_relation} ?mid .\n"
            intermediate_token = first_to_token
        elif first_to_role in {"anchor", "anchor_a", "anchor_b"}:
            first_pattern = f"  ?mid fb:{first_relation} ?anchor .\n"
            intermediate_token = first_from_token
        else:
            return -1

        second_from_token = self._normalize_variable_token(second_path.get("from"))
        second_to_token = self._normalize_variable_token(second_path.get("to"))
        if second_from_token == intermediate_token:
            second_pattern = f"  ?mid fb:{second_relation} ?answer .\n"
        elif second_to_token == intermediate_token:
            second_pattern = f"  ?answer fb:{second_relation} ?mid .\n"
        else:
            return -1
        anchor_match = self._build_probe_anchor_match_block(
            anchor_var="?anchor",
            label_var="?anchor_label",
            anchor_name=anchor_name,
        )

        sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?answer) AS ?count) WHERE {\n"
            f"{anchor_match}\n"
            f"{first_pattern}"
            f"{second_pattern}"
            "} LIMIT 1"
        )
        values = self._run_probe_sparql_query(
            endpoint=self._get_runtime_sparql_endpoint(),
            sparql=sparql,
            timeout_s=timeout_s,
        )
        for raw in values:
            try:
                return int(float(raw))
            except (ValueError, TypeError):
                pass
        return -1 if not values else 0

    def _probe_anchor_relation_chain_entity_ids(
        self,
        *,
        anchor_name: str,
        relation_paths: tuple[Mapping[str, Any], Mapping[str, Any]],
        timeout_s: float = 2.5,
    ) -> list[str]:
        if not _SPARQL_PROBE_AVAILABLE or not anchor_name.strip():
            return []
        first_path, second_path = relation_paths
        first_relation = str(first_path.get("relation") or "").strip()
        second_relation = str(second_path.get("relation") or "").strip()
        if not first_relation or not second_relation:
            return []

        first_from_role = self._normalize_relation_role(first_path.get("from_role"))
        first_to_role = self._normalize_relation_role(first_path.get("to_role"))
        first_from_token = self._normalize_variable_token(first_path.get("from"))
        first_to_token = self._normalize_variable_token(first_path.get("to"))
        if first_from_role in {"anchor", "anchor_a", "anchor_b"}:
            first_pattern = f"  ?anchor fb:{first_relation} ?mid .\n"
            intermediate_token = first_to_token
        elif first_to_role in {"anchor", "anchor_a", "anchor_b"}:
            first_pattern = f"  ?mid fb:{first_relation} ?anchor .\n"
            intermediate_token = first_from_token
        else:
            return []

        second_from_token = self._normalize_variable_token(second_path.get("from"))
        second_to_token = self._normalize_variable_token(second_path.get("to"))
        if second_from_token == intermediate_token:
            second_pattern = f"  ?mid fb:{second_relation} ?answer .\n"
        elif second_to_token == intermediate_token:
            second_pattern = f"  ?answer fb:{second_relation} ?mid .\n"
        else:
            return []

        anchor_match = self._build_probe_anchor_match_block(
            anchor_var="?anchor",
            label_var="?anchor_label",
            anchor_name=anchor_name,
        )
        sparql = (
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT ?anchor WHERE {\n"
            f"{anchor_match}\n"
            f"{first_pattern}"
            f"{second_pattern}"
            "} LIMIT 5"
        )
        values = self._run_probe_sparql_query(
            endpoint=self._get_runtime_sparql_endpoint(),
            sparql=sparql,
            timeout_s=timeout_s,
        )
        normalized_ids: list[str] = []
        for raw in values:
            entity_id = self._normalize_probe_entity_id(raw)
            if entity_id and entity_id not in normalized_ids:
                normalized_ids.append(entity_id)
        return normalized_ids

    def _run_anchor_existence_probes(
        self,
        *,
        query_plan: Mapping[str, Any],
        probe_paths: bool = False,
        timeout_s: float = 2.5,
    ) -> list[AnchorProbeResult]:
        """Probe KG existence for each anchor alias in the query plan.

        Runs entity-existence probes in parallel (up to 2 anchors).
        Path probes are run sequentially afterwards only when *probe_paths*
        is True and the anchor entity was found.

        Returns a list of AnchorProbeResult, one per probed anchor.
        """
        anchored_entities: list[Mapping[str, Any]] = [
            a for a in (query_plan.get("anchored_entities") or [])
            if isinstance(a, Mapping)
        ]
        if not anchored_entities:
            return []

        relation_paths: list[Mapping[str, Any]] = [
            rp
            for rp in (query_plan.get("relation_paths") or [])
            if isinstance(rp, Mapping)
        ]

        # Limit to first 2 anchors to keep probe cost bounded.
        probe_anchors = anchored_entities[:2]
        aliases: list[str] = []
        probe_labels: list[str] = []
        preset_resolved_ids: list[str | None] = []
        for anchored_entity in probe_anchors:
            chosen_alias = str(
                anchored_entity.get("chosen_alias") or anchored_entity.get("surface") or ""
            ).strip()
            surface = str(anchored_entity.get("surface") or "").strip()
            alias_is_mid = bool(re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", chosen_alias))
            resolved_entity_id = str(
                anchored_entity.get("resolved_entity_id") or ""
            ).strip()
            if not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", resolved_entity_id):
                resolved_entity_id = ""
            aliases.append(chosen_alias)
            probe_labels.append(surface if alias_is_mid and surface else chosen_alias)
            preset_resolved_ids.append(
                chosen_alias if alias_is_mid else (resolved_entity_id or None)
            )
        roles: list[str] = [
            self._normalize_relation_role(a.get("role")) for a in probe_anchors
        ]

        # Run entity-existence probes in parallel.
        entity_counts: list[int] = []
        with ThreadPoolExecutor(max_workers=max(1, len(aliases))) as executor:
            futures = [
                executor.submit(self._probe_entity_name_count, probe_label, timeout_s=timeout_s)
                for probe_label, preset_resolved_id in zip(probe_labels, preset_resolved_ids)
                if probe_label and not preset_resolved_id
            ]
            future_results: list[int] = []
            for future in futures:
                try:
                    future_results.append(future.result(timeout=timeout_s + 1.0))
                except Exception:
                    future_results.append(-1)
        future_idx = 0
        for probe_label, preset_resolved_id in zip(probe_labels, preset_resolved_ids):
            if preset_resolved_id:
                entity_counts.append(1)
            elif probe_label:
                entity_counts.append(future_results[future_idx])
                future_idx += 1
            else:
                entity_counts.append(-1)

        results: list[AnchorProbeResult] = []
        used_path_indexes: set[int] = set()
        for anchored_entity, alias, probe_label, preset_resolved_id, role, entity_count in zip(
            probe_anchors,
            aliases,
            probe_labels,
            preset_resolved_ids,
            roles,
            entity_counts,
        ):
            if not (alias or probe_label):
                continue
            relation = None
            anchor_position = None
            path_count: int | None = None
            resolved_entity_id: str | None = preset_resolved_id
            probe_binding = preset_resolved_id or probe_label or alias

            if probe_paths and entity_count > 0:
                chain_probe = self._resolve_anchor_count_chain_probe(
                    query_plan=query_plan,
                    anchored_entity=anchored_entity,
                    relation_paths=relation_paths,
                )
                if chain_probe is not None:
                    relation, anchor_position, chain_paths = chain_probe
                    path_count = self._probe_anchor_relation_chain_count(
                        anchor_name=probe_binding,
                        relation_paths=chain_paths,
                        timeout_s=timeout_s,
                    )
                    if path_count and path_count > 0:
                        resolved_ids = self._probe_anchor_relation_chain_entity_ids(
                            anchor_name=probe_binding,
                            relation_paths=chain_paths,
                            timeout_s=timeout_s,
                        )
                        if len(resolved_ids) == 1:
                            resolved_entity_id = resolved_ids[0]
                else:
                    relation, anchor_position = self._resolve_anchor_probe_target(
                        anchored_entity=anchored_entity,
                        query_plan=query_plan,
                        relation_paths=relation_paths,
                        used_path_indexes=used_path_indexes,
                    )
                    if relation:
                        path_count = self._probe_anchor_path_count(
                            probe_binding,
                            relation,
                            anchor_position=anchor_position or "subject",
                            timeout_s=timeout_s,
                        )
                        if path_count and path_count > 0:
                            resolved_ids = self._probe_anchor_entity_ids(
                                anchor_name=probe_binding,
                                relation=relation,
                                anchor_position=anchor_position or "subject",
                                timeout_s=timeout_s,
                            )
                            if len(resolved_ids) == 1:
                                resolved_entity_id = resolved_ids[0]
            if resolved_entity_id is None and entity_count == 1:
                resolved_ids = self._probe_anchor_entity_ids(
                    anchor_name=probe_binding,
                    timeout_s=timeout_s,
                )
                if len(resolved_ids) == 1:
                    resolved_entity_id = resolved_ids[0]

            results.append(
                AnchorProbeResult(
                    anchor_name=probe_label or alias,
                    entity_count=entity_count,
                    path_count=path_count,
                    relation_probed=relation if probe_paths else None,
                    anchor_position=anchor_position if probe_paths else None,
                    resolved_entity_id=resolved_entity_id,
                )
            )

        self._emit_generated_tools_event(
            {
                "event": "pal_anchor_probe_results",
                "mode": "pal",
                "probes": [
                    {
                        "anchor_name": r.anchor_name,
                        "entity_count": r.entity_count,
                        "found": r.found,
                        "path_count": r.path_count,
                        "relation_probed": r.relation_probed,
                        "anchor_position": r.anchor_position,
                        "resolved_entity_id": r.resolved_entity_id,
                    }
                    for r in results
                ],
            }
        )
        return results

    # ------------------------------------------------------------------
    # Dynamic predicate probe
    # ------------------------------------------------------------------

    #: Freebase namespace prefix stripped when normalising predicate IRIs.
    _FB_NS = "http://rdf.freebase.com/ns/"

    #: IRI prefixes that identify schema/metadata predicates to discard.
    _NOISE_PREDICATE_PREFIXES: tuple[str, ...] = (
        "http://rdf.freebase.com/ns/type.object",
        "http://rdf.freebase.com/ns/common.topic",
        "http://rdf.freebase.com/ns/kg.object",
        "http://rdf.freebase.com/key/",
        "http://rdf.freebase.com/ns/user.",
        "http://rdf.freebase.com/ns/base.",
        "http://www.w3.org/",
        "http://rdf.freebase.com/ns/dataworld.",
        "http://rdf.freebase.com/ns/community.",
        "http://rdf.freebase.com/ns/pipeline.",
        "http://rdf.freebase.com/ns/fbase.",
    )

    #: Freebase short-name suffixes that identify schema/metadata predicates.
    _NOISE_PREDICATE_SUFFIXES: tuple[str, ...] = (
        ".name",
        ".alias",
        ".description",
        ".image",
        ".mid",
        ".key",
        ".permission",
        ".notable_for",
        ".notable_types",
        ".guid",
        ".timestamp",
        ".attribution",
        ".text",
        ".topic_equivalent_webpage",
    )

    def _probe_dynamic_relation_candidates(
        self,
        *,
        entities: Sequence[str],
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str = "",
        probe_timeout_s: float = 5.0,
    ) -> list[dict[str, str]]:
        """Probe the live SPARQL endpoint for predicates around the first anchor entity.

        Returns an empty list if the endpoint is unavailable, the probe times out,
        or no usable predicates are found after filtering.
        """
        if not entities or not _SPARQL_PROBE_AVAILABLE:
            return []
        return self._probe_dynamic_relation_candidates_for_anchor(
            anchor_entity=entities[0],
            answer_target_phrase=answer_target_phrase,
            domain_hints=domain_hints,
            question_text=question_text,
            probe_timeout_s=probe_timeout_s,
            anchor_role="anchor",
            anchor_clue=self._infer_entity_clue(question_text, entities[0]),
        )

    def _probe_dynamic_relation_candidates_for_anchors(
        self,
        *,
        anchored_entities: Sequence[Mapping[str, Any]],
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str = "",
        probe_timeout_s: float = 5.0,
        max_anchors: int = 2,
    ) -> list[dict[str, str]]:
        merged_candidates: list[dict[str, str]] = []
        for anchored_entity in anchored_entities[:max_anchors]:
            if not isinstance(anchored_entity, Mapping):
                continue
            anchor_entity = str(
                anchored_entity.get("chosen_alias")
                or anchored_entity.get("surface")
                or ""
            ).strip()
            if not anchor_entity:
                continue
            anchor_role = self._normalize_relation_role(
                anchored_entity.get("role")
            ) or "anchor"
            anchor_candidates = self._probe_dynamic_relation_candidates_for_anchor(
                anchor_entity=anchor_entity,
                answer_target_phrase=answer_target_phrase,
                domain_hints=domain_hints,
                question_text=question_text,
                probe_timeout_s=probe_timeout_s,
                anchor_role=anchor_role,
                anchor_clue=self._infer_entity_clue(
                    question_text,
                    str(anchored_entity.get("surface") or anchor_entity),
                ),
            )
            merged_candidates = self._merge_relation_grounding_candidates(
                merged_candidates,
                anchor_candidates,
                prefer_extra=False,
            )
        return merged_candidates

    def _probe_dynamic_relation_candidates_for_anchor(
        self,
        *,
        anchor_entity: str,
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str = "",
        probe_timeout_s: float = 5.0,
        anchor_role: str = "anchor",
        anchor_clue: str = "",
    ) -> list[dict[str, str]]:
        if not anchor_entity or not _SPARQL_PROBE_AVAILABLE:
            return []

        alias_candidates = self._build_entity_alias_candidates(anchor_entity)
        # Use the safe (original) alias — index 0, not the singularized variant.
        probe_alias = alias_candidates[0] if alias_candidates else anchor_entity

        self._emit_generated_tools_event(
            {
                "event": "pal_grounding_dynamic_probe_start",
                "mode": "pal",
                "anchor": anchor_entity,
                "probe_alias": probe_alias,
            }
        )

        endpoint = self._get_runtime_sparql_endpoint()
        anchor_match = self._build_probe_anchor_match_block(
            anchor_var="?anchor",
            label_var="?anchor_label",
            anchor_name=probe_alias,
        )

        outgoing_sparql = (
            f"PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            f"SELECT DISTINCT ?p WHERE {{\n"
            f"{anchor_match}\n"
            f"  ?anchor ?p ?o .\n"
            f"}} LIMIT 40"
        )
        incoming_sparql = (
            f"PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            f"SELECT DISTINCT ?p WHERE {{\n"
            f"{anchor_match}\n"
            f"  ?s ?p ?anchor .\n"
            f"}} LIMIT 40"
        )

        outgoing_iris = self._run_probe_sparql_query(
            endpoint=endpoint,
            sparql=outgoing_sparql,
            timeout_s=probe_timeout_s,
        )
        incoming_iris = self._run_probe_sparql_query(
            endpoint=endpoint,
            sparql=incoming_sparql,
            timeout_s=probe_timeout_s,
        )

        self._emit_generated_tools_event(
            {
                "event": "pal_grounding_dynamic_probe_raw",
                "mode": "pal",
                "anchor": anchor_entity,
                "probe_alias": probe_alias,
                "outgoing_raw_count": len(outgoing_iris),
                "incoming_raw_count": len(incoming_iris),
            }
        )

        if not outgoing_iris and not incoming_iris:
            return []

        candidates = self._build_dynamic_candidate_list(
            outgoing_iris=outgoing_iris,
            incoming_iris=incoming_iris,
            anchor_entity=anchor_entity,
            anchor_role=anchor_role,
            answer_target_phrase=answer_target_phrase,
            domain_hints=domain_hints,
            question_text=question_text,
            anchor_clue=anchor_clue,
        )

        self._emit_generated_tools_event(
            {
                "event": "pal_grounding_dynamic_candidates_selected",
                "mode": "pal",
                "anchor": anchor_entity,
                "candidate_count": len(candidates),
                "relations": [c.get("relation") for c in candidates],
            }
        )
        return candidates

    def _run_probe_sparql_query(
        self,
        *,
        endpoint: str,
        sparql: str,
        timeout_s: float = 5.0,
    ) -> list[str]:
        """Execute a probe SPARQL query and return the list of result value strings.

        Returns an empty list on any error (timeout, connection failure, etc.).
        """
        if not _SPARQL_PROBE_AVAILABLE:
            return []

        def _execute() -> list[str]:
            wrapper = _ProbeSPARQLWrapper(endpoint)
            wrapper.setQuery(sparql)
            wrapper.setReturnFormat(_ProbeSPARQLJSON)
            result = wrapper.query().convert()
            bindings = result.get("results", {}).get("bindings", [])
            values: list[str] = []
            for binding in bindings:
                for cell in binding.values():
                    raw_val = str(cell.get("value", "")).strip()
                    if raw_val:
                        values.append(raw_val)
            return values

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute)
                return future.result(timeout=timeout_s)
        except Exception:
            return []

    def _build_dynamic_candidate_list(
        self,
        *,
        outgoing_iris: list[str],
        incoming_iris: list[str],
        anchor_entity: str,
        anchor_role: str,
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str,
        anchor_clue: str = "",
        max_per_direction: int = 8,
        max_total: int = 12,
    ) -> list[dict[str, str]]:
        """Filter, rank, and cap probed predicate IRIs into grounding candidates."""
        seen: set[str] = set()

        def _clean(iris: list[str]) -> list[str]:
            cleaned: list[str] = []
            for iri in iris:
                if self._is_noise_predicate(iri):
                    continue
                short = self._normalize_predicate_iri(iri)
                if short and short not in seen:
                    seen.add(short)
                    cleaned.append(short)
            return cleaned

        outgoing_clean = _clean(outgoing_iris)
        incoming_clean = _clean(incoming_iris)

        def _rank(names: list[str]) -> list[str]:
            return sorted(
                names,
                key=lambda s: self._score_probe_predicate(
                    s,
                    answer_target_phrase,
                    domain_hints,
                    question_text,
                    anchor_clue=anchor_clue,
                ),
                reverse=True,
            )[:max_per_direction]

        candidates: list[dict[str, str]] = []
        for short_name in _rank(outgoing_clean):
            _, target_label = self._infer_probe_relation_endpoint_labels(short_name)
            candidates.append(
                {
                    "relation": short_name,
                    "direction": "forward",
                    "from": anchor_entity,
                    "to": target_label or "?target",
                    "from_role": anchor_role,
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                    "support": "dynamic_probe_outgoing",
                    "use_when": (
                        f"traverse outgoing {short_name} from the anchor entity"
                        + (f" to reach {target_label}" if target_label else "")
                    ),
                }
            )
        for short_name in _rank(incoming_clean):
            source_label, _ = self._infer_probe_relation_endpoint_labels(short_name)
            candidates.append(
                {
                    "relation": short_name,
                    "direction": "reverse",
                    "from": source_label or "?source",
                    "to": anchor_entity,
                    "from_role": "candidate_set",
                    "to_role": anchor_role,
                    "grounding_source": "dynamic_probe",
                    "support": "dynamic_probe_incoming",
                    "use_when": (
                        f"find entities that link to the anchor via {short_name}"
                        + (f" from {source_label}" if source_label else "")
                    ),
                }
            )
        return candidates[:max_total]

    def _infer_probe_relation_endpoint_labels(
        self,
        short_name: str,
    ) -> tuple[str, str]:
        segments = [segment for segment in str(short_name or "").split(".") if segment]
        relation_tail = segments[-1] if segments else ""
        source_segment = segments[-2] if len(segments) >= 2 else ""
        if "_of_" in relation_tail:
            left, right = relation_tail.split("_of_", 1)
            return (
                self._semantic_label_from_relation_phrase(left),
                self._semantic_label_from_relation_phrase(right),
            )
        return (
            self._semantic_label_from_relation_phrase(source_segment),
            self._semantic_label_from_relation_phrase(relation_tail),
        )

    def _semantic_label_from_relation_phrase(
        self,
        phrase: str,
    ) -> str:
        raw_tokens = [
            token
            for token in str(phrase or "").strip().lower().split("_")
            if token
        ]
        if not raw_tokens:
            return ""
        stopwords = {
            "of",
            "the",
            "this",
            "these",
            "those",
            "in",
            "on",
            "for",
            "to",
            "by",
            "with",
            "from",
        }
        tokens = [token for token in raw_tokens if token not in stopwords]
        if not tokens:
            tokens = raw_tokens

        if tokens[0] in {"active", "dosage", "parent", "child"} and len(tokens) >= 2:
            return f"{tokens[0]}_{self._singularize_surface_token(tokens[1])}"
        if tokens[0] in {"fictional", "educational"} and len(tokens) >= 2:
            return f"{tokens[0]}_{self._singularize_surface_token(tokens[1])}"
        if tokens[0] == "drug" and len(tokens) >= 2 and tokens[1] in {
            "ingredient",
            "formulation",
            "class",
            "brand",
        }:
            return f"drug_{self._singularize_surface_token(tokens[1])}"

        return self._singularize_surface_token(tokens[-1])

    def _is_noise_predicate(self, iri: str) -> bool:
        """Return True for schema/metadata predicates that should not be offered to the planner."""
        for prefix in self._NOISE_PREDICATE_PREFIXES:
            if iri.startswith(prefix):
                return True
        # After stripping the FB namespace, check short-name suffixes.
        short = self._normalize_predicate_iri(iri)
        if short is None:
            # Not a Freebase predicate — discard.
            return True
        for suffix in self._NOISE_PREDICATE_SUFFIXES:
            if short.endswith(suffix):
                return True
        return False

    def _normalize_predicate_iri(self, iri: str) -> Optional[str]:
        """Strip the Freebase namespace prefix and return the short predicate name."""
        if iri.startswith(self._FB_NS):
            return iri[len(self._FB_NS):]
        return None

    def _score_probe_predicate(
        self,
        short_name: str,
        answer_target_phrase: str,
        domain_hints: Sequence[str],
        question_text: str,
        *,
        anchor_clue: str = "",
    ) -> int:
        """Return a relevance score for ranking probed predicates (higher = better)."""
        score = 0
        name_lower = short_name.lower()
        relation_tail = short_name.split(".")[-1].lower()
        source_label, target_label = self._infer_probe_relation_endpoint_labels(short_name)
        for word in re.split(r"[\s_.\-]+", (answer_target_phrase or "").lower()):
            if word and len(word) > 2 and word in name_lower:
                score += 100
            if word and len(word) > 2 and word == relation_tail:
                score += 40
        for hint in domain_hints:
            for word in re.split(r"[\s_.\-]+", hint.lower()):
                if word and len(word) > 2 and word in name_lower:
                    score += 50
        for word in re.split(r"[\s_.\-?]+", (question_text or "").lower()):
            if word and len(word) > 2 and word in name_lower:
                score += 20
            if word and len(word) > 2 and word == relation_tail:
                score += 10
            if word and len(word) > 2 and word == target_label:
                score += 12
            if word and len(word) > 2 and word == source_label:
                score += 8
        question_lower = str(question_text or "").lower()
        clue_keywords: dict[str, tuple[str, ...]] = {
            "active_ingredient": (
                "active_ingredient_of_formulation",
                "active_moiety_of_formulation",
                "active_ingredients",
                "active_ingredient_moieties",
                "ingredient",
                "moiety",
            ),
            "formulation_input": (
                "marketed_formulations",
                "formulation_of",
                "formulation",
            ),
            "source_constraint": (
                "source",
                "milk_source",
                "parent",
                "owner",
                "producer",
            ),
            "origin_constraint": (
                "origin",
                "country_of_origin",
                "originating_here",
                "breeds_originating_here",
                "breed_origin",
            ),
        }
        for keyword in clue_keywords.get(anchor_clue, ()):
            if keyword in name_lower:
                score += 70 if keyword in relation_tail else 35
        if anchor_clue == "origin_constraint":
            if target_label == "breed":
                score += 55
            if source_label == "breed":
                score += 30
            if any(token in relation_tail for token in ("origin", "originating_here")):
                score += 45
        if "profession" in question_lower:
            if "people_with_this_profession" in name_lower:
                score += 90
            if relation_tail == "profession":
                score += 35
            if "instrument" in name_lower:
                score -= 30
        if anchor_clue == "formulation_input" and target_label == "formulation":
            score += 45
        if anchor_clue == "active_ingredient" and target_label == "formulation":
            score += 55
        if anchor_clue.startswith("attribute_value.feature"):
            if any(
                token in name_lower
                for token in ("feature", "technique", "techniques", "strike", "attack", "move")
            ):
                score += 65 if any(
                    token in relation_tail
                    for token in ("feature", "technique", "techniques", "strike", "attack", "move")
                ) else 35
            if any(
                token in name_lower
                for token in ("category", "categories", "type", "instance")
            ):
                score -= 70
        if "dosage form" in str(question_text or "").lower() and target_label == "formulation":
            score += 45
        if (
            any(token in str(question_text or "").lower() for token in ("fictional world", "fictional setting", " world "))
            and target_label == "universe"
        ):
            score += 35
        if any(
            phrase in question_lower
            for phrase in (
                "preceded by",
                "proceeded by",
                "followed by",
                "succeeded by",
                "came after",
                "came before",
            )
        ):
            if any(
                token in name_lower
                for token in (
                    "preced",
                    "predecess",
                    "succeed",
                    "successor",
                    "follow",
                    "previous",
                    "next",
                )
            ):
                score += 80 if any(
                    token in relation_tail
                    for token in (
                        "preced",
                        "predecess",
                        "succeed",
                        "successor",
                        "follow",
                    )
                ) else 90
            if any(token in relation_tail for token in ("game", "games")):
                score -= 90
        if "engine" in question_lower:
            if "engine" in name_lower:
                score += 30
            if any(token in name_lower for token in ("predecess", "successor", "succeed")):
                score += 20
        if " has " in question_lower or re.search(
            r"\b(?:which|what|who)\b.+\bhas\b", question_lower
        ):
            if any(
                token in name_lower
                for token in ("parent", "containedby", "organization", "institution", "member")
            ):
                score += 35
            if any(token in name_lower for token in ("campus", "geolocation", "gallery", "image")):
                score -= 20
        if re.search(r"\b(?:which|what)\s+institution\s+has\b", question_lower):
            if "parent_institution" in name_lower or "parent_organization" in name_lower:
                score += 60
            if any(token in name_lower for token in ("campus", "campuses")):
                score -= 50
        return score

    def _normalize_grounding_entity(self, entity: str) -> str:
        return str(entity or "").strip()

    def _build_entity_alias_candidates(
        self,
        entity: str,
        *,
        entity_clue: str = "surface_constraint",
    ) -> list[str]:
        raw_entity = str(entity or "").strip()
        if not raw_entity:
            return []
        candidates: list[str] = []
        if entity_clue.startswith("attribute_value."):
            singular_entity = self._singularize_phrase(raw_entity)
            profession_aliases = {
                "educator": ["Teacher"],
                "teacher": ["Teacher"],
            }
            feature_aliases = {
                "bayer": ["Bayer filter"],
            }
            candidate_pool = (
                raw_entity,
                raw_entity.lower(),
                singular_entity,
                singular_entity.lower(),
                raw_entity.title(),
                singular_entity.title(),
            )
            profession_alias_candidates = [
                alias
                for token in {raw_entity.lower(), singular_entity.lower()}
                for alias in profession_aliases.get(token, [])
            ]
            feature_alias_candidates = [
                alias
                for token in {raw_entity.lower(), singular_entity.lower()}
                for alias in feature_aliases.get(token, [])
            ]
            candidate_pool = (
                *candidate_pool,
                *profession_alias_candidates,
                *feature_alias_candidates,
            )
        else:
            candidate_pool = [
                raw_entity,
                raw_entity.title(),
            ]
            lower_entity = raw_entity.lower()
            regional_org_match = re.match(
                r"^([a-z0-9&'.-]+(?: [a-z0-9&'.-]+)?) of "
                r"(japan|america|china|korea|europe|asia|canada|australia|taiwan)$",
                lower_entity,
            )
            if regional_org_match is not None:
                stripped_org = regional_org_match.group(1).strip()
                if stripped_org and stripped_org not in {
                    "bank",
                    "house",
                    "kingdom",
                    "republic",
                    "state",
                    "province",
                    "county",
                    "city",
                    "university",
                }:
                    candidate_pool.extend(
                        [
                            stripped_org,
                            stripped_org.title(),
                        ]
                    )
            for prefix in (
                "republic of ",
                "kingdom of ",
                "state of ",
                "province of ",
                "county of ",
                "city of ",
            ):
                if lower_entity.startswith(prefix):
                    stripped = raw_entity[len(prefix):].strip()
                    if stripped:
                        candidate_pool.extend([stripped.title(), stripped])
            compact_token = re.sub(r"[^A-Za-z0-9]", "", raw_entity)
            if compact_token.isalpha() and compact_token.islower() and len(compact_token) <= 5:
                candidate_pool.append(compact_token.upper())
            hyphen_parts = [part.strip() for part in raw_entity.split("-") if part.strip()]
            if len(hyphen_parts) >= 2:
                candidate_pool.append(" ".join(hyphen_parts))
                candidate_pool.append("".join(hyphen_parts))
                if len(hyphen_parts) == 2:
                    candidate_pool.append(" and ".join(hyphen_parts))
                else:
                    candidate_pool.append("-".join(hyphen_parts[:-1]) + " and " + hyphen_parts[-1])
            irregular_aliases = {
                "cows": ["Cattle"],
            }
            candidate_pool.extend(irregular_aliases.get(raw_entity.lower(), []))
            candidate_pool.append(self._singularize_surface_token(raw_entity))
            tokens = [token for token in raw_entity.split() if token]
            if len(tokens) >= 2 and tokens[-1].lower() in {
                "dog",
                "dogs",
                "cat",
                "cats",
            }:
                stripped_animal_class = " ".join(tokens[:-1]).strip()
                if stripped_animal_class:
                    candidate_pool.extend(
                        [
                            stripped_animal_class,
                            stripped_animal_class.title(),
                        ]
                    )
        for candidate in candidate_pool:
            cleaned_candidate = str(candidate or "").strip()
            if cleaned_candidate and cleaned_candidate not in candidates:
                candidates.append(cleaned_candidate)
        return candidates

    def _build_type_constraint_alias_candidates(
        self,
        phrase: str,
    ) -> list[str]:
        raw_phrase = str(phrase or "").strip()
        if not raw_phrase:
            return []
        singular_phrase = self._singularize_phrase(raw_phrase)
        candidate_pool = [
            singular_phrase,
            raw_phrase,
            singular_phrase.title(),
            raw_phrase.title(),
        ]
        candidates: list[str] = []
        for candidate in candidate_pool:
            cleaned_candidate = str(candidate or "").strip()
            if cleaned_candidate and cleaned_candidate not in candidates:
                candidates.append(cleaned_candidate)
        return candidates

    def _singularize_phrase(self, phrase: str) -> str:
        tokens = [token for token in str(phrase or "").split() if token]
        if not tokens:
            return ""
        tokens[-1] = self._singularize_surface_token(tokens[-1])
        return " ".join(tokens)

    def _should_surface_answer_target_as_type_constraint(
        self,
        *,
        answer_target_phrase: str,
        grounded_relation_candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        phrase = str(answer_target_phrase or "").strip()
        if not phrase:
            return False
        if len(phrase.split()) < 2 and not phrase.lower().endswith("s"):
            return False
        if self._answer_target_is_relation_encoded_type_projection(
            answer_target_phrase=phrase,
            relation_candidates=grounded_relation_candidates,
        ):
            return False
        for candidate in grounded_relation_candidates:
            if not isinstance(candidate, Mapping):
                continue
            relation = str(candidate.get("relation") or "").strip()
            from_role = self._normalize_relation_role(candidate.get("from_role"))
            to_role = self._normalize_relation_role(candidate.get("to_role"))
            if relation == "type.type.instance":
                return True
            if "type" in {from_role, to_role}:
                return True
            if {"type_set", "shared_type"} & {from_role, to_role}:
                return True
        return False

    def _singularize_surface_token(self, token: str) -> str:
        lower_token = token.lower()
        if lower_token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"
        if (
            lower_token.endswith("ses")
            and len(token) > 3
            and len(token) > 3
            and lower_token[-4] in {"a", "e", "i", "o", "u"}
        ):
            return token[:-1]
        if lower_token.endswith("ses") and len(token) > 3:
            return token[:-2]
        if lower_token.endswith("s") and not lower_token.endswith("ss") and len(token) > 1:
            return token[:-1]
        return token

    def _extract_answer_target_phrase(self, question_text: str) -> str:
        normalized_text = re.sub(r"^\s*Question:\s*", "", str(question_text or ""), flags=re.IGNORECASE)
        normalized_text = re.sub(
            r"\b(what|which|who|where|when|how)'s\b",
            lambda match: f"{match.group(1)} is",
            normalized_text,
            flags=re.IGNORECASE,
        )
        lower_text = normalized_text.lower()
        recent_superlative_match = re.search(
            r"^(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?(?:most|least)\s+recently\s+(.+?)(?:\s+(?:of|among|by|for|from|with|that|which|who|whose|in|on|at|to|is|are|was|were)\b|\?|$)",
            lower_text,
        )
        if recent_superlative_match is not None:
            return recent_superlative_match.group(1).strip()
        superlative_leading_match = re.search(
            r"^(?:what|which|who)\s+(.+?)\s+(?:is|are|was|were)\s+(?:the\s+)?(?:longest|shortest|largest|smallest|highest|lowest|earliest|latest|farthest|furthest|closest|nearest|maximum|minimum)\b",
            lower_text,
        )
        if superlative_leading_match is not None:
            return superlative_leading_match.group(1).strip()
        superlative_possessive_match = re.search(
            r"^[^?]+?'s\s+(?:the\s+)?(?:longest|shortest|largest|smallest|highest|lowest|earliest|latest|farthest|furthest|closest|nearest|maximum|minimum)\s+(.+?)(?:\s+(?:of|among|by|for|from|with|that|which|who|whose|in|on|at|to|is|are|was|were)\b|\?|$)",
            lower_text,
        )
        if superlative_possessive_match is not None:
            return superlative_possessive_match.group(1).strip()
        superlative_inline_match = re.search(
            r"\b(?:longest|shortest|largest|smallest|highest|lowest|earliest|latest|farthest|furthest|closest|nearest|maximum|minimum)\s+(.+?)(?:\s+(?:of|among|by|for|from|with|that|which|who|whose|in|on|at|to|is|are|was|were)\b|\?|$)",
            lower_text,
        )
        if superlative_inline_match is not None:
            return superlative_inline_match.group(1).strip()
        number_of_match = re.search(
            r"^(?:what|which)\s+(?:(?:is|was|were)\s+)?the\s+number\s+of\s+(.+?)(?:\s+(?:that|which|who|work|works|worked|have|has|had|can|could|are|is|was|were)\b|\?|$)",
            lower_text,
        )
        if number_of_match is not None:
            return number_of_match.group(1).strip()
        amount_of_match = re.search(
            r"^(?:what|which)\s+(?:(?:is|was|were)\s+)?(?:the\s+)?amount\s+of\s+(.+?)(?:\s+(?:that|which|who|work|works|worked|have|has|had|can|could|are|is|was|were)\b|\?|$)",
            lower_text,
        )
        if amount_of_match is not None:
            return amount_of_match.group(1).strip()
        total_number_of_match = re.search(
            r"^(?:what|which)\s+(?:is\s+)?the\s+total\s+number\s+of\s+(.+?)(?:\s+(?:that|which|who|work|works|worked|play|plays|played|have|has|had|can|could|are|is|was|were)\b|\?|$)",
            lower_text,
        )
        if total_number_of_match is not None:
            return total_number_of_match.group(1).strip()
        type_of_match = re.search(
            r"^(?:what|which)\s+(?:other\s+)?(?:type|types|kind|kinds|category|categories)\s+of\s+(.+?)(?:\s+(?:is|are|was|were|did|does|do|has|have|ran|run|used|uses|use|with|from|for|in|on|that|who|which)\b|\?|$)",
            lower_text,
        )
        if type_of_match is not None:
            return type_of_match.group(1).strip()
        embedded_type_of_match = re.search(
            r"\b(?:what|which)\s+(?:other\s+)?(?:type|types|kind|kinds|category|categories)\s+of\s+(.+?)(?:\s+(?:is|are|was|were|did|does|do|has|have|ran|run|used|uses|use|with|from|for|in|on|that|who|which)\b|\?|$)",
            lower_text,
        )
        if embedded_type_of_match is not None:
            return embedded_type_of_match.group(1).strip()
        name_of_relative_match = re.search(
            r"^(?:what|which)\s+(?:is|are|was|were)\s+the\s+name\s+of\s+(.+?)(?:\s+in\s+which\b|\s+(?:that|which|who|where)\b|\?|$)",
            lower_text,
        )
        if name_of_relative_match is not None:
            return re.sub(
                r"^(?:the|a|an)\s+",
                "",
                name_of_relative_match.group(1).strip(),
                flags=re.IGNORECASE,
            )
        trailing_which_match = re.search(
            r"\b(?:of|for|from|in|on|at|to|by)\s+which\s+(.+?)(?:\?|$)",
            lower_text,
        )
        if trailing_which_match is not None:
            return trailing_which_match.group(1).strip()
        if (
            "released where" in lower_text
            or "release where" in lower_text
            or "distributed where" in lower_text
        ):
            return "region"
        relational_clause_match = re.search(
            r"^(?:what|which|who)\s+(?:is\s+the\s+|is\s+|are\s+the\s+|are\s+)?(.+?)\s+(?:preceded|succeeded)\s+by\b",
            lower_text,
        )
        if relational_clause_match is not None:
            return relational_clause_match.group(1).strip()
        how_many_match = re.search(
            r"\bhow many\s+(.+?)(?:\s+(?:is|are|was|were|did|does|do|has|have|made|make|exist|exists|work|works|worked|serve|serves|served|apply|applies|applied|for|from|in|of|with|that|which|who)\b|[?]|$)",
            lower_text,
        )
        if how_many_match is not None:
            return how_many_match.group(1).strip()
        target_match = re.search(
            r"^(?:what|which|who)\s+(?:is\s+the\s+|is\s+|are\s+the\s+|are\s+|number of\s+|how many\s+)?(.+?)(?:\s+(?:is|are|was|were|did|does|do|has|have|made|make|exist|exists|for|from|in|of)\b|$)",
            lower_text,
        )
        if target_match is not None:
            return target_match.group(1).strip()
        embedded_target_match = re.search(
            r"\b(?:what|which|who)\s+(?:is\s+the\s+|is\s+|are\s+the\s+|are\s+|number of\s+|how many\s+)?(.+?)(?:\s+(?:is|are|was|were|did|does|do|has|have|made|make|exist|exists|for|from|in|of)\b|$)",
            lower_text,
        )
        if embedded_target_match is not None:
            return embedded_target_match.group(1).strip()
        keyword_targets = (
            "drug dosage form",
            "dosage form",
            "medical treatment",
            "media genre",
            "spacecraft",
            "monarch",
            "kingdom",
            "species",
            "release",
            "cheese",
        )
        for keyword in keyword_targets:
            pattern = rf"\b{re.escape(keyword)}(?:s)?\b"
            if re.search(pattern, lower_text) is not None:
                return keyword
        return ""

    def _is_count_question_text(self, question_text: str) -> bool:
        lower_text = re.sub(
            r"^\s*question:\s*",
            "",
            str(question_text or "").lower(),
            flags=re.IGNORECASE,
        )
        return (
            lower_text.startswith("how many")
            or re.search(r"\bhow many\b", lower_text) is not None
            or re.search(r"\bhow much\b", lower_text) is not None
            or "number of" in lower_text
            or "total number of" in lower_text
            or "amount of" in lower_text
        )

    def _should_treat_answer_target_as_count_class(
        self,
        *,
        question_text: str,
        answer_target_phrase: str,
    ) -> bool:
        if not self._is_count_question_text(question_text):
            return False
        target = str(answer_target_phrase or "").strip().lower()
        if not target or target in {"amount", "number", "total number"}:
            return False
        lower_text = re.sub(
            r"^\s*question:\s*",
            "",
            str(question_text or "").lower(),
            flags=re.IGNORECASE,
        )
        trailing_text = lower_text
        target_pos = trailing_text.find(target)
        if target_pos >= 0:
            trailing_text = trailing_text[target_pos + len(target):]
        class_cues = (
            " that ",
            " which ",
            " who ",
            " work ",
            " works ",
            " worked ",
            " have ",
            " has ",
            " had ",
            " associated with ",
            " profession ",
            " transmitted by ",
            " applies to ",
        )
        return any(cue in trailing_text for cue in class_cues)

    def _infer_domain_hints(self, question_text: str) -> list[str]:
        lower_text = str(question_text or "").lower()
        if "profession" in lower_text:
            return ["people", "profession"]
        if "cheese" in lower_text:
            return ["food", "dairy", "cheese"]
        if "dosage form" in lower_text or "drug" in lower_text:
            return ["medicine", "drug"]
        if any(token in lower_text for token in ("cyclone", "hurricane", "typhoon", "tropical storm")):
            return ["meteorology", "cyclone"]
        if any(token in lower_text for token in ("fictional universe", "fictional world", "universe", "short story", "written work", "copyright date")):
            hints = ["fictional_universe"]
            if any(token in lower_text for token in ("short story", "written work", "copyright date", "publication date", "book")):
                hints.append("book")
            return hints
        if any(
            token in lower_text
            for token in (
                "virtual console",
                "distribution system",
                "game version",
                "games distributed",
                "developed by",
                "released where",
                "video game",
            )
        ):
            return ["software", "video game", "cvg"]
        if "release" in lower_text or "song" in lower_text or "music" in lower_text:
            return ["music"]
        if "engine" in lower_text or "video game" in lower_text or "software" in lower_text:
            return ["software", "engine", "video game"]
        if "monarch" in lower_text or "kingdom" in lower_text:
            return ["royalty", "government"]
        if any(token in lower_text for token in ("digital camera", "camera sensor", "sensor type", "lumix")):
            return ["product", "camera", "digicams"]
        if any(token in lower_text for token in ("gregorian calendar", "day of the week", "days of the week")):
            return ["time", "calendar"]
        if any(token in lower_text for token in ("nebula", "farthest away", "furthest away", "cosmological distance")):
            return ["astronomy", "nebula"]
        if any(token in lower_text for token in ("star category", "flare star", "temperature")):
            return ["astronomy", "star"]
        if any(token in lower_text for token in ("producer of", "higher education", "broadcast content")):
            return ["broadcast", "content"]
        if any(token in lower_text for token in ("baronet", "nobility", "noble rank")):
            return ["royalty", "nobility"]
        if "exhibition" in lower_text:
            return ["exhibitions", "museum"]
        if any(token in lower_text for token in ("basketball", "played for", "sports team")):
            return ["sports", "team"]
        return []

    def _infer_relation_hints(self, question_text: str) -> list[str]:
        grounded_candidates = self._build_grounded_relation_candidates(question_text)
        return [
            str(candidate.get("relation") or "")
            for candidate in grounded_candidates
            if str(candidate.get("relation") or "")
        ]

    def _infer_entity_clue(self, question_text: str, entity: str) -> str:
        lower_question = str(question_text or "").lower()
        lower_entity = str(entity or "").lower()
        entity_index = lower_question.find(lower_entity)
        if entity_index >= 0:
            profession_tail_pattern = rf"\b{re.escape(lower_entity)}\b\s+(?:occupation|profession)\b"
            profession_head_pattern = rf"\b(?:occupation|profession)\b(?:\s+of)?\s+\b{re.escape(lower_entity)}\b"
            if (
                re.search(profession_tail_pattern, lower_question) is not None
                or re.search(profession_head_pattern, lower_question) is not None
            ):
                return "attribute_value.profession"
        if "texture" in lower_question or "textured" in lower_question:
            if "-" in lower_entity or "firm" in lower_entity or "soft" in lower_entity:
                return "attribute_value.texture"
        if entity_index >= 0:
            camera_attribute_patterns = (
                rf"\bcolor filter array(?: type)?\s+of\s+\b{re.escape(lower_entity)}\b",
                rf"\biso settings?\s+of\s+\b{re.escape(lower_entity)}\b",
                rf"\bwith\s+(?:the\s+)?color filter array(?: type)?\s+of\s+\b{re.escape(lower_entity)}\b",
                rf"\bwith\s+iso settings?\s+of\s+\b{re.escape(lower_entity)}\b",
                rf"\bhas\s+(?:the\s+)?color filter array(?: type)?\s+of\s+\b{re.escape(lower_entity)}\b",
                rf"\bhas\s+iso settings?\s+of\s+\b{re.escape(lower_entity)}\b",
            )
            if any(re.search(pattern, lower_question) is not None for pattern in camera_attribute_patterns):
                return "attribute_value.feature"
        if (
            entity_index >= 0
            and any(
                phrase in lower_question
                for phrase in ("same category as", "same type as", "of the same type as")
            )
            and lower_question[max(0, entity_index - 6):entity_index].strip().endswith("has")
        ):
            return "attribute_value.feature"
        if "made from" in lower_question or "products of" in lower_question:
            return "source_constraint"
        if entity_index >= 0:
            entity_end = entity_index + len(lower_entity)
            from_window_start = max(0, entity_index - 12)
            if (
                lower_question[from_window_start:entity_index].strip().endswith("from")
                and "made from" not in lower_question
                and "formulated from" not in lower_question
                and any(
                    token in lower_question
                    for token in (
                        "breed",
                        "breeds",
                        "origin",
                        "originating",
                        "species",
                        "animal",
                    )
                )
            ):
                return "origin_constraint"
            if (
                lower_question[from_window_start:entity_index].strip().endswith("of")
                and entity_end < len(lower_question)
                and "temperament" in lower_question[entity_end:]
                and any(token in lower_question for token in ("breed", "breeds", "species"))
            ):
                return "origin_constraint"
        active_index = lower_question.find("active ingredient")
        if entity_index >= 0 and active_index >= 0 and entity_index >= active_index:
            return "active_ingredient"
        if "formulated from" in lower_question:
            relation_index = lower_question.find("formulated from")
            if entity_index >= relation_index >= 0:
                return "formulation_input"
        return "surface_constraint"

    def _capture_pal_query_artifacts(
        self,
        *,
        generated_tool_name: str,
        generated_code: str,
        extracted_query_text: Optional[str],
        query_plan: Optional[Mapping[str, Any]] = None,
        query_validation_errors: Optional[Sequence[str]] = None,
    ) -> dict[str, str]:
        artifact_dir = self._get_pal_query_artifact_dir()
        python_path = artifact_dir / f"{generated_tool_name}.py"
        python_path.write_text(generated_code, encoding="utf-8")
        query_path = artifact_dir / f"{generated_tool_name}.sparql"
        query_text = extracted_query_text or ""
        if query_text:
            query_path.write_text(query_text, encoding="utf-8")
        plan_path: Optional[Path] = None
        if query_plan is not None:
            plan_path = self._capture_pal_query_plan_artifact(
                generated_tool_name=generated_tool_name,
                query_plan=query_plan,
            )
        projected_variables = self._extract_projected_variables(query_text)
        self._emit_generated_tools_event(
            {
                "event": "pal_query_artifact_saved",
                "mode": "pal",
                "tool_name": generated_tool_name,
                "python_artifact_path": str(python_path),
                "query_artifact_path": str(query_path),
                "plan_artifact_path": str(plan_path) if plan_path is not None else None,
                "projected_variables": projected_variables,
                "query_text": query_text,
                "query_text_summary": self._summarize_text(query_text, max_len=240),
                "validation_errors": list(query_validation_errors or []),
            }
        )
        return {
            "python_artifact_path": str(python_path),
            "query_artifact_path": str(query_path),
            "query_text": query_text,
            "plan_artifact_path": str(plan_path) if plan_path is not None else "",
        }

    def _refresh_query_artifacts_from_invocation(
        self,
        *,
        generated_tool_name: str,
        query_artifacts: Mapping[str, str],
        invocation_result: Any,
    ) -> dict[str, str]:
        refreshed = dict(query_artifacts)
        diagnostics = dict(getattr(invocation_result, "diagnostics", {}) or {})
        query_text = str(
            diagnostics.get("query_text")
            or refreshed.get("query_text")
            or ""
        )
        refreshed["query_text"] = query_text
        query_path_str = refreshed.get("query_artifact_path")
        if query_path_str and query_text:
            Path(query_path_str).write_text(query_text, encoding="utf-8")
        self._emit_generated_tools_event(
            {
                "event": "pal_query_runtime_captured",
                "mode": "pal",
                "tool_name": generated_tool_name,
                "endpoint_url": diagnostics.get("endpoint_url") or self._get_runtime_sparql_endpoint(),
                "generated_endpoint_url": diagnostics.get("generated_endpoint_url"),
                "query_artifact_path": refreshed.get("query_artifact_path"),
                "projected_variables": self._extract_projected_variables(query_text),
                "query_text": query_text,
                "query_text_summary": self._summarize_text(query_text, max_len=240),
            }
        )
        return refreshed

    def _get_pal_query_artifact_dir(self) -> Path:
        output_dir = Path(
            os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/pal_runtime")
        )
        artifact_dir = output_dir / "pal_query_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _extract_sparql_query_text(self, generated_code: str) -> Optional[str]:
        query_texts = self._extract_sparql_query_texts(generated_code)
        if query_texts:
            return query_texts[0]
        return self._extract_sparql_query_text_regex(generated_code)

    def _extract_sparql_query_texts(self, generated_code: str) -> list[str]:
        try:
            parsed = ast.parse(generated_code)
        except SyntaxError:
            fallback_query = self._extract_sparql_query_text_regex(generated_code)
            return [fallback_query] if fallback_query else []

        string_assignments: dict[str, str] = {}
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign):
                name_targets = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
                if not name_targets:
                    continue
                literal_string = self._resolve_python_string_literal(node.value)
                if literal_string is not None:
                    for target_name in name_targets:
                        string_assignments[target_name] = literal_string

        query_texts: list[str] = []

        class _SetQueryCollector(ast.NodeVisitor):
            def __init__(self, outer: "PALAgentController") -> None:
                self.outer = outer

            def visit_Call(self, node: ast.Call) -> Any:
                if isinstance(node.func, ast.Attribute) and node.func.attr == "setQuery":
                    if node.args:
                        query_arg = node.args[0]
                        literal_string = self.outer._resolve_python_string_literal(query_arg)
                        if literal_string is None and isinstance(query_arg, ast.Name):
                            literal_string = string_assignments.get(query_arg.id)
                        cleaned_query = str(literal_string or "").strip()
                        if cleaned_query and cleaned_query not in query_texts:
                            query_texts.append(cleaned_query)
                return self.generic_visit(node)

        _SetQueryCollector(self).visit(parsed)
        if query_texts:
            return query_texts
        fallback_query = self._extract_sparql_query_text_regex(generated_code)
        return [fallback_query] if fallback_query else []

    def _extract_sparql_query_text_regex(self, generated_code: str) -> Optional[str]:
        match = re.search(
            r"query\s*=\s*(?:f)?(?P<quote>'''|\"\"\"|'|\")(?P<query>.*?)(?P=quote)",
            generated_code,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        return match.group("query")

    def _resolve_python_string_literal(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    return None
            return "".join(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._resolve_python_string_literal(node.left)
            right = self._resolve_python_string_literal(node.right)
            if left is None or right is None:
                return None
            return left + right
        return None

    def _extract_projected_variables(self, query_text: str) -> list[str]:
        if not query_text:
            return []
        select_match = re.search(
            r"SELECT\s+(?:DISTINCT\s+)?(?P<projection>.*?)\s+WHERE\b",
            query_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if select_match is None:
            return []
        projection = select_match.group("projection")
        ordered: list[str] = []
        for variable in re.findall(r"\?([A-Za-z_][A-Za-z0-9_]*)", projection):
            if variable not in ordered:
                ordered.append(variable)
        return ordered

    def _resolve_projected_variables(
        self,
        *,
        query_text: str,
        invocation_result: Any,
    ) -> list[str]:
        projected_variables = self._extract_projected_variables(query_text)
        if projected_variables:
            return projected_variables
        payload = getattr(invocation_result, "payload", None)
        if not isinstance(payload, Mapping):
            return []
        head_vars = payload.get("head", {}).get("vars", [])
        if isinstance(head_vars, list):
            return [str(item) for item in head_vars]
        return []

    def _build_pal_query_execution_event(
        self,
        *,
        generated_tool_name: str,
        query_artifacts: Mapping[str, str],
        invocation_result: Any,
        result_payload: Optional[Mapping[str, Any]],
        projected_variables: Sequence[str],
        result_kind: str,
    ) -> Mapping[str, Any]:
        endpoint_url = self._resolve_invocation_endpoint_url(invocation_result)
        binding_count = (
            self._get_result_binding_count(result_payload)
            if isinstance(result_payload, Mapping)
            else None
        )
        raw_result_summary = (
            self._summarize_text(
                json.dumps(result_payload, ensure_ascii=False, default=str),
                max_len=240,
            )
            if isinstance(result_payload, Mapping)
            else self._summarize_text(
                getattr(invocation_result, "error", "") or "",
                max_len=240,
            )
        )
        return {
            "event": "pal_query_execution_result",
            "mode": "pal",
            "tool_name": generated_tool_name,
            "python_artifact_path": query_artifacts.get("python_artifact_path"),
            "query_artifact_path": query_artifacts.get("query_artifact_path"),
            "endpoint_url": endpoint_url,
            "projected_variables": list(projected_variables),
            "binding_count": binding_count,
            "result_kind": result_kind,
            "result_empty": result_kind == "empty",
            "raw_result_summary": raw_result_summary,
        }

    def _extract_macro_pointer(self, content: str) -> Optional[str]:
        if "Macro result:" not in (content or ""):
            return None
        if not self._macro_result_is_trusted_final(content):
            return None
        pointer_match = re.search(r"Final variable:\s*(#\d+)", content)
        if pointer_match is None:
            return None
        return pointer_match.group(1)

    def _log_macro_result(self, content: str) -> None:
        run_id = self._get_run_id()
        pending = self._pending_macro_runs.pop(run_id, {})
        tool_name_match = re.search(
            r"Macro result:\s*([A-Za-z0-9_]+)\s*->\s*([A-Z_]+)",
            content or "",
        )
        tool_name = pending.get("tool_name")
        status = "UNKNOWN"
        if tool_name_match is not None:
            tool_name = tool_name_match.group(1)
            status = tool_name_match.group(2)
        success = status == "SUCCESS" and self._macro_result_is_trusted_final(content)
        pointer_match = re.search(r"Final variable:\s*(#\d+)", content or "")
        result_errors = None if success else [content]
        self._emit_generated_tools_event(
            {
                "event": "invoke",
                "tool_name": tool_name or PAL_BENCHMARK_BRIDGE_TOOL_NAME,
                "run_id": str(pending.get("run_id") or run_id),
                "state_dir": pending.get("state_dir"),
                "asked_for": pending.get("asked_for"),
                "actions_spec_keys": [],
                "trace": [],
                "invocation_context": {
                    "environment": str(
                        getattr(
                            getattr(self, "_current_session", None),
                            "task_name",
                            "knowledge_graph",
                        )
                    )
                },
                "success": success,
                "duration_ms": None,
                "result_status": status.lower(),
                "result_contract_ok": success,
                "result_errors": result_errors,
                "result_answer_recommendation": pointer_match.group(1)
                if success and pointer_match is not None
                else None,
            }
        )

    def _macro_result_is_trusted_final(self, content: str) -> bool:
        text = str(content or "")
        return bool(
            re.search(r"Trusted final:\s*yes\b", text, flags=re.IGNORECASE)
            and re.search(r"Solves task:\s*yes\b", text, flags=re.IGNORECASE)
        )

    def _build_tool_result_semantics(
        self,
        *,
        materialization: BenchmarkMaterialization,
        trust_contract: TrustContractEvaluation,
    ) -> dict[str, Any]:
        trusted_final = bool(
            trust_contract.materialization_allowed
            and materialization.solves_task
            and materialization.trusted_for_materialization
        )
        useful_intermediate = bool(
            materialization.useful_intermediate
            and not trusted_final
            and str(materialization.semantic_description or "").strip()
        )
        status = "success" if trusted_final else ("partial" if useful_intermediate else "failed")
        failure_reason = str(materialization.failure_reason or "").strip()
        if not failure_reason and not trusted_final:
            failure_reason = ",".join(
                str(item).strip()
                for item in trust_contract.denial_reasons
                if str(item).strip()
            )
        return {
            "tool_result_status": status,
            "tool_result_semantic_description": str(
                materialization.semantic_description or ""
            ).strip(),
            "tool_result_solves_task": bool(materialization.solves_task),
            "tool_result_trusted_for_materialization": trusted_final,
            "tool_result_useful_intermediate": useful_intermediate,
            "tool_result_failure_reason": failure_reason,
            "tool_result_confidence": materialization.confidence,
        }

    def _build_tool_advisory_text(
        self,
        *,
        materialization: BenchmarkMaterialization,
        trust_contract: TrustContractEvaluation,
    ) -> str:
        semantics = self._build_tool_result_semantics(
            materialization=materialization,
            trust_contract=trust_contract,
        )
        lines = [
            f"PAL tool status: {semantics.get('tool_result_status')}.",
        ]
        semantic_description = str(
            semantics.get("tool_result_semantic_description") or ""
        ).strip()
        if semantic_description:
            lines.append(f"Semantic description: {semantic_description}.")
        failure_reason = str(semantics.get("tool_result_failure_reason") or "").strip()
        if failure_reason:
            lines.append(f"Failure reason: {failure_reason}.")
        if semantics.get("tool_result_useful_intermediate"):
            lines.append(
                "This is bounded advisory context only. Do not treat it as a final answer."
            )
        else:
            lines.append(
                "Ignore the PAL tool output if it does not help, and continue solving manually."
            )
        return " ".join(line.strip() for line in lines if line.strip())

    def _manual_fallback_active_for_current_run(self) -> bool:
        return self._get_run_id() in self._manual_fallback_active_runs

    def _activate_manual_fallback(
        self,
        *,
        chat_history: ChatHistory,
        advisory_text: Optional[str] = None,
    ) -> None:
        self._manual_fallback_active_runs.add(self._get_run_id())
        if advisory_text and chat_history.get_value_length() > 0:
            last_item = chat_history.get_item_deep_copy(-1)
            if last_item.role == Role.USER and advisory_text not in (last_item.content or ""):
                separator = "\n\nPAL tool advisory (optional):\n"
                chat_history.set(
                    -1,
                    ChatHistoryItem(
                        role=Role.USER,
                        content=(last_item.content or "") + separator + advisory_text,
                    ),
                )

    def _get_manual_fallback_agent(self) -> SelfEvolvingController:
        if self._manual_fallback_agent is None:
            output_dir = Path(
                os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/pal_runtime")
            )
            tool_registry_path = str(output_dir / "tool_library")
            environment_label = str(
                getattr(getattr(self, "_current_session", None), "task_name", "knowledge_graph")
            )
            self._manual_fallback_agent = SelfEvolvingController(
                language_model=self._language_model,
                tool_registry_path=tool_registry_path,
                inference_config_dict=dict(self._inference_config_dict),
                environment_label=environment_label,
            )
        return self._manual_fallback_agent

    def _delegate_to_manual_solver(
        self,
        *,
        chat_history: ChatHistory,
    ) -> ChatHistoryItem:
        fallback_agent = self._get_manual_fallback_agent()
        setattr(fallback_agent, "_current_session", getattr(self, "_current_session", None))
        response = fallback_agent._inference(chat_history)
        self._tool_invoked_in_last_inference = getattr(
            fallback_agent,
            "_tool_invoked_in_last_inference",
            None,
        )
        self._emit_generated_tools_event(
            {
                "event": "pal_manual_solver_fallback_used",
                "mode": "pal",
                "run_id": self._get_run_id(),
            }
        )
        return response

    def _build_adapter_context(self, task_question: str) -> BenchmarkAdapterContext:
        return BenchmarkAdapterContext(
            task_question=task_question,
            run_id=self._get_run_id(),
            state_dir=self._get_macro_state_dir(),
            bridge_tool_name=PAL_BENCHMARK_BRIDGE_TOOL_NAME,
        )

    def _ensure_bridge_tool(self, tool_name: str) -> str:
        tool_path = self._get_bridge_tool_path(tool_name)
        tool_path.parent.mkdir(parents=True, exist_ok=True)
        if not tool_path.exists():
            tool_code = build_pal_benchmark_bridge_tool_code()
            assert_bridge_tool_code_narrow(tool_code)
            tool_path.write_text(tool_code, encoding="utf-8")
            self._registered_bridge_tools.add(tool_name)
            self._emit_generated_tools_event(
                {
                    "event": "register",
                    "tool_name": tool_name,
                    "signature": "run(payload: dict) -> dict",
                    "description": "PAL benchmark adapter bridge macro.",
                    "docstring": "Bridge typed PAL artifacts into deterministic benchmark variables.",
                    "tool_type": "utility",
                    "tool_category": "utility",
                    "input_schema": {
                        "type": "object",
                        "required": [
                            "pal_artifact_type",
                            "pal_artifact_value",
                            "run_id",
                            "state_dir",
                        ],
                        "properties": {
                            "pal_artifact_type": {"type": "string"},
                            "pal_artifact_value": {},
                            "pal_artifact_source": {"type": "string"},
                            "pal_artifact_diagnostics": {"type": "object"},
                            "run_id": {"type": "string"},
                            "state_dir": {"type": "string"},
                        },
                    },
                    "required_keys": [
                        "pal_artifact_type",
                        "pal_artifact_value",
                        "run_id",
                        "state_dir",
                    ],
                    "optional_keys": [
                        "pal_artifact_source",
                        "pal_artifact_diagnostics",
                            "pal_semantic_description",
                            "pal_solves_task",
                            "pal_trusted_for_materialization",
                            "pal_tool_status",
                            "pal_failure_reason",
                            "pal_confidence",
                            "variable_list",
                        ],
                    "property_types": {
                        "pal_artifact_type": "string",
                        "run_id": "string",
                        "state_dir": "string",
                    },
                    "capabilities": [],
                    "path": str(tool_path),
                    "code_len": len(tool_code),
                    "code_sha256": hashlib.sha256(tool_code.encode("utf-8")).hexdigest(),
                }
            )
        return tool_name

    def _get_bridge_tool_path(self, tool_name: str) -> Path:
        output_dir = Path(
            os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/pal_runtime")
        )
        return output_dir / "tool_library" / "generated_tools" / f"{tool_name}.py"

    def _get_macro_state_dir(self) -> str:
        output_dir = Path(
            os.environ.get("LIFELONG_OUTPUT_DIR", "outputs/pal_runtime")
        )
        state_dir = output_dir / "tool_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return str(state_dir)

    def _get_run_id(self) -> str:
        current_session = getattr(self, "_current_session", None)
        if current_session is None:
            return "pal"
        return f"{current_session.task_name}_{current_session.sample_index}"

    def _get_runtime_sparql_endpoint(self) -> str:
        return os.environ.get(
            "PAL_SPARQL_ENDPOINT_URL",
            "http://127.0.0.1:3001/kb/sparql",
        )

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        return self._language_model.role_dict
