#controller.py

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import threading
import time


from typing_extensions import override

from src.agents.agent import Agent
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.language_models import LanguageModel
from src.utils.output_paths import get_output_dir_override, prefix_filename
from src.typings import (
    AgentContextLimitException,
    AgentOutOfMemoryException,
    AgentUnknownException,
    ChatHistory,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    LanguageModelOutOfMemoryException,
    LanguageModelUnknownException,
    Role,
)
from src.typings.config import get_predefined_timestamp_structure

from .controller_history import ControllerHistoryMixin
from .controller_logging import ControllerLoggingMixin
from .controller_orchestrator import ControllerOrchestratorMixin
from .controller_prompts import (
    SOLVER_SYSTEM_PROMPT,
    TOOLGEN_DEBUG_APPENDIX,
    TOOLGEN_USER_APPENDIX,
    TOOLGEN_SYSTEM_PROMPT_MARKERS,
    TOOL_INVOKER_SYSTEM_PROMPT,
    TOOL_ORCHESTRATOR_SYSTEM_PROMPT,
    TOP_LEVEL_ORCHESTRATOR_SYSTEM_PROMPT,
)
from .controller_toolgen import ControllerToolgenMixin
from .controller_tools import ControllerToolsMixin
from .tool_registry import ToolMetadata, ToolResult, get_registry
from .toolgen_debug_logger import ToolgenDebugLogger, toolgen_debug_enabled


USE_PACKAGED_AGENT = False


class SelfEvolvingController(
    ControllerLoggingMixin,
    ControllerHistoryMixin,
    ControllerToolgenMixin,
    ControllerToolsMixin,
    ControllerOrchestratorMixin,
    Agent,
):
    _INTERNAL_TOOL_PATTERN = re.compile(
        r"<internal_tool\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</internal_tool>",
        re.MULTILINE,
    )

    def __init__(
        self,
        language_model: LanguageModel,
        tool_registry_path: str,
        max_generated_tools_per_run: int = 3,
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        bootstrap_tools: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: str = (
            "You are a helpful assistant that can emit <action> blocks to either "
            "call previously generated tools or request new ones. Always keep task-"
            "specific output formats (e.g., Action: Operation) intact."
        ),
        force_tool_generation_if_missing: bool = True,
        tool_match_min_score: float = 0.7,
        include_registry_in_prompt: bool = True,
        use_orchestrator: bool = True,
        environment_label: str = "unknown",
        retrieval_top_k: int = 5,
        reuse_top_k: int = 3,
        reuse_similarity_threshold: Optional[float] = None,
        reuse_min_reliability: float = 0.0,
        canonical_tool_naming: bool = True,
        use_packaged_agent: bool = USE_PACKAGED_AGENT,
    ):
        self._use_packaged_agent = use_packaged_agent
        self._packaged_shim = None
        if self._use_packaged_agent:
            from .packaged_shim import PackagedSelfEvolvingShim

            self._packaged_shim = PackagedSelfEvolvingShim(
                language_model=language_model,
                tool_registry_path=tool_registry_path,
                max_generated_tools_per_run=max_generated_tools_per_run,
                inference_config_dict=inference_config_dict,
                bootstrap_tools=bootstrap_tools,
                system_prompt=system_prompt,
                force_tool_generation_if_missing=force_tool_generation_if_missing,
                tool_match_min_score=tool_match_min_score,
                include_registry_in_prompt=include_registry_in_prompt,
                environment_label=environment_label,
                retrieval_top_k=retrieval_top_k,
                reuse_top_k=reuse_top_k,
                reuse_similarity_threshold=reuse_similarity_threshold,
                reuse_min_reliability=reuse_min_reliability,
                canonical_tool_naming=canonical_tool_naming,
            )
            return

        solver_cfg = dict(inference_config_dict) if inference_config_dict else {}
        for k in ("tools", "tool_choice", "functions", "function_call"):
            solver_cfg.pop(k, None)

        solver_cfg["tool_choice"] = "none"

        self._language_model_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=system_prompt,
            inference_config_dict=solver_cfg,
            agent_name="solver",
        )

        base_cfg = dict(inference_config_dict) if inference_config_dict else {}
        for k in ("tools", "tool_choice", "functions", "function_call"):
            base_cfg.pop(k, None)

        base_cfg["tool_choice"] = "none"
        base_cfg["toolgen_extract_tool_calls"] = True
        base_cfg["ollama_force_tool_calls"] = False
        toolgen_system_prompt = TOOLGEN_USER_APPENDIX
        if toolgen_debug_enabled():
            toolgen_system_prompt = (
                f"{toolgen_system_prompt}\n\n{TOOLGEN_DEBUG_APPENDIX}".strip()
            )
        self._toolgen_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=toolgen_system_prompt,
            inference_config_dict={
                **base_cfg,
                "temperature": 0.1,
            },
            agent_name="toolgen",
        )

        self._use_orchestrator = use_orchestrator
        self._orchestrator_agent: Optional[LanguageModelAgent] = None
        self._tool_orchestrator_agent: Optional[LanguageModelAgent] = None
        self._tool_invoker_agent: Optional[LanguageModelAgent] = None
        if self._use_orchestrator:
            orchestrator_cfg = dict(inference_config_dict) if inference_config_dict else {}
            for k in ("tools", "tool_choice", "functions", "function_call"):
                orchestrator_cfg.pop(k, None)

            orchestrator_cfg["tool_choice"] = "none"
            orchestrator_cfg["response_format"] = {"type": "json_object"}

            self._orchestrator_agent = LanguageModelAgent(
                language_model=language_model,
                system_prompt=TOP_LEVEL_ORCHESTRATOR_SYSTEM_PROMPT,
                inference_config_dict={
                    **orchestrator_cfg,
                    "temperature": 0.0,
                },
                agent_name="top_orchestrator",
            )
            self._tool_orchestrator_agent = LanguageModelAgent(
                language_model=language_model,
                system_prompt=TOOL_ORCHESTRATOR_SYSTEM_PROMPT,
                inference_config_dict={
                    **orchestrator_cfg,
                    "temperature": 0.0,
                },
                agent_name="tool_orchestrator",
            )
            self._tool_invoker_agent = LanguageModelAgent(
                language_model=language_model,
                system_prompt=TOOL_INVOKER_SYSTEM_PROMPT,
                inference_config_dict={
                    **orchestrator_cfg,
                    "temperature": 0.0,
                },
                agent_name="tool_invoker",
            )

        self._registry = get_registry(tool_registry_path)
        self._registry.set_run_snapshot(
            get_predefined_timestamp_structure()["TIMESTAMP"]
        )
        self._tool_invocation_log_path: Optional[Path] = None
        self._generated_tools_log_path: Optional[Path] = None
        self._flow_session_log_path: Optional[Path] = None
        self._flow_full_log_path: Optional[Path] = None
        self._agent_system_prompt_path: Optional[Path] = None
        self._toolgen_debug_logger: Optional[ToolgenDebugLogger] = None
        self._run_task_label: Optional[str] = None
        try:
            output_dir = get_output_dir_override()
            if output_dir:
                self._tool_invocation_log_path = (
                    Path(output_dir) / prefix_filename("tool_invocations.log")
                )
                self._generated_tools_log_path = (
                    Path(output_dir) / prefix_filename("generated_tools.log")
                )
                self._flow_session_log_path = (
                    Path(output_dir) / prefix_filename("flow_session.json")
                )
                self._flow_full_log_path = (
                    Path(output_dir).parent / prefix_filename("flow_full.json")
                )
                self._agent_system_prompt_path = (
                    Path(output_dir) / prefix_filename("agent_system_prompts.json")
                )
            else:
                run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
                self._tool_invocation_log_path = (
                    Path("outputs") / run_id / "tool_invocations.log"
                )
                self._generated_tools_log_path = (
                    Path("outputs") / run_id / "generated_tools.log"
                )
                self._flow_session_log_path = (
                    Path("outputs") / run_id / "flow_session.json"
                )
                self._flow_full_log_path = Path("outputs") / run_id / "flow_full.json"
                self._agent_system_prompt_path = (
                    Path("outputs") / run_id / "agent_system_prompts.json"
                )
        except Exception:
            self._tool_invocation_log_path = None
            self._generated_tools_log_path = None
            self._flow_session_log_path = None
            self._flow_full_log_path = None
            self._agent_system_prompt_path = None
        try:
            debug_enabled = toolgen_debug_enabled()
            if debug_enabled:
                if self._generated_tools_log_path is not None:
                    debug_path = self._generated_tools_log_path.parent / "toolgen_debug.log"
                    run_id = self._generated_tools_log_path.parent.name
                else:
                    run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
                    debug_path = Path("outputs") / "tool_library" / "toolgen_debug.log"
                self._toolgen_debug_logger = ToolgenDebugLogger(
                    debug_path,
                    enabled=True,
                    run_id=run_id,
                    environment=environment_label,
                )
                self._registry.add_event_listener(
                    self._toolgen_debug_logger.log_registry_event
                )
        except Exception:
            self._toolgen_debug_logger = None
        self._max_generated_tools_per_run = max_generated_tools_per_run
        self._generated_tool_counter = 0
        self._force_tool_generation_if_missing = force_tool_generation_if_missing
        self._tool_match_min_score = tool_match_min_score
        self._include_registry_in_prompt = include_registry_in_prompt
        self._retrieval_top_k = retrieval_top_k
        self._reuse_top_k = reuse_top_k
        self._reuse_similarity_threshold = (
            tool_match_min_score
            if reuse_similarity_threshold is None
            else reuse_similarity_threshold
        )
        self._reuse_min_reliability = reuse_min_reliability
        self._canonical_tool_naming = canonical_tool_naming
        self._min_reliability = 0.2
        self._internal_tool_max_steps = 3
        self._tool_creation_attempts = 0
        self._tool_creation_successes = 0
        self._tool_invocation_attempts = 0
        self._tool_invocation_successes = 0
        self._tool_invoked_in_last_inference = False
        self._environment_label = environment_label
        self._toolgen_attempted_queries: set[str] = set()
        self._toolgen_debug_registered_tools: dict[str, int] = {}
        self._last_toolgen_parse_source: Optional[str] = None
        self._toolgen_last_recommendation: Optional[str] = None
        self._run_task_metadata: Optional[dict[str, Any]] = None
        self._last_solver_output: Optional[str] = None
        self._last_solver_context_key: Optional[str] = None
        self._solver_repeat_count = 0
        if hasattr(self._registry, "set_canonical_naming"):
            self._registry.set_canonical_naming(self._canonical_tool_naming)

        self._bootstrap_tools(bootstrap_tools or [])
        base_solver_prompt = (system_prompt or "").strip() or "You are a helpful assistant."
        self._solver_system_prompt = f"{base_solver_prompt}\n\n{SOLVER_SYSTEM_PROMPT}".strip()


    def _default_payload_dict(
        self,
        *,
        task_text: str,
        candidate_output: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_text": task_text or "",
            "asked_for": "",          # safe default
            "trace": [],              # safe default
            "actions_spec": {},       # safe default
            "constraints": [],
            "output_contract": {},
            "draft_response": None,
            "candidate_output": candidate_output,
            "env_observation": None,
        }
        # Optional: only include if you actually want tools to see it
        if chat_history is not None:
            payload["chat_history"] = self._toolgen_render_history(chat_history, max_chars_per_item=None)
        return payload

    def _orchestrated_inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.inference(chat_history)

        working_history = self._clone_history(self._prune_for_current_task(chat_history))
        self._tool_invoked_in_last_inference = False

        user_items = [item for item in self._history_items(working_history) if item.role == Role.USER]
        if len(user_items) >= 2:
            original_query = user_items[1].content
        else:
            orig_last_user = self._get_last_user_item(working_history)
            original_query = orig_last_user.content if orig_last_user else ""
        task_query = (original_query or "").strip()

        def _record_tool_error(stage: str, message: str, tb: Optional[str] = None) -> None:
            self._trace("tool_pipeline_error", f"{stage}: {message}")
            if tb:
                self._trace("tool_pipeline_traceback", tb)

        def _requires_action_format() -> bool:
            return self._is_db_bench_env()

        def _fallback_response() -> str:
            if _requires_action_format():
                return "Action: Answer\nFinal Answer: []"
            return "OK."

        def _ensure_solver_output(content: str) -> str:
            text = (content or "").strip()
            if not text:
                return _fallback_response()
            if _requires_action_format() and not re.match(r"^Action:\s*(Operation|Answer)\b", text):
                return _fallback_response()
            return content

        tool_traces: list[dict[str, Any]] = []
        tool_error: Optional[str] = None
        tool_result_injected = False
        tool_agent_traced = False
        solver_sidecar: list[str] = []
        solver_recommendation: Optional[str] = None

        initial_solver_prompt = self._solver_prompt_no_tools()
        self._write_agent_system_prompt("solver", initial_solver_prompt)
        initial_payload = {
            "system_prompt": initial_solver_prompt,
            "history": self._toolgen_render_history(
                working_history, max_chars_per_item=None
            ),
            "stage": "initial_recommendation",
        }
        self._log_flow_event(
            "solver_input",
            chat_history=working_history,
            payload=initial_payload,
        )
        initial_response = self._solver_inference_with_retry(
            working_history, system_prompt=initial_solver_prompt
        )
        solver_recommendation = (getattr(initial_response, "content", "") or "")
        self._toolgen_last_recommendation = solver_recommendation
        initial_solver_output = _ensure_solver_output(solver_recommendation)
        if self._contains_internal_tool(initial_solver_output):
            initial_solver_output = _fallback_response()

        try:
            decision = self._orchestrate_decision(task_query, working_history)
            action = decision.get("action", "no_tool")
            if action != "use_tool":
                self._trace("solver_result", initial_solver_output)
                self._log_flow_event(
                    "final_response",
                    chat_history=working_history,
                    content=initial_solver_output,
                )
                self._flush_tool_traces(tool_traces, initial_solver_output)
                return ChatHistoryItem(role=Role.AGENT, content=initial_solver_output)
            if action == "use_tool":
                tool_agent_traced = True
                tool_decision = self._tool_orchestrate_decision(
                    task_query,
                    working_history,
                    solver_recommendation=solver_recommendation,
                )
                tool_action = tool_decision.get("action", "create_tool")
                tool_suggestion = {
                    "tool_name": tool_decision.get("tool_name"),
                    "reason": tool_decision.get("reason"),
                }

                if tool_action == "create_tool":
                    self._toolgen_last_recommendation = solver_recommendation
                    selected_tool = self._maybe_generate_tool_for_query(
                        task_query, working_history, allow_reuse=False, force=True
                    )
                    if selected_tool is None:
                        self._trace("registry_add_failed", "tool generation failed")
                        tool_error = tool_error or "tool generation failed"
                        _record_tool_error("create_tool", tool_error)
                        tool_result = ToolResult.failure(tool_error)
                        solver_sidecar.append(
                            self._format_tool_result("create_tool", tool_result)
                        )
                        tool_result_injected = True
                    else:
                        self._trace("registry_add", selected_tool.name)
                        tool_suggestion["tool_name"] = selected_tool.name

                if not tool_result_injected:
                    tool_invoker = self._tool_invoker_decision(
                        task_query, working_history, suggestion=tool_suggestion
                    )
                    tool_name = tool_invoker.get("tool_name") or tool_suggestion.get("tool_name")
                    tool_args = tool_invoker.get("tool_args")
                    if tool_name:
                        args_auto_built = False
                        if tool_args is None or tool_args == {} or tool_args == []:
                            resolved_name = (
                                self._registry.resolve_name(tool_name)
                                if hasattr(self._registry, "resolve_name")
                                else None
                            )
                            resolved_name = resolved_name or tool_name
                            tool_meta = self._get_tool_metadata(resolved_name)
                            if tool_meta is not None:
                                tool_args = self._auto_build_tool_args(
                                    tool_meta, query=task_query, chat_history=working_history
                                )
                                args_auto_built = tool_args is not None
                        if tool_args is None or tool_args == {} or tool_args == []:
                            tool_args = {"args": [ {"task_text": task_query, "candidate_output": None} ]}
                            args_auto_built = True
                        self._log_flow_event(
                            "tool_agent_input",
                            chat_history=working_history,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            reason="tool_invoker",
                        )
                        tool_result = self._invoke_tool_by_payload(
                            tool_name,
                            tool_args,
                            reason="tool_invoker",
                            args_auto_built=args_auto_built,
                            decision_action=tool_action,
                        )
                        self._log_flow_event(
                            "tool_agent_output",
                            chat_history=working_history,
                            tool_name=tool_name,
                            success=tool_result.success,
                            error=tool_result.error,
                            output=tool_result.output,
                        )
                        tool_traces.append(
                            self._build_tool_trace(
                                summary="Tool invoker executed a tool.",
                                tool_name=tool_name,
                                args=[],
                                kwargs={"tool_args": tool_args},
                                result=tool_result,
                            )
                        )
                        solver_sidecar.append(
                            self._format_tool_result(tool_name, tool_result)
                        )
                        tool_result_injected = True
                    else:
                        tool_error = "tool_invoker_missing_tool_name"
                        _record_tool_error("tool_invoker", tool_error)
                        tool_result = ToolResult.failure(tool_error)
                        solver_sidecar.append(
                            self._format_tool_result("tool_invoker", tool_result)
                        )
                        tool_result_injected = True
        except Exception as exc:
            tool_error = f"{type(exc).__name__}: {exc}"
            _record_tool_error("tool_pipeline", tool_error, traceback.format_exc())

        if not tool_agent_traced:
            self._trace("tool_agent_input", "none")
            self._trace("tool_agent_result", "none")

        repeat_info = self._detect_repeated_env_action(working_history)
        last_action_info = self._get_last_action_info(working_history)
        task_intent = self._infer_task_intent(task_query)
        context_msg = self._build_solver_context_message(
            task_query=task_query,
            tool_result_injected=tool_result_injected,
            last_action_info=last_action_info,
            task_intent=task_intent,
            repeat_info=repeat_info,
        )
        if context_msg:
            solver_sidecar.append("CONTEXT:\n" + context_msg)
        if solver_recommendation:
            solver_sidecar.insert(
                0, "SOLVER_RECOMMENDATION:\n" + solver_recommendation
            )

        for _ in range(3):
            solver_prompt = self._solver_prompt_no_tools()
            if solver_sidecar:
                solver_prompt = (
                    solver_prompt
                    + "\n\nINTERNAL TOOL CONTEXT:\n"
                    + "\n\n".join(solver_sidecar)
                )
            solver_payload = {
                "system_prompt": solver_prompt,
                "history": self._toolgen_render_history(
                    working_history, max_chars_per_item=None
                ),
                "tool_error": tool_error,
                "tool_result_injected": tool_result_injected,
            }
            self._write_agent_system_prompt(
                "solver",
                solver_prompt,
            )
            self._trace("solver_input", json.dumps(solver_payload, ensure_ascii=True, default=str))
            self._log_flow_event(
                "solver_input",
                chat_history=working_history,
                payload=solver_payload,
            )
            solver_response = self._solver_inference_with_retry(
                working_history, system_prompt=solver_prompt
            )
            content = getattr(solver_response, "content", "") or ""
            if self._contains_internal_tool(content):
                self._trace("solver_result", content)
                working_history = self._safe_inject(working_history, solver_response)
                continue
            content = _ensure_solver_output(content)
            context_key = self._solver_context_key(working_history)
            if content == self._last_solver_output and context_key == self._last_solver_context_key:
                self._solver_repeat_count += 1
                if self._solver_repeat_count >= 2:
                    fallback = _fallback_response()
                    self._trace("solver_result", fallback)
                    self._log_flow_event(
                        "final_response",
                        chat_history=working_history,
                        content=fallback,
                    )
                    self._flush_tool_traces(tool_traces, fallback)
                    return ChatHistoryItem(role=Role.AGENT, content=fallback)
                working_history = self._safe_inject(working_history, solver_response)
                working_history = self._safe_inject(
                    working_history,
                    ChatHistoryItem(
                        role=Role.USER,
                        content=(
                            "Stop repeating; use latest TOOL_RESULT/env output and produce final answer now."
                        ),
                    ),
                )
                continue
            self._solver_repeat_count = 0
            self._last_solver_output = content
            self._last_solver_context_key = context_key
            self._trace("solver_result", content)
            self._log_flow_event(
                "final_response",
                chat_history=working_history,
                content=content,
            )
            self._flush_tool_traces(tool_traces, content)
            self._log_flow_event(
                "final_response",
                chat_history=working_history,
                content=content,
            )
            return ChatHistoryItem(role=Role.AGENT, content=content)

        fallback = _fallback_response()
        self._trace("solver_result", fallback)
        self._log_flow_event(
            "final_response",
            chat_history=working_history,
            content=fallback,
        )
        self._flush_tool_traces(tool_traces, fallback)
        return ChatHistoryItem(role=Role.AGENT, content=fallback)

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        """
        Controller inference:
        - Returns the solver model's output EXACTLY as-is (no env-format parsing/repair).
        - BUT: <internal_tool ...> blocks must NEVER be returned to the environment.
        If present, we execute them internally and loop until the solver returns
        a response with no internal_tool blocks.

        Key fixes:
        1) Pin ORIGINAL_TASK once so tool results don't become the "query" driver.
        2) Dedupe internal tool calls so the model can't re-run the same call forever.
        3) Lightly anchor ORIGINAL_TASK into the system prompt each step.
        """
        if self._use_orchestrator:
            return self._orchestrated_inference(chat_history)
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.inference(chat_history)

        working_history = self._clone_history(self._prune_for_current_task(chat_history))
        self._tool_invoked_in_last_inference = False

        user_items = [item for item in self._history_items(working_history) if item.role == Role.USER]
        if len(user_items) >= 2:
            original_query = user_items[1].content
        else:
            orig_last_user = self._get_last_user_item(working_history)
            original_query = orig_last_user.content if orig_last_user else ""

        auto_invoked_tools: set[str] = set()
        tool_traces: list[dict[str, Any]] = []
        solver_sidecar: list[str] = []
        seen_internal_calls: set[str] = set()

        for _ in range(self._internal_tool_max_steps):
            last_user = self._get_last_user_item(working_history)
            query = last_user.content if last_user else ""

            task_query = (original_query or "").strip() or (query or "").strip()

            self._consider_tool_generation(task_query, working_history)

            if task_query:
                preprocess_tool = self._select_tool_for_query(
                    task_query, categories={"parser", "normalizer", "planner"}
                )
                if preprocess_tool and preprocess_tool.name not in auto_invoked_tools:
                    preprocess_payload = self._invoke_tool_for_query(
                        preprocess_tool, task_query, reason="auto_preprocess"
                    )
                    if preprocess_payload is not None:
                        preprocess_result, tool_args, tool_kwargs = preprocess_payload
                        auto_invoked_tools.add(preprocess_tool.name)
                        tool_traces.append(
                            self._build_tool_trace(
                                summary="Auto-invoked tool to extract or normalize the task.",
                                tool_name=preprocess_tool.name,
                                args=tool_args,
                                kwargs=tool_kwargs,
                                result=preprocess_result,
                            )
                        )
                        solver_sidecar.append(
                            self._format_tool_result(preprocess_tool.name, preprocess_result)
                        )

            try:
                solver_prompt = self._solver_system_prompt or ""
                if solver_sidecar:
                    solver_prompt = (
                        solver_prompt
                        + "\n\nINTERNAL TOOL CONTEXT:\n"
                        + "\n\n".join(solver_sidecar)
                    )
                self._write_agent_system_prompt(
                    "solver",
                    solver_prompt,
                )
                solver_response = self._solver_inference_with_retry(
                    working_history, system_prompt=solver_prompt
                )
            except LanguageModelContextLimitException as e:
                raise AgentContextLimitException(str(e)) from e
            except LanguageModelOutOfMemoryException as e:
                raise AgentOutOfMemoryException(str(e)) from e
            except LanguageModelUnknownException as e:
                raise AgentUnknownException(str(e)) from e

            content = getattr(solver_response, "content", "") or ""

            if self._contains_internal_tool(content):
                calls = self._extract_internal_tool_calls(content)

                working_history = self._safe_inject(working_history, solver_response)

                if not calls:
                    working_history = self._safe_inject(
                        working_history,
                        ChatHistoryItem(
                            role=Role.USER,
                            content=(
                                'If you are calling an internal tool, output ONLY a valid '
                                '<internal_tool name="...">{...}</internal_tool> block and nothing else.'
                            ),
                        ),
                    )
                    continue

                for tool_name, payload in calls:
                    if isinstance(payload, (dict, list)):
                        payload_key = json.dumps(payload, sort_keys=True, default=str)
                    else:
                        payload_key = str(payload)
                    call_key = f"{tool_name}::{payload_key}"

                    if call_key in seen_internal_calls:
                        working_history = self._safe_inject(
                            working_history,
                            ChatHistoryItem(
                                role=Role.USER,
                                content=(
                                    f"You already called {tool_name} with the same arguments and received the result above. "
                                    "Do NOT call it again. Continue solving ORIGINAL_TASK using the available results. "
                                    "Return a non-internal response next."
                                ),
                            ),
                        )
                        continue

                    seen_internal_calls.add(call_key)

                    tool_result, tool_args, tool_kwargs, resolved_name = self._handle_internal_tool_call(
                        tool_name, payload, working_history
                    )

                    tool_traces.append(
                        self._build_tool_trace(
                            summary="Model invoked an internal tool (kept internal).",
                            tool_name=resolved_name,
                            args=tool_args,
                            kwargs=tool_kwargs,
                            result=tool_result,
                        )
                    )

                    solver_sidecar.append(
                        self._format_tool_result(tool_name, tool_result)
                    )

                continue

            self._flush_tool_traces(tool_traces, content)
            return ChatHistoryItem(role=Role.AGENT, content=content)

        self._flush_tool_traces(tool_traces, "Exceeded internal tool steps.")
        raise AgentUnknownException(
            "Exceeded internal tool steps without producing a non-internal output."
        )

    def _solver_prompt_no_tools(self) -> str:
        return (self._solver_system_prompt or "").strip() or "You are a helpful assistant."

    def _solver_inference_with_retry(
        self, chat_history: ChatHistory, system_prompt: Optional[str] = None
    ) -> ChatHistoryItem:
        return self._infer_with_system_prompt(chat_history, system_prompt)

    def _infer_with_system_prompt(
        self, chat_history: ChatHistory, system_prompt: Optional[str] = None
    ) -> ChatHistoryItem:
        if system_prompt is None:
            return self._language_model_agent._inference(chat_history)
        original_prompt = self._language_model_agent._system_prompt
        if system_prompt == original_prompt:
            return self._language_model_agent._inference(chat_history)
        self._language_model_agent._system_prompt = system_prompt
        try:
            return self._language_model_agent._inference(chat_history)
        finally:
            self._language_model_agent._system_prompt = original_prompt

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.get_role_dict()
        return self._language_model_agent.get_role_dict()
