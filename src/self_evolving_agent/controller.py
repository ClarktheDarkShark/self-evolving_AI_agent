#controller.py

import json
import os
import re
import traceback
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import threading
import time
import hashlib


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
    TOOLGEN_SYSTEM_PROMPT_MARKERS,
    TOOL_INVOKER_SYSTEM_PROMPT,
    COMBINED_ORCHESTRATOR_SYSTEM_PROMPT,
    TOOLGEN_VALIDATOR_SYSTEM_PROMPT,
)
from .controller_toolgen import ControllerToolgenMixin
from .controller_tools import ControllerToolsMixin
from .tool_registry import ToolMetadata, ToolResult, get_registry
from .toolgen_debug_logger import ToolgenDebugLogger, toolgen_debug_enabled
from src.toolgen.config import get_toolgen_pipeline_config
from src.toolgen.prompts import get_toolgen_system_prompt
from src.toolgen.prompting.build_task_pack import build_task_pack
from src.toolgen.pipelines import build_toolgen_pipeline


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
        tool_match_min_score: float = 0.55,
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
        self._episode_task_sig_cache: dict[tuple[str, str], dict[str, str]] = {}
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
        self._orchestrator_system_prompt = COMBINED_ORCHESTRATOR_SYSTEM_PROMPT
        self._tool_invoker_prompt = TOOL_INVOKER_SYSTEM_PROMPT

        output_dir_override = get_output_dir_override()
        if output_dir_override:
            tool_registry_root = str(Path(output_dir_override) / "tool_library")
        else:
            run_stamp = get_predefined_timestamp_structure()["TIMESTAMP"]
            tool_registry_root = str(Path("outputs") / run_stamp / "tool_library")

        tool_registry_root = os.path.abspath(tool_registry_root)
        pipeline_config = get_toolgen_pipeline_config(tool_registry_root)
        tool_registry_path = os.path.abspath(pipeline_config.registry_dir)
        self._registry_dir = tool_registry_path
        self._toolgen_pipeline_name = pipeline_config.pipeline
        self._toolgen_name_prefix = pipeline_config.name_prefix
        self._toolgen_agg_n = pipeline_config.agg_n
        self._toolgen_registry_root = os.path.abspath(pipeline_config.registry_root)
        self._toolgen_registry_dir = tool_registry_path
        self._toolgen_registry_root_from_env = pipeline_config.registry_root_from_env
        if self._toolgen_pipeline_name == "aggregate3":
            print(
                "[ToolGen] aggregate3 pipeline enabled; registry_dir="
                f"{self._toolgen_registry_dir}"
            )
        self._toolgen_preboot_envs: set[str] = set()
        self._toolgen_agg_bootstrapped_envs: set[str] = set()
        self._toolgen_preaggregate_envs: set[str] = set()
        self._force_toolgen_always_on = (
            os.getenv("FORCE_TOOLGEN_ALWAYS_ON", "0") == "1"
        )
        self._toolgen_off = os.getenv("TOOLGEN_OFF", "1") != "0"
        if self._toolgen_pipeline_name == "aggregate3":
            # Hard-enable ToolGen for aggregate3 runs to avoid env loss in subprocesses.
            self._force_toolgen_always_on = True
        if self._force_toolgen_always_on:
            self._toolgen_off = False

        base_cfg = dict(inference_config_dict) if inference_config_dict else {}
        for k in ("tools", "tool_choice", "functions", "function_call"):
            base_cfg.pop(k, None)

        base_cfg["tool_choice"] = "none"
        base_cfg["toolgen_extract_tool_calls"] = True
        base_cfg["ollama_force_tool_calls"] = False
        toolgen_timeout_s = os.getenv("LIFELONG_TOOLGEN_TIMEOUT_S", "").strip()
        if toolgen_timeout_s:
            base_cfg["request_timeout_s"] = toolgen_timeout_s
        toolgen_system_prompt = get_toolgen_system_prompt(
            self._toolgen_pipeline_name, environment_label
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
        validator_cfg = dict(base_cfg)
        validator_cfg["response_format"] = {"type": "json_object"}
        self._toolgen_validator_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=TOOLGEN_VALIDATOR_SYSTEM_PROMPT,
            inference_config_dict={
                **validator_cfg,
                "temperature": 0.0,
            },
            agent_name="toolgen_validator",
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
                system_prompt=COMBINED_ORCHESTRATOR_SYSTEM_PROMPT,
                inference_config_dict={
                    **orchestrator_cfg,
                    "temperature": 0.0,
                },
                agent_name="top_orchestrator",
            )
            self._tool_orchestrator_agent = None
            self._tool_invoker_agent = LanguageModelAgent(
                language_model=language_model,
                system_prompt=TOOL_INVOKER_SYSTEM_PROMPT,
                inference_config_dict={
                    **orchestrator_cfg,
                    "allow_internal_tool_protocol": False,
                    "temperature": 0.0,
                },
                agent_name="tool_invoker",
            )

        self._registry = get_registry(tool_registry_path)
        if hasattr(self, "set_registry_dir"):
            try:
                self.set_registry_dir(self._registry_dir)
            except Exception:
                pass
        self._registry.set_run_snapshot(
            get_predefined_timestamp_structure()["TIMESTAMP"]
        )
        self._toolgen_pipeline = build_toolgen_pipeline(
            pipeline=self._toolgen_pipeline_name,
            agg_n=self._toolgen_agg_n,
            build_user_prompt=self._toolgen_request_prompt,
            compact_task=self._toolgen_compact_query,
            invoke_toolgen=self._toolgen_generate_from_prompt,
            system_prompt_selector=get_toolgen_system_prompt,
            name_prefix=self._toolgen_name_prefix,
        )
        if self._toolgen_pipeline is not None:
            try:
                setattr(self._toolgen_pipeline, "registry_dir", self._registry_dir)
                setattr(self._toolgen_pipeline, "output_dir", self._registry_dir)
            except Exception:
                pass
        if hasattr(self._toolgen_pipeline, "set_registry_dir"):
            try:
                self._toolgen_pipeline.set_registry_dir(self._registry_dir)
            except Exception:
                pass
        try:
            print(
                f"[PATH_CHECK] Production Registry Dir: {self._registry_dir}",
                file=sys.stderr,
                flush=True,
            )
            pipe_registry = getattr(self._toolgen_pipeline, "registry_dir", None)
            print(
                f"[PATH_CHECK] ToolGen internal path: {pipe_registry}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        self._tool_invocation_log_path: Optional[Path] = None
        self._generated_tools_log_path: Optional[Path] = None
        self._flow_session_log_path: Optional[Path] = None
        self._flow_full_log_path: Optional[Path] = None
        self._agent_system_prompt_path: Optional[Path] = None
        self._loop_log_path: Optional[Path] = None
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
                self._loop_log_path = (
                    Path(output_dir) / prefix_filename("orchestrator_loop.log")
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
                self._loop_log_path = (
                    Path("outputs") / run_id / "orchestrator_loop.log"
                )
        except Exception:
            self._tool_invocation_log_path = None
            self._generated_tools_log_path = None
            self._flow_session_log_path = None
            self._flow_full_log_path = None
            self._agent_system_prompt_path = None
            self._loop_log_path = None
        self._run_id: Optional[str] = None
        self._state_dir: Optional[str] = None
        try:
            run_id = None
            state_dir = None
            if self._generated_tools_log_path is not None:
                for parent in self._generated_tools_log_path.parents:
                    if parent.name.startswith("run_all_"):
                        run_id = parent.name
                        break
                run_id = run_id or self._generated_tools_log_path.parent.name
                state_dir = str(self._generated_tools_log_path.parent / "tool_state")
            else:
                run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
                state_dir = str(Path("outputs") / run_id / "tool_state")
            self._run_id = run_id
            self._state_dir = state_dir
        except Exception:
            self._run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
            self._state_dir = str(Path("outputs") / self._run_id / "tool_state")
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
        self._tool_usage_logged: set[str] = set()
        self._outcome_scored: set[str] = set()
        self._failure_forged: set[str] = set()
        self._run_task_metadata: Optional[dict[str, Any]] = None
        self._last_solver_output: Optional[str] = None
        self._last_solver_context_key: Optional[str] = None
        self._solver_repeat_count = 0
        if hasattr(self._registry, "set_canonical_naming"):
            self._registry.set_canonical_naming(self._canonical_tool_naming)

        self._bootstrap_tools(bootstrap_tools or [])
        base_solver_prompt = (system_prompt or "").strip() or "You are a helpful assistant."
        self._solver_system_prompt = f"{base_solver_prompt}\n\n{SOLVER_SYSTEM_PROMPT}".strip()
        if getattr(self, "_language_model_agent", None) is not None:
            self._language_model_agent._system_prompt = self._solver_system_prompt
        if getattr(self, "_orchestrator_agent", None) is not None:
            self._orchestrator_agent._system_prompt = self._orchestrator_system_prompt
        if getattr(self, "_tool_invoker_agent", None) is not None:
            self._tool_invoker_agent._system_prompt = self._tool_invoker_prompt


    def _scoped_run_id_state_dir(
        self,
        *,
        task_text_full: str,
        asked_for: str,
        sample_index: Optional[str | int] = None,
        task_name: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """
        Returns (run_id, state_dir, task_sig) that are STABLE for a given episode.

        Problem: asked_for and even task_text_full can drift across steps (LLM rephrasing),
        which causes task_sig/run_id/state_dir to change mid-sample.

        Fix: cache per (task_name, sample_index) so the signature is pinned for the episode.
        Also remove asked_for from the signature source (too unstable).
        """
        # Include sample_index to prevent collisions across samples in a batch run.
        si = "" if sample_index is None else str(sample_index)

        # Prefer explicit task_name; fall back to environment label if not provided.
        tn = (task_name or self._run_task_label or self._environment_label or "task").strip()

        cache_key = (tn, si)
        cached = getattr(self, "_episode_task_sig_cache", None)
        if cached is None:
            self._episode_task_sig_cache = {}
            cached = self._episode_task_sig_cache

        if cache_key in cached:
            entry = cached[cache_key]
            try:
                task_text_hash = hashlib.sha1(task_text_full.encode("utf-8")).hexdigest()[:12]
                self._log_flow_event(
                    "run_scope",
                    cached=True,
                    task_text_hash=task_text_hash,
                    task_sig=entry["task_sig"],
                    run_id=entry["run_id"],
                    state_dir=entry["state_dir"],
                    task_name=tn,
                    sample_index=si,
                )
            except Exception:
                pass
            return entry["run_id"], entry["state_dir"], entry["task_sig"]

        # IMPORTANT: do NOT include asked_for (it drifts) in the signature.
        # You can include task_text_full, but it can also drift; the cache pins the result anyway.
        task_sig_src = f"{self._environment_label}||{tn}||{si}||{task_text_full}"
        task_sig = hashlib.sha1(task_sig_src.encode("utf-8")).hexdigest()[:12]

        run_id = f"{self._run_id or 'run'}_{task_sig}"
        base_state_dir = Path(self._state_dir or "/tmp/state")
        state_dir = str(base_state_dir / task_sig)

        cached[cache_key] = {"run_id": run_id, "state_dir": state_dir, "task_sig": task_sig}
        try:
            task_text_hash = hashlib.sha1(task_text_full.encode("utf-8")).hexdigest()[:12]
            self._log_flow_event(
                "run_scope",
                cached=False,
                task_text_hash=task_text_hash,
                task_sig=task_sig,
                run_id=run_id,
                state_dir=state_dir,
                task_name=tn,
                sample_index=si,
            )
        except Exception:
            pass
        return run_id, state_dir, task_sig



    def _default_payload_dict(
        self,
        *,
        task_text: str,
        candidate_output: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
        sample_index: Optional[str | int] = None,
        task_name: Optional[str] = None,
    ) -> dict[str, Any]:

        task_text_full = task_text or ""
        if chat_history is not None:
            user_items = [
                item
                for item in self._history_items(chat_history)
                if item.role == Role.USER and (item.content or "").strip()
            ]
            if user_items:
                first_user = (user_items[0].content or "").strip()
                last_user = (user_items[-1].content or "").strip()
                if first_user and last_user and last_user not in first_user:
                    task_text_full = f"{first_user}\n\n{last_user}"
                else:
                    task_text_full = first_user or last_user or task_text_full

        # Prefer explicit params; fall back to metadata if not provided
        si = sample_index
        if si is None:
            si = getattr(self, "_run_task_metadata", {}).get("sample_index")

        tn = task_name
        if not tn:
            tn = getattr(self, "_run_task_metadata", {}).get("task_name") or self._environment_label

        run_id, state_dir, task_sig = self._scoped_run_id_state_dir(
            task_text_full=task_text_full,
            asked_for="",
            sample_index=si,
            task_name=tn,
        )

        payload: dict[str, Any] = {
            "task_text": task_text_full,
            "constraints": [],
            "output_contract": {},
            "draft_response": None,
            "candidate_output": candidate_output,
            "env_observation": None,
            "run_id": run_id,
            "state_dir": state_dir,
        }
        if chat_history is not None:
            payload["chat_history"] = self._toolgen_render_history(chat_history, max_chars_per_item=None)
        return payload

    def _parse_action_line(self, content: str) -> Optional[tuple[str, list[str]]]:
        text = (content or "").strip()
        if not text:
            return None
        match = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?", text)
        if not match:
            return None
        name = match.group(1).strip()
        args_raw = (match.group(2) or "").strip()
        if not args_raw:
            return name, []
        parts = [part.strip() for part in args_raw.split(",")]
        args = [part for part in parts if part]
        return name, args

    def _parse_env_observation(self, content: str) -> tuple[Optional[str], Optional[str], bool]:
        text = (content or "").strip()
        if not text:
            return None, None, False
        error = None
        output = None
        if "Error:" in text:
            error = text.split("Error:", 1)[1].strip()
        elif "error:" in text:
            error = text.split("error:", 1)[1].strip()
        if "Observation:" in text:
            output = text.split("Observation:", 1)[1].strip()
        else:
            output = text
        ok = error is None
        return output, error, ok

    def _build_structured_trace(
        self, chat_history: ChatHistory
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        steps: list[dict[str, Any]] = []
        pending: Optional[dict[str, Any]] = None
        for item in self._history_items(chat_history):
            content = item.content or ""
            if item.role == Role.AGENT:
                parsed = self._parse_action_line(content)
                if parsed:
                    action, args = parsed
                    pending = {"action": action, "args": args}
            elif item.role == Role.USER and pending:
                output, error, ok = self._parse_env_observation(content)
                step = {
                    "action": pending.get("action"),
                    "args": pending.get("args") or [],
                    "ok": ok,
                    "output": output,
                    "error": error,
                }
                steps.append(step)
                pending = None

        loop_detected = False
        repeat_count = 1
        if len(steps) >= 2:
            last = steps[-1]
            prev = steps[-2]
            if (
                last.get("action") == prev.get("action")
                and last.get("args") == prev.get("args")
                and last.get("output") == prev.get("output")
                and last.get("error") == prev.get("error")
            ):
                loop_detected = True
                repeat_count = 2
                for idx in range(len(steps) - 3, -1, -1):
                    cur = steps[idx]
                    if (
                        cur.get("action") == last.get("action")
                        and cur.get("args") == last.get("args")
                        and cur.get("output") == last.get("output")
                        and cur.get("error") == last.get("error")
                    ):
                        repeat_count += 1
                    else:
                        break

        last_obs = dict(steps[-1]) if steps else None
        if last_obs is not None:
            last_obs["loop_detected"] = loop_detected
            last_obs["repeat_count"] = repeat_count
        return steps, last_obs

    def _detect_derailment(
        self,
        structured_trace: list[dict[str, Any]],
        last_obs: Optional[dict[str, Any]],
    ) -> Optional[dict[str, str]]:
        """Heuristic derailment detector.

        Returns ``None`` when no derailment is detected, or a dict with
        ``type``, ``severity`` ("high"/"medium"), and ``message`` keys.
        Checked in priority order — first match wins.
        """
        if not structured_trace:
            return None

        # 1. action_loop — last 3 steps have identical (action, args)
        if len(structured_trace) >= 3:
            tail3 = structured_trace[-3:]
            keys = [(s.get("action"), str(s.get("args"))) for s in tail3]
            if keys[0] == keys[1] == keys[2]:
                return {
                    "type": "action_loop",
                    "severity": "high",
                    "message": (
                        f"Agent repeated identical action 3 times: "
                        f"{tail3[0].get('action')}({tail3[0].get('args')})"
                    ),
                }

        # 2. error_streak — last 2 steps both failed
        if len(structured_trace) >= 2:
            tail2 = structured_trace[-2:]
            if tail2[0].get("ok") is False and tail2[1].get("ok") is False:
                return {
                    "type": "error_streak",
                    "severity": "high",
                    "message": (
                        f"Last 2 actions both failed: "
                        f"{tail2[0].get('action')}, {tail2[1].get('action')}"
                    ),
                }

        # 3. stall_no_new_vars — last 3 ok steps produced no new Variable #N
        if len(structured_trace) >= 3:
            tail3 = structured_trace[-3:]
            all_ok = all(s.get("ok") is True for s in tail3)
            if all_ok:
                has_new_var = any(
                    re.search(r"Variable\s+#\d+", str(s.get("output") or ""))
                    for s in tail3
                )
                if not has_new_var:
                    return {
                        "type": "stall_no_new_vars",
                        "severity": "medium",
                        "message": "Last 3 successful actions produced no new variables.",
                    }

        # 4. excessive_turns — long trace with no tool assistance
        if len(structured_trace) >= 12:
            tool_used = getattr(self, "_tool_invoked_in_last_inference", False)
            if not tool_used:
                return {
                    "type": "excessive_turns",
                    "severity": "medium",
                    "message": (
                        f"Trace has {len(structured_trace)} steps with no tool assistance."
                    ),
                }

        return None

    def _truncate_for_log(
        self, value: Any, max_chars: int = 400, max_list: int = 8
    ) -> Any:
        if isinstance(value, str):
            if len(value) <= max_chars:
                return value
            return value[:max_chars] + f"...(+{len(value) - max_chars} chars)"
        if isinstance(value, list):
            if len(value) <= max_list:
                return [self._truncate_for_log(v, max_chars, max_list) for v in value]
            head = [self._truncate_for_log(v, max_chars, max_list) for v in value[:max_list]]
            head.append(f"...(+{len(value) - max_list} items)")
            return head
        if isinstance(value, dict):
            return {k: self._truncate_for_log(v, max_chars, max_list) for k, v in value.items()}
        return value

    def _run_dir_from_state_dir(self, state_dir: Optional[str]) -> Optional[Path]:
        if not state_dir:
            return None
        try:
            path = Path(state_dir).resolve()
        except Exception:
            return None
        if path.parent is None or path.parent.parent is None:
            return None
        return path.parent.parent

    def _get_run_dir(self) -> Optional[Path]:
        if self._generated_tools_log_path:
            return self._generated_tools_log_path.parent
        return None

    def _record_tool_usage(
        self,
        *,
        tool_name: Optional[str],
        run_id: Optional[str],
        state_dir: Optional[str],
    ) -> None:
        if not tool_name or not run_id:
            return
        meta = self._get_run_task_metadata()
        task_name = meta.get("task_name")
        sample_index = meta.get("sample_index")
        if task_name is None or sample_index is None:
            return
        key = f"{task_name}|{sample_index}|{run_id}|{tool_name}"
        if key in self._tool_usage_logged:
            return
        run_dir = self._run_dir_from_state_dir(state_dir) or self._get_run_dir()
        if run_dir is None:
            return
        usage_path = run_dir / "tool_usage.jsonl"
        payload = {
            "task_name": task_name,
            "sample_index": sample_index,
            "run_id": run_id,
            "tool_name": tool_name,
            "ts": time.time(),
            # Advisory metadata for granular blame assignment
            "confidence_score": getattr(self, "_last_tool_confidence", None),
            "recommendation": getattr(self, "_last_tool_recommendation", None),
            "solver_followed_recommendation": getattr(
                self, "_solver_followed_recommendation", None
            ),
        }
        try:
            with open(usage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
            self._tool_usage_logged.add(key)
        except Exception:
            return

    def _apply_outcome_penalties(self) -> None:
        run_dir = self._get_run_dir()
        if run_dir is None:
            return
        runs_path = run_dir / "runs.json"
        usage_path = run_dir / "tool_usage.jsonl"
        processed_path = run_dir / "tool_outcome_marks.jsonl"
        if not runs_path.exists() or not usage_path.exists():
            return

        usage_map: dict[str, dict[str, Any]] = {}
        try:
            with open(usage_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    task_name = entry.get("task_name")
                    sample_index = entry.get("sample_index")
                    tool_name = entry.get("tool_name")
                    if task_name is None or sample_index is None or not tool_name:
                        continue
                    key = f"{task_name}|{sample_index}"
                    usage_map[key] = {
                        "tool_name": tool_name,
                        "confidence_score": entry.get("confidence_score"),
                        "solver_followed_recommendation": entry.get(
                            "solver_followed_recommendation"
                        ),
                    }
        except Exception:
            return

        if processed_path.exists():
            try:
                with open(processed_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._outcome_scored.add(line)
            except Exception:
                pass

        try:
            with open(runs_path, "r", encoding="utf-8") as f:
                runs_data = json.load(f)
        except Exception:
            return
        if not isinstance(runs_data, list):
            return

        new_marks: list[str] = []
        for entry in runs_data:
            if not isinstance(entry, dict):
                continue
            task_name = entry.get("task_name")
            sample_index = entry.get("sample_index")
            if task_name is None or sample_index is None:
                continue
            outcome = None
            eval_record = entry.get("evaluation_record")
            if isinstance(eval_record, Mapping):
                outcome = eval_record.get("outcome")
            if outcome is None:
                outcome = entry.get("outcome")
            if outcome is None:
                outcome = entry.get("sample_status")
            if outcome != "incorrect":
                continue
            usage_entry = usage_map.get(f"{task_name}|{sample_index}")
            if not usage_entry:
                continue
            tool_name = usage_entry["tool_name"]
            key = f"{task_name}|{sample_index}|{tool_name}|{outcome}"
            if key in self._outcome_scored:
                continue

            # Granular blame assignment based on advisory metadata
            should_penalize = True  # default: penalize (backward compat)
            confidence = usage_entry.get("confidence_score")
            solver_followed = usage_entry.get("solver_followed_recommendation")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = None
            if confidence is not None:
                if confidence >= 0.7 and solver_followed is False:
                    # High confidence recommendation ignored by solver → solver fault
                    should_penalize = False
                elif confidence < 0.5:
                    # Low confidence → ambiguous, don't penalize either party
                    should_penalize = False

            if should_penalize:
                self._registry.record_task_outcome(tool_name, success=False)
            self._outcome_scored.add(key)
            new_marks.append(key)

        if new_marks:
            try:
                with open(processed_path, "a", encoding="utf-8") as f:
                    for key in new_marks:
                        f.write(key + "\n")
            except Exception:
                pass


    # ------------------------------------------------------------------
    # Dynamic quality feedback
    # ------------------------------------------------------------------
    def update_tool_quality_for_outcome(self, *, success: bool) -> None:
        """Reward or penalize all tools invoked in the current sample.

        Called from the experiment loop after the sample outcome is known.
        +0.1 for correct, -0.2 for incorrect (clamped to [0, 10]).
        Persists immediately to metadata.json.
        """
        if not hasattr(self, "_registry"):
            return
        tool_names: set[str] = set()
        for key in self._tool_usage_logged:
            parts = key.split("|")
            if len(parts) >= 4:
                tool_names.add(parts[3])
        for name in tool_names:
            try:
                self._registry.record_task_outcome(name, success=success)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reflection Forge: post-task failure tool generation
    # ------------------------------------------------------------------
    def generate_tool_from_failure(
        self,
        session,
        *,
        agent_output: Optional[Any] = None,
        expected_answer: Optional[Any] = None,
    ) -> None:
        """Generate a tool from a failed task session (Reflection Forge).

        Called after ``task.complete(session)`` when the evaluation outcome
        is ``incorrect``.  Builds an enriched ToolGen query from the
        session trace and triggers tool generation.
        """
        if self._toolgen_off and not getattr(self, "_force_toolgen_always_on", False):
            return

        chat_history = getattr(session, "chat_history", None)
        if chat_history is None:
            return

        # Dedupe: one forge per (task_name, sample_index)
        task_name = str(getattr(session, "task_name", ""))
        sample_index = str(getattr(session, "sample_index", ""))
        forge_key = f"{task_name}|{sample_index}"
        if forge_key in self._failure_forged:
            return
        self._failure_forged.add(forge_key)

        # Extract the original task query
        user_items = [
            item for item in self._history_items(chat_history)
            if item.role == Role.USER
        ]
        task_query = user_items[1].content if len(user_items) >= 2 else ""

        # Build structured trace
        trace_steps, last_obs = self._build_structured_trace(chat_history)
        trace_tail = trace_steps[-5:] if trace_steps else []

        # Determine failure pattern
        patterns: list[str] = []
        if last_obs and last_obs.get("loop_detected"):
            patterns.append(f"loop_detected (repeat_count={last_obs.get('repeat_count')})")
        if last_obs and last_obs.get("ok") is False:
            patterns.append(f"last_action_failed: {last_obs.get('action')}")
        if last_obs and not str(last_obs.get("output") or "").strip():
            patterns.append("empty_result")
        if not patterns:
            patterns.append("incorrect_final_answer")

        def _stringify(value: Any, max_len: int = 800) -> str:
            if value is None:
                return ""
            try:
                if isinstance(value, (dict, list)):
                    text = json.dumps(value, ensure_ascii=True, default=str)
                else:
                    text = str(value)
            except Exception:
                text = str(value)
            return self._truncate(text, max_len)

        agent_output_value = agent_output
        if agent_output_value is None:
            task_output = getattr(session, "task_output", None)
            if isinstance(task_output, Mapping):
                if "answer" in task_output:
                    agent_output_value = task_output.get("answer")
                elif task_output:
                    agent_output_value = task_output
            elif task_output is not None:
                agent_output_value = task_output
        if agent_output_value is None:
            try:
                for item in reversed(self._history_items(chat_history)):
                    if item.role != Role.AGENT:
                        continue
                    content = (item.content or "").strip()
                    if content:
                        agent_output_value = content
                        break
            except Exception:
                agent_output_value = None

        expected_answer_value = expected_answer
        if expected_answer_value is None:
            expected_answer_value = getattr(session, "expected_answer", None)

        agent_output_text = (
            _stringify(agent_output_value, 800) or "unknown"
        )
        expected_answer_text = (
            _stringify(expected_answer_value, 800) or "unknown"
        )

        enriched_query = (
            f"REFLECTION FORGE: The agent failed this task.\n"
            f"TASK: {(task_query or '')[:500]}\n"
            f"AGENT_OUTPUT: {agent_output_text}\n"
            f"EXPECTED_ANSWER (use to reverse-engineer the general ontology "
            f"path — do NOT hardcode): {expected_answer_text}\n"
            f"TRACE SUMMARY (last {len(trace_tail)} steps): "
            f"{json.dumps(trace_tail, ensure_ascii=True, default=str)[:1500]}\n"
            f"FAILURE PATTERN: {', '.join(patterns)}\n"
            f"DESIRED TOOL: A parametric MACRO tool that reverse-engineers the "
            f"correct ontology path and produces the expected answer "
            f"deterministically. It must dynamically extract entities from "
            f"payload['task_text']. Do NOT generate an advisory tool."
        )
        reflection_header = (
            "=== POST-TASK REFLECTION ===\n"
            "The agent failed the following task.\n"
            f"Task: {task_query}\n"
            f"Agent Output: {agent_output_text}\n"
            f"Expected Correct Answer (use ONLY to reverse-engineer the "
            f"general ontology path — do NOT hardcode this value): "
            f"{expected_answer_text}\n"
            "Review the trace below to see where the agent took the wrong "
            "ontology path. You MUST write a 'Macro Tool' (Expert Router) "
            "that reads the trace and entities from the payload, determines "
            "the correct multi-step ontology path using internal logic, and "
            "returns precise action directives in answer_recommendation. "
            "The tool CANNOT query the KG directly (no network, stdlib only). "
            "Do NOT write an advisory tool.\n"
            "MACRO TOOL DEFINITION: In this offline environment, tools "
            "CANNOT make network requests. A Macro Tool acts as an Expert "
            "Router. It reads the trace and returns the EXACT next action "
            "string (e.g., 'Action: get_neighbors(#0, base.dog.breeds)') "
            "in its recommendation. Do not write dummy network code.\n"
            "CRITICAL ABSTRACTION RULE: You are generating a tool for a "
            "class of problems, not a specific task. You MUST write "
            "PARAMETRIC code. Do NOT hardcode entities (like 'Naloxone', "
            "'goats', 'cows') into the script. Your script must dynamically "
            "extract entities from 'payload[\"task_text\"]' or "
            "'payload[\"asked_for\"]' and execute the search abstractly. "
            "Give the tool a generic, descriptive name "
            "(e.g., 'multi_entity_intersection_macro_tool').\n"
            "============================\n\n"
        )

        self._append_loop_log(
            f"REFLECTION FORGE | sample={sample_index} patterns={patterns}"
        )

        try:
            self._append_generated_tools_log(
                {
                    "event": "reflection_forge_start",
                    "task_name": task_name,
                    "sample_index": sample_index,
                    "failure_patterns": patterns,
                }
            )
        except Exception:
            pass

        try:
            # NOTE: force_strict=True ensures the full 8-round gen→val→repair
            # loop runs.  No relaxed-mode bypass for reflection forges.
            if getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3":
                env_name = self._resolved_environment_label()
                env_contract = ""
                context = getattr(self, "_toolgen_agg_context", None)
                if isinstance(context, Mapping):
                    env_contract = str(context.get("env_contract") or "")
                if not env_contract:
                    try:
                        for item in self._history_items(chat_history):
                            if item.role == Role.USER:
                                env_contract = (item.content or "").strip()
                                break
                    except Exception:
                        env_contract = ""
                user_prompt = build_task_pack(
                    env_name, env_contract, [enriched_query]
                )
                final_user_prompt = reflection_header + user_prompt
                exec_payload = self._build_toolgen_execution_payload(
                    task_text=enriched_query,
                    trace=trace_tail,
                    failure_context=last_obs.get("output") if isinstance(last_obs, dict) else "",
                    active_variables=None,
                )
                prev_exec_payload = getattr(self, "_toolgen_execution_payload", None)
                setattr(self, "_toolgen_execution_payload", exec_payload)
                system_prompt = get_toolgen_system_prompt("aggregate3", env_name)
                try:
                    result = self._toolgen_generate_from_prompt(
                        user_prompt=final_user_prompt,
                        system_prompt=system_prompt,
                        chat_history=chat_history,
                        name_prefix=getattr(self, "_toolgen_name_prefix", ""),
                        force_strict=True,
                        force_max_rounds=8,
                    )
                finally:
                    setattr(self, "_toolgen_execution_payload", prev_exec_payload)
            else:
                env_name = self._resolved_environment_label()
                env_contract = ""
                try:
                    for item in self._history_items(chat_history):
                        if item.role == Role.USER:
                            env_contract = (item.content or "").strip()
                            break
                except Exception:
                    env_contract = ""
                user_prompt = build_task_pack(env_name, env_contract, [enriched_query])
                final_user_prompt = reflection_header + user_prompt
                system_prompt = get_toolgen_system_prompt(
                    getattr(self, "_toolgen_pipeline_name", "baseline"),
                    env_name,
                )
                prev_exec_payload = getattr(self, "_toolgen_execution_payload", None)
                exec_payload = self._build_toolgen_execution_payload(
                    task_text=enriched_query,
                    trace=trace_tail,
                    failure_context=last_obs.get("output") if isinstance(last_obs, dict) else "",
                    active_variables=None,
                )
                setattr(self, "_toolgen_execution_payload", exec_payload)
                try:
                    result = self._toolgen_generate_from_prompt(
                        user_prompt=final_user_prompt,
                        system_prompt=system_prompt,
                        chat_history=chat_history,
                        name_prefix=getattr(self, "_toolgen_name_prefix", ""),
                        force_strict=True,
                        force_max_rounds=8,
                    )
                finally:
                    setattr(self, "_toolgen_execution_payload", prev_exec_payload)

            tool_name = None
            if result is not None:
                if isinstance(result, Mapping):
                    tool_name = result.get("tool_name") or result.get("name")
                else:
                    tool_name = getattr(result, "name", None)

            self._append_loop_log(
                f"REFLECTION FORGE RESULT | tool={tool_name or 'FAILED'}"
            )
            try:
                self._append_generated_tools_log(
                    {
                        "event": "reflection_forge_result",
                        "task_name": task_name,
                        "sample_index": sample_index,
                        "tool_name": tool_name,
                        "success": tool_name is not None,
                    }
                )
            except Exception:
                pass

            # Auto-refresh registry so the new tool is embedded and
            # available for the very next sample.
            if tool_name is not None:
                if getattr(self, "_registry", None) is not None and hasattr(
                    self._registry, "refresh"
                ):
                    try:
                        self._registry.refresh()
                    except Exception:
                        pass
        except Exception:
            self._append_loop_log("REFLECTION FORGE | ERROR (exception during toolgen)")

    def _orchestrated_inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.inference(chat_history)

        working_history = self._clone_history(self._prune_for_current_task(chat_history))
        self._tool_invoked_in_last_inference = False
        self._apply_outcome_penalties()
        # --- TROUBLESHOOTING LOGS ---
        try:
            print(
                f"[DEBUG_PATH] Controller Registry Dir: {getattr(self, '_registry_dir', None)}",
                file=sys.stderr,
                flush=True,
            )
            if getattr(self, "_registry", None) is not None:
                base_path = getattr(self._registry, "base_path", "Unknown")
                print(
                    f"[DEBUG_PATH] Orchestrator is watching: {base_path}",
                    file=sys.stderr,
                    flush=True,
                )
            pipeline = getattr(self, "_toolgen_pipeline", None)
            if pipeline is not None:
                pipe_registry = getattr(pipeline, "registry_dir", None)
                print(
                    f"[DEBUG_PATH] ToolGen is writing to: {pipe_registry}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:
            pass
        if getattr(self, "_registry", None) is not None and hasattr(self._registry, "refresh"):
            try:
                self._registry.refresh()
            except Exception:
                pass

        user_items = [item for item in self._history_items(working_history) if item.role == Role.USER]
        if len(user_items) >= 2:
            original_query = user_items[1].content
        else:
            orig_last_user = self._get_last_user_item(working_history)
            original_query = orig_last_user.content if orig_last_user else ""
        task_query = (original_query or "").strip()

        print("\n[DEBUG] --- TURN START ---", file=sys.stderr, flush=True)
        print("[DEBUG] Prompt Name Tracing active.", file=sys.stderr, flush=True)
        # Pre-boot ToolGen must run before any orchestrator decision on first loop.
        try:
            preboot_tasks = getattr(self, "_preboot_tasks", None)
            if isinstance(preboot_tasks, list) and preboot_tasks:
                self._toolgen_prebootstrap_once(
                    task_query, working_history, tasks=preboot_tasks
                )
                self._preboot_tasks = []
            else:
                self._toolgen_prebootstrap_once(task_query, working_history)
            if getattr(self, "_registry", None) is not None and hasattr(
                self._registry, "refresh"
            ):
                try:
                    self._registry.refresh()
                except Exception:
                    pass
        except Exception:
            pass
        forced_decision: Optional[Mapping[str, Any]] = None
        try:
            forced_decision = self._orchestrate_decision(
                task_query=task_query,
                chat_history=chat_history,
            )
        except TypeError:
            forced_decision = self._orchestrate_decision(task_query, chat_history)
        print(
            f"[DEBUG] Orchestrator Identity Verified. Decision: {forced_decision}",
            file=sys.stderr,
            flush=True,
        )
        if isinstance(forced_decision, Mapping):
            action_name = forced_decision.get("action")
            if action_name == "request_new_tool":
                print(
                    f"[LATM_FLOW] Desired Behavior for Forge: "
                    f"{forced_decision.get('desired_behavior')}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"[!] ESCAPE HATCH TRIGGERED: "
                    f"{forced_decision.get('reasoning')}",
                    file=sys.stderr,
                    flush=True,
                )
                escape_result = self._handle_escape_hatch(
                    forced_decision, task_query, working_history
                )
                if isinstance(escape_result, Mapping):
                    solver_recommendation = str(
                        escape_result.get("observation") or ""
                    )
                    forced_decision = self._orchestrate_decision(
                        task_query,
                        working_history,
                        solver_recommendation=solver_recommendation,
                    )

        _loop_start = time.perf_counter()
        turn_start_time = time.time()
        _hist_len = chat_history.get_value_length()
        _task_meta = self._get_run_task_metadata() or {}
        turn_id = _task_meta.get("sample_index") if isinstance(_task_meta, dict) else None
        if turn_id is None:
            turn_id = _hist_len
        self._last_turn_data = {"turn_id": turn_id, "orchestrator_decisions": []}
        self._append_loop_log(
            f"{'=' * 60}\n"
            f"TURN START | env={self._resolved_environment_label()} "
            f"sample={_task_meta.get('sample_index', '?')} "
            f"history_len={_hist_len} "
            f"toolgen_off={self._toolgen_off} "
            f"force_toolgen_on={self._force_toolgen_always_on}\n"
            f"  query: {task_query[:120]}{'...' if len(task_query) > 120 else ''}"
        )

        def _record_tool_error(stage: str, message: str, tb: Optional[str] = None) -> None:
            self._trace("tool_pipeline_error", f"{stage}: {message}")
            if tb:
                self._trace("tool_pipeline_traceback", tb)

        tool_traces: list[dict[str, Any]] = []
        tool_error: Optional[str] = None
        tool_result_injected = False
        tool_agent_traced = False
        solver_sidecar: list[str] = []
        last_tool_result: Optional[ToolResult] = None
        last_tool_actions_spec: Optional[Mapping[str, Any]] = None
        solver_recommendation: Optional[str] = None
        last_tool_next_action: Optional[str] = None
        last_tool_run_id: Optional[str] = None
        last_tool_state_dir: Optional[str] = None
        last_tool_name: Optional[str] = None
        self._last_tool_confidence: Optional[float] = None
        self._last_tool_recommendation: Optional[str] = None
        self._solver_followed_recommendation: Optional[bool] = None
        last_structured_trace: list[dict[str, Any]] = []
        last_structured_obs: Optional[dict[str, Any]] = None

        # Sidecar paradigm: orchestrator must intercept observations before any solver pass.
        init_solver = False
        solver_recommendation = ""
        initial_solver_output = ""
        self._toolgen_last_recommendation = ""

        # Phase 1: Extract latest observation immediately (before any solver call)
        pre_orch_trace, pre_orch_obs = self._build_structured_trace(working_history)
        observation_triggers: list[dict[str, Any]] = []
        if pre_orch_obs is not None:
            obs_output = str(pre_orch_obs.get("output") or "")
            obs_error = pre_orch_obs.get("error")
            if obs_output.startswith("[") or "[" in obs_output[:200]:
                try:
                    bracket_start = obs_output.index("[")
                    bracket_end = obs_output.rindex("]") + 1
                    inner = obs_output[bracket_start + 1 : bracket_end - 1]
                    items = [x.strip() for x in inner.split(",") if x.strip()]
                    if len(items) > 15:
                        observation_triggers.append(
                            {
                                "type": "size_trigger",
                                "item_count": len(items),
                                "reason": f"observation has {len(items)} items (>15 threshold)",
                            }
                        )
                except (ValueError, IndexError):
                    pass
            if obs_error or (obs_output and "Error" in obs_output[:100]):
                observation_triggers.append(
                    {
                        "type": "error_trigger",
                        "error": obs_error or obs_output[:200],
                        "reason": "observation contains error",
                    }
                )

        # Derailment detection — broader heuristic over the full trace
        derailment = self._detect_derailment(pre_orch_trace, pre_orch_obs)
        if derailment is not None:
            observation_triggers.append(
                {
                    "type": "derailment_trigger",
                    "severity": derailment["severity"],
                    "reason": derailment["message"],
                }
            )

        if observation_triggers:
            self._append_loop_log(
                f"  triggers: {[t.get('type') for t in observation_triggers]}"
            )

        try:
            print(
                "\n[DEBUG] Entering Orchestrator Decision Phase",
                file=sys.stderr,
                flush=True,
            )
            if forced_decision is None:
                decision = self._orchestrate_decision(
                    task_query,
                    working_history,
                    solver_recommendation=solver_recommendation,
                    observation_triggers=observation_triggers,
                    last_observation=pre_orch_obs,
                )
            else:
                decision = dict(forced_decision)
            print(
                f"[DEBUG] Orchestrator Decision: {decision}",
                file=sys.stderr,
                flush=True,
            )
            action = decision.get("action", "no_tool")
            if action == "request_new_tool":
                decision["tool_name"] = "request_new_tool"
                action = "use_tool"
            if (
                self._toolgen_off
                and not self._force_toolgen_always_on
                and action in {"create_tool", "request_new_tool"}
            ):
                action = "use_tool"
            if isinstance(self._last_turn_data, dict):
                decision_record = dict(decision)
                decision_record["action"] = action
                decision_record["_stage"] = "initial"
                self._last_turn_data["orchestrator_decision"] = decision_record
                if isinstance(self._last_turn_data.get("orchestrator_decisions"), list):
                    self._last_turn_data["orchestrator_decisions"].append(decision_record)

            self._append_loop_log(
                f"  orchestrator: action={action} "
                f"tool={decision.get('tool_name') or '-'} "
                f"reason={self._truncate(str(decision.get('reason') or ''), 80)}"
            )

            # Phase 4: Escape hatch intercept — request_new_tool
            if (
                action == "use_tool"
                and str(decision.get("tool_name") or "").strip() == "request_new_tool"
            ):
                print(
                    f"[LATM_FLOW] Desired Behavior for Forge: "
                    f"{decision.get('desired_behavior')}",
                    file=sys.stderr,
                    flush=True,
                )
                self._append_loop_log("  >>> ESCAPE HATCH TRIGGERED <<<")
                escape_result = self._handle_escape_hatch(
                    decision, task_query, working_history
                )
                escape_message = ""
                if isinstance(escape_result, Mapping):
                    escape_message = str(escape_result.get("observation") or "")
                if not escape_message:
                    escape_message = str(escape_result)
                escape_status = "success" if "successfully created" in escape_message else "failure"
                self._append_loop_log(
                    {
                        "event": "escape_hatch_complete",
                        "turn": turn_id,
                        "status": escape_status,
                        "message": escape_message,
                    }
                )
                self._append_loop_log(
                    f"  escape_hatch: success={escape_result.get('success')} "
                    f"new_tool={escape_result.get('tool_name') or '-'}"
                )
                # Re-run the Orchestrator with the ToolGen observation so it
                # can see the refreshed catalog and select the new tool.
                print(
                    "\n[DEBUG] Entering Orchestrator Decision Phase (post-escape)",
                    file=sys.stderr,
                    flush=True,
                )
                decision = self._orchestrate_decision(
                    task_query,
                    working_history,
                    solver_recommendation=escape_result["observation"],
                    observation_triggers=observation_triggers,
                    last_observation=pre_orch_obs,
                )
                print(
                    f"[DEBUG] Orchestrator Decision (post-escape): {decision}",
                    file=sys.stderr,
                    flush=True,
                )
                action = decision.get("action", "no_tool")
                if action == "request_new_tool":
                    decision["tool_name"] = "request_new_tool"
                    action = "use_tool"
                if (
                    self._toolgen_off
                    and not self._force_toolgen_always_on
                    and action in {"create_tool", "request_new_tool"}
                ):
                    action = "use_tool"
                if isinstance(self._last_turn_data, dict):
                    decision_record = dict(decision)
                    decision_record["action"] = action
                    decision_record["_stage"] = "post_escape"
                    self._last_turn_data["orchestrator_decision"] = decision_record
                    if isinstance(
                        self._last_turn_data.get("orchestrator_decisions"), list
                    ):
                        self._last_turn_data["orchestrator_decisions"].append(
                            decision_record
                        )
                self._append_loop_log(
                    f"  orchestrator (post-escape): action={action} "
                    f"tool={decision.get('tool_name') or '-'}"
                )

            if action == "no_tool":
                if not init_solver:
                    solver_prompt = self._solver_prompt_no_tools()
                    self._write_agent_system_prompt("solver", solver_prompt)
                    history_text = self._toolgen_render_history(
                        working_history,
                        max_chars_per_item=1200,
                        preserve_first_user_n=2,
                    )
                    self._append_solver_io_log(
                        {
                            "event": "solver_input",
                            "stage": "no_tool",
                            "system_prompt": solver_prompt,
                            "history": history_text,
                            "sidecar": "",
                        }
                    )
                    solver_response = self._solver_inference_with_retry(
                        working_history, system_prompt=solver_prompt
                    )
                    self._append_loop_log(
                        {
                            "event": "solver_execution",
                            "turn": turn_id,
                            "has_sidecar": False,
                            "solver_response_preview": str(solver_response)[:150],
                        }
                    )
                    self._append_solver_io_log(
                        {
                            "event": "solver_output",
                            "stage": "no_tool",
                            "content": getattr(solver_response, "content", "") or "",
                        }
                    )
                    content = getattr(solver_response, "content", "") or ""
                else:
                    content = initial_solver_output
                self._trace("solver_result", content)
                self._log_flow_event(
                    "final_response",
                    chat_history=working_history,
                    content=content,
                )
                self._flush_tool_traces(tool_traces, content)
                self._append_loop_log(
                    {
                        "event": "turn_complete",
                        "turn": turn_id,
                        "total_turn_duration_s": round(time.time() - turn_start_time, 2),
                        "final_action": "no_tool",
                    }
                )
                _dur = time.perf_counter() - _loop_start
                self._append_loop_log(
                    f"  solver(no_tool): {self._truncate(content, 100)}\n"
                    f"TURN END | duration={_dur:.1f}s branch=no_tool\n"
                )
                return ChatHistoryItem(role=Role.AGENT, content=content)
            if action in {"use_tool", "create_tool"}:
                tool_agent_traced = True
                tool_action = action
                self._toolgen_last_orchestrator_reason = decision.get("reason")
                self._toolgen_last_orchestrator_gap = decision.get("insufficiency")
                self._toolgen_last_orchestrator_needed = decision.get("needed_capabilities")
                tool_suggestion = {
                    "tool_name": decision.get("tool_name"),
                    "reason": decision.get("reason"),
                }

                if tool_action == "create_tool":
                    self._append_loop_log("  create_tool: generating new tool via ToolGen...")
                    self._toolgen_last_recommendation = solver_recommendation
                    selected_tool = self._maybe_generate_tool_for_query(
                        task_query, working_history, allow_reuse=False, force=True
                    )
                    if selected_tool is None:
                        self._trace("registry_add_failed", "tool generation failed")
                        self._append_loop_log("  create_tool: FAILED (toolgen returned None)")
                        tool_error = tool_error or "tool generation failed"
                        _record_tool_error("create_tool", tool_error)
                        tool_result = ToolResult.failure(tool_error)
                        solver_sidecar.append(
                            self._format_tool_result("create_tool", tool_result)
                        )
                        tool_result_injected = True
                    else:
                        self._trace("registry_add", selected_tool.name)
                        self._append_loop_log(f"  create_tool: OK → {selected_tool.name}")
                        tool_suggestion["tool_name"] = selected_tool.name

                if not tool_result_injected:
                    tool_invoker = self._tool_invoker_decision(
                        task_query, working_history, suggestion=tool_suggestion
                    )
                    tool_name = tool_invoker.get("tool_name")
                    payload = tool_invoker.get("payload")
                    self._append_loop_log(
                        f"  tool_invoker: selected={tool_name or '-'}"
                    )
                    if tool_name:
                        args_auto_built = False
                        if not tool_result_injected:
                            contract = self._parse_tool_invoke_contract(tool_name)
                            invoke_with = contract.get("invoke_with") if contract else None
                            required_keys = list(contract.get("required") or []) if contract else []
                            optional_keys = list(contract.get("optional") or []) if contract else []
                            if not invoke_with:
                                tool_meta = self._get_tool_metadata(tool_name)
                                invoke_with = '{"args":[<RUN_PAYLOAD>], "kwargs":{}}'
                                if tool_meta:
                                    if not required_keys:
                                        required_keys = list(tool_meta.required_keys or [])
                                    if not optional_keys:
                                        optional_keys = list(tool_meta.optional_keys or [])
                            payload_map: dict[str, Any] = dict(payload) if isinstance(payload, Mapping) else {}
                            structured_trace, last_obs = self._build_structured_trace(working_history)
                            last_structured_trace = structured_trace
                            last_structured_obs = last_obs
                            payload_map["trace"] = structured_trace
                            if last_obs is not None:
                                payload_map["env_observation"] = last_obs
                            if isinstance(payload_map.get("actions_spec"), Mapping):
                                last_tool_actions_spec = payload_map.get("actions_spec")
                            payload_keys = sorted(str(k) for k in payload_map.keys())
                            try:
                                self._append_generated_tools_log(
                                    {
                                        "event": "tool_observation_payload",
                                        "tool_name": tool_name,
                                        "run_id": payload_map.get("run_id"),
                                        "trace_len": len(structured_trace),
                                        "trace_tail": self._truncate_for_log(structured_trace[-3:]),
                                        "env_observation": self._truncate_for_log(last_obs),
                                    }
                                )
                            except Exception:
                                pass
                            payload_for_tool: dict[str, Any] = {}
                            missing_keys: list[str] = []
                            for key in required_keys:
                                if key in payload_map:
                                    payload_for_tool[key] = payload_map[key]
                                else:
                                    missing_keys.append(key)
                            for key in optional_keys:
                                if key in payload_map:
                                    payload_for_tool[key] = payload_map[key]
                            try:
                                self._append_generated_tools_log(
                                    {
                                        "event": "tool_invoke_contract",
                                        "tool_name": tool_name,
                                        "invoke_with": invoke_with,
                                        "required_keys": required_keys,
                                        "optional_keys": optional_keys,
                                        "payload_keys": payload_keys,
                                        "payload_missing": missing_keys,
                                        "wrapper_shape": invoke_with,
                                    }
                                )
                            except Exception:
                                pass
                            if missing_keys:
                                run_id_hint = payload_map.get("run_id") if isinstance(payload_map, Mapping) else None
                                tool_error = (
                                    "missing_required_keys:"
                                    + ",".join(sorted(str(k) for k in missing_keys))
                                    + f"|tool={tool_name}|run_id={run_id_hint or ''}"
                                )
                                self._log_failed_invoke_event(
                                    tool_name=tool_name,
                                    reason=tool_error,
                                )
                                _record_tool_error("tool_invoker", tool_error)
                                tool_result = ToolResult.failure(tool_error)
                                solver_sidecar.append(
                                    self._format_tool_result("tool_invoker", tool_result)
                                )
                                tool_result_injected = True
                            else:
                                wrapped = self._wrap_payload_with_contract(
                                    invoke_with, payload_for_tool
                                )
                                if not wrapped:
                                    tool_error = "invalid_invoke_wrapper"
                                    self._log_failed_invoke_event(
                                        tool_name=tool_name,
                                        reason=tool_error,
                                    )
                                    _record_tool_error("tool_invoker", tool_error)
                                    tool_result = ToolResult.failure(tool_error)
                                    solver_sidecar.append(
                                        self._format_tool_result("tool_invoker", tool_result)
                                    )
                                    tool_result_injected = True
                                else:
                                    args, kwargs = wrapped
                                    if (
                                        isinstance(args, list)
                                        and len(args) == 1
                                        and isinstance(args[0], Mapping)
                                        and set(args[0].keys()) == {"payload"}
                                        and isinstance(args[0].get("payload"), Mapping)
                                    ):
                                        args = [dict(args[0].get("payload") or {})]
                                    tool_args = {"args": args, "kwargs": kwargs}
                                    self._log_flow_event(
                                        "tool_agent_input",
                                        chat_history=working_history,
                                        tool_name=tool_name,
                                        tool_args=tool_args,
                                        reason="tool_invoker",
                                    )
                                    # Now invoke
                                    tool_result = self._invoke_tool_by_payload(
                                        tool_name,
                                        tool_args,
                                        reason="tool_invoker",
                                        chat_history=working_history,
                                        args_auto_built=args_auto_built,
                                        decision_action=tool_action,
                                    )
                                    last_tool_result = tool_result
                                    _conf = None
                                    _rec_preview = ""
                                    if isinstance(tool_result.output, Mapping):
                                        _conf = tool_result.output.get("confidence_score")
                                        _rec_preview = self._truncate(
                                            str(tool_result.output.get("answer_recommendation") or ""), 80
                                        )
                                    self._append_loop_log(
                                        f"  tool_exec: success={tool_result.success} "
                                        f"confidence={_conf} "
                                        f"rec={_rec_preview or '-'}"
                                    )

                            if not tool_result_injected:
                                self._log_flow_event(
                                    "tool_agent_output",
                                    chat_history=working_history,
                                    tool_name=tool_name,
                                    success=tool_result.success,
                                    error=tool_result.error,
                                    output=tool_result.output,
                                )
                                if isinstance(tool_result.output, Mapping):
                                    caps = getattr(self, "_tool_confidence_caps", {})
                                    cap = caps.get(tool_name) if isinstance(caps, dict) else None
                                    if cap is not None:
                                        current = tool_result.output.get("confidence_score")
                                        if isinstance(current, (int, float)):
                                            tool_result.output["confidence_score"] = min(
                                                float(current), float(cap)
                                            )
                                        else:
                                            tool_result.output["confidence_score"] = float(cap)
                                    # Advisory paradigm fields
                                    self._last_tool_confidence = tool_result.output.get(
                                        "confidence_score"
                                    )
                                    self._last_tool_recommendation = tool_result.output.get(
                                        "answer_recommendation"
                                    )
                                    # Backward compat: legacy next_action extraction
                                    next_action = tool_result.output.get("next_action")
                                    if isinstance(next_action, Mapping):
                                        action_name = next_action.get("name")
                                        if not isinstance(action_name, str):
                                            action_name = next_action.get("action")
                                        if isinstance(action_name, str):
                                            last_tool_next_action = action_name
                                last_tool_name = tool_name
                                if isinstance(tool_args, dict):
                                    args_list = tool_args.get("args")
                                    if (
                                        isinstance(args_list, list)
                                        and args_list
                                        and isinstance(args_list[0], dict)
                                    ):
                                        last_tool_run_id = args_list[0].get("run_id")
                                        last_tool_state_dir = args_list[0].get("state_dir")
                                self._record_tool_usage(
                                    tool_name=last_tool_name,
                                    run_id=last_tool_run_id,
                                    state_dir=last_tool_state_dir,
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
                                # Route to advisory formatter if tool uses advisory schema
                                if self._is_advisory_result(tool_result):
                                    solver_sidecar.append(
                                        self._format_advisory_result(tool_name, tool_result)
                                    )
                                else:
                                    solver_sidecar.append(
                                        self._format_tool_result(tool_name, tool_result)
                                    )
                                tool_result_injected = True
                    else:
                        tool_error = "tool_invoker_missing_tool_name"
                        self._log_failed_invoke_event(
                            tool_name=None,
                            reason=tool_error,
                        )
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
        if last_structured_obs is not None:
            solver_sidecar.append(
                "STRUCTURED_OBSERVATION:\n"
                + json.dumps(last_structured_obs, ensure_ascii=True, default=str)
            )
        if last_structured_trace:
            solver_sidecar.append(
                "STRUCTURED_TRACE_TAIL:\n"
                + json.dumps(last_structured_trace[-3:], ensure_ascii=True, default=str)
            )

        for attempt in range(3):
            solver_prompt = self._solver_prompt_no_tools()
            if solver_sidecar:
                # Detect advisory vs legacy sidecar content
                sidecar_joined = "\n\n".join(solver_sidecar)
                has_advisory = any(
                    "TOOL_ADVISORY" in s or "Recommendation:" in s or "Confidence:" in s
                    for s in solver_sidecar
                )
                if has_advisory:
                    solver_prompt = (
                        solver_prompt
                        + "\n\n### TOOL ADVISORY (CRITICAL)\n"
                        + "A helper tool analyzed the data and provided this recommendation.\n"
                        + "You MUST prioritize this recommendation when choosing your next action.\n\n"
                        + sidecar_joined
                    )
                else:
                    solver_prompt = (
                        solver_prompt
                        + "\n\nINTERNAL TOOL CONTEXT:\n"
                        + "Tool results may include 'recommended_next_action' with args; consider using it if it fits.\n\n"
                        + sidecar_joined
                    )
            history_text = self._toolgen_render_history(
                working_history,
                max_chars_per_item=1200,
                preserve_first_user_n=2,
            )
            sidecar_text = "\n\n".join(solver_sidecar)
            solver_payload = {
                "system_prompt": solver_prompt,
                "history": history_text,
                "tool_error": tool_error,
                "tool_result_injected": tool_result_injected,
                "system_prompt_chars": len(solver_prompt),
                "history_chars": len(history_text),
                "history_items": len(self._history_items(working_history)),
                "sidecar_chars": len(sidecar_text),
                "attempt": attempt + 1,
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
            self._append_solver_io_log(
                {
                    "event": "solver_input",
                    "stage": "final_response",
                    "attempt": attempt + 1,
                    "system_prompt": solver_prompt,
                    "history": history_text,
                    "sidecar": sidecar_text,
                }
            )
            start_ts = time.perf_counter()
            solver_response = self._solver_inference_with_retry(
                working_history, system_prompt=solver_prompt
            )
            self._append_loop_log(
                {
                    "event": "solver_execution",
                    "turn": turn_id,
                    "has_sidecar": bool(solver_sidecar),
                    "solver_response_preview": str(solver_response)[:150],
                }
            )
            self._append_solver_io_log(
                {
                    "event": "solver_output",
                    "stage": "final_response",
                    "attempt": attempt + 1,
                    "content": getattr(solver_response, "content", "") or "",
                }
            )
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._log_flow_event(
                "solver_metrics",
                chat_history=working_history,
                payload={
                    "stage": "final_response",
                    "attempt": attempt + 1,
                    "solver_latency_ms": round(elapsed_ms, 2),
                    "system_prompt_chars": len(solver_prompt),
                    "history_chars": len(history_text),
                    "history_items": len(self._history_items(working_history)),
                    "sidecar_chars": len(sidecar_text),
                },
            )
            content = getattr(solver_response, "content", "") or ""
            if self._contains_internal_tool(content):
                self._trace("solver_result", content)
                working_history = self._safe_inject(working_history, solver_response)
                continue
            content = content
            invalid_action = False
            action_name = None
            allowed_actions: set[str] = set()
            match = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", content)
            if match:
                action_name = match.group(1)
                if action_name.strip().lower() in {"none", "null", "nil"}:
                    invalid_action = True
                actions_spec = (
                    last_tool_actions_spec
                    if isinstance(last_tool_actions_spec, Mapping)
                    else self._available_actions_spec()
                )
                if isinstance(actions_spec, Mapping) and actions_spec:
                    allowed_actions = {
                        str(name).strip().lower()
                        for name in actions_spec.keys()
                        if str(name).strip()
                    }
                    if action_name.lower() not in allowed_actions:
                        invalid_action = True
            elif "Action:" in content:
                invalid_action = True
            if invalid_action:
                allowed_hint = ""
                if allowed_actions:
                    allowed_hint = " Allowed actions: " + ", ".join(sorted(allowed_actions)) + "."
                solver_sidecar.append(
                    f"INVALID_ACTION: {action_name or 'missing'}.{allowed_hint}"
                )
                continue
            context_key = self._solver_context_key(working_history)
            if content == self._last_solver_output and context_key == self._last_solver_context_key:
                self._solver_repeat_count += 1
                if self._solver_repeat_count >= 2:
                    self._trace("solver_result", content)
                    self._log_flow_event(
                        "final_response",
                        chat_history=working_history,
                        content=content,
                    )
                    self._flush_tool_traces(tool_traces, content)
                    self._append_loop_log(
                        {
                            "event": "turn_complete",
                            "turn": turn_id,
                            "total_turn_duration_s": round(time.time() - turn_start_time, 2),
                            "final_action": action_name,
                        }
                    )
                    return ChatHistoryItem(role=Role.AGENT, content=content)
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
            if last_tool_next_action or content or self._last_tool_recommendation:
                executed_action = None
                match = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", content)
                if match:
                    executed_action = match.group(1)
                # Detect if solver followed the tool advisory
                self._solver_followed_recommendation = None
                if getattr(self, "_last_tool_recommendation", None) and content:
                    rec_lower = self._last_tool_recommendation.lower()
                    if "final answer" in rec_lower and "final answer" in content.lower():
                        self._solver_followed_recommendation = True
                    elif executed_action and executed_action.lower() in rec_lower:
                        self._solver_followed_recommendation = True
                    else:
                        self._solver_followed_recommendation = False
                if last_tool_next_action or executed_action or self._last_tool_recommendation:
                    self._log_flow_event(
                        "tool_advisory_compare",
                        tool_name=last_tool_name,
                        tool_next_action=last_tool_next_action,
                        tool_recommendation=self._last_tool_recommendation,
                        tool_confidence=self._last_tool_confidence,
                        executed_action=executed_action,
                        solver_followed=self._solver_followed_recommendation,
                        run_id=last_tool_run_id,
                        state_dir=last_tool_state_dir,
                    )
            self._log_flow_event(
                "final_response",
                chat_history=working_history,
                content=content,
            )
            self._flush_tool_traces(tool_traces, content)
            self._append_loop_log(
                {
                    "event": "turn_complete",
                    "turn": turn_id,
                    "total_turn_duration_s": round(time.time() - turn_start_time, 2),
                    "final_action": action_name,
                }
            )
            self._log_flow_event(
                "final_response",
                chat_history=working_history,
                content=content,
            )
            return ChatHistoryItem(role=Role.AGENT, content=content)

        last_output = self._last_solver_output or ""
        self._trace("solver_result", last_output)
        self._log_flow_event(
            "final_response",
            chat_history=working_history,
            content=last_output,
        )
        self._flush_tool_traces(tool_traces, last_output)
        self._append_loop_log(
            {
                "event": "turn_complete",
                "turn": turn_id,
                "total_turn_duration_s": round(time.time() - turn_start_time, 2),
                "final_action": None,
            }
        )
        return ChatHistoryItem(role=Role.AGENT, content=last_output)

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
            self._toolgen_prebootstrap_once(task_query, working_history)
            # Refresh registry so freshly registered pre-boot tool is visible
            if getattr(self, "_registry", None) is not None and hasattr(self._registry, "refresh"):
                try:
                    self._registry.refresh()
                except Exception:
                    pass

            self._consider_tool_generation(task_query, working_history)

            if task_query:
                preprocess_tool = self._select_tool_for_query(
                    task_query, categories={"parser", "normalizer", "planner"}
                )
                if preprocess_tool and preprocess_tool.name not in auto_invoked_tools:
                    preprocess_payload = self._invoke_tool_for_query(
                        preprocess_tool,
                        task_query,
                        chat_history=working_history,
                        reason="auto_preprocess",
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
                        + "Tool results may include 'recommended_next_action' with args; consider using it if it fits.\n\n"
                        + "\n\n".join(solver_sidecar)
                    )
                history_text = self._toolgen_render_history(
                    working_history,
                    max_chars_per_item=1200,
                    preserve_first_user_n=2,
                )
                sidecar_text = "\n\n".join(solver_sidecar)
                self._write_agent_system_prompt(
                    "solver",
                    solver_prompt,
                )
                self._log_flow_event(
                    "solver_input",
                    chat_history=working_history,
                    payload={
                        "system_prompt": solver_prompt,
                        "history": history_text,
                        "system_prompt_chars": len(solver_prompt),
                        "history_chars": len(history_text),
                        "history_items": len(self._history_items(working_history)),
                        "sidecar_chars": len(sidecar_text),
                    },
                )
                self._append_solver_io_log(
                    {
                        "event": "solver_input",
                        "stage": "non_orchestrated",
                        "system_prompt": solver_prompt,
                        "history": history_text,
                        "sidecar": sidecar_text,
                    }
                )
                start_ts = time.perf_counter()
                solver_response = self._solver_inference_with_retry(
                    working_history, system_prompt=solver_prompt
                )
                self._append_solver_io_log(
                    {
                        "event": "solver_output",
                        "stage": "non_orchestrated",
                        "content": getattr(solver_response, "content", "") or "",
                    }
                )
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                self._log_flow_event(
                    "solver_metrics",
                    chat_history=working_history,
                    payload={
                        "stage": "non_orchestrated",
                        "solver_latency_ms": round(elapsed_ms, 2),
                        "system_prompt_chars": len(solver_prompt),
                        "history_chars": len(history_text),
                        "history_items": len(self._history_items(working_history)),
                        "sidecar_chars": len(sidecar_text),
                    },
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

    def get_last_turn_data(self) -> dict[str, Any]:
        data = getattr(self, "_last_turn_data", None)
        return dict(data) if isinstance(data, dict) else {}

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.get_role_dict()
        return self._language_model_agent.get_role_dict()
