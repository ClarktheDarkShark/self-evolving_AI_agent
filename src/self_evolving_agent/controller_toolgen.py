#controller_toolgen.py

import datetime
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import ast

from src.typings import ChatHistory, ChatHistoryItem, Role

from .tool_registry import ToolMetadata
from .tool_spec import ToolSpec
from .tool_validation import validate_tool_code
from .tool_retrieval import retrieve_tools
from .toolgen_debug_logger import toolgen_debug_enabled
from .controller_prompts import TOOLGEN_DEBUG_APPENDIX
from src.toolgen.prompts import get_toolgen_system_prompt
from src.toolgen.prompting.build_task_pack import build_task_pack


class ControllerToolgenMixin:
    _GENERIC_TOOL_NAMES = {
        "generated_tool",
        "agg3_generated_tool",
        "analysis_tool",
        "analysis_generated_tool",
        "analysis_tool_generated_tool",
    }
    _NAME_STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "with",
        "by",
        "from",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "tool",
        "utility",
    }

    def _toolgen_build_task_prompt(self, env_name: str, dataset_item: Any) -> str:
        if env_name == "knowledge_graph":
            if isinstance(dataset_item, Mapping):
                question = str(dataset_item.get("question", "") or "")
                entity_dict = dataset_item.get("entity_dict") or {}
            else:
                question = str(getattr(dataset_item, "question", "") or "")
                entity_dict = getattr(dataset_item, "entity_dict", {}) or {}
            if isinstance(entity_dict, str):
                try:
                    parsed = ast.literal_eval(entity_dict)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    entity_dict = parsed
            entity_list = list(entity_dict.keys()) if isinstance(entity_dict, dict) else []
            return f"Question: {question}, Entities: {entity_list}".strip()
        if env_name == "db_bench":
            if isinstance(dataset_item, Mapping):
                question_prefix = str(dataset_item.get("instruction", "") or "").strip()
                table_info = dataset_item.get("table_info") or {}
                table_name = str(table_info.get("name", "") or "").strip()
                columns = []
                for column in table_info.get("column_info_list") or []:
                    if isinstance(column, Mapping):
                        name = str(column.get("name", "") or "").strip()
                        if name:
                            columns.append(name)
                if question_prefix and table_name and columns:
                    question_suffix = (
                        f"The name of this table is {table_name}, and the headers of this table are "
                        f"{', '.join(columns)}."
                    )
                    return f"{question_prefix} {question_suffix}".strip()
        if isinstance(dataset_item, Mapping):
            instruction = dataset_item.get("instruction", "") or ""
        else:
            instruction = getattr(dataset_item, "instruction", "") or ""
        if instruction:
            return str(instruction)
        question = getattr(dataset_item, "question", "") or ""
        return str(question)

    def preaggregate_toolgen(
        self,
        task: Any,
        sample_indices: Sequence[Any],
        *,
        raw_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if getattr(self, "_toolgen_pipeline_name", "baseline") != "aggregate3":
            return
        if raw_config is None:
            return
        task_name = raw_config.get("assignment_config", {}).get("task")
        if not isinstance(task_name, str) or not task_name:
            return
        preagg_envs = getattr(self, "_toolgen_preaggregate_envs", None)
        if not isinstance(preagg_envs, set):
            preagg_envs = set()
            self._toolgen_preaggregate_envs = preagg_envs
        if task_name in preagg_envs:
            return
        task_def = raw_config.get("task_dict", {}).get(task_name)
        if not isinstance(task_def, Mapping):
            return
        parameters = task_def.get("parameters") or {}
        data_file_path = parameters.get("data_file_path")
        chat_history_factory = parameters.get("chat_history_item_factory") or {}
        chat_history_params = chat_history_factory.get("parameters") or {}
        chat_history_path = chat_history_params.get("chat_history_item_dict_path")
        dataset_map: dict[str, Any] | None = None
        if not isinstance(data_file_path, str) or not os.path.exists(data_file_path):
            getter = None
            try:
                getter = object.__getattribute__(task, "_Task__get_dataset_item")
            except AttributeError:
                getter = None
            if callable(getter):
                dataset_map = {}
                for sample_index in sample_indices:
                    try:
                        dataset_map[str(sample_index)] = getter(sample_index)
                    except Exception:
                        continue
        env_contract = ""
        if isinstance(chat_history_path, str) and os.path.exists(chat_history_path):
            try:
                with open(chat_history_path, "r") as handle:
                    env_contract = (
                        (json.load(handle).get("value", {}).get("0", {}) or {})
                        .get("content", "")
                        .strip()
                    )
            except Exception:
                env_contract = ""
        if not isinstance(data_file_path, str) or not os.path.exists(data_file_path):
            if not dataset_map:
                return
        self._toolgen_agg_context = {
            "env_name": task_name,
            "data_file_path": data_file_path,
            "dataset_map": dataset_map,
            "env_contract": env_contract,
            "sample_indices": list(sample_indices),
        }
        preagg_envs.add(task_name)

    def _toolgen_output_mode(self) -> str:
        return "markers"

    def _normalize_toolgen_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                if isinstance(part, Mapping)
                else str(part)
                for part in content
            )
        return ""

    def _extract_marked_python(self, text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("###TOOL_START")
        if start < 0:
            return None
        end = text.find("###TOOL_END", start + len("###TOOL_START"))
        if end < 0:
            return None
        return text[start + len("###TOOL_START"):end].strip() or None

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse_creation_payload(self, payload: str) -> Optional[Mapping[str, Any]]:
        try:
            obj = json.loads(payload)
        except Exception:
            return None
        return obj if isinstance(obj, Mapping) else None

    def _extract_tool_call_arguments(self, response_obj: Any) -> Optional[str]:
        if response_obj is None:
            return None
        if isinstance(response_obj, Mapping):
            raw = response_obj.get("raw_output") or response_obj.get("arguments")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            tool_calls = response_obj.get("tool_calls")
            if tool_calls:
                first = tool_calls[0]
                if isinstance(first, Mapping):
                    func = first.get("function")
                    if isinstance(func, Mapping):
                        args = func.get("arguments")
                        if isinstance(args, str) and args.strip():
                            return args.strip()
            func_call = response_obj.get("function_call")
            if isinstance(func_call, Mapping):
                args = func_call.get("arguments")
                if isinstance(args, str) and args.strip():
                    return args.strip()
            return None
        tool_calls = getattr(response_obj, "tool_calls", None)
        if tool_calls:
            first = tool_calls[0]
            func = getattr(first, "function", None)
            args = getattr(func, "arguments", None) if func is not None else None
            if isinstance(args, str) and args.strip():
                return args.strip()
        func_call = getattr(response_obj, "function_call", None)
        if func_call is not None:
            args = getattr(func_call, "arguments", None)
            if isinstance(args, str) and args.strip():
                return args.strip()
        return None

    def extract_tool_spec(self, raw_text: str, response_obj: Any) -> dict[str, Any]:
        raw_text_full = raw_text or ""
        if not isinstance(raw_text_full, str):
            raise ValueError(f"raw_text must be str (got {type(raw_text_full).__name__})")
        obj = self._parse_creation_payload(raw_text_full)
        if obj is not None:
            return dict(obj)
        obj_text = self._extract_first_json_object(raw_text_full)
        if obj_text:
            obj = self._parse_creation_payload(obj_text)
            if obj is not None:
                return dict(obj)
        tool_args = self._extract_tool_call_arguments(response_obj)
        if tool_args:
            obj = self._parse_creation_payload(tool_args)
            if obj is not None:
                return dict(obj)
        preview = raw_text_full[:200]
        raise ValueError(f"tool_spec_parse_failed preview={preview!r}")

    def _toolgen_compact_existing_tools(self) -> list[dict[str, Any]]:
        # Filter tools by current environment
        current_env = self._resolved_environment_label()
        tools = (
            self._registry.list_latest_tools(environment=current_env)
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools(environment=current_env)
        )
        print(f"[TOOLGEN_COMPACT] Found {len(tools)} tools for environment '{current_env}'")
        if toolgen_debug_enabled():
            return [{"name": tool.name} for tool in tools[-5:]]
        return [
            {
                "name": tool.name,
                "signature": tool.signature,
                "docstring": tool.docstring,
            }
            for tool in tools[-50:]
        ]

    def _toolgen_default_name(self) -> str:
        return "generated_tool"

    def _toolgen_default_description(self) -> str:
        query = getattr(self, "_toolgen_last_query", "") or ""
        summary = query.strip()
        return f"Utility tool for: {summary[:120]}" if summary else "Utility tool for the current task."

    def _apply_tool_name_prefix(self, name: str, prefix: str) -> str:
        if not prefix:
            return name
        if not name:
            return name
        if name.startswith(prefix):
            return name
        return f"{prefix}{name}"

    def _is_generic_tool_name(self, name: str) -> bool:
        normalized = (name or "").strip().lower()
        if not normalized:
            return True
        base = normalized
        if base.endswith("_generated_tool"):
            base = base[: -len("_generated_tool")]
        if base in self._GENERIC_TOOL_NAMES:
            return True
        if base.startswith("generated_tool"):
            return True
        if base.startswith("agg3_generated_tool"):
            return True
        return False

    def _toolgen_name_from_description(self, description: str) -> Optional[str]:
        if not description:
            return None
        words = re.findall(r"[a-zA-Z0-9]+", description.lower())
        keywords = [w for w in words if w not in self._NAME_STOPWORDS]
        if not keywords:
            return None
        name = "_".join(keywords[:4]) + "_tool"
        return name[:50]

    def _parse_schema_keys_from_code(self, python_code: str) -> tuple[list[str], list[str]] | None:
        if not python_code:
            return None

        def _parse_list(match: re.Match) -> list[str] | None:
            raw = match.group(1).strip()
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return None
            if not isinstance(parsed, list):
                return None
            return [str(item).strip() for item in parsed if str(item).strip()]

        req_match = re.search(r"RUN_PAYLOAD_REQUIRED:\s*(\[[^\r\n]*\])", python_code)
        opt_match = re.search(r"RUN_PAYLOAD_OPTIONAL:\s*(\[[^\r\n]*\])", python_code)
        required = _parse_list(req_match) if req_match else None
        optional = _parse_list(opt_match) if opt_match else None
        if required is not None:
            return required, (optional or [])

        req_match = re.search(r"input_schema_required:\s*([^\r\n]+)", python_code, flags=re.IGNORECASE)
        opt_match = re.search(r"input_schema_optional:\s*([^\r\n]+)", python_code, flags=re.IGNORECASE)
        if req_match:
            required = [item.strip() for item in req_match.group(1).split(",") if item.strip()]
            optional = [item.strip() for item in opt_match.group(1).split(",")] if opt_match else []
            optional = [item for item in optional if item]
            return required, optional

        doc_match = re.search(
            r"INPUT_SCHEMA:.*?required=([^;]+);\\s*optional=([^\\n]+)",
            python_code,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if doc_match:
            required = [item.strip() for item in doc_match.group(1).split(",") if item.strip()]
            optional = [item.strip() for item in doc_match.group(2).split(",") if item.strip()]
            return required, optional
        return None

    def _build_input_schema(self, required_keys: list[str], optional_keys: list[str]) -> dict[str, Any]:
        type_map: dict[str, Any] = {
            "task_text": {"type": "string"},
            "asked_for": {"type": "string"},
            "trace": {"type": "array"},
            "actions_spec": {"type": "object"},
            "constraints": {"type": "array"},
            "output_contract": {"type": "object"},
            "draft_response": {"type": ["string", "null"]},
            "candidate_output": {},
            "env_observation": {},
            "run_id": {"type": "string"},
            "state_dir": {"type": "string"},
        }
        properties: dict[str, Any] = {}
        for key in required_keys + optional_keys:
            properties[key] = type_map.get(key, {})
        return {
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {
                    "type": "object",
                    "required": required_keys,
                    "properties": properties,
                }
            },
        }

    def _toolgen_prebootstrap_once(
        self, task_query: str, chat_history: ChatHistory
    ) -> None:
        if not task_query.strip():
            print("[TOOLGEN_PREBOOT] Skipping: empty task_query")
            return
        if getattr(self, "_toolgen_agent", None) is None:
            print("[TOOLGEN_PREBOOT] Skipping: _toolgen_agent is None")
            return
        env_name = self._resolved_environment_label()
        preboot_envs = getattr(self, "_toolgen_preboot_envs", None)

        print()
        print("=" * 70)
        print(f"[TOOLGEN_PREBOOT] env_name: {env_name}")
        print(f"[TOOLGEN_PREBOOT] preboot_envs: {preboot_envs}")
        print(f"[TOOLGEN_PREBOOT] pipeline: {getattr(self, '_toolgen_pipeline_name', 'baseline')}")
        print("=" * 70)
        print()


        if not isinstance(preboot_envs, set):
            preboot_envs = set()
            self._toolgen_preboot_envs = preboot_envs
        if env_name in preboot_envs:
            print(f"[TOOLGEN_PREBOOT] Env '{env_name}' already bootstrapped, skipping")
            return
        if getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3":
            # Check if context is ready
            context = getattr(self, "_toolgen_agg_context", None)
            prompt = None

            # Try to build aggregate prompt if context is ready
            if isinstance(context, dict):
                print(f"[TOOLGEN_PREBOOT] Context available, building aggregate prompt for env '{env_name}'")
                prompt = self._toolgen_build_aggregate_prompt_for_env(
                    env_name, task_query=task_query, chat_history=chat_history
                )
                if prompt:
                    print(f"[TOOLGEN_PREBOOT] Aggregate prompt built successfully")
                else:
                    print(f"[TOOLGEN_PREBOOT] Aggregate prompt failed (likely env mismatch)")
            else:
                print(f"[TOOLGEN_PREBOOT] WARNING: _toolgen_agg_context not ready (type={type(context).__name__})")

            # Fallback: Use simple prompt if aggregate prompt not available
            if not prompt:
                print(f"[TOOLGEN_PREBOOT] Falling back to simple prompt for env '{env_name}'")
                # Build a simple prompt from the current task query and env contract
                env_contract = ""
                if chat_history is not None:
                    try:
                        for item in self._history_items(chat_history):
                            if item.role == Role.USER:
                                env_contract = (item.content or "").strip()
                                break
                    except Exception:
                        pass

                if env_contract and task_query:
                    prompt = (
                        f"ENVIRONMENT: {env_name}\n\n"
                        f"ENVIRONMENT CONTRACT:\n{env_contract}\n\n"
                        f"CURRENT TASK:\n{task_query}\n\n"
                        f"Create a utility tool that will help solve tasks in this environment. "
                        f"The tool should be reusable across multiple similar tasks."
                    )
                    print(f"[TOOLGEN_PREBOOT] Simple prompt built (length={len(prompt)})")
                else:
                    print(f"[TOOLGEN_PREBOOT] Cannot build simple prompt: env_contract={bool(env_contract)}, task_query={bool(task_query)}")

            if not prompt:
                print(f"[TOOLGEN_PREBOOT] No prompt available for env '{env_name}', skipping tool generation")
                return

            print(f"[TOOLGEN_PREBOOT] Generating tool for env '{env_name}'")
            tool = self._toolgen_generate_from_prompt(
                user_prompt=prompt,
                system_prompt=get_toolgen_system_prompt("aggregate3", env_name),
                chat_history=chat_history,
                name_prefix=getattr(self, "_toolgen_name_prefix", ""),
            )

            if tool:
                print(f"[TOOLGEN_PREBOOT] SUCCESS: Tool '{tool.name}' generated for env '{env_name}'")
            else:
                print(f"[TOOLGEN_PREBOOT] WARNING: Tool generation returned None for env '{env_name}'")

            preboot_envs.add(env_name)
            agg_envs = getattr(self, "_toolgen_agg_bootstrapped_envs", None)
            if not isinstance(agg_envs, set):
                agg_envs = set()
            agg_envs.add(env_name)
            self._toolgen_agg_bootstrapped_envs = agg_envs
            return
        pipeline = getattr(self, "_toolgen_pipeline", None)
        if pipeline is not None:
            tools = pipeline.maybe_generate_tools(env_name, task_query, chat_history)
            if tools:
                preboot_envs.add(env_name)
                return
            return
        system_prompt = get_toolgen_system_prompt(
            getattr(self, "_toolgen_pipeline_name", "baseline"),
            self._resolved_environment_label(),
        )
        prompt = self._toolgen_request_prompt(task_query, chat_history)
        self._toolgen_generate_from_prompt(
            user_prompt=prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            name_prefix=getattr(self, "_toolgen_name_prefix", ""),
        )
        preboot_envs.add(env_name)

    def _toolgen_build_system_prompt(self, base_prompt: str) -> str:
        prompt = (base_prompt or "").strip()
        if toolgen_debug_enabled():
            prompt = f"{prompt}\n\n{TOOLGEN_DEBUG_APPENDIX}".strip()
        prompt = f"{prompt}\n\n{self._toolgen_tool_list_appendix()}".strip()
        return prompt

    def _toolgen_generate_from_prompt(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        chat_history: ChatHistory,
        name_prefix: str,
    ) -> Optional[ToolMetadata]:
        if getattr(self, "_toolgen_agent", None) is None:
            print("[TOOLGEN] ERROR: _toolgen_agent is None, cannot generate tool")
            return None

        if not user_prompt or not user_prompt.strip():
            print("[TOOLGEN] ERROR: user_prompt is empty, cannot generate tool")
            return None

        self._trace("tool_agent_input", user_prompt)
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=user_prompt)
        )
        final_system_prompt = self._toolgen_build_system_prompt(system_prompt)
        debug_prompt = os.getenv("TOOLGEN_DEBUG_PROMPT") == "1"
        if debug_prompt and not getattr(self, "_toolgen_first_prompt_printed", False):
            print("[ToolGen] first_run system_prompt:\n" + final_system_prompt)
            print("[ToolGen] first_run user_prompt:\n" + user_prompt)
            self._toolgen_first_prompt_printed = True
        self._write_agent_system_prompt("toolgen", final_system_prompt)
        original_prompt = getattr(self._toolgen_agent, "_system_prompt", "") or ""
        self._toolgen_agent._system_prompt = final_system_prompt
        print(f"[TOOLGEN] Calling toolgen agent inference...")
        try:
            response = self._toolgen_agent._inference(tool_history)
            print(f"[TOOLGEN] Toolgen agent inference completed")
        except Exception as e:
            print(f"[TOOLGEN] ERROR: Toolgen agent inference failed: {e}")
            raise
        finally:
            self._toolgen_agent._system_prompt = original_prompt

        self._trace("tool_agent_result", response.content)
        raw_text_full = self._normalize_toolgen_content(response.content)
        extracted = self._extract_marked_python(raw_text_full)
        if not extracted:
            self._write_failed_tool_artifact(
                stage="toolgen_markers_missing",
                error="marker_block_not_found",
                raw_output=raw_text_full,
            )
            return None
        tool_spec = self._wrap_marker_tool_spec(extracted)
        tool_name = str(tool_spec.get("name") or "")
        tool_spec["name"] = self._apply_tool_name_prefix(tool_name, name_prefix)
        return self._register_tool_from_payload(tool_spec, chat_history)

    def _toolgen_build_aggregate_prompt_for_env(
        self,
        env_name: str,
        *,
        task_query: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> Optional[str]:
        print(f"[BUILD_AGG_PROMPT] Building prompt for env '{env_name}'")
        context = getattr(self, "_toolgen_agg_context", None)
        if not isinstance(context, Mapping):
            print(f"[BUILD_AGG_PROMPT] ERROR: _toolgen_agg_context is not a Mapping (type={type(context).__name__})")
            return None
        context_env = context.get("env_name")
        if context_env != env_name:
            print(f"[BUILD_AGG_PROMPT] ERROR: Context env mismatch - expected '{env_name}', got '{context_env}'")
            return None

        data_file_path = context.get("data_file_path")
        dataset_map = context.get("dataset_map")
        env_contract = context.get("env_contract") or ""

        print(f"[BUILD_AGG_PROMPT] data_file_path: {data_file_path}")
        print(f"[BUILD_AGG_PROMPT] dataset_map type: {type(dataset_map).__name__ if dataset_map else 'None'}")
        print(f"[BUILD_AGG_PROMPT] env_contract length: {len(env_contract) if env_contract else 0}")

        if not env_contract and chat_history is not None:
            print(f"[BUILD_AGG_PROMPT] env_contract empty, extracting from chat_history")
            try:
                for item in self._history_items(chat_history):
                    if item.role == Role.USER:
                        env_contract = (item.content or "").strip()
                        break
                print(f"[BUILD_AGG_PROMPT] Extracted env_contract length: {len(env_contract)}")
            except Exception as e:
                print(f"[BUILD_AGG_PROMPT] WARNING: Failed to extract env_contract from history: {e}")
                env_contract = ""

        sample_indices = context.get("sample_indices") or []
        print(f"[BUILD_AGG_PROMPT] sample_indices: {sample_indices}")

        if not isinstance(data_file_path, str) or not os.path.exists(data_file_path):
            if not isinstance(dataset_map, Mapping):
                print(f"[BUILD_AGG_PROMPT] ERROR: No valid data_file_path and dataset_map is not a Mapping")
                return None
            print(f"[BUILD_AGG_PROMPT] Using dataset_map (data_file_path not available)")

        if not env_contract:
            print(f"[BUILD_AGG_PROMPT] ERROR: env_contract is empty")
            return None
        if not isinstance(sample_indices, list):
            print(f"[BUILD_AGG_PROMPT] ERROR: sample_indices is not a list (type={type(sample_indices).__name__})")
            return None
        if len(sample_indices) == 0:
            print(f"[BUILD_AGG_PROMPT] ERROR: sample_indices is empty")
            return None
        dataset: Mapping[str, Any]
        if isinstance(dataset_map, Mapping):
            dataset = dataset_map
        else:
            try:
                with open(data_file_path, "r") as handle:
                    dataset = json.load(handle)
            except Exception:
                return None
            if not isinstance(dataset, Mapping):
                return None
        candidates: list[Any] = []
        for sample_index in sample_indices:
            key = str(sample_index)
            entry = dataset.get(key)
            if entry is None:
                continue
            candidates.append(sample_index)
        agg_n = int(getattr(self, "_toolgen_agg_n", 10) or 10)
        if agg_n < 1:
            agg_n = 1
        if len(candidates) < agg_n:
            agg_n = len(candidates)
        if agg_n < 1:
            return None
        selected_indices = random.sample(candidates, agg_n)
        tasks: list[str] = []
        used_indices: list[Any] = []
        for sample_index in selected_indices:
            key = str(sample_index)
            entry = dataset.get(key)
            if entry is None:
                continue
            task_prompt = self._toolgen_build_task_prompt(env_name, entry)
            if not task_prompt:
                continue
            tasks.append(task_prompt.strip())
            used_indices.append(sample_index)
            print(
                f"agg3 buffer env={env_name} size={len(tasks)} sample_index={sample_index}"
            )
        if len(tasks) < agg_n:
            return None
        print(f"agg3 trigger env={env_name} sample_indexes={used_indices}")
        user_prompt = build_task_pack(env_name, env_contract, tasks)
        gap_reason = getattr(self, "_toolgen_last_orchestrator_reason", "") or ""
        gap_detail = getattr(self, "_toolgen_last_orchestrator_gap", "") or ""
        needed = getattr(self, "_toolgen_last_orchestrator_needed", "") or ""
        gap_lines = []
        if gap_reason:
            gap_lines.append(f"reason: {gap_reason}")
        if gap_detail:
            gap_lines.append(f"insufficiency: {gap_detail}")
        if needed:
            gap_lines.append(f"needed_capabilities: {needed}")
        if gap_lines:
            user_prompt = (
                user_prompt
                + "\n\nTOOL_ORCHESTRATOR_GAPS:\n"
                + "\n".join(gap_lines)
            )
        task_query = (task_query or "").strip()
        if task_query:
            history_text = ""
            if chat_history is not None:
                history_text = self._toolgen_render_history(
                    chat_history,
                    max_chars_per_item=1200,
                    preserve_first_user_n=2,
                )
            context_block = (
                "CURRENT TASK CONTEXT:\n"
                "Here is the current task you need to create a tool to help solve. "
                "The chat history provides an overview of recent actions taken. "
                "A tool is needed to ensure rapid and accurate task completion.\n"
                f"TASK:\n{task_query}"
            )
            if history_text:
                context_block = (
                    context_block + "\n\nCHAT HISTORY (truncated):\n" + history_text
                )
            user_prompt = user_prompt + "\n\n" + context_block

        print(f"[BUILD_AGG_PROMPT] SUCCESS: Prompt built for env '{env_name}' (length={len(user_prompt)})")
        return user_prompt

    def _toolgen_tool_list_appendix(self) -> str:
        # Filter tools by current environment
        current_env = self._resolved_environment_label()
        tools = (
            self._registry.list_latest_tools(environment=current_env)
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools(environment=current_env)
        )
        if not tools:
            return f"CURRENT TOOLS FOR {current_env}: none"
        lines = [f"CURRENT TOOLS FOR {current_env} (name | signature):"]
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            signature = str(getattr(tool, "signature", "") or "").strip()
            if not name:
                continue
            if signature:
                lines.append(f"- {name} | {signature}")
            else:
                lines.append(f"- {name}")
        return "\n".join(lines)

    def _extract_tool_name_from_code(self, python_code: str) -> Optional[str]:
        if not python_code:
            return None
        for raw_line in python_code.splitlines():
            line = raw_line.strip()
            if not line.startswith("#"):
                continue
            match = re.match(r"^#\s*tool_name\s*:\s*([a-zA-Z0-9_]+)\s*$", line)
            if not match:
                continue
            name = match.group(1).strip().lower()
            if name:
                if not name.endswith("_generated_tool"):
                    return f"{name}_generated_tool"
                return name
        return None

    def _toolgen_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        debug_enabled = toolgen_debug_enabled()
        existing = []
        try:
            existing = self._toolgen_compact_existing_tools()
        except Exception:
            existing = []

        max_obs_len = 120 if debug_enabled else 300
        max_line_len = 300 if debug_enabled else 1000

        def _shorten_history_line(text: str) -> str:
            if not text:
                return ""
            if "Observation:" in text or "executes successfully" in text:
                head, sep, tail = text.partition("Observation:")
                if sep:
                    trimmed_tail = tail.strip()
                    if len(trimmed_tail) > max_obs_len:
                        trimmed_tail = trimmed_tail[:max_obs_len] + "...[truncated_observation]"
                    return (head.strip() + " Observation: " + trimmed_tail).strip()
            if len(text) > max_line_len:
                return text[:max_line_len] + "...[truncated]"
            return text

        history_text_full = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=1200,
            preserve_first_user_n=2,
        )
        history_lines_full = history_text_full.splitlines()
        history_lines = history_lines_full[-8:]
        if debug_enabled:
            last_user = self._get_last_user_item(chat_history)
            if last_user:
                content = (last_user.content or "").strip()
                content = _shorten_history_line(content)
                history_lines = ["0:{}:{}".format(last_user.role.value, content)]
            else:
                history_lines = history_lines_full[-2:]

        payload = {
            "task": self._toolgen_compact_query(query) if debug_enabled else (query or "").strip(),
            "task_requirement": (
                "Task-specific tool required."
                if debug_enabled
                else "Tool must directly help solve the current task; generic tools are invalid."
            ),
            "history": "\n".join(history_lines),
            "existing_tools": existing,
        }
        solver_recommendation = (
            getattr(self, "_toolgen_last_recommendation", "") or ""
        ).strip()
        if solver_recommendation:
            if debug_enabled and len(solver_recommendation) > 200:
                solver_recommendation = solver_recommendation[:200] + "...[truncated]"
            payload["solver_recommendation"] = solver_recommendation
            payload["recommendation_note"] = (
                "Use solver_recommendation if helpful."
                if debug_enabled
                else "Solver provided a draft response. Use it to design a tool that "
                "validates or strengthens the draft for this task."
            )

        json_kwargs = {"ensure_ascii": True, "default": str}
        if debug_enabled:
            json_kwargs["separators"] = (",", ":")
        return json.dumps(payload, **json_kwargs)

    def _normalize_retrieval_query(self, query: str) -> str:
        return (query or "").strip()[:1200]

    def _join_code_lines(self, code_lines: Sequence[Any]) -> Optional[str]:
        normalized_lines: list[str] = []
        for line in code_lines:
            text = str(line)
            if "\r\n" in text:
                text = text.replace("\r\n", "\n")
            if "\n" in text:
                normalized_lines.extend(text.splitlines())
            else:
                normalized_lines.append(text)
        if not normalized_lines:
            return None
        return "\n".join(normalized_lines).rstrip() + "\n"

    def _wrap_marker_tool_spec(self, python_code: str) -> dict[str, Any]:
        name = self._extract_tool_name_from_code(python_code) or self._toolgen_default_name()
        description = self._toolgen_default_description()
        signature = "run(payload: dict) -> dict"
        tool_type = "utility"
        tool_category = "utility"
        schema_keys = self._parse_schema_keys_from_code(python_code)
        if schema_keys:
            input_schema = self._build_input_schema(schema_keys[0], schema_keys[1])
        else:
            input_schema = self._build_input_schema(
                ["task_text", "asked_for", "trace", "actions_spec"],
                ["constraints", "output_contract", "draft_response", "candidate_output", "env_observation"],
            )


        capabilities = []
        return {
            "name": name,
            "description": description,
            "signature": signature,
            "tool_type": tool_type,
            "tool_category": tool_category,
            "input_schema": input_schema,
            "capabilities": capabilities,
            "code_lines": python_code.splitlines(),
        }

    def _normalize_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(spec)
        normalized.setdefault("name", self._toolgen_default_name())
        name = str(normalized.get("name") or "").strip().lower()
        description = str(normalized.get("description") or "")
        if self._is_generic_tool_name(name):
            derived = self._toolgen_name_from_description(description)
            if derived:
                normalized["name"] = derived
                name = derived
        if name and not name.endswith("_generated_tool"):
            normalized["name"] = f"{name}_generated_tool"
        normalized.setdefault("description", self._toolgen_default_description())
        normalized.setdefault("signature", "run(payload: dict) -> dict")
        normalized.setdefault("tool_type", "utility")
        normalized.setdefault("tool_category", "utility")
        if "input_schema" not in normalized:
            normalized["input_schema"] = self._build_input_schema(
                ["task_text", "asked_for", "trace", "actions_spec"],
                ["constraints", "output_contract", "draft_response", "candidate_output", "env_observation"],
            )
        normalized.setdefault("capabilities", [])
        code_lines = normalized.get("code_lines")
        if isinstance(code_lines, list):
            code = "\n".join(str(line) for line in code_lines)
            schema_keys = self._parse_schema_keys_from_code(code)
            if schema_keys:
                normalized["input_schema"] = self._build_input_schema(schema_keys[0], schema_keys[1])
        schema = normalized.get("input_schema")
        if isinstance(schema, Mapping):
            props = schema.get("properties") or {}
            payload_schema = props.get("payload")
            if isinstance(payload_schema, Mapping):
                required = payload_schema.get("required") or []
                if not isinstance(required, list):
                    required = []
                if not required:
                    required_keys = ["task_text", "asked_for", "trace", "actions_spec"]
                    for key in required_keys:
                        if key not in required:
                            required.append(key)
                payload_schema["required"] = required
                properties = payload_schema.get("properties") or {}
                type_defaults = {
                    "task_text": {"type": "string"},
                    "asked_for": {"type": "string"},
                    "trace": {"type": "array"},
                    "actions_spec": {"type": "object"},
                    "constraints": {"type": "array"},
                    "output_contract": {"type": "object"},
                    "draft_response": {"type": ["string", "null"]},
                    "candidate_output": {},
                    "env_observation": {},
                    "run_id": {"type": "string"},
                    "state_dir": {"type": "string"},
                }
                for key in required:
                    if key not in properties:
                        properties[key] = type_defaults.get(key, {})
                payload_schema["properties"] = properties
                props["payload"] = payload_schema
                schema["properties"] = props
                normalized["input_schema"] = schema
        return normalized

    def _validate_spec_alignment(self, spec: ToolSpec) -> Optional[str]:
        if spec.signature.strip() != "run(payload: dict) -> dict":
            return "signature_mismatch"
        schema = spec.input_schema
        if not isinstance(schema, Mapping):
            return "missing_input_schema"
        if schema.get("type") != "object":
            return "input_schema_type_mismatch"
        required = schema.get("required") or []
        if required != ["payload"]:
            return "input_schema_required_mismatch"
        payload_schema = (schema.get("properties") or {}).get("payload")
        if not isinstance(payload_schema, Mapping):
            return "payload_schema_missing"
        if payload_schema.get("type") != "object":
            return "payload_schema_type_mismatch"
        if "required" not in payload_schema:
            return "payload_required_missing"
        return None

    def _validate_run_ast(self, code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
        except Exception as exc:
            return f"ast_parse_failed: {exc}"
        run_fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_fn = node
                break
        if run_fn is None:
            return "run_not_found"
        args = run_fn.args
        total_args = list(args.posonlyargs) + list(args.args)
        if len(total_args) != 1 or total_args[0].arg != "payload":
            return "run_signature_mismatch"
        if args.vararg or args.kwarg or args.kwonlyargs:
            return "run_signature_mismatch"
        return None

    def _failed_tool_log_dir(self) -> Path:
        base_path = None
        log_path = getattr(self, "_generated_tools_log_path", None)
        if log_path is not None:
            base_path = Path(log_path).parent
        if base_path is None:
            base_path = Path("outputs")
        return base_path / "callback_state" / "callback_generated_tool_logging"

    def _write_failed_tool_artifact(
        self,
        *,
        stage: str,
        error: str,
        spec: Optional[ToolSpec] = None,
        code: Optional[str] = None,
        raw_spec: Optional[Mapping[str, Any]] = None,
        raw_output: Optional[str] = None,
    ) -> None:
        try:
            tool_name = (spec.name if spec else None) or "unknown_tool"
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
            suffix = "py" if code else "txt"
            filename = f"{tool_name}__{stage}__{ts}.{suffix}"
            out_dir = self._failed_tool_log_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            header = (
                f"# stage: {stage}\n"
                f"# tool_name: {tool_name}\n"
                f"# signature: {(spec.signature if spec else '')}\n"
                f"# error: {error}\n"
            )
            if code:
                content = header + "\n" + code
            else:
                meta = {
                    "stage": stage,
                    "tool_name": tool_name,
                    "signature": spec.signature if spec else None,
                    "error": error,
                    "raw_spec": raw_spec,
                    "raw_output": raw_output,
                }
                content = header + "\n" + json.dumps(meta, ensure_ascii=True, default=str, indent=2)
            (out_dir / filename).write_text(content, encoding="utf-8")
        except Exception:
            return

    def _validate_and_register_tool(
        self,
        spec: ToolSpec,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolMetadata]:
        alignment_error = self._validate_spec_alignment(spec)
        if alignment_error:
            self._trace("tool_generation_error", f"stage=spec_alignment error={alignment_error}")
            self._write_failed_tool_artifact(
                stage="spec_alignment",
                error=alignment_error,
                spec=spec,
                raw_spec=raw_spec,
            )
            return None
        code = self._join_code_lines(spec.code_lines or [])
        if not code:
            self._trace("tool_generation_error", "stage=code_join error=empty_code")
            self._write_failed_tool_artifact(
                stage="code_join",
                error="empty_code",
                spec=spec,
                raw_spec=raw_spec,
            )
            return None
        ast_error = self._validate_run_ast(code)
        if ast_error:
            self._trace("tool_generation_error", f"stage=run_signature error={ast_error}")
            self._write_failed_tool_artifact(
                stage="run_signature",
                error=ast_error,
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        try:
            result = validate_tool_code(code)
        except Exception as exc:
            self._trace("tool_generation_error", f"stage=validate_exception error={exc}")
            self._write_failed_tool_artifact(
                stage="validate_exception",
                error=str(exc),
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        if not result or not result.success:
            error = getattr(result, "error", None) or "validate failed"
            self._trace("tool_generation_error", f"stage=validate error={error}")
            self._write_failed_tool_artifact(
                stage="validate",
                error=error,
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        try:
            # Get current environment to organize tools by env
            current_env = self._resolved_environment_label()
            print(f"[TOOL_REGISTER] Registering tool '{spec.name}' for environment '{current_env}'")

            metadata = self._registry.register_tool(
                name=spec.name,
                code=code,
                signature=spec.signature,
                description=spec.description,
                tool_type=spec.tool_type,
                tool_category=spec.tool_category,
                input_schema=spec.input_schema,
                capabilities=spec.capabilities,
                environment=current_env,
            )
        except Exception as exc:
            self._trace("tool_generation_error", f"stage=register_exception error={exc}")
            self._write_failed_tool_artifact(
                stage="register_exception",
                error=str(exc),
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        if metadata:
            self._tool_creation_successes += 1
            try:
                payload = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "event": "create",
                    "tool_name": metadata.name,
                    "signature": metadata.signature,
                    "description": metadata.description,
                    "tool_type": metadata.tool_type,
                    "tool_category": metadata.tool_category,
                    "input_schema": metadata.input_schema,
                    "capabilities": metadata.capabilities,
                    "path": getattr(
                        self._registry,
                        "_get_tool_path",
                        lambda n, environment=None: None
                    )(metadata.name, environment=getattr(metadata, "environment", None)),
                }
                payload.update(self._get_run_task_metadata())
                self._append_generated_tools_log(payload)
            except Exception:
                pass
        return metadata

    def _register_tool_from_payload(
        self, creation_request: Mapping[str, Any], chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            return None
        if not isinstance(creation_request, Mapping):
            return None
        tool_spec = self._normalize_tool_spec(dict(creation_request))
        code_lines = tool_spec.get("code_lines") or []
        if isinstance(code_lines, str):
            tool_spec["code_lines"] = code_lines.splitlines()
        spec = ToolSpec.from_payload(tool_spec)
        metadata = self._validate_and_register_tool(spec, chat_history, raw_spec=tool_spec)
        if metadata:
            self._generated_tool_counter += 1
            self._mark_tool_invoked()
        return metadata

    def _consider_tool_generation(
        self, query: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if (
            getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3"
            and self._resolved_environment_label()
            not in getattr(self, "_toolgen_agg_bootstrapped_envs", set())
        ):
            return None
        if not query.strip():
            return None
        if not self._force_tool_generation_if_missing:
            return None
        normalized = self._normalize_retrieval_query(query)
        if not normalized or normalized in self._toolgen_attempted_queries:
            return None
        self._toolgen_attempted_queries.add(normalized)
        return self._maybe_generate_tool_for_query(query, chat_history)

    def _maybe_generate_tool_for_query(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        allow_reuse: bool = True,
        force: bool = False,
        use_pipeline: bool = True,
    ) -> Optional[ToolMetadata]:
        if getattr(self, "_toolgen_off", False):
            return None
        if not query.strip():
            return None
        if (
            getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3"
            and True
        ):
            env_name = self._resolved_environment_label()
            agg_envs = getattr(self, "_toolgen_agg_bootstrapped_envs", None)
            if not isinstance(agg_envs, set):
                agg_envs = set()
                self._toolgen_agg_bootstrapped_envs = agg_envs
            if env_name not in agg_envs:
                prompt = self._toolgen_build_aggregate_prompt_for_env(
                    env_name, task_query=query, chat_history=chat_history
                )
                if not prompt:
                    return None
                tool = self._toolgen_generate_from_prompt(
                    user_prompt=prompt,
                    system_prompt=get_toolgen_system_prompt("aggregate3", env_name),
                    chat_history=chat_history,
                    name_prefix=getattr(self, "_toolgen_name_prefix", ""),
                )
                if tool:
                    print(f"agg3 bootstrapped env={env_name} tools=1")
                agg_envs.add(env_name)
                return tool
        if (
            getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3"
            and self._resolved_environment_label()
            in getattr(self, "_toolgen_agg_bootstrapped_envs", set())
        ):
            use_pipeline = False
        if allow_reuse:
            reuse_query = self._normalize_retrieval_query(query)
            candidate_output = self._get_candidate_output(chat_history, query)
            reuse = self._reuse_existing_tool(
                reuse_query, candidate_output=candidate_output, needed_archetype=None
            )
            if reuse is not None and self._reuse_matches_request(reuse):
                return reuse

        if not self._force_tool_generation_if_missing and not force:
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run and not force:
            return None

        self._toolgen_last_query = query
        pipeline = getattr(self, "_toolgen_pipeline", None)
        env_name = self._resolved_environment_label()
        if use_pipeline and pipeline is not None:
            tools = pipeline.maybe_generate_tools(env_name, query, chat_history)
            if tools:
                return tools[0]
            if hasattr(pipeline, "should_fallback") and pipeline.should_fallback(env_name):
                return self._maybe_generate_tool_for_query(
                    query,
                    chat_history,
                    allow_reuse=allow_reuse,
                    force=force,
                    use_pipeline=False,
                )
            return None

        prompt = self._toolgen_request_prompt(query, chat_history)
        return self._toolgen_generate_from_prompt(
            user_prompt=prompt,
            system_prompt=getattr(self._toolgen_agent, "_system_prompt", "") or "",
            chat_history=chat_history,
            name_prefix="",
        )

    def _reuse_matches_request(self, tool: ToolMetadata) -> bool:
        return True
