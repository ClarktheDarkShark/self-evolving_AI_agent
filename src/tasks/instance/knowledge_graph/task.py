import ast
import json
import logging
import os
import re
import time
from typing import Optional, Callable, Any, Sequence, Mapping
from pydantic import field_validator
import inspect

from src.tasks.task import (
    Task,
    DatasetItem,
    SkillUtility,
    AgentResponseParserResult,
    AgentAction,
)
from src.typings import (
    SampleIndex,
    SampleStatus,
    TaskEnvironmentException,
    Session,
    TaskName,
    Role,
    SessionEvaluationOutcome,
    MetricDict,
    SessionMetricCalculationPartial,
)
from src.factories.chat_history_item import ChatHistoryItemFactory
from src.utils.struct_coercion import coerce_struct
from .api import KnowledgeGraphAPI, Variable, KnowledgeGraphAPIException
from .utils.sparql_executor import SparqlExecutor


class KnowledgeGraphSkillUtility(SkillUtility):
    _SKILL_TO_LEVEL_DICT = {}


class KnowledgeGraphDatasetItem(DatasetItem):
    question: str
    entity_dict: dict[str, str]
    answer_set: set[str]

    @field_validator("entity_dict", mode="before")  # noqa
    @classmethod
    def _validate_parentheses_pair(cls, entity_dict: dict[str, str]) -> dict[str, str]:
        if isinstance(entity_dict, str):
            entity_dict = coerce_struct(entity_dict)
        if not isinstance(entity_dict, dict):
            return entity_dict
        for entity in entity_dict.keys():
            error_message = f"Invalid parentheses pair in entity: {entity}"
            left_parentheses_count = 0
            for char in entity:
                if char == "(":
                    left_parentheses_count += 1
                elif char == ")":
                    left_parentheses_count -= 1
                    if left_parentheses_count < 0:
                        raise ValueError(error_message)
            if left_parentheses_count != 0:
                raise ValueError(error_message)
        return entity_dict

    @field_validator("entity_dict", mode="before")  # noqa
    @classmethod
    def _validate_entity_name(cls, entity_dict: dict[str, str]) -> dict[str, str]:
        if isinstance(entity_dict, str):
            entity_dict = coerce_struct(entity_dict)
        if not isinstance(entity_dict, dict):
            return entity_dict
        for entity in entity_dict.keys():
            if "#" in entity:
                raise ValueError(f"Invalid entity name: {entity}")
        return entity_dict

    def get_skill_list(self) -> list[str]:
        return []

    def get_difficulty_level(self) -> int:
        return 0


class KnowledgeGraph(Task[KnowledgeGraphDatasetItem]):
    def __init__(
        self,
        task_name: TaskName,
        chat_history_item_factory: ChatHistoryItemFactory,
        sparql_url: str,
        ontology_dir_path: str,
        data_file_path: str,
        max_round: int,
        data_source: str = "auto",
        hf_dataset_name: str = "csyq/LifelongAgentBench",
        hf_dataset_config: Optional[str] = None,
        hf_data_dir: Optional[str] = "knowledge_graph",
        hf_split: str = "train",
        hf_cache_dir: Optional[str] = None,
    ):
        super().__init__(task_name, chat_history_item_factory, max_round)
        sparql_executor = SparqlExecutor(sparql_url)
        self.knowledge_graph_api = KnowledgeGraphAPI(ontology_dir_path, sparql_executor)
        raw_dataset = self._load_data(
            data_file_path=data_file_path,
            data_source=data_source,
            hf_dataset_name=hf_dataset_name,
            hf_dataset_config=hf_dataset_config,
            hf_data_dir=hf_data_dir,
            hf_split=hf_split,
            hf_cache_dir=hf_cache_dir,
        )
        dataset: dict[SampleIndex, KnowledgeGraphDatasetItem] = {}
        for key, item in raw_dataset.items():
            row = dict(item)
            row["entity_dict"] = coerce_struct(row.get("entity_dict"))
            row["action_list"] = coerce_struct(row.get("action_list"))
            row["answer_list"] = coerce_struct(row.get("answer_list"))
            row["skill_list"] = coerce_struct(row.get("skill_list"))

            idx = row.get("sample_index", "<unknown>")
            if not isinstance(row.get("entity_dict"), dict):
                raise TypeError(
                    "knowledge_graph: entity_dict not dict after coercion "
                    f"(sample_index={idx}, type={type(row.get('entity_dict'))})"
                )
            if not isinstance(row.get("action_list"), list):
                raise TypeError(
                    "knowledge_graph: action_list not list after coercion "
                    f"(sample_index={idx}, type={type(row.get('action_list'))})"
                )
            if not isinstance(row.get("answer_list"), list):
                raise TypeError(
                    "knowledge_graph: answer_list not list after coercion "
                    f"(sample_index={idx}, type={type(row.get('answer_list'))})"
                )
            if not isinstance(row.get("skill_list"), list):
                raise TypeError(
                    "knowledge_graph: skill_list not list after coercion "
                    f"(sample_index={idx}, type={type(row.get('skill_list'))})"
                )

            question = row["question"]
            entity_dict = row["entity_dict"]
            answer_set = set(row["answer_list"])
            dataset[key] = KnowledgeGraphDatasetItem(
                question=question,
                entity_dict=entity_dict,
                answer_set=answer_set,
            )
        self._set_dataset(dataset)
        self.variable_list: Optional[list[Variable]] = None

    @staticmethod
    def _load_data(
        *,
        data_file_path: str,
        data_source: str,
        hf_dataset_name: str,
        hf_dataset_config: Optional[str],
        hf_data_dir: Optional[str],
        hf_split: str,
        hf_cache_dir: Optional[str],
    ) -> dict[str, dict[str, Any]]:
        source = data_source.lower()
        if source not in {"local", "huggingface", "auto"}:
            raise ValueError(
                f"Unsupported data_source: {data_source}. Use local, huggingface, or auto."
            )
        if source in {"local", "auto"} and os.path.exists(data_file_path):
            with open(data_file_path, "r") as f:
                return json.load(f)
        if source == "local":
            raise FileNotFoundError(
                "KnowledgeGraph local data file not found. "
                f"Missing path: {data_file_path}. "
                "Set data_source to 'huggingface' or 'auto' to load from HF."
            )
        try:
            from src.utils.hf_dataset_loader import load_hf_env_parquet, HfEnvLoadSpec
        except Exception as exc:
            raise RuntimeError(
                "Failed to import HF dataset loader. "
                "Ensure src/utils/hf_dataset_loader.py is available."
            ) from exc
        try:
            dataset = load_hf_env_parquet(
                HfEnvLoadSpec(
                    dataset_name=hf_dataset_name,
                    env="knowledge_graph",
                    split=hf_split,
                    cache_dir=hf_cache_dir,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load KnowledgeGraph dataset from Hugging Face: "
                f"{hf_dataset_name} [{hf_split}]."
            ) from exc

        logger = logging.getLogger(__name__)
        logger.info(
            "Loaded HF knowledge_graph: rows=%d cols=%s",
            len(dataset),
            dataset.column_names,
        )
        logger.info(
            "First row keys=%s",
            list(dataset[0].keys()) if len(dataset) else None,
        )

        required = ["question", "entity_dict", "answer_list"]
        missing = [key for key in required if key not in dataset.column_names]
        if missing:
            raise RuntimeError(
                "knowledge_graph dataset missing required columns: "
                f"{missing}. cols={dataset.column_names}"
            )

        data: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(dataset):
            row_dict = dict(row)
            sample_key = row_dict.get("sample_index")
            if sample_key is None:
                sample_key = str(idx)
            else:
                sample_key = str(sample_key)
            data[sample_key] = row_dict
        return data

    def _get_default_task_output(self) -> dict[str, Optional[str]]:
        return {"answer": None}

    @staticmethod
    def _get_action_pattern() -> str:
        return r"Action: (\w+)\((.+?)\)"

    @staticmethod
    def _extract_argument_str_from_agent_response(agent_response: str) -> Optional[str]:
        # AgentBench returns
        # re.findall(rf"{api_name}\((.+?)\)", agent_response)[0]
        # directly.
        action_pattern = KnowledgeGraph._get_action_pattern()
        argument_list_match = re.search(action_pattern, agent_response)
        if argument_list_match is None:
            return None
        start_index = argument_list_match.start(2) - 1
        left_parentheses_count = 0
        in_quote: Optional[str] = None
        escape_next = False
        for index in range(start_index, len(agent_response)):
            ch = agent_response[index]
            if in_quote is not None:
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ("'", '"'):
                in_quote = ch
                continue
            if ch == "(":
                left_parentheses_count += 1
            elif ch == ")":
                left_parentheses_count -= 1
                if left_parentheses_count == 0:
                    return agent_response[start_index + 1 : index]
            else:
                continue
        # Incomplete parentheses
        return None

    @staticmethod
    def _extract_argument_list_from_argument_str(
        argument_str: str, entity_list: list[str]
    ) -> list[str]:
        # AgentBench returns re.split(r "\s*,\s*", argument_str) directly.
        if len(argument_str) == 0:
            return []
        for entity in entity_list:
            if "," not in entity:
                continue
            entity_index = argument_str.find(entity)
            if entity_index == -1:
                # entity not found
                continue
            left_str = argument_str[:entity_index]
            left_argument_list = (
                KnowledgeGraph._extract_argument_list_from_argument_str(
                    left_str, entity_list
                )
            )
            right_str = argument_str[entity_index + len(entity) :]
            right_argument_list = (
                KnowledgeGraph._extract_argument_list_from_argument_str(
                    right_str, entity_list
                )
            )
            return left_argument_list + [entity] + right_argument_list
        # no entity found, split in the same way as AgentBench
        argument_list = re.split(r"\s*,\s*", argument_str)
        # ", hello_world" will be split into ["", "hello_world"], remove ""
        argument_list = [argument for argument in argument_list if argument != ""]
        return argument_list

    @staticmethod
    def _extract_variable_index_from_argument(raw_argument: str) -> Optional[int]:
        # The original implementation of AgentBench contains bugs.
        # The method will not check whether the variable index exists.
        possible_lower_prefix_list = ["#", "variable#", "variable #", "var#", "var #"]
        for prefix in possible_lower_prefix_list:
            if raw_argument.lower().startswith(prefix):
                variable_index_str = raw_argument[len(prefix) :]
                try:
                    variable_index = int(variable_index_str)
                    return variable_index
                except Exception as e:
                    raise e
        return None

    @staticmethod
    def _parse_agent_response(agent_response: str) -> AgentResponseParserResult:
        # AgentBench final_answer_pattern: r"Final Answer: #(\d+)"'
        final_answer_pattern = r"Final [Aa]nswer:\s*(?:[Vv]ar(?:iable)?\s*)?#(\d+)"
        action_pattern = KnowledgeGraph._get_action_pattern()
        if (
            final_answer_match := re.search(final_answer_pattern, agent_response)
        ) is not None:
            final_answer = final_answer_match.group(1)
            return AgentResponseParserResult(
                action=AgentAction.FINISH,
                content=final_answer,
                finish_reason=None,
            )
        if (action_match := re.search(action_pattern, agent_response)) is None:
            return AgentResponseParserResult(
                action=AgentAction.INVALID,
                content=None,
                finish_reason=(
                    f"Cannot find the pattern of action in agent response. "
                    f"final_answer_pattern: {final_answer_pattern} "
                    f"action_pattern: {action_pattern}"
                ),
            )
        api_name = action_match.group(1)
        argument_str = KnowledgeGraph._extract_argument_str_from_agent_response(
            agent_response
        )
        if argument_str is None:
            # Analogous to the DBBench environment, it can be regarded as the agent outputting a syntactically
            # incorrect SQL statement.
            return AgentResponseParserResult(
                action=AgentAction.EXECUTE,
                content=f"{api_name}()",  # Cannot find argument list
                finish_reason=None,
            )
        else:
            return AgentResponseParserResult(
                action=AgentAction.EXECUTE,
                content=f"{api_name}({argument_str})",
                finish_reason=None,
            )

    def _get_nonexistent_variable_error_message(self, variable_index: int) -> str:
        error_message = f"Variable #{variable_index} is not found in variable list. "
        assert self.variable_list is not None
        if len(self.variable_list) > 0:
            variable_list_str = "["
            for i in range(len(self.variable_list)):
                variable_list_str += f"#{i}, "
            variable_list_str = variable_list_str[:-2] + "]"
            error_message += f"Current variable list: {variable_list_str}."
        else:
            create_variable_prompt = (
                "The variable list is empty. You can use the following process to create the first variable: "
                "Use get_relations(var: Variable | str) to retrieve the relations connected to the input str or Variable, "
                "then use get_neighbors(var: Variable | str, relation: str) to retrieve the entities connected via the specified relation. "
            )
            error_message += create_variable_prompt
        return error_message

    def _reset(self, session: Session) -> None:
        current_dataset_item: KnowledgeGraphDatasetItem = (
            self._get_current_dataset_item()
        )
        prompt_item = self.chat_history_item_factory.construct(
            0, expected_role=Role.USER
        )
        prompt_content = prompt_item.content or ""
        # NOTE: execute_macro is intentionally NOT injected into the prompt.
        # The backend (_route_macro_to_server) is the sole caller of
        # execute_macro.  Hiding it from the chat prompt prevents the
        # Solver Agent from hallucinating rogue execute_macro calls.
        # execute_macro remains in get_valid_api_name_list() so _interact
        # still accepts the backend's injected Action string.
        session.chat_history.inject(prompt_item)
        session.chat_history.inject(
            self.chat_history_item_factory.construct(1, expected_role=Role.AGENT)
        )
        question = current_dataset_item.question
        entity_list = list(current_dataset_item.entity_dict.keys())
        session.chat_history.inject(
            {
                "role": Role.USER,
                "content": f"Question: {question}, Entities: {entity_list}",
            }
        )
        self.variable_list = []
        self.knowledge_graph_api.reset_cache()

    @staticmethod
    def _ensure_execute_macro_action(prompt: str) -> str:
        if not prompt or "execute_macro(" in prompt:
            return prompt
        macro_block = (
            "8. **execute_macro(tool_name: str, entities: list) -> Variable**\n"
            "    - Executes a registered macro tool to solve complex multi-hop queries.\n"
            "    - **Input**: The exact name of the tool, and a list of the exact entity strings from the Entities list.\n"
            "    - **Example**: Action: execute_macro(\"multi_entity_intersection_macro_generated_tool\", [\"Goat\", \"cows\", \"semi-firm\"])"
        )
        marker = "### Final Answer Format"
        if marker in prompt:
            return prompt.replace(marker, f"{macro_block}\n\n{marker}", 1)
        return f"{prompt}\n\n{macro_block}"

    @staticmethod
    def _parse_execute_macro_args(
        raw: str,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Parse execute_macro arguments.

        Accepts two formats:
        - Lightweight: execute_macro("tool_name", ["Entity1", "Entity2"])
          Returns (tool_name, {"entities": [...]})
        - Legacy:      execute_macro("tool_name", {"key": "value", ...})
          Returns (tool_name, {payload_dict})
        """
        text = (raw or "").strip()
        if not text:
            return None, None
        try:
            parsed = json.loads(f"[{text}]")
            if isinstance(parsed, list) and len(parsed) == 2:
                tool_name, second_arg = parsed
                if isinstance(tool_name, str):
                    # Lightweight format: (tool_name, [entities])
                    if isinstance(second_arg, list):
                        return tool_name, {"entities": second_arg}
                    # Pipe-delimited format: (tool_name, "E1|E2|E3")
                    if isinstance(second_arg, str):
                        entities = [e.strip() for e in second_arg.split("|") if e.strip()]
                        return tool_name, {"entities": entities}
                    # Legacy format: (tool_name, {payload})
                    if isinstance(second_arg, Mapping):
                        return tool_name, dict(second_arg)
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(f"({text})")
        except Exception:
            return None, None
        if isinstance(parsed, tuple) and len(parsed) == 2:
            tool_name, second_arg = parsed
            if isinstance(tool_name, str):
                if isinstance(second_arg, (list, tuple)):
                    return tool_name, {"entities": list(second_arg)}
                if isinstance(second_arg, str):
                    entities = [e.strip() for e in second_arg.split("|") if e.strip()]
                    return tool_name, {"entities": entities}
                if isinstance(second_arg, Mapping):
                    return tool_name, dict(second_arg)
        if isinstance(parsed, dict):
            tool_name = parsed.get("tool_name") or parsed.get("name")
            payload = parsed.get("payload")
            if isinstance(tool_name, str) and isinstance(payload, Mapping):
                return tool_name, dict(payload)
        return None, None

    def _build_macro_trace(
        self, session: Session, entity_list: list[str]
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        items = session.chat_history.get_value_length()
        for idx in range(items - 1):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role != Role.AGENT:
                continue
            content = item.content or ""
            if not content.startswith("Action:"):
                continue
            action_match = re.search(self._get_action_pattern(), content)
            if not action_match:
                continue
            action_name = action_match.group(1)
            arg_str = KnowledgeGraph._extract_argument_str_from_agent_response(content)
            args = []
            if arg_str is not None:
                args = KnowledgeGraph._extract_argument_list_from_argument_str(
                    arg_str, entity_list
                )
            next_item = session.chat_history.get_item_deep_copy(idx + 1)
            output = next_item.content if next_item.role == Role.USER else ""
            ok = True
            error = None
            if output.startswith("Error") or "Error in executing" in output:
                ok = False
                error = output
            trace.append(
                {
                    "action": action_name,
                    "args": args,
                    "ok": ok,
                    "output": output,
                    "error": error,
                }
            )
        return trace

    def _build_macro_actions_spec(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self.variable_list is not None

        def _resolve_arg(arg):
            if isinstance(arg, str):
                m = re.match(r"^#(\d+)$", arg.strip())
                if m:
                    idx = int(m.group(1))
                    if 0 <= idx < len(self.variable_list):
                        return self.variable_list[idx]
            return arg

        def _track_variable(new_var, msg):
            if new_var is not None and "<<NEW_VARIABLE>>" in msg:
                msg = msg.replace("<<NEW_VARIABLE>>", f"#{len(self.variable_list)}")
                self.variable_list.append(new_var)
            return msg

        def _safe_msg(msg, raw_args):
            for i, a in enumerate(raw_args):
                msg = msg.replace(f"<<ARGUMENT{i}>>", str(a))
            return msg

        def get_relations_fn(entity_or_var):
            raw = str(entity_or_var)
            resolved = _resolve_arg(entity_or_var)
            try:
                _, msg = self.knowledge_graph_api.get_relations(resolved)
                msg = _safe_msg(msg, [raw])
                return msg.replace("<<API_STR>>", f"get_relations({raw})")
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [raw])}"

        def get_neighbors_fn(entity_or_var, relation):
            raw = str(entity_or_var)
            resolved = _resolve_arg(entity_or_var)
            try:
                new_var, msg = self.knowledge_graph_api.get_neighbors(
                    resolved, relation
                )
                msg = _safe_msg(msg, [raw, relation])
                msg = msg.replace("<<API_STR>>", f"get_neighbors({raw}, {relation})")
                return _track_variable(new_var, msg)
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [raw, relation])}"

        def intersection_fn(var1, var2):
            r1, r2 = _resolve_arg(var1), _resolve_arg(var2)
            try:
                new_var, msg = self.knowledge_graph_api.intersection(r1, r2)
                msg = _safe_msg(msg, [str(var1), str(var2)])
                msg = msg.replace("<<API_STR>>", f"intersection({var1}, {var2})")
                return _track_variable(new_var, msg)
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [str(var1), str(var2)])}"

        def get_attributes_fn(var):
            resolved = _resolve_arg(var)
            try:
                _, msg = self.knowledge_graph_api.get_attributes(resolved)
                msg = _safe_msg(msg, [str(var)])
                return msg.replace("<<API_STR>>", f"get_attributes({var})")
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [str(var)])}"

        def argmax_fn(var, attribute):
            resolved = _resolve_arg(var)
            try:
                new_var, msg = self.knowledge_graph_api.argmax(resolved, attribute)
                msg = _safe_msg(msg, [str(var), attribute])
                msg = msg.replace("<<API_STR>>", f"argmax({var}, {attribute})")
                return _track_variable(new_var, msg)
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [str(var), attribute])}"

        def argmin_fn(var, attribute):
            resolved = _resolve_arg(var)
            try:
                new_var, msg = self.knowledge_graph_api.argmin(resolved, attribute)
                msg = _safe_msg(msg, [str(var), attribute])
                msg = msg.replace("<<API_STR>>", f"argmin({var}, {attribute})")
                return _track_variable(new_var, msg)
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [str(var), attribute])}"

        def count_fn(var):
            resolved = _resolve_arg(var)
            try:
                new_var, msg = self.knowledge_graph_api.count(resolved)
                msg = _safe_msg(msg, [str(var)])
                msg = msg.replace("<<API_STR>>", f"count({var})")
                return _track_variable(new_var, msg)
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [str(var)])}"

        actions_spec = {
            "get_relations": get_relations_fn,
            "get_neighbors": get_neighbors_fn,
            "intersection": intersection_fn,
            "get_attributes": get_attributes_fn,
            "argmax": argmax_fn,
            "argmin": argmin_fn,
            "count": count_fn,
        }
        raw_actions_spec = {
            "get_relations": self.knowledge_graph_api.get_relations,
            "get_neighbors": self.knowledge_graph_api.get_neighbors,
            "intersection": self.knowledge_graph_api.intersection,
            "get_attributes": self.knowledge_graph_api.get_attributes,
            "argmax": self.knowledge_graph_api.argmax,
            "argmin": self.knowledge_graph_api.argmin,
            "count": self.knowledge_graph_api.count,
        }
        return actions_spec, raw_actions_spec

    def _execute_macro(
        self,
        session: Session,
        tool_name: Optional[str],
        payload: Optional[Mapping[str, Any]],
        current_dataset_item: KnowledgeGraphDatasetItem,
        api_str: str,
    ) -> None:
        if not tool_name or not isinstance(payload, Mapping):
            session.chat_history.inject(
                {
                    "role": Role.USER,
                    "content": f"Error in executing '{api_str}'. Error: invalid_execute_macro_args",
                }
            )
            return
        payload_map: dict[str, Any] = dict(payload)
        state_dir_value = payload_map.get("state_dir")
        if not isinstance(state_dir_value, str) or not state_dir_value:
            state_dir_value = os.path.join("outputs", "macro_state")

        def _resolve_macro_tool_path(
            name: str, state_dir: str
        ) -> Optional[str]:
            filename = f"{name}.py"
            search_roots: list[str] = []
            if state_dir:
                search_roots.append(
                    os.path.abspath(os.path.join(state_dir, os.pardir, "tool_library"))
                )
            env_root = os.environ.get("TOOL_REGISTRY_ROOT")
            if env_root:
                search_roots.append(os.path.abspath(env_root))
            search_roots.append(os.path.abspath("outputs"))
            matches: list[str] = []
            for root in search_roots:
                if not root or not os.path.isdir(root):
                    continue
                for dirpath, _, files in os.walk(root):
                    if filename in files and "tool_library" in dirpath:
                        matches.append(os.path.join(dirpath, filename))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        tool_path = _resolve_macro_tool_path(tool_name, state_dir_value)
        if not tool_path:
            session.chat_history.inject(
                {
                    "role": Role.USER,
                    "content": f"Error in executing '{api_str}'. Error: macro_tool_not_found:{tool_name}",
                }
            )
            return
        try:
            with open(tool_path, "r", encoding="utf-8") as handle:
                code = handle.read()
        except Exception as exc:
            session.chat_history.inject(
                {
                    "role": Role.USER,
                    "content": f"Error in executing '{api_str}'. Error: macro_tool_read_failed:{exc}",
                }
            )
            return
        entity_list = list(current_dataset_item.entity_dict.keys())
        trace = self._build_macro_trace(session, entity_list)
        actions_spec, raw_actions_spec = self._build_macro_actions_spec()
        run_id = payload_map.get("run_id") or f"{session.task_name}_{session.sample_index}"
        os.makedirs(state_dir_value, exist_ok=True)
        last_user = session.chat_history.get_item_deep_copy(
            session.chat_history.get_value_length() - 1
        )
        env_observation = (
            last_user.content if last_user and last_user.role == Role.USER else ""
        )
        payload_map.setdefault("task_text", current_dataset_item.question)
        payload_map.setdefault("asked_for", current_dataset_item.question)
        payload_map["trace"] = trace
        payload_map["actions_spec"] = actions_spec
        payload_map["actions_spec_raw"] = raw_actions_spec
        payload_map["run_id"] = run_id
        payload_map["state_dir"] = state_dir_value
        payload_map.setdefault("env_observation", env_observation)
        payload_map["variable_list"] = self.variable_list
        # Ensure entities are always available. If the lightweight format
        # already provided them via the parser, keep those; otherwise fall
        # back to the full entity list from the dataset item.
        if not payload_map.get("entities"):
            payload_map["entities"] = entity_list

        allowed_modules = {
            "os",
            "json",
            "re",
            "math",
            "statistics",
            "itertools",
            "functools",
            "collections",
            "datetime",
            "typing",
        }

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in allowed_modules:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"module '{name}' is not allowed in execute_macro")

        safe_builtins = {
            "__import__": _safe_import,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "iter": iter,
            "next": next,
            "float": float,
            "int": int,
            "str": str,
            "dict": dict,
            "list": list,
            "set": set,
            "tuple": tuple,
            "abs": abs,
            "all": all,
            "any": any,
            "zip": zip,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "print": print,
            "map": map,
            "filter": filter,
            "round": round,
            "callable": callable,
            "Exception": Exception,
        }
        safe_globals = {
            "__builtins__": safe_builtins,
            "json": json,
            "re": re,
            "os": os,
        }
        safe_locals: dict[str, Any] = {}
        try:
            exec(code, safe_globals, safe_locals)
            run_fn = safe_locals.get("run") or safe_globals.get("run")
            if not callable(run_fn):
                raise RuntimeError("macro_run_not_found")
            result = run_fn(payload_map)
        except Exception as exc:
            session.chat_history.inject(
                {
                    "role": Role.USER,
                    "content": f"Error in executing '{api_str}'. Error: {exc}",
                }
            )
            return

        observation = None
        if isinstance(result, dict):
            for key in (
                "final_var",
                "result_var",
                "answer",
                "final_answer",
                "answer_recommendation",
                "result",
                "output",
            ):
                if key in result:
                    observation = result.get(key)
                    break
            if observation is None:
                observation = json.dumps(result, ensure_ascii=True, default=str)
        elif result is None:
            observation = "Macro executed."
        else:
            observation = str(result)

        execution_message = KnowledgeGraphAPI._construct_execution_message(
            str(observation)
        )
        execution_message = execution_message.replace("<<API_STR>>", api_str)
        session.chat_history.inject(
            {"role": Role.USER, "content": execution_message}
        )

    def _interact(self, session: Session) -> None:
        logger = logging.getLogger(__name__)
        # region Parse agent response, ensure the code pass the type check
        parser_result = KnowledgeGraph._parse_agent_response(
            session.chat_history.get_item_deep_copy(-1).content
        )
        assert self.variable_list is not None
        # endregion
        # region Execute action
        match parser_result.action:
            case AgentAction.EXECUTE:
                api_str = parser_result.content
                assert api_str is not None  # Type narrowing
                current_dataset_item = self._get_current_dataset_item()
                # region Get API name
                api_name = api_str.split("(")[0]
                if api_name not in KnowledgeGraphAPI.get_valid_api_name_list():
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": (
                                f"Unknown API name: {api_name}. "
                                f"Available API name list: {KnowledgeGraphAPI.get_valid_api_name_list()}"
                            ),
                        }
                    )
                    return
                # endregion
                # region Get argument list
                # + 1, -1 are used to remove the parentheses
                argument_str = api_str[len(api_name) + 1 : -1]
                if api_name == "execute_macro":
                    tool_name, payload = KnowledgeGraph._parse_execute_macro_args(
                        argument_str
                    )
                    self._execute_macro(
                        session=session,
                        tool_name=tool_name,
                        payload=payload,
                        current_dataset_item=current_dataset_item,
                        api_str=api_str,
                    )
                    return
                raw_argument_list = (
                    KnowledgeGraph._extract_argument_list_from_argument_str(
                        argument_str, list(current_dataset_item.entity_dict.keys())
                    )
                )
                processed_argument_list: list[str | Variable] = []
                for raw_argument in raw_argument_list:
                    processed_argument: str | Variable
                    if raw_argument in current_dataset_item.entity_dict.keys():
                        # region Argument is an entity
                        processed_argument = current_dataset_item.entity_dict[
                            raw_argument
                        ]
                        # endregion
                    else:
                        # region Extract variable index
                        try:
                            variable_index: Optional[int] = (
                                KnowledgeGraph._extract_variable_index_from_argument(
                                    raw_argument
                                )
                            )
                        except:  # noqa
                            session.chat_history.inject(
                                {
                                    "role": Role.USER,
                                    "content": (
                                        f"Cannot extract the variable index from the following API argument: "
                                        f"{raw_argument}"
                                    ),
                                }
                            )
                            return
                        # endregion
                        if variable_index is not None:
                            # region Change variable index to variable
                            try:
                                processed_argument = self.variable_list[variable_index]
                            except:  # noqa
                                error_message = (
                                    self._get_nonexistent_variable_error_message(
                                        variable_index
                                    )
                                )
                                session.chat_history.inject(
                                    {
                                        "role": Role.USER,
                                        "content": error_message,
                                    }
                                )
                                return
                            logger.info(
                                "KG resolve var: token=%s idx=%d type=%s program=%s",
                                raw_argument,
                                variable_index,
                                processed_argument.type,
                                processed_argument.program[:500],
                            )
                            # endregion
                        else:
                            # region Argument is neither an entity nor a variable
                            processed_argument = raw_argument
                            # endregion
                    processed_argument_list.append(processed_argument)
                # endregion
                # region Get callable API
                api: Callable[..., tuple[Variable | None, str]]
                match api_name:
                    case "get_relations":
                        api = self.knowledge_graph_api.get_relations
                    case "get_neighbors":
                        api = self.knowledge_graph_api.get_neighbors
                    case "intersection":
                        api = self.knowledge_graph_api.intersection
                    case "get_attributes":
                        api = self.knowledge_graph_api.get_attributes
                    case "argmax":
                        api = self.knowledge_graph_api.argmax
                    case "argmin":
                        api = self.knowledge_graph_api.argmin
                    case "count":
                        api = self.knowledge_graph_api.count
                    case _:
                        raise ValueError(f"An API name is not handled: {api_name}")
                # endregion
                # region Check argument count
                api_parameter_list: list[str] = inspect.getfullargspec(api).args
                if api_parameter_list[0] == "self":
                    api_parameter_list = api_parameter_list[1:]
                if len(api_parameter_list) != len(processed_argument_list):
                    # region Construct error_message
                    if len(api_parameter_list) > 1:
                        error_message = f"API {api_name} requires {len(api_parameter_list)} arguments, "
                    else:
                        error_message = f"API {api_name} requires {len(api_parameter_list)} argument, "
                    if len(processed_argument_list) > 1:
                        error_message += f"but {len(processed_argument_list)} arguments are provided."
                    else:
                        error_message += (
                            f"but {len(processed_argument_list)} argument is provided."
                        )
                    # endregion
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": error_message,
                        }
                    )
                    return
                # endregion
                # region Call API with arguments
                try:
                    call_start = time.monotonic()
                    new_variable, execution_message = api(*processed_argument_list)
                except TaskEnvironmentException:
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": (
                                "Observation: Error - SPARQL query timed out or node "
                                "explosion detected."
                            ),
                        }
                    )
                    return
                except KnowledgeGraphAPIException as e:
                    error_message = str(e)
                    # region Replace the template in error_message with real value
                    for raw_argument_index, raw_argument in enumerate(
                        raw_argument_list
                    ):
                        error_message = error_message.replace(
                            f"<<ARGUMENT{raw_argument_index}>>", raw_argument
                        )
                    callable_variable_str_list: list[str] = []
                    not_callable_variable_str_list: list[str] = []
                    for variable_index, variable in enumerate(self.variable_list):
                        if variable.is_callable():
                            callable_variable_str_list.append(f"#{variable_index}")
                        else:
                            not_callable_variable_str_list.append(f"#{variable_index}")
                    callable_variable_list_str = (
                        f"[{', '.join(callable_variable_str_list)}]"
                    )
                    not_callable_variable_list_str = (
                        f"[{', '.join(not_callable_variable_str_list)}]"
                    )
                    error_message = error_message.replace(
                        "<<CALLABLE_VARIABLE_LIST_STR>>", callable_variable_list_str
                    )
                    error_message = error_message.replace(
                        "<<NOT_CALLABLE_VARIABLE_LIST_STR>>",
                        not_callable_variable_list_str,
                    )
                    # endregion
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": f"Error in executing '{api_str}'. Error: {error_message}",
                        }
                    )
                    return
                except Exception as e:
                    session.task_output = self._get_default_task_output()
                    raise TaskEnvironmentException(str(e))
                elapsed_ms = (time.monotonic() - call_start) * 1000.0
                result_size = None
                if api_name == "get_relations":
                    cache_key = processed_argument_list[0]
                    result_size = len(
                        self.knowledge_graph_api.variable_to_relations_cache.get(
                            cache_key, []
                        )
                    )
                elif api_name == "get_attributes":
                    cache_key = processed_argument_list[0]
                    result_size = len(
                        self.knowledge_graph_api.variable_to_attributes_cache.get(
                            cache_key, []
                        )
                    )
                if isinstance(new_variable, Variable):
                    result_meta = f"type={new_variable.type} program_len={len(new_variable.program)}"
                else:
                    result_meta = "none"
                logger.info(
                    "KG tool exec: api=%s elapsed_ms=%.1f result_size=%s result_meta=%s",
                    api_name,
                    elapsed_ms,
                    result_size,
                    result_meta,
                )
                execution_message = execution_message.replace("<<API_STR>>", api_str)
                if "<<NEW_VARIABLE>>" in execution_message:
                    # the execution message contains a variable
                    execution_message = execution_message.replace(
                        "<<NEW_VARIABLE>>", f"#{len(self.variable_list)}"
                    )
                    assert isinstance(new_variable, Variable)  # Type narrowing
                    self.variable_list.append(new_variable)
                    logger.info(
                        "KG store var: new_idx=%d type=%s program=%s",
                        len(self.variable_list) - 1,
                        new_variable.type,
                        new_variable.program[:500],
                    )
                session.chat_history.inject(
                    {"role": Role.USER, "content": execution_message}
                )
                return
                # endregion
            case AgentAction.FINISH:
                assert parser_result.content is not None
                try:
                    answer_variable_index = int(parser_result.content)
                except:  # noqa
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": f"Cannot find variable index in final answer.",
                        }
                    )
                    return
                try:
                    answer_variable: Variable = self.variable_list[
                        answer_variable_index
                    ]
                except:  # noqa
                    error_message = self._get_nonexistent_variable_error_message(
                        answer_variable_index
                    )
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": error_message,
                        }
                    )
                    return
                try:
                    answer = self.knowledge_graph_api.final_execute(answer_variable)
                except Exception as e:
                    session.task_output = self._get_default_task_output()
                    raise TaskEnvironmentException(str(e))
                session.task_output = {"answer": "<SEP>".join(answer)}
                session.sample_status = SampleStatus.COMPLETED
                return
            case AgentAction.INVALID:
                session.sample_status = SampleStatus.AGENT_VALIDATION_FAILED
                session.finish_reason = parser_result.finish_reason
                session.task_output = self._get_default_task_output()
                return
        # endregion

    def _complete(self, session: Session) -> None:
        # region Preparation
        current_dataset_item: KnowledgeGraphDatasetItem = (
            self._get_current_dataset_item()
        )
        if session.task_output is None:
            # Handle extreme case, such as SampleStatus.TASK_UNKNOWN_ERROR.
            session.task_output = {}
        if session.task_output.get("answer", None) is not None:
            agent_answer_list = str(session.task_output["answer"]).split("<SEP>")
        else:
            agent_answer_list = None
        ground_truth_answer_set = current_dataset_item.answer_set
        session.expected_answer = {
            "answer_list": sorted(ground_truth_answer_set),
        }
        # endregion
        # region Calculate metrics
        # region Calculate f1_score
        f1_score: float
        if agent_answer_list is None:
            f1_score = 0
        else:
            agent_answer_set = set(agent_answer_list)
            true_positive = len(ground_truth_answer_set.intersection(agent_answer_set))
            false_positive = len(agent_answer_set - ground_truth_answer_set)
            false_negative = len(ground_truth_answer_set - agent_answer_set)
            if true_positive == 0:
                f1_score = 0
            else:
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)
                f1_score = 2 * precision * recall / (precision + recall)
        # endregion
        # region Calculate exact_match
        if agent_answer_list is None:
            exact_match = False
        else:
            exact_match = ground_truth_answer_set == set(agent_answer_list)
        # endregion
        # endregion
        # region Record evaluation results
        session.evaluation_record.outcome = SessionEvaluationOutcome.from_bool(
            exact_match
        )
        session.evaluation_record.detail_dict = {
            "f1_score": f1_score,
            "executable_flag": session.task_output.get("answer", None) is not None,
            # Do not set exact_match here, because it is already included in the outcome
        }
        # endregion
        # region Clean up
        self.variable_list = None
        # endregion

    def _release(self) -> None:
        return  # Do nothing

    def calculate_metric(
        self, session_partial_list: Sequence[SessionMetricCalculationPartial]
    ) -> MetricDict:
        # region Calculate general metrics
        skill_metric_dict = self._calculate_metric_based_on_skill(
            KnowledgeGraphSkillUtility, session_partial_list
        )
        difficulty_level_metric_dict = self._calculate_metric_based_on_difficulty_level(
            session_partial_list
        )
        overall_metric_dict = Task._calculate_overall_metric(session_partial_list)
        # endregion
        # region Calculate task-specific metrics
        f1_score_numerator: float = 0
        executable_rate_numerator: int = 0
        for session_partial in session_partial_list:
            if session_partial.evaluation_record.detail_dict is not None:
                f1_score = session_partial.evaluation_record.detail_dict.get(
                    "f1_score", 0.0
                )
                if isinstance(
                    f1_score, (int, float)
                ):  # The if statement is used for type narrowing
                    f1_score_numerator += float(f1_score)
                executable_flag = session_partial.evaluation_record.detail_dict.get(
                    "executable_flag", False
                )
                if isinstance(
                    executable_flag, bool
                ):  # The if statement is used for type narrowing
                    executable_rate_numerator += int(executable_flag)
        additional_metric_dict = {
            "f1_score": f1_score_numerator / len(session_partial_list),
            "executable_rate": executable_rate_numerator / len(session_partial_list),
        }
        overall_metric_dict["additional"] = additional_metric_dict
        # endregion
        metric_dict = {
            "skill": skill_metric_dict,
            "difficulty_level": difficulty_level_metric_dict,
            "overall": overall_metric_dict,
        }
        return metric_dict
