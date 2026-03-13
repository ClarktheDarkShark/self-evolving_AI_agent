import ast
import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
from src.self_evolving_agent import kg_utils as _kg_utils


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
        entity_dict: Optional[dict[str, str]] = None,
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

        def _resolve_entity(arg: Any) -> Any:
            """Translate a string entity name to its canonical Freebase MID.

            _interact does this via current_dataset_item.entity_dict before
            calling the API.  Macro wrapper functions must do the same so that
            entity names like "cows" are resolved to the correct KG node instead
            of going through the unreliable SPARQL name-search fallback.
            """
            if isinstance(arg, str) and entity_dict and arg in entity_dict:
                return entity_dict[arg]
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
            resolved = _resolve_entity(_resolve_arg(entity_or_var))
            try:
                _, msg = self.knowledge_graph_api.get_relations(resolved)
                msg = _safe_msg(msg, [raw])
                return msg.replace("<<API_STR>>", f"get_relations({raw})")
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [raw])}"

        def get_neighbors_fn(entity_or_var, relation):
            raw = str(entity_or_var)
            resolved = _resolve_entity(_resolve_arg(entity_or_var))
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

        def union_fn(var1, var2):
            r1, r2 = _resolve_arg(var1), _resolve_arg(var2)
            try:
                new_var, msg = self.knowledge_graph_api.union(r1, r2)
                msg = _safe_msg(msg, [str(var1), str(var2)])
                msg = msg.replace("<<API_STR>>", f"union({var1}, {var2})")
                return _track_variable(new_var, msg)
            except Exception as exc:
                return f"Error: {_safe_msg(str(exc), [str(var1), str(var2)])}"

        def difference_fn(var1, var2):
            r1, r2 = _resolve_arg(var1), _resolve_arg(var2)
            try:
                new_var, msg = self.knowledge_graph_api.difference(r1, r2)
                msg = _safe_msg(msg, [str(var1), str(var2)])
                msg = msg.replace("<<API_STR>>", f"difference({var1}, {var2})")
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
            "union": union_fn,
            "difference": difference_fn,
            "get_attributes": get_attributes_fn,
            "argmax": argmax_fn,
            "argmin": argmin_fn,
            "count": count_fn,
        }
        raw_actions_spec = {
            "get_relations": self.knowledge_graph_api.get_relations,
            "get_neighbors": self.knowledge_graph_api.get_neighbors,
            "intersection": self.knowledge_graph_api.intersection,
            "union": self.knowledge_graph_api.union,
            "difference": self.knowledge_graph_api.difference,
            "get_attributes": self.knowledge_graph_api.get_attributes,
            "argmax": self.knowledge_graph_api.argmax,
            "argmin": self.knowledge_graph_api.argmin,
            "count": self.knowledge_graph_api.count,
        }
        return actions_spec, raw_actions_spec

    def evaluate_generated_macro(
        self, tool_code: str, execution_payload_json: str
    ) -> str:
        """Server-side live execution sandbox for a generated Macro tool.

        The controller cannot access self.knowledge_graph_api directly (it
        lives behind an HTTP proxy that cannot serialize KnowledgeGraphAPI).
        This method runs on the task server where the KG objects are native,
        builds a shadow Proxy Interceptor that mirrors the real KG API without
        mutating self.variable_list, exec()s the tool code, and returns the
        result as a JSON string.

        Args:
            tool_code: Python source code string for the macro tool.
            execution_payload_json: JSON-serialized execution payload.  Must
                NOT contain an 'actions_spec' key — this method builds its own.

        Returns:
            JSON string of the run() result dict, or an error-description JSON.
        """
        logger = logging.getLogger(__name__)
        def _ssot_crash(exc: Exception) -> str:
            return json.dumps(
                {
                    "status": "ERROR",
                    "final_variable": None,
                    "observation": f"Crash: {str(exc)}",
                }
            )
        try:
            base_payload: dict = json.loads(execution_payload_json)
        except Exception as exc:
            return _ssot_crash(exc)

        kg_api = self.knowledge_graph_api
        real_var_list = self.variable_list or []

        # Resolve the canonical entity_dict for this sample so that proxy
        # functions apply the same MID substitution that _interact performs.
        # Without this, string entity names (e.g. "cows") go through the
        # unreliable SPARQL name-search fallback and may resolve to the wrong
        # Freebase node (e.g. a dog-breed entity instead of cattle).
        try:
            _entity_dict: dict[str, str] = self._get_current_dataset_item().entity_dict
        except Exception:
            _entity_dict = {}

        # --- Shadow Proxy Interceptor (same design as controller_toolgen.py) ---
        shadow_var_list: list = []
        shadow_base: int = len(real_var_list)

        # Cache key snapshots for cleanup.
        pre_cache_rel_keys: set = set(kg_api.variable_to_relations_cache.keys())
        pre_cache_attr_keys: set = set(
            getattr(kg_api, "variable_to_attributes_cache", {}).keys()
        )

        # Tripwire: prevent runaway tool from hammering the KG backend.
        proxy_call_count = 0
        MAX_PROXY_CALLS = 15

        def _check_tripwire() -> None:
            nonlocal proxy_call_count
            proxy_call_count += 1
            if proxy_call_count > MAX_PROXY_CALLS:
                raise RuntimeError(
                    f"Proxy Execution Tripwire: tool exceeded {MAX_PROXY_CALLS} "
                    "live KG calls. Refactor to reduce query volume."
                )

        def _resolve(arg: Any) -> Any:
            s = str(arg).strip()
            m = re.match(r"^#(\d+)$", s)
            if m:
                idx = int(m.group(1))
                shadow_idx = idx - shadow_base
                if 0 <= shadow_idx < len(shadow_var_list):
                    return shadow_var_list[shadow_idx]
                if 0 <= idx < len(real_var_list):
                    return real_var_list[idx]
            return arg

        def _track(new_var: Any, msg: str) -> str:
            if new_var is not None and "<<NEW_VARIABLE>>" in msg:
                idx = shadow_base + len(shadow_var_list)
                msg = msg.replace("<<NEW_VARIABLE>>", f"#{idx}")
                shadow_var_list.append(new_var)
            return msg

        def _sub(msg: str, raw_args: list) -> str:
            for i, a in enumerate(raw_args):
                msg = msg.replace(f"<<ARGUMENT{i}>>", str(a))
            return msg

        def _resolve_entity_mid(arg: Any) -> Any:
            """Translate a string entity name to its canonical Freebase MID,
            mirroring the substitution _interact performs via entity_dict."""
            if isinstance(arg, str) and _entity_dict and arg in _entity_dict:
                return _entity_dict[arg]
            return arg

        def proxy_get_relations(entity_or_var: Any) -> str:
            _check_tripwire()
            raw = str(entity_or_var)
            resolved = _resolve_entity_mid(_resolve(entity_or_var))
            try:
                _, msg = kg_api.get_relations(resolved)
                return _sub(msg, [raw]).replace("<<API_STR>>", f"get_relations({raw})")
            except Exception as exc:
                return f"Error: {_sub(str(exc), [raw])}"

        def proxy_get_neighbors(entity_or_var: Any, relation: str) -> str:
            _check_tripwire()
            raw = str(entity_or_var)
            resolved = _resolve_entity_mid(_resolve(entity_or_var))
            try:
                new_var, msg = kg_api.get_neighbors(resolved, relation)
                msg = _sub(msg, [raw, relation]).replace(
                    "<<API_STR>>", f"get_neighbors({raw}, {relation})"
                )
                return _track(new_var, msg)
            except Exception as exc:
                return f"Error: {_sub(str(exc), [raw, relation])}"

        def proxy_intersection(var1: Any, var2: Any) -> str:
            _check_tripwire()
            try:
                new_var, msg = kg_api.intersection(_resolve(var1), _resolve(var2))
                msg = _sub(msg, [str(var1), str(var2)]).replace(
                    "<<API_STR>>", f"intersection({var1}, {var2})"
                )
                return _track(new_var, msg)
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var1), str(var2)])}"

        def proxy_union(var1: Any, var2: Any) -> str:
            _check_tripwire()
            try:
                new_var, msg = kg_api.union(_resolve(var1), _resolve(var2))
                msg = _sub(msg, [str(var1), str(var2)]).replace(
                    "<<API_STR>>", f"union({var1}, {var2})"
                )
                return _track(new_var, msg)
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var1), str(var2)])}"

        def proxy_difference(var1: Any, var2: Any) -> str:
            _check_tripwire()
            try:
                new_var, msg = kg_api.difference(_resolve(var1), _resolve(var2))
                msg = _sub(msg, [str(var1), str(var2)]).replace(
                    "<<API_STR>>", f"difference({var1}, {var2})"
                )
                return _track(new_var, msg)
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var1), str(var2)])}"

        def proxy_get_attributes(var: Any) -> str:
            _check_tripwire()
            try:
                _, msg = kg_api.get_attributes(_resolve(var))
                return _sub(msg, [str(var)]).replace("<<API_STR>>", f"get_attributes({var})")
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var)])}"

        def proxy_argmax(var: Any, attribute: str) -> str:
            _check_tripwire()
            try:
                new_var, msg = kg_api.argmax(_resolve(var), attribute)
                msg = _sub(msg, [str(var), attribute]).replace(
                    "<<API_STR>>", f"argmax({var}, {attribute})"
                )
                return _track(new_var, msg)
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var), attribute])}"

        def proxy_argmin(var: Any, attribute: str) -> str:
            _check_tripwire()
            try:
                new_var, msg = kg_api.argmin(_resolve(var), attribute)
                msg = _sub(msg, [str(var), attribute]).replace(
                    "<<API_STR>>", f"argmin({var}, {attribute})"
                )
                return _track(new_var, msg)
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var), attribute])}"

        def proxy_count(var: Any) -> str:
            _check_tripwire()
            try:
                new_var, msg = kg_api.count(_resolve(var))
                return _track(new_var, _sub(msg, [str(var)]).replace(
                    "<<API_STR>>", f"count({var})"
                ))
            except Exception as exc:
                return f"Error: {_sub(str(exc), [str(var)])}"

        proxy_actions_spec = {
            "get_relations": proxy_get_relations,
            "get_neighbors": proxy_get_neighbors,
            "intersection": proxy_intersection,
            "union": proxy_union,
            "difference": proxy_difference,
            "get_attributes": proxy_get_attributes,
            "argmax": proxy_argmax,
            "argmin": proxy_argmin,
            "count": proxy_count,
        }

        # Build payload with proxy injected.
        payload = dict(base_payload)
        payload["actions_spec"] = proxy_actions_spec

        result: dict = {}
        safe_globals = {
            "__builtins__": __builtins__,
            "json": json,
            "re": re,
            "os": os,
            "kg_utils": _kg_utils,
        }
        try:
            exec(tool_code, safe_globals)  # noqa: S102
            run_fn = safe_globals.get("run")
            if not callable(run_fn):
                return _ssot_crash(RuntimeError("run() not found in generated tool"))
            # Run in a thread with a generous timeout. The tripwire (15 calls)
            # is the primary safeguard; the timeout is the backstop for pure
            # CPU-bound runaway loops.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_fn, payload)
                result = future.result(timeout=20.0)
            # Shape Verifier: catch non-COUNT tasks that return a raw scalar instead
            # of a Variable ID pointer.  COUNT/COUNTING_INTERSECTOR tasks are
            # explicitly excluded: KnowledgeGraphAPI.count() returns a new
            # Variable(type="type.int") whose string representation IS a pointer
            # (e.g. "#5").  Rejecting that pointer as a "shape mismatch" was
            # incorrect and caused valid counter tools to be silenced.
            if isinstance(result, dict) and result.get("status") == "SUCCESS":
                archetype_upper = str(payload.get("target_archetype", "")).upper()
                is_count_task = "COUNT" in archetype_upper
                is_extractor = "EXTRACTOR" in archetype_upper
                var_val = str(result.get("final_variable", "")).strip()
                is_pointer = var_val.startswith("#")
                if not is_count_task and not is_extractor and not is_pointer:
                    result["status"] = "ERROR"
                    result["final_variable"] = None
                    result["observation"] = (
                        "Shape Mismatch: Task requires an entity node, "
                        "but the tool returned a numeric count."
                    )
        except FutureTimeoutError:
            result = {
                "status": "ERROR",
                "final_variable": None,
                "observation": "Crash: run() exceeded 20s server-side timeout",
            }
        except Exception as exc:
            result = {
                "status": "ERROR",
                "final_variable": None,
                "observation": f"Crash: {str(exc)}",
            }
        finally:
            # Purge shadow cache entries to keep kg_api.variable_to_relations_cache clean.
            for k in list(kg_api.variable_to_relations_cache.keys()):
                if k not in pre_cache_rel_keys:
                    try:
                        del kg_api.variable_to_relations_cache[k]
                    except KeyError:
                        pass
            attr_cache = getattr(kg_api, "variable_to_attributes_cache", None)
            if isinstance(attr_cache, dict):
                for k in list(attr_cache.keys()):
                    if k not in pre_cache_attr_keys:
                        try:
                            del attr_cache[k]
                        except KeyError:
                            pass

        try:
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.warning("evaluate_generated_macro: result serialization failed: %s", exc)
            return _ssot_crash(exc)

    def _execute_macro(
        self,
        session: Session,
        tool_name: Optional[str],
        payload: Optional[Mapping[str, Any]],
        current_dataset_item: KnowledgeGraphDatasetItem,
        api_str: str,
    ) -> None:
        logger = logging.getLogger(__name__)
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
            # TOCTOU guard: Reflection Forge may delete files between the
            # os.walk and this getmtime call.  Filter out any path that has
            # already disappeared rather than letting sort() throw.
            surviving: list[tuple[float, str]] = []
            for p in matches:
                try:
                    surviving.append((os.path.getmtime(p), p))
                except FileNotFoundError:
                    pass
            if not surviving:
                return None
            surviving.sort(key=lambda x: x[0], reverse=True)
            return surviving[0][1]

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
        actions_spec, raw_actions_spec = self._build_macro_actions_spec(
            current_dataset_item.entity_dict
        )
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

        # ── Inject semantic keys stripped by the lightweight action string ──
        # execute_macro("tool", "E1|E2") carries only entity names; the Tool
        # Invoker's target_concept and domain_hints never reach this point.
        # Derive them here from the dataset question so score_relations
        # receives a meaningful answer-type noun rather than None / the full
        # sentence, which causes wrong-relation selection (e.g., permaculture
        # over food.cheese_milk_source when target_concept="products").
        #
        # Use setdefault throughout: if the Tool Invoker ever starts passing
        # these keys in the action string, those values will be preserved.
        if not payload_map.get("target_concept"):
            _exec_q = current_dataset_item.question or ""
            _exec_q_part = (
                _exec_q.split("Entities:")[0].strip()
                if "Entities:" in _exec_q
                else _exec_q
            )
            _exec_wh_m = re.search(
                r"(?i)\bwhat\s+(?:\w[\w\-]*\s+){0,8}(\w[\w\-]*)"
                r"\s+(?:is|are|was|were|does|do|exist\b)",
                _exec_q_part,
            )
            if _exec_wh_m:
                payload_map["target_concept"] = _exec_wh_m.group(1).strip().lower()

        payload_map.setdefault("domain_hints", [])

        # For ATTRIBUTE_INTERSECTOR: the last entity is the attribute literal
        # and should be resolved against itself as target_concept.
        _exec_entities = payload_map.get("entities") or entity_list
        # The lightweight action string execute_macro("tool", "E1|E2") carries
        # no target_archetype — infer it from the tool name so the injection
        # guards below fire correctly.
        if not payload_map.get("target_archetype") and tool_name:
            _tn = tool_name.upper()
            if "ATTRIBUTE_INTERSECTOR" in _tn:
                payload_map["target_archetype"] = "ATTRIBUTE_INTERSECTOR"
            elif "COUNTING_INTERSECTOR" in _tn:
                payload_map["target_archetype"] = "COUNTING_INTERSECTOR"
            elif "INTERSECT" in _tn:
                payload_map["target_archetype"] = "INTERSECTOR"
            elif "SHARED_TRAIT" in _tn or "PIVOT" in _tn:
                payload_map["target_archetype"] = "SHARED_TRAIT_PIVOT"
            elif "ATTRIBUTE_EXTRACTOR" in _tn:
                payload_map["target_archetype"] = "ATTRIBUTE_EXTRACTOR"
            elif "COUNTER" in _tn:
                payload_map["target_archetype"] = "COUNTER"
        _exec_archetype = str(payload_map.get("target_archetype", "")).upper()
        if (
            "ATTRIBUTE_INTERSECTOR" in _exec_archetype
            and _exec_entities
            and not payload_map.get("attribute_target_concept")
        ):
            payload_map["attribute_target_concept"] = (
                payload_map.get("target_concept") or _exec_entities[-1]
            )
        payload_map.setdefault(
            "attribute_domain_hints", payload_map.get("domain_hints") or []
        )
        # ── end semantic key injection ───────────────────────────────────────

        safe_globals = {
            "__builtins__": __builtins__,
            "json": json,
            "re": re,
            "os": os,
            "kg_utils": _kg_utils,
        }
        logger.info(
            f"[MACRO EXEC] Tool: {tool_name} | Concept: {payload_map.get('target_concept')} | Hints: {payload_map.get('domain_hints')}"
        )
        try:
            exec(code, safe_globals)
            run_fn = safe_globals.get("run")
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

        if not isinstance(result, dict):
            session.chat_history.inject(
                {
                    "role": Role.USER,
                    "content": (
                        f"Error in executing '{api_str}'. Error: "
                        "macro_invalid_output_schema: expected SSOT dict"
                    ),
                }
            )
            return

        ssot_keys = {"status", "final_variable", "observation"}
        if set(result.keys()) != ssot_keys:
            session.chat_history.inject(
                {
                    "role": Role.USER,
                    "content": (
                        f"Error in executing '{api_str}'. Error: "
                        "macro_invalid_output_schema: expected exact keys "
                        "['status','final_variable','observation']"
                    ),
                }
            )
            return

        status = str(result.get("status") or "")
        final_variable = result.get("final_variable")
        if status == "SUCCESS" and final_variable not in (None, ""):
            obs_text = result.get("observation", "")
            observation = f"{final_variable} - {obs_text}" if obs_text else str(final_variable)
        else:
            observation = json.dumps(result, ensure_ascii=True, default=str)

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
                    case "union":
                        api = self.knowledge_graph_api.union
                    case "difference":
                        api = self.knowledge_graph_api.difference
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
