import ast
import json
import logging
import os
import re
from pydantic import BaseModel
from typing import Optional, Any, Sequence

from .container import OSInteractionContainer
from .utility import (
    CommandItem,
    CommandName,
)
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
    TaskReleaseException,
    Session,
    TaskName,
    Role,
    SessionEvaluationOutcome,
    MetricDict,
    SessionMetricCalculationPartial,
)
from src.factories.chat_history_item import ChatHistoryItemFactory


class OSInteractionSkillUtility(SkillUtility):
    _SKILL_TO_LEVEL_DICT = {
        key: 0
        for key in sorted(
            [
                "addgroup",
                "awk",
                "cat",
                "cd",
                "chage",
                "chgrp",
                "chmod",
                "chown",
                "chsh",
                "cp",
                "echo",
                "exit",
                "find",
                "gpasswd",
                "grep",
                "groupadd",
                "ln",
                "ls",
                "mkdir",
                "mv",
                "rm",
                "sed",
                "sleep",
                "tee",
                "touch",
                "useradd",
                "usermod",
                "vi",
                "wc",
            ]
        )
    }


class EvaluationInfo(BaseModel):
    ground_truth_command_item: CommandItem
    evaluation_command_item: CommandItem


class OSInteractionDatasetItem(DatasetItem):
    instruction: str
    initialization_command_item: CommandItem
    evaluation_info: EvaluationInfo
    skill_list: list[str]

    def get_skill_list(self) -> list[str]:
        return self.skill_list

    def get_difficulty_level(self) -> int:
        return 0


class OSInteraction(Task[OSInteractionDatasetItem]):
    def __init__(
        self,
        task_name: TaskName,
        chat_history_item_factory: ChatHistoryItemFactory,
        data_file_path: str,
        max_round: int,
        command_execution_timeout: int,
        data_source: str = "auto",
        hf_dataset_name: str = "csyq/LifelongAgentBench",
        hf_dataset_config: Optional[str] = None,
        hf_data_dir: Optional[str] = "os_interaction",
        hf_split: str = "train",
        hf_cache_dir: Optional[str] = None,
    ):
        super().__init__(task_name, chat_history_item_factory, max_round)
        data = self._load_data(
            data_file_path=data_file_path,
            data_source=data_source,
            hf_dataset_name=hf_dataset_name,
            hf_dataset_config=hf_dataset_config,
            hf_data_dir=hf_data_dir,
            hf_split=hf_split,
            hf_cache_dir=hf_cache_dir,
        )
        # self.dataset can also be implemented as a list, but it is implemented as a dict for forward compatibility
        dataset: dict[SampleIndex, OSInteractionDatasetItem] = {}
        for key, item in data.items():
            dataset_item = OSInteraction._construct_dataset_item(item)
            dataset[key] = dataset_item
            for skill in dataset_item.get_skill_list():
                assert OSInteractionSkillUtility.is_valid_skill(skill)
        self._set_dataset(dataset)
        self.container: Optional[OSInteractionContainer] = None
        self.command_execution_timeout = command_execution_timeout

    @staticmethod
    def _construct_dataset_item(entry: dict[str, Any]) -> OSInteractionDatasetItem:
        entry = dict(entry)
        entry["initialization_command_item"] = OSInteraction._coerce_struct(
            entry.get("initialization_command_item")
        )
        entry["evaluation_info"] = OSInteraction._coerce_struct(
            entry.get("evaluation_info")
        )
        entry["skill_list"] = OSInteraction._coerce_struct(entry.get("skill_list"))
        sample_index = entry.get("sample_index", "<unknown>")
        if not isinstance(entry.get("initialization_command_item"), dict):
            raise ValueError(
                "os_interaction: init_command_item not dict after coercion "
                f"(sample_index={sample_index}, type={type(entry.get('initialization_command_item'))})"
            )
        if not isinstance(entry.get("evaluation_info"), dict):
            raise ValueError(
                "os_interaction: evaluation_info not dict after coercion "
                f"(sample_index={sample_index}, type={type(entry.get('evaluation_info'))})"
            )
        if not isinstance(entry.get("skill_list"), list):
            raise ValueError(
                "os_interaction: skill_list not list after coercion "
                f"(sample_index={sample_index}, type={type(entry.get('skill_list'))})"
            )
        # Use del instead of pop to promote early detection of errors
        del entry["raw_entry_hash"]
        return OSInteractionDatasetItem.model_validate(entry)

    @staticmethod
    def _coerce_struct(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if value is None:
            return value
        if not isinstance(value, str):
            return value
        s = value.strip()
        if not s:
            return value
        try:
            if s[0] in "{[" and s[-1] in "}]":
                return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return value

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
                "OSInteraction local data file not found. "
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
                    env="os_interaction",
                    split=hf_split,
                    cache_dir=hf_cache_dir,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load OSInteraction dataset from Hugging Face: "
                f"{hf_dataset_name} [{hf_split}]."
            ) from exc

        logger = logging.getLogger(__name__)
        logger.info(
            "Loaded HF os_interaction: rows=%d cols=%s",
            len(dataset),
            dataset.column_names,
        )
        logger.info(
            "First row keys=%s",
            list(dataset[0].keys()) if len(dataset) else None,
        )

        required = list(OSInteractionDatasetItem.model_fields.keys()) + [
            "raw_entry_hash"
        ]
        missing = [key for key in required if key not in dataset.column_names]
        if missing:
            raise RuntimeError(
                "os_interaction dataset missing required columns: "
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
    def _parse_agent_response(
        agent_response: str,
    ) -> AgentResponseParserResult:
        action_match = re.search(r"Act:\s*(.+)", agent_response)
        if action_match is None:
            return AgentResponseParserResult(
                action=AgentAction.INVALID,
                content=None,
                finish_reason=r'Cannot extract action from agent response. Pattern: "Act:\s*(.+)"',
            )
        action_str = action_match.group(1)
        if action_str.lower().startswith("bash"):
            content_list = re.findall(r"```bash\n(.*?)\n```", agent_response, re.DOTALL)
            content = "\n\n".join(content_list)
            return AgentResponseParserResult(
                action=AgentAction.EXECUTE,
                content=content,
                finish_reason=None,
            )
        elif action_str.lower().startswith("finish"):
            return AgentResponseParserResult(
                action=AgentAction.FINISH,
                content=None,
                finish_reason=None,
            )
        else:
            return AgentResponseParserResult(
                action=AgentAction.INVALID,
                content=None,
                finish_reason=r'Invalid action string matched by pattern "Act:\s*(.+)"',
            )

    def _reset(self, session: Session) -> None:
        current_dataset_item: OSInteractionDatasetItem = (
            self._get_current_dataset_item()
        )
        self.container = OSInteractionContainer(self.command_execution_timeout)
        command_item = current_dataset_item.initialization_command_item
        try:
            execution_result = self.container.execute_independent(command_item)
        except Exception as e:
            raise TaskEnvironmentException(str(e))
            # region Handle initialization failure
        if execution_result.timeout_flag or execution_result.exit_code != 0:
            raise TaskEnvironmentException(
                f"Initialization failed with exit code {execution_result.exit_code}\n"
                f"Output: {execution_result.output}\n"
                f"Command Item: {command_item}"
            )
        # endregion
        session.chat_history.inject(
            self.chat_history_item_factory.construct(0, expected_role=Role.USER)
        )
        session.chat_history.inject(
            self.chat_history_item_factory.construct(1, expected_role=Role.AGENT)
        )
        session.chat_history.inject(
            {"role": Role.USER, "content": current_dataset_item.instruction}
        )

    def _interact(self, session: Session) -> None:
        # region Parse agent response, ensure the code pass the type check
        parser_response: AgentResponseParserResult = (
            OSInteraction._parse_agent_response(
                session.chat_history.get_item_deep_copy(-1).content
            )
        )
        assert self.container is not None
        # endregion
        # region Execute action
        match parser_response.action:
            case AgentAction.EXECUTE:
                try:
                    assert (
                        parser_response.content is not None
                    ), "The content of the bash command is not provided."
                    command_execution_result = self.container.execute_independent(
                        CommandItem(
                            command_name=CommandName.BASH,
                            script=parser_response.content,
                        )
                    )
                    command_output: str | None
                    if command_execution_result.timeout_flag:
                        command_output = (
                            f"The command is marked as timeout since it did not finish "
                            f"within {self.command_execution_timeout} seconds."
                        )
                    else:
                        command_output = command_execution_result.output
                except Exception as e:
                    session.task_output = self._get_default_task_output()
                    raise TaskEnvironmentException(str(e))
                session.chat_history.inject(
                    {
                        "role": Role.USER,
                        "content": (
                            "The output of the OS:\n\n" + command_output
                            if command_output
                            else "The output of the OS is empty."
                        ),
                    }
                )
                return
            case AgentAction.FINISH:
                session.sample_status = SampleStatus.COMPLETED
                session.task_output = {"answer": parser_response.content}
                return
            case AgentAction.INVALID:
                session.sample_status = SampleStatus.AGENT_VALIDATION_FAILED
                session.finish_reason = parser_response.finish_reason
                session.task_output = self._get_default_task_output()
                return
            case _:
                raise NotImplementedError(
                    f'Unhandled action: "{parser_response.action}"'
                )
        # endregion

    def _complete(self, session: Session) -> None:
        # region Prepare Variables, ensure the code pass the type check
        current_dataset_item: OSInteractionDatasetItem = (
            self._get_current_dataset_item()
        )
        assert self.container is not None
        session.expected_answer = {
            "ground_truth_command": current_dataset_item.evaluation_info.ground_truth_command_item.model_dump(),
            "evaluation_command": current_dataset_item.evaluation_info.evaluation_command_item.model_dump(),
        }
        # endregion
        # region Check the correctness of the answer
        # If the command times out, exit_code will be 0 and the answer is considered as incorrect.
        try:
            correct_flag = (
                self.container.execute_independent(
                    current_dataset_item.evaluation_info.evaluation_command_item,
                ).exit_code
                == 0
            )
        except Exception as e:
            raise TaskEnvironmentException(str(e))
        session.evaluation_record.outcome = SessionEvaluationOutcome.from_bool(
            correct_flag
        )
        # endregion
        # region Clean the container for next sample
        try:
            self.container.terminate()
        except:  # noqa
            pass
        finally:
            self.container = None
        # endregion

    def _release(self) -> None:
        if self.container is not None:
            try:
                self.container.terminate()
            except Exception as e:
                raise TaskReleaseException(str(e))

    def calculate_metric(
        self, session_partial_list: Sequence[SessionMetricCalculationPartial]
    ) -> MetricDict:
        skill_metric_dict = self._calculate_metric_based_on_skill(
            OSInteractionSkillUtility, session_partial_list
        )
        difficulty_level_metric_dict = self._calculate_metric_based_on_difficulty_level(
            session_partial_list
        )
        overall_metric_dict = Task._calculate_overall_metric(session_partial_list)
        metric_dict = {
            "skill": skill_metric_dict,
            "difficulty_level": difficulty_level_metric_dict,
            "overall": overall_metric_dict,
        }
        return metric_dict
