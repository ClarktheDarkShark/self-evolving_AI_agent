from pydantic import BaseModel, field_validator
from typing import Literal, Optional, Sequence, Mapping
import datetime
import re

from .general import SampleIndex
from .instance_factory import GeneralInstanceFactory


SampleOrderDescription = Literal["default"]  # Execute all the samples one by one


class LoggerConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    # See ConfigUtility.read_raw_config for the default value of log_file_path
    log_file_path: str
    logger_name: str


_TIMESTAMP_ANCHOR = datetime.datetime.now()


def get_predefined_timestamp_structure() -> dict[str, str]:
    """Return a stable timestamp mapping for formatting output paths."""

    return {
        "TIMESTAMP": _TIMESTAMP_ANCHOR.strftime("%Y-%m-%d-%H-%M-%S"),
        "TIMESTAMP_DATE": _TIMESTAMP_ANCHOR.strftime("%Y-%m-%d"),
        "TIMESTAMP_TIME": _TIMESTAMP_ANCHOR.strftime("%H-%M-%S"),
    }


class AssignmentConfig(BaseModel):
    task: GeneralInstanceFactory
    agent: GeneralInstanceFactory
    language_model_dict: Mapping[str, GeneralInstanceFactory]
    callback_dict: Mapping[str, GeneralInstanceFactory]
    output_dir: str
    sample_order: Sequence[SampleIndex] | SampleOrderDescription

    @field_validator("output_dir", mode="before")  # noqa
    @classmethod
    def output_path_validation(cls, value: str) -> str:
        assert isinstance(
            value, str
        ), f"'output_dir' must be a string, but got {type(value)}"
        return value.format(**get_predefined_timestamp_structure())

    @staticmethod
    def is_output_dir_equal(output_dir_1: str, output_dir_2: str) -> bool:
        def recover_output_dir(_output_dir: str) -> str:
            timestamp_pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
            date_pattern = r"\d{4}-\d{2}-\d{2}"
            time_pattern = r"\d{2}-\d{2}-\d{2}"
            _output_dir = re.sub(timestamp_pattern, "{TIMESTAMP}", _output_dir)
            _output_dir = re.sub(date_pattern, "{TIMESTAMP_DATE}", _output_dir)
            _output_dir = re.sub(time_pattern, "{TIMESTAMP_TIME}", _output_dir)
            return _output_dir

        return recover_output_dir(output_dir_1) == recover_output_dir(output_dir_2)


class EnvironmentConfig(BaseModel):
    """
    task_client: in config file, if environment_config.use_task_client_flag is set to False,
        then task_client will be None.
    """

    task_client: Optional[GeneralInstanceFactory]
    chat_history_item_factory_client: Optional[GeneralInstanceFactory]
    server_side_controller_address: Optional[str]
    interpreter_path: Optional[str]


class PathConfig(BaseModel):
    """
    The output path of the logger and the callbacks are not included in the PathConfig.
    """

    exception_record_file_path: str
    config_output_path: str
    session_list_output_path: str
    metric_output_path: str
    coredumpy_output_dir: str
