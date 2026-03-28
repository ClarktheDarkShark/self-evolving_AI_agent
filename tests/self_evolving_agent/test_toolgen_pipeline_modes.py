import os
import pathlib
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.typings import ChatHistory
from src.self_evolving_agent.tool_registry import ToolMetadata
from src.toolgen.config import get_toolgen_pipeline_config
from src.toolgen.prompts import get_toolgen_system_prompt
from src.toolgen.pipelines.baseline import BaselineToolgenPipeline


def _make_tool(name: str) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        signature="run(payload: dict) -> dict",
        description="test tool",
        creation_time="now",
    )


class _Invoker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        chat_history: ChatHistory,
        name_prefix: str,
    ) -> ToolMetadata:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "name_prefix": name_prefix,
            }
        )
        return _make_tool(f"{name_prefix}dummy_generated_tool")


def test_baseline_pipeline_invokes_every_task() -> None:
    invoker = _Invoker()
    pipeline = BaselineToolgenPipeline(
        build_user_prompt=lambda task, history: f"prompt::{task}",
        invoke_toolgen=invoker,
        system_prompt_selector=lambda pipeline_name, env: "BASELINE_PROMPT",
        name_prefix="",
    )
    history = ChatHistory()
    assert pipeline.maybe_generate_tools("db_bench", "task1", history)
    assert pipeline.maybe_generate_tools("db_bench", "task2", history)
    assert pipeline.maybe_generate_tools("db_bench", "task3", history)
    assert len(invoker.calls) == 3


def test_toolgen_prompt_selector_returns_string() -> None:
    prompt = get_toolgen_system_prompt("baseline")
    assert isinstance(prompt, str)


def test_toolgen_pipeline_config_registry_paths(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "registry"
    monkeypatch.delenv("TOOLGEN_PIPELINE", raising=False)
    monkeypatch.delenv("TOOL_REGISTRY_ROOT", raising=False)
    config = get_toolgen_pipeline_config(str(registry_root))
    assert config.pipeline == "baseline"
    assert config.registry_dir == str(registry_root)
