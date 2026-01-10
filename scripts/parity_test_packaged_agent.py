from __future__ import annotations

from typing import Any, Mapping, Sequence
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
pkg_src = repo_root / "packages" / "lifelong_agent_core" / "src"
if pkg_src.exists():
    pkg_path = str(pkg_src)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

from src.language_models.language_model import LanguageModel
from src.self_evolving_agent.controller import SelfEvolvingController
from src.typings import ChatHistory, ChatHistoryItem, Role

from lifelong_agent_core import AgentConfig, EnvStepResult, run_episode
from lifelong_agent_core.model import ModelResponse
from lifelong_agent_core.types import Message


class DummyLanguageModel(LanguageModel):
    def __init__(self, output: str) -> None:
        super().__init__({"user": "user", "agent": "assistant"})
        self._output = output

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        return [
            ChatHistoryItem(role=Role.AGENT, content=self._output)
            for _ in batch_chat_history
        ]


class DummyModelAdapter:
    def __init__(self, output: str) -> None:
        self._output = output

    def complete(self, messages, *, system_prompt: str, inference_config=None):
        return ModelResponse(content=self._output)


class DummyEnvAdapter:
    def initialize(self, task_input: Any) -> Sequence[Message]:
        return [Message(role="user", content=str(task_input))]

    def step(self, agent_output: str) -> EnvStepResult:
        return EnvStepResult(observation=None, done=True, final_answer=agent_output)


def main() -> None:
    output = "Action: Answer\nFinal Answer: ok"

    legacy_model = DummyLanguageModel(output)
    legacy_agent = SelfEvolvingController(
        language_model=legacy_model,
        tool_registry_path="outputs/tool_library_parity",
        force_tool_generation_if_missing=False,
        use_packaged_agent=False,
    )

    legacy_history = ChatHistory()
    legacy_history.inject(ChatHistoryItem(role=Role.USER, content="Ping"))
    legacy_resp = legacy_agent._inference(legacy_history)

    config = AgentConfig(
        model_adapter=DummyModelAdapter(output),
        tool_registry_path="outputs/tool_library_parity_pkg",
        force_tool_generation_if_missing=False,
    )
    result = run_episode("Ping", DummyEnvAdapter(), config)

    same_output = legacy_resp.content == result.final_answer
    print("Legacy output:", legacy_resp.content)
    print("Packaged output:", result.final_answer)
    print("Same output:", same_output)
    print("Tool calls:", len(result.tool_calls))
    print("Generated tools:", len(result.generated_tools))


if __name__ == "__main__":
    main()
