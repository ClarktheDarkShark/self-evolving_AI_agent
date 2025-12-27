import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_evolving_agent.controller import SelfEvolvingController
from src.self_evolving_agent.tool_registry import get_registry
from src.language_models import LanguageModel
from src.typings import Role


class _NoOpLanguageModel(LanguageModel):
    def __init__(self) -> None:
        super().__init__({Role.USER: "user", Role.AGENT: "assistant"})

    def _inference(self, batch_chat_history, inference_config_dict, system_prompt):
        # This class is only used to satisfy the controller constructor in tests.
        raise RuntimeError("_inference should not be called in bootstrap tests")


def test_bootstrap_tools_registered(tmp_path) -> None:
    registry_base = tmp_path / "registry"
    # Reset any shared registry state so the test is deterministic.
    get_registry(str(registry_base), force_reset=True)

    controller = SelfEvolvingController(
        language_model=_NoOpLanguageModel(),
        tool_registry_path=str(registry_base),
        max_generated_tools_per_run=5,
        bootstrap_tools=[
            {
                "name": "echo_value",
                "description": "Returns the provided value.",
                "signature": "run(value)",
                "code": "def run(value):\n    return value\n",
            }
        ],
    )

    registry = get_registry(str(registry_base))
    assert registry.has_tool("echo_value")
    assert (registry_base / "generated_tools" / "echo_value.py").exists()
    assert controller._generated_tool_counter >= 1  # noqa: SLF001
