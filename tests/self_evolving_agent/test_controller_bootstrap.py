import os
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
    # The controller ignores tool_registry_path and derives its registry path
    # from LIFELONG_OUTPUT_DIR. Set it so all registry access is isolated.
    prev = os.environ.get("LIFELONG_OUTPUT_DIR")
    os.environ["LIFELONG_OUTPUT_DIR"] = str(tmp_path)
    try:
        registry_dir = tmp_path / "tool_library"
        get_registry(str(registry_dir), force_reset=True)

        controller = SelfEvolvingController(
            language_model=_NoOpLanguageModel(),
            tool_registry_path=str(registry_dir),
            max_generated_tools_per_run=5,
            bootstrap_tools=[
                {
                    "name": "echo_value",
                    "description": "Returns the provided value.",
                    "signature": "run(value)",
                    "code": '"""Returns the provided value."""\n\ndef run(value):\n    return value\n',
                }
            ],
        )

        registry = controller._registry  # noqa: SLF001
        assert registry.has_tool("echo_value")
        assert pathlib.Path(registry.tools_dir, "echo_value.py").exists()
        assert controller._generated_tool_counter >= 1  # noqa: SLF001
    finally:
        if prev is None:
            os.environ.pop("LIFELONG_OUTPUT_DIR", None)
        else:
            os.environ["LIFELONG_OUTPUT_DIR"] = prev
        get_registry(force_reset=True)
