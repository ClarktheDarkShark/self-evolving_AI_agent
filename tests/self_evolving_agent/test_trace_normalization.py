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
        raise RuntimeError("_inference should not be called in this test")


def test_trace_normalization_and_actions_spec(tmp_path) -> None:
    registry_base = tmp_path / "registry"
    get_registry(str(registry_base), force_reset=True)

    controller = SelfEvolvingController(
        language_model=_NoOpLanguageModel(),
        tool_registry_path=str(registry_base),
        use_orchestrator=False,
    )
    registry = get_registry(str(registry_base))

    code = (
        '"""Trace normalization checker."""\n'
        "\n"
        "def run(payload: dict) -> dict:\n"
        "    try:\n"
        "        trace = payload.get('trace') or []\n"
        "        actions = payload.get('actions_spec') or {}\n"
        "        first_type = type(trace[0]).__name__ if trace else 'none'\n"
        "        return {'trace_item_type': first_type, 'actions_spec_keys': list(actions.keys())}\n"
        "    except Exception as exc:\n"
        "        return {'trace_item_type': 'error', 'error': str(exc)}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    try:\n"
        "        out = run({'trace': [], 'actions_spec': {}})\n"
        "        return isinstance(out, dict)\n"
        "    except Exception:\n"
        "        return False\n"
    )

    metadata = registry.register_tool(
        name="payload_analyzer_generated_tool",
        code=code,
        signature="run(payload: dict) -> dict",
        description="Trace normalization checker.",
        tool_type="utility",
        tool_category="validator",
        input_schema={"type": "object", "required": ["trace"], "properties": {"trace": {"type": "array"}}},
        capabilities=[],
    )
    assert metadata is not None

    tool_args = {
        "args": [
            {
                "payload": {
                    "task_text": "x",
                    "asked_for": "y",
                    "trace": ["Action: foo(bar)", "baz(qux)"],
                    "actions_spec": {},
                }
            }
        ]
    }
    result = controller._invoke_tool_by_payload(  # noqa: SLF001
        "payload_analyzer_generated_tool", tool_args, reason="test"
    )
    assert result.success
    assert isinstance(result.output, dict)
    assert result.output.get("trace_item_type") == "str"
    assert result.output.get("actions_spec_keys", []) == []
