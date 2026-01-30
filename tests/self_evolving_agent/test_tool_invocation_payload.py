import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_evolving_agent.controller import SelfEvolvingController
from src.self_evolving_agent.tool_registry import get_registry
from src.language_models import LanguageModel
from src.typings import ChatHistory, Role


class _NoOpLanguageModel(LanguageModel):
    def __init__(self) -> None:
        super().__init__({Role.USER: "user", Role.AGENT: "assistant"})

    def _inference(self, batch_chat_history, inference_config_dict, system_prompt):
        raise RuntimeError("_inference should not be called in this test")


def test_payload_tool_invocation_contract(tmp_path) -> None:
    registry_base = tmp_path / "registry"
    get_registry(str(registry_base), force_reset=True)

    controller = SelfEvolvingController(
        language_model=_NoOpLanguageModel(),
        tool_registry_path=str(registry_base),
        use_orchestrator=False,
    )
    registry = get_registry(str(registry_base))

    code = (
        '"""Payload echo tool."""\n'
        "\n"
        "def run(payload: dict) -> dict:\n"
        "    try:\n"
        "        missing = []\n"
        "        for key in ('task_text', 'asked_for', 'trace', 'actions_spec'):\n"
        "            if key not in payload:\n"
        "                missing.append(key)\n"
        "        if missing:\n"
        "            return {\n"
        "                'status': 'blocked',\n"
        "                'errors': ['missing_payload_key:' + k for k in missing],\n"
        "                'warnings': [],\n"
        "            }\n"
        "        return {'status': 'ok', 'errors': [], 'warnings': []}\n"
        "    except Exception as exc:\n"
        "        return {'status': 'error', 'errors': [str(exc)], 'warnings': []}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    try:\n"
        "        out = run({'task_text': 'x', 'asked_for': 'y', 'trace': [], 'actions_spec': {}})\n"
        "        return isinstance(out, dict)\n"
        "    except Exception:\n"
        "        return False\n"
    )

    input_schema = {
        "type": "object",
        "required": ["payload"],
        "properties": {
            "payload": {
                "type": "object",
                "required": ["task_text", "asked_for", "trace", "actions_spec"],
                "properties": {
                    "task_text": {"type": "string"},
                    "asked_for": {"type": "string"},
                    "trace": {"type": "array"},
                    "actions_spec": {"type": "object"},
                },
            }
        },
    }

    metadata = registry.register_tool(
        name="payload_echo_tool",
        code=code,
        signature="run(payload: dict) -> dict",
        description="Echo payload tool.",
        tool_type="utility",
        tool_category="validator",
        input_schema=input_schema,
        capabilities=[],
    )
    assert metadata is not None

    tool_meta = controller._get_tool_metadata("payload_echo_tool")  # noqa: SLF001
    assert tool_meta is not None
    assert tool_meta.input_schema == input_schema

    tool_args = controller._auto_build_tool_args(  # noqa: SLF001
        tool_meta, query="hello", chat_history=ChatHistory()
    )
    assert tool_args is not None
    assert "args" in tool_args
    assert isinstance(tool_args["args"][0], dict)
    assert "payload" in tool_args["args"][0]

    result = controller._invoke_tool_by_payload(  # noqa: SLF001
        "payload_echo_tool", tool_args, reason="test"
    )
    assert result.success
    assert isinstance(result.output, dict)
    assert result.output.get("status") == "ok"

    direct = registry.invoke_tool(
        "payload_echo_tool",
        {"payload": {"task_text": "t", "asked_for": "a", "trace": [], "actions_spec": {}}},
    )
    assert direct.success
    assert isinstance(direct.output, dict)
    assert direct.output.get("status") == "ok"
