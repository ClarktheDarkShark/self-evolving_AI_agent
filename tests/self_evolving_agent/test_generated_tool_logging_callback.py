import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.callbacks.callback import CallbackArguments
from src.self_evolving_agent.callbacks import GeneratedToolLoggingCallback
from src.self_evolving_agent.tool_registry import get_registry
from src.typings import ChatHistory, Role, Session, TaskName
from src.agents.agent import Agent


class _DummyAgent(Agent):
    def _inference(self, chat_history: ChatHistory):
        raise RuntimeError("Should not be called")

    def get_role_dict(self) -> dict[Role, str]:
        return {Role.USER: "user", Role.AGENT: "assistant"}


def test_generated_tool_logging_callback(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    state_dir = output_dir / "callback_state" / "logging_callback"
    callback = GeneratedToolLoggingCallback()
    callback.set_state_dir(str(state_dir))

    # Initialize registry and callback arguments
    registry = get_registry(str(output_dir / "registry"), force_reset=True)
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0")
    callback_args = CallbackArguments(
        current_session=session,
        task=object(),  # Unused by the callback
        agent=_DummyAgent(),
        session_list=[session],
    )

    callback.on_session_create(callback_args)

    registry.register_tool(
        name="echo",
        code="def run(value):\n    return value\n",
        signature="run(value)",
        description="Echoes a value.",
    )
    registry.invoke_tool("echo", "ping")

    log_path = output_dir / "generated_tools.log"
    assert log_path.exists()
    with open(log_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f.readlines()]
    events = {entry["event"] for entry in entries}
    assert {"callback_initialized", "register", "invoke"} <= events
