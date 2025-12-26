import datetime
import json
import os
from typing import Any, Mapping, Optional

from src.callbacks import Callback, CallbackArguments
from src.typings import Role

from .tool_registry import get_registry


class GeneratedToolLoggingCallback(Callback):
    """
    Logs generated tool lifecycle events (creation/invocation) into
    `outputs/<run>/generated_tools.log`.
    """

    def __init__(self, log_filename: str = "generated_tools.log"):
        super().__init__()
        self._subscribed = False
        self._log_filename = log_filename

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_log_path(self) -> str:
        # callback_state/<id> -> outputs/<run>
        state_dir = self.get_state_dir()
        run_dir = os.path.abspath(os.path.join(state_dir, os.pardir, os.pardir))
        return os.path.join(run_dir, self._log_filename)

    def _append_log(self, payload: Mapping[str, Any]) -> None:
        log_path = self._get_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _listener(self, payload: dict[str, Any]) -> None:
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        payload_with_time = {"timestamp": timestamp, **payload}
        self._append_log(payload_with_time)
        print(f"[GeneratedToolLogging] {json.dumps(payload_with_time)}")

    def _ensure_subscription(self) -> None:
        if self._subscribed:
            return
        registry = get_registry()
        registry.add_event_listener(self._listener)
        self._subscribed = True

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        # Only subscribe when the first session is created to ensure the registry is ready.
        self._ensure_subscription()
        # Log the initial role dictionary for traceability.
        role_dict = callback_args.session_context.agent.get_role_dict()
        self._append_log(
            {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "event": "callback_initialized",
                "role_dict": {str(k): v for k, v in role_dict.items()},
                "task": str(callback_args.current_session.task_name),
                "agent_role": Role.AGENT.value,
            }
        )
