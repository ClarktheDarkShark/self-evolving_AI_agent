import datetime
import json
import os
from typing import Any, Mapping, Optional

from src.callbacks import Callback, CallbackArguments
from src.typings import Role
from src.utils.output_paths import prefix_filename

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
        return os.path.join(run_dir, prefix_filename(self._log_filename))

    def _append_log(self, payload: Mapping[str, Any]) -> None:
        log_path = self._get_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def _truncate_text(self, text: str, max_len: int) -> str:
        if text is None:
            return ""
        text = str(text)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _truncate_payload(self, obj: Any, max_len: int) -> Any:
        if isinstance(obj, str):
            return self._truncate_text(obj, max_len)
        if isinstance(obj, list):
            return [self._truncate_payload(item, max_len) for item in obj]
        if isinstance(obj, dict):
            return {k: self._truncate_payload(v, max_len) for k, v in obj.items()}
        return obj

    def _listener(self, payload: dict[str, Any]) -> None:
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        payload_with_time = {"timestamp": timestamp, **payload}
        self._append_log(self._truncate_payload(payload_with_time, 500))
        # print(f"[GeneratedToolLogging] {json.dumps(payload_with_time)}")

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
