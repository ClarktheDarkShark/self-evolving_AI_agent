import hashlib
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
        self._run_headers: set[str] = set()
        self._trace_state: dict[str, int] = {}
        self._step_index: dict[str, int] = {}
        self._logged_callback_runs: set[str] = set()
        self._log_level = os.environ.get("GENERATED_TOOLS_LOG_LEVEL", "INFO").upper()

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
        payload_with_seq = self._assign_sequence(payload, log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload_with_seq, default=str) + "\n")

    def _assign_sequence(self, payload: Mapping[str, Any], log_path: str) -> Mapping[str, Any]:
        seq = 1
        try:
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        last_t = obj.get("t")
                        if isinstance(last_t, int):
                            seq = last_t + 1
                    break
        except Exception:
            seq = 1
        data = dict(payload)
        if "t" in data:
            data.pop("t", None)
        data["t"] = seq
        return data

    def _truncate_text(self, text: str, max_len: int) -> str:
        if text is None:
            return ""
        text = str(text)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _sha1_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _summarize_text(self, text: Optional[str], max_len: int = 120) -> dict[str, Any]:
        safe_text = "" if text is None else str(text)
        return {
            "len": len(safe_text),
            "sha1": self._sha1_text(safe_text),
            "preview": self._truncate_text(safe_text, max_len),
        }

    def _summarize_list(self, values: list[Any], max_len: int = 120) -> dict[str, Any]:
        rendered = ",".join(str(v) for v in values)
        return {
            "len": len(values),
            "sha1": self._sha1_text(rendered),
            "preview": self._truncate_text(rendered, max_len),
        }

    def _trace_delta(self, run_key: str, trace: list[Any]) -> dict[str, Any]:
        trace_len_before = self._trace_state.get(run_key, 0)
        trace_len_after = len(trace)
        trace_new = trace[trace_len_before:] if trace_len_after > trace_len_before else []
        compact_new = []
        def _truncate_arg(value: Any, max_len: int = 80) -> Any:
            if value is None:
                return None
            text = str(value)
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        def _truncate_args(value: Any) -> Any:
            if isinstance(value, list):
                return [_truncate_arg(item) for item in value]
            return _truncate_arg(value)

        def _truncate_output(value: Any, max_len: int = 200) -> Any:
            if value is None:
                return None
            return self._truncate_text(str(value), max_len)

        for step in trace_new:
            if isinstance(step, dict):
                compact_new.append(
                    {
                        "action": step.get("action"),
                        "args": _truncate_args(step.get("args")),
                        "ok": step.get("ok"),
                        "output": _truncate_output(step.get("output")),
                        "error": step.get("error"),
                    }
                )
            else:
                compact_new.append(step)
        self._trace_state[run_key] = trace_len_after
        return {
            "trace_len_before": trace_len_before,
            "trace_len_after": trace_len_after,
            "trace_new": compact_new,
        }

    def _listener(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "invoke":
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                return
            run_key = run_id
            if run_id not in self._run_headers:
                inv_ctx = payload.get("invocation_context") or {}
                asked_for = payload.get("asked_for")
                actions_spec_keys = payload.get("actions_spec_keys") or []
                state_dir = payload.get("state_dir") or ""
                header = {
                    "event": "run_header",
                    "run_id": run_id,
                    "environment": inv_ctx.get("environment"),
                    "agent_role": Role.AGENT.value,
                    "tool_name": payload.get("tool_name"),
                    "asked_for": self._summarize_text(asked_for),
                    "actions_spec_keys": self._summarize_list(list(actions_spec_keys)),
                    "state_dir_basename": os.path.basename(str(state_dir)) if state_dir else "",
                }
                self._append_log(header)
                self._run_headers.add(run_id)

            step_idx = self._step_index.get(run_key, 0) + 1
            self._step_index[run_key] = step_idx
            trace = payload.get("trace") if isinstance(payload.get("trace"), list) else []
            trace_delta = self._trace_delta(run_key, trace)
            status = payload.get("result_status")
            if not isinstance(status, str):
                status = "success" if payload.get("success") else "error"

            # Build error summary from multiple sources
            error_parts = []
            if payload.get("error"):
                error_parts.append(str(payload.get("error")))
            if payload.get("error_type"):
                error_parts.append(f"type:{payload.get('error_type')}")
            result_errors = payload.get("result_errors")
            if isinstance(result_errors, list) and result_errors:
                error_parts.append(f"errors:{','.join(str(e) for e in result_errors[:3])}")
            error_short = "; ".join(error_parts) if error_parts else ""

            step_event = {
                "event": "step_event",
                "run_id": run_id,
                "step_idx": step_idx,
                "tool": payload.get("tool_name"),
                "status": status,
                "next_action": payload.get("result_next_action"),
                "contract_ok": payload.get("result_contract_ok"),
                "error_short": self._truncate_text(error_short, 200) if error_short else "",
                "duration_ms": payload.get("duration_ms"),
                "trace_full": trace,
                **trace_delta,
            }

            # Always log errors and warnings when present
            if isinstance(result_errors, list) and result_errors:
                step_event["result_errors"] = result_errors
            result_warnings = payload.get("result_warnings")
            if isinstance(result_warnings, list) and result_warnings:
                step_event["result_warnings"] = result_warnings

            # Log validation details when contract validation fails
            if not payload.get("result_contract_ok"):
                result_validation = payload.get("result_validation")
                if isinstance(result_validation, dict):
                    step_event["result_validation"] = result_validation

            # Log rationale for need_step/blocked states
            if status in {"need_step", "blocked"}:
                rationale = payload.get("result_rationale")
                if isinstance(rationale, list) and rationale:
                    step_event["result_rationale"] = rationale
                next_action_reason = payload.get("result_next_action_reason")
                if isinstance(next_action_reason, str):
                    step_event["next_action_reason"] = next_action_reason
                # Log when next_action is None but status suggests we need one
                if not payload.get("result_next_action") and status == "need_step":
                    step_event["next_action_missing"] = True

            # Log advisory paradigm fields
            result_confidence = payload.get("result_confidence_score")
            if isinstance(result_confidence, (int, float)):
                step_event["confidence_score"] = result_confidence
            result_pruned = payload.get("result_pruned_observation")
            if result_pruned is not None:
                step_event["has_pruned_observation"] = True

            # Log answer_recommendation for done and advisory statuses
            if status in {"done", "advisory"}:
                answer_rec = payload.get("result_answer_recommendation")
                if answer_rec is not None:
                    step_event["answer_recommendation"] = self._summarize_text(str(answer_rec), 150)

            # Add failure pattern flags for easier analysis
            failure_patterns = []
            if status == "error" and isinstance(result_errors, list):
                for err in result_errors:
                    err_str = str(err)
                    if "hallucinated_action" in err_str:
                        failure_patterns.append("hallucinated_action")
                    if "missing_payload_key" in err_str:
                        failure_patterns.append("missing_payload_key")
                    if "missing_actions_spec" in err_str:
                        failure_patterns.append("missing_actions_spec")
                    if "invalid_next_action" in err_str:
                        failure_patterns.append("invalid_next_action")
            if status == "need_step" and not payload.get("result_next_action"):
                failure_patterns.append("no_viable_next_action")
            if not payload.get("result_contract_ok") and status != "done":
                failure_patterns.append("contract_validation_failed")
            if failure_patterns:
                step_event["failure_patterns"] = list(set(failure_patterns))

            # Log plan when present for debugging complex reasoning
            result_plan = payload.get("result_plan")
            if result_plan is not None and self._log_level in {"DEBUG", "VERBOSE"}:
                step_event["result_plan"] = self._summarize_text(str(result_plan), 200)
            if self._log_level == "DEBUG":
                debug_payload = {
                    "result": self._summarize_text(payload.get("result_preview")),
                    "task_text": self._summarize_text(payload.get("task_text")),
                    "chat_history": self._summarize_text(payload.get("chat_history")),
                    "actions_spec": self._summarize_text(str(payload.get("actions_spec"))),
                }
                step_event["debug"] = debug_payload
            self._append_log(step_event)
            return

        if event == "callback_initialized":
            log_key = self._get_log_path()
            if log_key in self._logged_callback_runs:
                if self._log_level != "DEBUG":
                    return
            self._logged_callback_runs.add(log_key)
        payload_with_time = dict(payload)
        payload_with_time.pop("timestamp", None)
        self._append_log(payload_with_time)
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
        payload = {
            "event": "callback_initialized",
            "role_dict": {str(k): v for k, v in role_dict.items()},
            "task": str(callback_args.current_session.task_name),
            "agent_role": Role.AGENT.value,
        }
        self._append_log(payload)
