#tool_validation.py

from __future__ import annotations

import ast
import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError


@dataclass
class ToolValidationResult:
    success: bool
    error: Optional[str] = None
    smoke_output: Any = None
    self_test_passed: bool = False


def _dummy_value(param: inspect.Parameter) -> Any:
    annotation = param.annotation
    name = param.name.lower()
    if annotation in (str,):
        return "test"
    if annotation in (int,):
        return 0
    if annotation in (float,):
        return 0.0
    if annotation in (bool,):
        return False
    if annotation in (dict,):
        return {}
    if annotation in (list,):
        return []
    if "text" in name or "query" in name or "task" in name:
        return "test"
    return None


def _build_smoke_args(run_fn: Callable[..., Any]) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(run_fn)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not param.empty:
            continue
        value = _dummy_value(param)
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            args.append(value)
        else:
            kwargs[param.name] = value
    return args, kwargs


def _build_variation_args(
    run_fn: Callable[..., Any], base_args: list[Any], base_kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(run_fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    args = list(base_args)
    kwargs = dict(base_kwargs)
    if not params:
        return args, kwargs

    def _alt(val: Any) -> Any:
        if isinstance(val, str):
            return val + " alt"
        if isinstance(val, bool):
            return not val
        if isinstance(val, int):
            return val + 1
        if isinstance(val, float):
            return val + 1.0
        if isinstance(val, dict):
            return {**val, "alt": True}
        if isinstance(val, list):
            return val + ["alt"]
        if val is None:
            return "alt"
        return val

    # Prefer modifying the first required parameter.
    if args:
        args[0] = _alt(args[0])
        return args, kwargs

    # Otherwise modify the first kwarg.
    for key in list(kwargs.keys()):
        kwargs[key] = _alt(kwargs[key])
        return args, kwargs

    # If no args/kwargs were generated, fallback to a single string arg.
    return ["alt"], {}


def validate_tool_code(
    code: str, *, timeout_s: float = 2.0
) -> ToolValidationResult:
    for heading in ("INVOKE_WITH:", "RUN_PAYLOAD_REQUIRED:", "RUN_PAYLOAD_OPTIONAL:"):
        if heading not in (code or ""):
            return ToolValidationResult(
                success=False,
                error=f"missing_invoke_contract:{heading}",
            )
    try:
        compiled = compile(code, "<generated_tool>", "exec")
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"compile failed: {exc}")

    module = types.ModuleType("generated_tool")
    try:
        exec(compiled, module.__dict__)
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"exec failed: {exc}")

    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        return ToolValidationResult(success=False, error="run() not found or not callable")

    self_test_fn = getattr(module, "self_test", None)
    if not callable(self_test_fn):
        return ToolValidationResult(
            success=False,
            error="self_test() not found or not callable",
        )

    sig = inspect.signature(run_fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    required = [
        p
        for p in params
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if len(required) != 1 or required[0].name != "payload":
        return ToolValidationResult(
            success=False,
            error="run() must accept exactly one required parameter named 'payload'",
        )

    # Build a smoke payload that includes mock callable actions_spec so
    # Macro tools pass callable() checks during validation.
    # NOTE: include 'entities' and other optional keys that generated tools
    # may declare as required, to avoid false-negative failures during smoke
    # testing.  The payload must be a superset of what real Orchestrator
    # payloads provide.
    smoke_payload: dict = {
        "_smoke": True,
        "task_text": "smoke test",
        "asked_for": "smoke test",
        "trace": [],
        "run_id": "smoke",
        "state_dir": "/tmp",
        "entities": [],
        "env_observation": "",
        "constraints": {},
        "actions_spec": {
            "get_relations": lambda *_args: "Relations of mock: [mock.rel]",
            "get_neighbors": lambda *_args: "Variable #99 = get_neighbors(mock, mock.rel)",
            "intersection": lambda *_args: "Variable #100 = intersection(#98, #99)",
            "get_attributes": lambda *_args: "Attributes of #99: [mock.attr]",
            "argmax": lambda *_args: "Variable #101 = argmax(#99, mock.attr)",
            "argmin": lambda *_args: "Variable #102 = argmin(#99, mock.attr)",
            "count": lambda *_args: "Count of #99 is 42",
        },
    }
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_fn, smoke_payload)
            result = future.result(timeout=timeout_s)
    except TimeoutError:
        return ToolValidationResult(success=False, error="smoke test timed out")
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"smoke test failed: {exc}")

    if not isinstance(result, dict):
        return ToolValidationResult(
            success=False,
            error="run() must return a dict",
        )
    advisory_keys = {"pruned_observation", "answer_recommendation", "confidence_score"}
    legacy_keys = {"next_action", "next_action_candidates", "why_stuck"}
    has_advisory = any(k in result for k in advisory_keys)
    has_legacy = any(k in result for k in legacy_keys)
    if has_advisory:
        missing = [k for k in advisory_keys if k not in result]
        if missing:
            return ToolValidationResult(
                success=False,
                error=f"missing_advisory_keys:{','.join(missing)}",
            )
        if not isinstance(result.get("answer_recommendation"), str):
            return ToolValidationResult(
                success=False,
                error="answer_recommendation must be str",
            )
        confidence = result.get("confidence_score")
        if not isinstance(confidence, (int, float)):
            return ToolValidationResult(
                success=False,
                error="confidence_score must be float",
            )
    elif not has_legacy:
        return ToolValidationResult(
            success=False,
            error="missing_output_schema_keys",
        )

    self_test_passed = False
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self_test_fn)
            self_test_passed = bool(future.result(timeout=timeout_s))
    except TimeoutError:
        self_test_passed = False
    except Exception:
        self_test_passed = False

    return ToolValidationResult(
        success=True, smoke_output=result, self_test_passed=self_test_passed
    )
