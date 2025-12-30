from __future__ import annotations

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


def validate_tool_code(
    code: str, *, timeout_s: float = 2.0
) -> ToolValidationResult:
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

    args, kwargs = _build_smoke_args(run_fn)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_fn, *args, **kwargs)
            result = future.result(timeout=timeout_s)
    except TimeoutError:
        return ToolValidationResult(success=False, error="smoke test timed out")
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"smoke test failed: {exc}")

    self_test_passed = False
    self_test_fn = getattr(module, "self_test", None)
    if callable(self_test_fn):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self_test_fn)
                self_test_passed = bool(future.result(timeout=timeout_s))
        except TimeoutError:
            return ToolValidationResult(success=False, error="self_test timed out")
        except Exception as exc:
            return ToolValidationResult(success=False, error=f"self_test failed: {exc}")

    return ToolValidationResult(
        success=True, smoke_output=result, self_test_passed=self_test_passed
    )
