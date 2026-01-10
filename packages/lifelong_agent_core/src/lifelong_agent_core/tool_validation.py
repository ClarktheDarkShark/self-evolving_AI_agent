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

    if args:
        args[0] = _alt(args[0])
        return args, kwargs

    for key in list(kwargs.keys()):
        kwargs[key] = _alt(kwargs[key])
        return args, kwargs

    return ["alt"], {}


def _run_uses_input(code: str) -> tuple[bool, Optional[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    run_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_fn = node
            break
    if run_fn is None:
        return False, "run() definition not found"

    param_names: list[str] = []
    for arg in run_fn.args.args:
        param_names.append(arg.arg)
    if not param_names:
        return False, "run() must accept at least one input parameter"

    used_names: set[str] = set()
    for node in ast.walk(run_fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)

    if not any(name in used_names for name in param_names):
        return False, "run() does not reference its input parameters"

    return True, None


def validate_tool_code(
    code: str, *, timeout_s: float = 2.0
) -> ToolValidationResult:
    try:
        compiled = compile(code, "<generated_tool>", "exec")
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"compile failed: {exc}")

    uses_input, input_issue = _run_uses_input(code)
    if not uses_input:
        return ToolValidationResult(success=False, error=input_issue)

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

    if len(inspect.signature(run_fn).parameters) == 1 and isinstance(result, str):
        alt_args, alt_kwargs = _build_variation_args(run_fn, args, kwargs)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_fn, *alt_args, **alt_kwargs)
                alt_result = future.result(timeout=timeout_s)
            if isinstance(alt_result, str) and alt_result == result:
                return ToolValidationResult(
                    success=False,
                    error="run() returned identical output for distinct inputs",
                )
        except TimeoutError:
            return ToolValidationResult(success=False, error="variance test timed out")
        except Exception as exc:
            return ToolValidationResult(
                success=False, error=f"variance test failed: {exc}"
            )

    self_test_fn = getattr(module, "self_test", None)
    if not callable(self_test_fn):
        return ToolValidationResult(
            success=False,
            error="self_test() not found or not callable",
        )
    self_test_passed = False
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
