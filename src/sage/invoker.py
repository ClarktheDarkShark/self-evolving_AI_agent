from __future__ import annotations

import builtins
import inspect
import json
import multiprocessing
import os
import queue as queue_module
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError

from .parser import ParsedSAGEProgram, parse_sage_code

DEFAULT_SPARQL_ENDPOINT_URL = os.environ.get(
    "SAGE_SPARQL_ENDPOINT_URL",
    "http://127.0.0.1:3001/kb/sparql",
)
_SAFE_BUILTIN_NAMES = (
    "Exception",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "AssertionError",
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


class SAGEInvokerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SAGEInvocationResult:
    success: bool
    payload: dict[str, Any] | None = None
    error: str | None = None
    failure_kind: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def invoke_sage_program(
    parsed_program: ParsedSAGEProgram,
    *,
    endpoint_url: str = DEFAULT_SPARQL_ENDPOINT_URL,
    timeout_s: float = 10.0,
    sparqlwrapper_module: Any | None = None,
) -> SAGEInvocationResult:
    queue: multiprocessing.Queue[dict[str, Any]] = multiprocessing.get_context(
        "spawn"
    ).Queue(maxsize=1)
    process = multiprocessing.get_context("spawn").Process(
        target=_execute_program_in_subprocess,
        kwargs={
            "parsed_program": parsed_program,
            "endpoint_url": endpoint_url,
            "sparqlwrapper_module": sparqlwrapper_module,
            "queue": queue,
        },
    )
    process.start()
    try:
        process.join(timeout=timeout_s)
        if process.is_alive():
            _terminate_process(process)
            return SAGEInvocationResult(
                success=False,
                error=f"execution_timed_out:{timeout_s}",
                failure_kind="endpoint_timeout",
                diagnostics={
                    "endpoint_url": endpoint_url,
                    "timeout_s": timeout_s,
                },
            )

        if process.exitcode not in (0, None):
            return SAGEInvocationResult(
                success=False,
                error=f"execution_process_failed:{process.exitcode}",
                failure_kind="execution_error",
                diagnostics={
                    "endpoint_url": endpoint_url,
                    "exitcode": process.exitcode,
                },
            )

        try:
            message = queue.get(timeout=0.2)
        except queue_module.Empty:
            return SAGEInvocationResult(
                success=False,
                error="execution_result_missing",
                failure_kind="execution_error",
                diagnostics={
                    "endpoint_url": endpoint_url,
                    "exitcode": process.exitcode,
                },
            )

        status = str(message.get("status") or "").strip().lower()
        if status == "ok":
            return SAGEInvocationResult(
                success=True,
                payload=message.get("payload"),
                diagnostics=dict(message.get("diagnostics") or {}),
            )
        if status == "error":
            return SAGEInvocationResult(
                success=False,
                error=str(message.get("error") or "sage_execution_failed"),
                failure_kind=str(message.get("failure_kind") or "execution_error"),
                diagnostics=dict(message.get("diagnostics") or {}),
            )
        return SAGEInvocationResult(
            success=False,
            error="execution_result_invalid",
            failure_kind="execution_error",
            diagnostics={
                "endpoint_url": endpoint_url,
            },
        )
    finally:
        try:
            queue.close()
        except Exception:
            pass
        try:
            queue.join_thread()
        except Exception:
            pass


def _execute_program_in_subprocess(
    *,
    parsed_program: ParsedSAGEProgram,
    endpoint_url: str,
    sparqlwrapper_module: Any | None,
    queue: multiprocessing.Queue,
) -> None:
    try:
        payload, runtime_diagnostics = _execute_program(
            parsed_program=parsed_program,
            endpoint_url=endpoint_url,
            sparqlwrapper_module=sparqlwrapper_module,
        )
        queue.put(
            {
                "status": "ok",
                "payload": payload,
                "diagnostics": runtime_diagnostics,
            }
        )
    except Exception as exc:
        failure_kind, diagnostics = _classify_invocation_exception(
            exc,
            endpoint_url=endpoint_url,
        )
        queue.put(
            {
                "status": "error",
                "error": str(exc),
                "failure_kind": failure_kind,
                "diagnostics": diagnostics,
            }
        )


def _terminate_process(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _execute_program(
    *,
    parsed_program: ParsedSAGEProgram,
    endpoint_url: str,
    sparqlwrapper_module: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_diagnostics: dict[str, Any] = {
        "endpoint_url": endpoint_url,
    }
    resolved_sparqlwrapper_module = _build_redirected_sparqlwrapper_module(
        sparqlwrapper_module
        or _resolve_sparqlwrapper_module(None),
        endpoint_url=endpoint_url,
        runtime_diagnostics=runtime_diagnostics,
    )
    sandbox_globals = {
        "__builtins__": _build_safe_builtins(resolved_sparqlwrapper_module),
        "json": json,
        "SPARQLWrapper": resolved_sparqlwrapper_module,
        "JSON": getattr(resolved_sparqlwrapper_module, "JSON", None),
    }
    compiled = compile(parsed_program.code, "<sage_program>", "exec")
    exec(compiled, sandbox_globals, sandbox_globals)

    entrypoint = sandbox_globals.get(parsed_program.entrypoint)
    if not callable(entrypoint):
        raise SAGEInvokerError(
            f"entrypoint_not_callable:{parsed_program.entrypoint}"
        )

    args, kwargs = _build_invocation(entrypoint, endpoint_url)
    raw_payload = entrypoint(*args, **kwargs)
    return _coerce_json_payload(raw_payload), dict(runtime_diagnostics)


def _build_safe_builtins(sparqlwrapper_module: Any) -> dict[str, Any]:
    safe_builtins = {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTIN_NAMES
    }
    safe_builtins["__import__"] = _build_safe_import(sparqlwrapper_module)
    return safe_builtins


def _resolve_sparqlwrapper_module(sparqlwrapper_module: Any | None) -> Any:
    if sparqlwrapper_module is not None:
        return sparqlwrapper_module
    try:
        import SPARQLWrapper as resolved_sparqlwrapper_module
    except Exception as exc:
        raise SAGEInvokerError(
            "SPARQLWrapper_import_failed"
        ) from exc
    return resolved_sparqlwrapper_module


def _build_safe_import(sparqlwrapper_module: Any):
    allowed_modules = {
        "SPARQLWrapper": sparqlwrapper_module,
        "json": json,
    }

    def _safe_import(
        name: str,
        globals_dict: dict[str, Any] | None = None,
        locals_dict: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise SAGEInvokerError("relative_import_not_allowed")
        if name not in allowed_modules:
            raise SAGEInvokerError(f"import_not_allowed:{name}")
        return allowed_modules[name]

    return _safe_import


def _build_redirected_sparqlwrapper_module(
    sparqlwrapper_module: Any,
    *,
    endpoint_url: str,
    runtime_diagnostics: dict[str, Any],
) -> Any:
    wrapper_cls = getattr(sparqlwrapper_module, "SPARQLWrapper", None)
    json_constant = getattr(sparqlwrapper_module, "JSON", None)
    if wrapper_cls is None:
        raise SAGEInvokerError("SPARQLWrapper_class_missing")

    class _RedirectedSPARQLWrapper:
        def __init__(self, _ignored_endpoint_url: str) -> None:
            runtime_diagnostics.setdefault(
                "generated_endpoint_url",
                _ignored_endpoint_url,
            )
            runtime_diagnostics["endpoint_url"] = endpoint_url
            self._wrapped = wrapper_cls(endpoint_url)

        def setQuery(self, query: str) -> None:
            runtime_diagnostics["query_text"] = query
            self._wrapped.setQuery(query)

        def setReturnFormat(self, return_format: Any) -> None:
            runtime_diagnostics["return_format"] = str(return_format)
            self._wrapped.setReturnFormat(return_format)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    return type(
        "RedirectedSPARQLWrapperModule",
        (),
        {
            "SPARQLWrapper": _RedirectedSPARQLWrapper,
            "JSON": json_constant,
        },
    )()


def _build_invocation(
    entrypoint: Any, endpoint_url: str
) -> tuple[list[Any], dict[str, Any]]:
    signature = inspect.signature(entrypoint)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    if len(parameters) > 1:
        raise SAGEInvokerError("entrypoint_accepts_too_many_parameters")
    if not parameters:
        return [], {}

    parameter = parameters[0]
    if parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return [endpoint_url], {}
    if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
        return [], {parameter.name: endpoint_url}
    raise SAGEInvokerError("unsupported_entrypoint_signature")


def _coerce_json_payload(raw_payload: Any) -> dict[str, Any]:
    try:
        json_payload = json.loads(json.dumps(raw_payload))
    except TypeError as exc:
        raise SAGEInvokerError("payload_not_json_serializable") from exc
    if not isinstance(json_payload, dict):
        raise SAGEInvokerError("payload_must_be_json_object")
    return json_payload


def execute_sage_code(
    code: str,
    *,
    endpoint_url: str = DEFAULT_SPARQL_ENDPOINT_URL,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    result = execute_sage_code_with_result(
        code,
        endpoint_url=endpoint_url,
        timeout_s=timeout_s,
    )
    if not result.success or result.payload is None:
        raise SAGEInvokerError(result.error or "sage_execution_failed")
    return result.payload


def execute_sage_code_with_result(
    code: str,
    *,
    endpoint_url: str = DEFAULT_SPARQL_ENDPOINT_URL,
    timeout_s: float = 10.0,
) -> SAGEInvocationResult:
    parsed_program = parse_sage_code(code)
    return invoke_sage_program(
        parsed_program,
        endpoint_url=endpoint_url,
        timeout_s=timeout_s,
    )


def build_sage_execution_failure_payload(
    invocation_result: SAGEInvocationResult,
    *,
    endpoint_url: str,
) -> dict[str, Any]:
    return {
        "sage_execution_status": "error",
        "sage_failure_kind": invocation_result.failure_kind or "execution_error",
        "sage_error": invocation_result.error or "sage_execution_failed",
        "endpoint_url": endpoint_url,
        "sage_diagnostics": dict(invocation_result.diagnostics),
    }


def _classify_invocation_exception(
    exc: Exception,
    *,
    endpoint_url: str,
) -> tuple[str, dict[str, Any]]:
    exception_chain = list(_iter_exception_chain(exc))
    diagnostics: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "exception_chain": [
            {
                "type": chain_exc.__class__.__name__,
                "message": str(chain_exc),
            }
            for chain_exc in exception_chain
        ],
    }
    for chain_exc in exception_chain:
        errno_value = getattr(chain_exc, "errno", None)
        if errno_value is not None:
            diagnostics.setdefault("errno", errno_value)
        reason = getattr(chain_exc, "reason", None)
        if reason is not None:
            diagnostics.setdefault("reason", str(reason))

    if _looks_like_timeout(exception_chain):
        return "endpoint_timeout", diagnostics
    if _looks_like_endpoint_unavailable(exception_chain):
        return "endpoint_unavailable", diagnostics
    if _looks_like_transport_error(exception_chain):
        return "transport_error", diagnostics
    return "execution_error", diagnostics


def _iter_exception_chain(exc: Exception) -> list[Exception]:
    chain: list[Exception] = []
    seen: set[int] = set()
    current: Exception | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, Exception) else None
    return chain


def _looks_like_timeout(exception_chain: list[Exception]) -> bool:
    timeout_like_types = (TimeoutError, socket.timeout)
    for exc in exception_chain:
        if isinstance(exc, timeout_like_types):
            return True
        if "timed out" in str(exc).lower():
            return True
    return False


def _looks_like_endpoint_unavailable(exception_chain: list[Exception]) -> bool:
    unavailable_tokens = (
        "connection refused",
        "failed to establish a new connection",
        "actively refused",
        "nodename nor servname provided",
        "name or service not known",
        "no route to host",
    )
    unavailable_errnos = {61, 111, 113}
    for exc in exception_chain:
        errno_value = getattr(exc, "errno", None)
        if errno_value in unavailable_errnos:
            return True
        text = str(exc).lower()
        if any(token in text for token in unavailable_tokens):
            return True
        reason = getattr(exc, "reason", None)
        if reason is not None and any(
            token in str(reason).lower() for token in unavailable_tokens
        ):
            return True
    return False


def _looks_like_transport_error(exception_chain: list[Exception]) -> bool:
    transport_like_types = (ConnectionError, OSError, URLError, socket.timeout)
    for exc in exception_chain:
        if isinstance(exc, transport_like_types):
            return True
        if any(
            token in str(exc).lower()
            for token in (
                "network is unreachable",
                "connection reset",
                "transport endpoint",
                "temporary failure in name resolution",
            )
        ):
            return True
    return False
