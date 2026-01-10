import ast
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .utils import get_predefined_timestamp_structure


@dataclass
class ToolMetadata:
    name: str
    signature: str
    description: str
    creation_time: str
    usage_count: int = 0
    docstring: str = ""
    tool_type: Optional[str] = None
    input_schema: Optional[Any] = None
    capabilities: Optional[Any] = None
    base_name: str = ""
    version: int = 1
    code_hash: str = ""
    reliability_score: float = 0.0
    validation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used_time: Optional[str] = None
    environment_usage: Optional[dict[str, int]] = None


@dataclass
class ToolResult:
    success: bool
    output: Any | None = None
    error: str | None = None

    @classmethod
    def success_result(cls, output: Any) -> "ToolResult":
        return cls(success=True, output=output, error=None)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(success=False, output=None, error=error)


class ToolRegistry:
    """
    Persistent registry for generated tools.

    Tools are stored under `<tool_registry_path>/generated_tools/<tool_name>.py`
    and metadata is persisted as JSON.

    NOTE (format support):
    - Tools may define run(...) with any valid signature and any return type.
    - Generated tools are expected (by prompt) to include:
        * a module docstring
        * a run() docstring explaining usage
    """

    def __init__(self, tool_registry_path: str):
        self.base_path = os.path.abspath(tool_registry_path)
        self.tools_dir = os.path.join(self.base_path, "generated_tools")
        self.metadata_path = os.path.join(self.base_path, "metadata.json")
        self.fingerprint_map_path = os.path.join(self.base_path, "fingerprint_map.json")
        self.snapshots_dir = os.path.join(self.base_path, "run_snapshots")
        os.makedirs(self.tools_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        print(f"[ToolRegistry] Initialized registry at '{self.base_path}'")
        self._metadata: Dict[str, ToolMetadata] = {}
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._run_snapshot_path: Optional[str] = None
        self._canonical_naming = True
        self._fingerprint_map: Dict[str, str] = {}
        self._load_metadata()
        self._load_fingerprint_map()

    # ---------------------------
    # Metadata persistence
    # ---------------------------
    def _load_metadata(self) -> None:
        if not os.path.exists(self.metadata_path):
            return
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data: Iterable[dict[str, Any]] = json.load(f)
            for entry in data:
                tool_metadata = ToolMetadata(**entry)
                if not tool_metadata.docstring:
                    tool_metadata.docstring = tool_metadata.description
                if not tool_metadata.base_name:
                    tool_metadata.base_name = tool_metadata.name.split("__v")[0]
                if not tool_metadata.code_hash:
                    tool_metadata.code_hash = ""
                if tool_metadata.environment_usage is None:
                    tool_metadata.environment_usage = {}
                self._metadata[tool_metadata.name] = tool_metadata
        except Exception:
            # Corrupted metadata should not crash the run; start from an empty registry.
            self._metadata = {}

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self._metadata.values()], f, indent=2)

    def _load_fingerprint_map(self) -> None:
        if os.path.exists(self.fingerprint_map_path):
            try:
                with open(self.fingerprint_map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._fingerprint_map = {str(k): str(v) for k, v in data.items()}
                    return
            except Exception:
                self._fingerprint_map = {}
        self._fingerprint_map = {}
        self._hydrate_fingerprint_map()

    def _save_fingerprint_map(self) -> None:
        try:
            with open(self.fingerprint_map_path, "w", encoding="utf-8") as f:
                json.dump(self._fingerprint_map, f, indent=2)
        except Exception:
            pass

    def _hydrate_fingerprint_map(self) -> None:
        updated = False
        for meta in self._metadata.values():
            tool_path = self._get_tool_path(meta.name)
            if not os.path.exists(tool_path):
                continue
            try:
                with open(tool_path, "r", encoding="utf-8") as f:
                    code = f.read()
                fingerprint = self._compute_fingerprint(
                    meta.tool_type, meta.signature, code
                )
            except Exception:
                continue
            if fingerprint and fingerprint not in self._fingerprint_map:
                self._fingerprint_map[fingerprint] = meta.name
                updated = True
        if updated:
            self._save_fingerprint_map()

    # ---------------------------
    # Utility helpers
    # ---------------------------
    def _sanitize_name(self, name: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        return sanitized or "tool"

    def _normalize_base_name(self, name: str) -> str:
        base = self._sanitize_name(name)
        base = re.sub(r"(__v\\d+)+$", "", base)
        base = re.sub(r"(_v\\d+)+$", "", base)
        return base or "tool"

    def _unwrap_code_block(self, code: str) -> str:
        fenced_pattern = re.compile(r"```(?:python)?\s*(?P<body>[\s\S]*?)```", re.DOTALL)
        if match := fenced_pattern.search(code or ""):
            return match.group("body").strip()
        return code or ""

    def _extract_docstrings(self, code: str) -> tuple[str, str]:
        try:
            tree = ast.parse(code)
        except Exception:
            return "", ""
        module_doc = (ast.get_docstring(tree) or "").strip()
        run_doc = ""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_doc = (ast.get_docstring(node) or "").strip()
                break
        return module_doc, run_doc

    def set_canonical_naming(self, enabled: bool) -> None:
        self._canonical_naming = bool(enabled)

    def _get_tool_path(self, name: str) -> str:
        filename = f"{name}.py"
        return os.path.join(self.tools_dir, filename)

    def _compute_code_hash(self, signature: str, code: str) -> str:
        base = f"{signature}\n{code}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _normalize_code_for_fingerprint(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except Exception:
            return "\n".join(
                line.strip()
                for line in code.splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        self._strip_docstrings(tree)
        try:
            normalized = ast.unparse(tree)
        except Exception:
            normalized = code
        normalized = "\n".join(
            line.strip()
            for line in normalized.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        return normalized

    def _strip_docstrings(self, tree: ast.AST) -> None:
        if isinstance(tree, ast.Module) and tree.body:
            first = tree.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                tree.body = tree.body[1:]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    node.body = node.body[1:]

    def _compute_fingerprint(
        self, tool_type: Optional[str], signature: str, code: str
    ) -> str:
        normalized = self._normalize_code_for_fingerprint(code)
        base = f"{tool_type or ''}\n{signature.strip()}\n{normalized}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _canonical_name(self, base_name: str, fingerprint: str) -> str:
        base = re.sub(r"__[0-9a-fA-F]{6,}$", "", base_name)
        short = fingerprint[:10]
        return f"{base}__{short}"

    def _find_by_hash(self, code_hash: str) -> Optional[ToolMetadata]:
        for meta in self._metadata.values():
            if meta.code_hash == code_hash and code_hash:
                return meta
        return None

    def _next_version(self, base_name: str) -> int:
        versions = [
            meta.version
            for meta in self._metadata.values()
            if meta.base_name == base_name and isinstance(meta.version, int)
        ]
        return max(versions, default=0) + 1

    def set_run_snapshot(self, run_id: str) -> None:
        self._run_snapshot_path = os.path.join(self.snapshots_dir, f"{run_id}.jsonl")

    def _append_snapshot(self, payload: dict[str, Any]) -> None:
        if not self._run_snapshot_path:
            return
        try:
            with open(self._run_snapshot_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass

    @staticmethod
    def _calculate_reliability(meta: ToolMetadata) -> float:
        success = meta.success_count
        failure = meta.failure_count
        validation = meta.validation_count
        total = success + failure
        if total > 0:
            return success / total
        if validation > 0:
            return 0.6
        return 0.0

    def record_validation_result(
        self, name: str, success: bool, *, self_test_passed: bool = False
    ) -> None:
        with self._lock:
            meta = self._metadata.get(name)
            if not meta:
                return
            meta.validation_count += 1
            meta.reliability_score = self._calculate_reliability(meta)
            if success:
                meta.reliability_score = max(meta.reliability_score, 0.7)
                if self_test_passed:
                    meta.reliability_score = max(meta.reliability_score, 0.8)
            self._save_metadata()

    @staticmethod
    def _preview(obj: Any, max_len: int = 300) -> str:
        try:
            s = repr(obj)
        except Exception:
            s = f"<unreprable {type(obj).__name__}>"
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    def add_event_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if listener not in self._event_listeners:
                self._event_listeners.append(listener)

    def _notify(self, payload: dict[str, Any]) -> None:
        for listener in list(self._event_listeners):
            try:
                listener(payload)
            except Exception:
                # Callbacks should not break tool execution.
                continue

    # ---------------------------
    # Validation / stubs
    # ---------------------------
    def _validate_tool_source(self, code: str) -> list[str]:
        """
        Lightweight static validation that avoids importing/executing code.

        Enforces (per new system prompt):
        - module docstring exists
        - a top-level function named run exists
        - run has a docstring
        """
        issues: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]

        if ast.get_docstring(tree) is None:
            issues.append("Missing module docstring.")

        run_fn: ast.FunctionDef | None = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_fn = node
                break

        if run_fn is None:
            issues.append("Missing top-level function 'run'.")
            return issues

        if ast.get_docstring(run_fn) is None:
            issues.append("Missing docstring for function 'run'.")

        return issues

    def _make_invalid_tool_stub(self, issues: list[str]) -> str:
        """
        Create a safe stub tool with proper docstrings that returns an error string.
        """
        issue_text = "\\n".join(f"- {i}" for i in issues) if issues else "- Unknown issue"
        return (
            '"""\\n'
            "Auto-generated stub tool (invalid tool replaced).\\n\\n"
            "This file was created because the generated tool code failed validation.\\n"
            "It provides a safe `run()` entrypoint that returns an error message.\\n\\n"
            "Usage example:\\n"
            "    result = run('your input')\\n"
            '"""\\n\\n'
            "def run(*args, **kwargs):\\n"
            '    """\\n'
            "    Return an error message explaining why the generated tool was invalid.\\n\\n"
            "    Parameters:\\n"
            "        *args: Any positional arguments.\\n"
            "        **kwargs: Any keyword arguments.\\n\\n"
            "    Returns:\\n"
            "        str: Human-readable validation error details.\\n"
            "    Example:\\n"
            "        run('test')\\n"
            '    """\\n'
            f"    return 'Invalid generated tool. Validation issues:\\n{issue_text}'\\n"
        )

    # ---------------------------
    # Public API
    # ---------------------------
    def register_tool(
        self,
        name: str,
        code: str,
        signature: str,
        description: str,
        tool_type: Optional[str] = None,
        input_schema: Optional[Any] = None,
        capabilities: Optional[Any] = None,
    ) -> Optional[ToolMetadata]:
        base_name = self._normalize_base_name(name)
        normalized_code = self._unwrap_code_block(code)

        if not normalized_code.strip():
            normalized_code = (
                '"""\\n'
                "Auto-generated default tool.\\n\\n"
                "Usage example:\\n"
                "    run('hello')\\n"
                '"""\\n\\n'
                "def run(*args, **kwargs):\\n"
                '    """Return a default message (tool code was empty)."""\\n'
                "    return 'Error: empty tool code.'\\n"
            )

        # Enforce docstring + run() presence (new format). If invalid, replace with safe stub.
        issues = self._validate_tool_source(normalized_code)
        if issues:
            print(
                f"[ToolRegistry] Tool '{base_name}' failed validation; skipping. "
                f"Issues: {issues}"
            )
            self._notify(
                {
                    "event": "register_skipped",
                    "tool_name": base_name,
                    "signature": signature,
                    "description": description,
                    "issues": issues,
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                }
            )
            return None

        fingerprint = ""
        try:
            fingerprint = self._compute_fingerprint(tool_type, signature, normalized_code)
        except Exception:
            fingerprint = ""
        if fingerprint:
            existing_name = self._fingerprint_map.get(fingerprint)
            if existing_name:
                existing_meta = self._metadata.get(existing_name)
                if existing_meta:
                    print(
                        "[ToolRegistry] Deduped tool generation: "
                        f"fingerprint={fingerprint[:10]} name={existing_name}"
                    )
                    self._notify(
                        {
                            "event": "register_deduped",
                            "tool_name": existing_name,
                            "fingerprint": fingerprint,
                            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                        }
                    )
                    return existing_meta

        code_hash = self._compute_code_hash(signature, normalized_code)
        existing = self._find_by_hash(code_hash)
        if existing is not None:
            if fingerprint and fingerprint not in self._fingerprint_map:
                self._fingerprint_map[fingerprint] = existing.name
                self._save_fingerprint_map()
            print(
                f"[ToolRegistry] Skipping duplicate tool '{base_name}'; "
                f"matches existing '{existing.name}'."
            )
            return existing

        version = 1
        tool_name = ""
        if self._canonical_naming:
            fingerprint_short = fingerprint or code_hash[:10]
            tool_name = self._canonical_name(base_name, fingerprint_short)
            if tool_name != base_name:
                print(
                    "[ToolRegistry] Canonicalized tool name "
                    f"'{base_name}' -> '{tool_name}'."
                )
        else:
            version = self._next_version(base_name)
            tool_name = f"{base_name}__v{version}"
        if tool_name in self._metadata:
            existing_meta = self._metadata[tool_name]
            if fingerprint and fingerprint not in self._fingerprint_map:
                self._fingerprint_map[fingerprint] = tool_name
                self._save_fingerprint_map()
            print(
                "[ToolRegistry] Tool already registered; reusing "
                f"'{tool_name}'."
            )
            return existing_meta
        tool_path = self._get_tool_path(tool_name)
        module_doc, run_doc = self._extract_docstrings(normalized_code)
        docstring = run_doc or module_doc or description
        with self._lock:
            # print(f"[ToolRegistry] Persisting tool '{tool_name}' to '{tool_path}'")
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(normalized_code)

            metadata = self._metadata.get(
                tool_name,
                ToolMetadata(
                    name=tool_name,
                    signature=signature,
                    description=description,
                    creation_time=datetime.datetime.now(datetime.UTC).isoformat(),
                    usage_count=0,
                ),
            )
            # Overwrite signature/description while preserving creation_time/usage_count.
            metadata.signature = signature
            metadata.description = description
            metadata.docstring = docstring
            metadata.tool_type = tool_type
            metadata.input_schema = input_schema
            metadata.capabilities = capabilities
            metadata.base_name = base_name if not self._canonical_naming else tool_name
            metadata.version = version
            metadata.code_hash = code_hash
            if metadata.environment_usage is None:
                metadata.environment_usage = {}
            metadata.reliability_score = self._calculate_reliability(metadata)
            self._metadata[tool_name] = metadata
            self._save_metadata()
            if fingerprint:
                self._fingerprint_map[fingerprint] = tool_name
                self._save_fingerprint_map()

        code_len = len(normalized_code)
        code_sha256 = hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()
        self._notify(
            {
                "event": "register",
                "tool_name": tool_name,
                "signature": signature,
                "description": description,
                "docstring": docstring,
                "tool_type": tool_type,
                "input_schema": input_schema,
                "capabilities": capabilities,
                "path": tool_path,
                "code_len": code_len,
                "code_sha256": code_sha256,
                "code_preview": self._preview(normalized_code),
                "fingerprint": fingerprint,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        return metadata

    def has_tool(self, name: str) -> bool:
        return name in self._metadata

    def list_latest_tools(self) -> List[ToolMetadata]:
        latest: dict[str, ToolMetadata] = {}
        for meta in self._metadata.values():
            base = meta.base_name or meta.name.split("__v")[0]
            if base not in latest or meta.version > latest[base].version:
                latest[base] = meta
        return list(latest.values())

    def resolve_name(self, name: str) -> Optional[str]:
        if name in self._metadata:
            return name
        base = self._normalize_base_name(name)
        candidates = [meta for meta in self._metadata.values() if meta.base_name == base]
        if not candidates:
            candidates = [
                meta
                for meta in self._metadata.values()
                if meta.name.startswith(f"{base}__")
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda m: m.version, reverse=True)
        return candidates[0].name

    def list_tools(self) -> List[ToolMetadata]:
        return list(self._metadata.values())

    def get_tool_docstring(self, name: str) -> str:
        metadata = self._metadata.get(name)
        return metadata.docstring if metadata else ""

    def describe_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": m.name,
                "signature": m.signature,
                "description": m.description,
                "docstring": m.docstring,
                "usage_count": m.usage_count,
                "tool_type": m.tool_type,
                "input_schema": m.input_schema,
                "capabilities": m.capabilities,
            }
            for m in self._metadata.values()
        ]

    # ---------------------------
    # Import / invocation
    # ---------------------------
    def _load_tool_module(self, name: str, tool_path: str):
        """
        Load a tool module from disk. Uses a unique module name per load to avoid
        stale code if a tool is overwritten during a run.
        """
        unique_tag = f"{int(time.time() * 1000)}_{os.path.getmtime(tool_path)}"
        module_name = f"generated_tools.{name}.{unique_tag}"

        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import tool '{name}' from '{tool_path}'.")

        module = importlib.util.module_from_spec(spec)
        # Ensure no accidental reuse
        sys.modules.pop(module_name, None)
        spec.loader.exec_module(module)
        return module

    def _signature_accepts_kw(self, sig: inspect.Signature, key: str) -> bool:
        if key in sig.parameters:
            p = sig.parameters[key]
            return p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        # If **kwargs exists, accept any kw
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def _auto_adapt_call(
        self, run_fn: Callable[..., Any], query: str
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Adapt a single string query into args/kwargs that match run_fn's signature.
        """
        sig = inspect.signature(run_fn)
        params = list(sig.parameters.values())

        # Common payload used for dict-style tools or **kwargs tools
        payload = {"task_text": query, "query": query, "text": query, "input": query}

        # If run() takes no parameters (or only optional), try no-arg call.
        required = [
            p for p in params
            if p.default is inspect._empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(required) == 0 and len(params) == 0:
            return (), {}

        # If it has **kwargs, prefer passing payload as kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return (), payload

        # If it has a single parameter, choose best mapping
        if len(params) == 1:
            p = params[0]
            name = p.name.lower()

            # If the param name strongly suggests dict/payload, pass dict
            if name in {"payload", "data", "inputs", "params", "context", "request", "body"}:
                return (payload,), {}

            # If annotation suggests Mapping/dict, pass dict (best-effort)
            ann = p.annotation
            if ann in (dict, Mapping) or str(ann).lower() in {"dict", "mapping", "typing.mapping"}:
                return (payload,), {}

            # Otherwise pass query string directly
            return (query,), {}

        # If it has *args, simplest: pass query as first arg
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return (query,), {}

        # If it has named parameters and accepts "query"/"task_text"/"text", pass that kw
        for key in ("task_text", "query", "text", "input"):
            if self._signature_accepts_kw(sig, key):
                return (), {key: query}

        # If first param looks payload-like, pass dict to it (only if remaining are optional)
        first = params[0]
        remaining = params[1:]
        remaining_required = [
            p for p in remaining
            if p.default is inspect._empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if not remaining_required:
            return (payload,), {}

        # Give up: cannot adapt safely
        raise TypeError(
            "Cannot auto-invoke tool from a single query string because the tool "
            "requires multiple non-optional parameters."
        )


    def get_tool_usage(self, name: str, max_chars: int = 1200) -> str:
        """Return module + run() docstrings for a tool, trimmed for prompt use."""
        if not self.has_tool(name):
            return ""

        tool_path = self._get_tool_path(name)
        try:
            module = self._load_tool_module(name, tool_path)
            mod_doc = (getattr(module, "__doc__", "") or "").strip()
            run_doc = ""
            if hasattr(module, "run") and callable(getattr(module, "run")):
                run_doc = (getattr(getattr(module, "run"), "__doc__", "") or "").strip()

            usage = ""
            if mod_doc:
                usage += f"MODULE DOC:\n{mod_doc}\n"
            if run_doc:
                usage += f"\nRUN() DOC:\n{run_doc}\n"

            usage = usage.strip()
            if len(usage) > max_chars:
                usage = usage[: max_chars - 3] + "..."
            return usage
        except Exception:
            return ""


    def invoke_tool(
        self,
        name: str,
        *args: Any,
        invocation_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        resolved_name = self.resolve_name(name)
        if resolved_name is None:
            resolved_name = name
        # print(
        #     f"[ToolRegistry] invoke_tool start name={resolved_name} "
        #     f"args={self._preview(args)} kwargs={self._preview(kwargs)}"
        # )
        start_time = time.monotonic()
        inv_ctx = dict(invocation_context or {})

        if not self.has_tool(resolved_name):
            # print(f"[ToolRegistry] invoke_tool missing tool name={resolved_name}")
            outcome = ToolResult.failure(f"Tool '{resolved_name}' not found.")
            duration_ms = int((time.monotonic() - start_time) * 1000)
            self._notify(
                {
                    "event": "invoke",
                    "tool_name": resolved_name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "args_preview": self._preview(args),
                    "kwargs_preview": self._preview(kwargs),
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "success": outcome.success,
                    "error": outcome.error,
                    "error_type": "ToolNotFoundError",
                    "duration_ms": duration_ms,
                    "invocation_context": inv_ctx,
                }
            )
            return outcome

        tool_path = self._get_tool_path(resolved_name)
        # print(f"[ToolRegistry] invoke_tool path name={resolved_name} path={tool_path}")

        error_type: str | None = None
        error_traceback: str | None = None

        try:
            module = self._load_tool_module(resolved_name, tool_path)
            # print(
            #     f"[ToolRegistry] invoke_tool loaded name={resolved_name} "
            #     f"module={getattr(module, '__name__', '<unknown>')}"
            # )

            if not hasattr(module, "run"):
                raise AttributeError(
                    f"Tool '{name}' does not expose a callable 'run' entrypoint."
                )
            run_fn = getattr(module, "run")
            if not callable(run_fn):
                raise TypeError(f"Tool '{name}'.run exists but is not callable.")

            result = run_fn(*args, **kwargs)

            # print(
            #     f"[ToolRegistry] invoke_tool success name={resolved_name} "
            #     f"result_type={type(result).__name__} result={self._preview(result)}"
            # )
            outcome = ToolResult.success_result(result)

        except Exception as e:
            print(
                f"[ToolRegistry] invoke_tool failure name={resolved_name} "
                f"error_type={type(e).__name__} error={e}"
            )
            print(traceback.format_exc())
            outcome = ToolResult.failure(str(e))
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()

        with self._lock:
            metadata = self._metadata.get(resolved_name)
            if metadata:
                metadata.usage_count += 1
                metadata.last_used_time = datetime.datetime.now(datetime.UTC).isoformat()
                if outcome.success:
                    metadata.success_count += 1
                else:
                    metadata.failure_count += 1
                env_name = inv_ctx.get("environment")
                if env_name:
                    if metadata.environment_usage is None:
                        metadata.environment_usage = {}
                    metadata.environment_usage[env_name] = (
                        metadata.environment_usage.get(env_name, 0) + 1
                    )
                metadata.reliability_score = self._calculate_reliability(metadata)
                self._save_metadata()

        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._notify(
            {
                "event": "invoke",
                "tool_name": resolved_name,
                "args": list(args),
                "kwargs": kwargs,
                "args_preview": self._preview(args),
                "kwargs_preview": self._preview(kwargs),
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "success": outcome.success,
                "error": outcome.error,
                "error_type": error_type,
                "traceback": error_traceback,
                "result_preview": self._preview(outcome.output),
                "duration_ms": duration_ms,
                "invocation_context": inv_ctx,
            }
        )
        self._append_snapshot(
            {
                "tool_name": resolved_name,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "success": outcome.success,
                "environment": inv_ctx.get("environment"),
            }
        )
        return outcome


_REGISTRY_INSTANCE: Optional[ToolRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def _resolve_registry_path(tool_registry_path: str) -> str:
    """Expand timestamp placeholders to match the output directory structure."""
    predefined_structure = get_predefined_timestamp_structure()
    try:
        return tool_registry_path.format(**predefined_structure)
    except Exception:
        return tool_registry_path


def get_registry(
    tool_registry_path: Optional[str] = None, *, force_reset: bool = False
) -> ToolRegistry:
    global _REGISTRY_INSTANCE
    with _REGISTRY_LOCK:
        if force_reset or _REGISTRY_INSTANCE is None:
            base_path = _resolve_registry_path(tool_registry_path or os.getcwd())
            print(f"[ToolRegistry] Creating new registry at '{base_path}'")
            _REGISTRY_INSTANCE = ToolRegistry(base_path)
        elif tool_registry_path and os.path.abspath(_resolve_registry_path(tool_registry_path)) != _REGISTRY_INSTANCE.base_path:
            resolved_path = _resolve_registry_path(tool_registry_path)
            print(f"[ToolRegistry] Switching registry base path to '{resolved_path}'")
            _REGISTRY_INSTANCE = ToolRegistry(resolved_path)
    return _REGISTRY_INSTANCE
