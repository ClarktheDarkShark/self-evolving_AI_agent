import datetime
import importlib.util
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.typings.config import get_predefined_timestamp_structure


@dataclass
class ToolMetadata:
    name: str
    signature: str
    description: str
    creation_time: str
    usage_count: int = 0


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
    """

    def __init__(self, tool_registry_path: str):
        self.base_path = os.path.abspath(tool_registry_path)
        self.tools_dir = os.path.join(self.base_path, "generated_tools")
        self.metadata_path = os.path.join(self.base_path, "metadata.json")
        os.makedirs(self.tools_dir, exist_ok=True)
        print(f"[ToolRegistry] Initialized registry at '{self.base_path}'")
        self._metadata: Dict[str, ToolMetadata] = {}
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not os.path.exists(self.metadata_path):
            return
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data: Iterable[dict[str, Any]] = json.load(f)
            for entry in data:
                tool_metadata = ToolMetadata(**entry)
                self._metadata[tool_metadata.name] = tool_metadata
        except Exception:
            # Corrupted metadata should not crash the run; start from an empty registry.
            self._metadata = {}

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self._metadata.values()], f, indent=2)

    def _sanitize_name(self, name: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        return sanitized or "tool"

    def _unwrap_code_block(self, code: str) -> str:
        fenced_pattern = re.compile(r"```(?:python)?\s*(?P<body>[\s\S]*?)```", re.DOTALL)
        if match := fenced_pattern.search(code):
            return match.group("body").strip()
        return code

    def _get_tool_path(self, name: str) -> str:
        filename = f"{name}.py"
        return os.path.join(self.tools_dir, filename)

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

    def register_tool(
        self, name: str, code: str, signature: str, description: str
    ) -> ToolMetadata:
        sanitized_name = self._sanitize_name(name)
        normalized_code = self._unwrap_code_block(code)
        if not normalized_code.strip():
            normalized_code = "def run(*args, **kwargs):\n    return None\n"
        tool_path = self._get_tool_path(sanitized_name)
        with self._lock:
            print(
                f"[ToolRegistry] Persisting tool '{sanitized_name}' to '{tool_path}'"
            )
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(normalized_code)
            metadata = self._metadata.get(
                sanitized_name,
                ToolMetadata(
                    name=sanitized_name,
                    signature=signature,
                    description=description,
                    creation_time=datetime.datetime.now(
                        datetime.UTC
                    ).isoformat(),
                    usage_count=0,
                ),
            )
            # Allow the latest description/signature to overwrite previous entries while preserving creation time.
            metadata.signature = signature
            metadata.description = description
            self._metadata[sanitized_name] = metadata
            self._save_metadata()
        self._notify(
            {
                "event": "register",
                "tool_name": sanitized_name,
                "signature": signature,
                "description": description,
                "path": tool_path,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        return metadata

    def has_tool(self, name: str) -> bool:
        return name in self._metadata

    def list_tools(self) -> List[ToolMetadata]:
        return list(self._metadata.values())

    def invoke_tool(self, name: str, *args: Any, **kwargs: Any) -> ToolResult:
        if not self.has_tool(name):
            return ToolResult.failure(f"Tool '{name}' not found.")
        tool_path = self._get_tool_path(name)
        try:
            spec = importlib.util.spec_from_file_location(f"generated_tools.{name}", tool_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to import tool '{name}' from '{tool_path}'.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "run"):
                raise AttributeError(
                    f"Tool '{name}' does not expose a callable 'run' entrypoint."
                )
            result = getattr(module, "run")(*args, **kwargs)
            outcome = ToolResult.success_result(result)
        except Exception as e:
            outcome = ToolResult.failure(str(e))
        with self._lock:
            metadata = self._metadata.get(name)
            if metadata:
                metadata.usage_count += 1
                self._save_metadata()
        self._notify(
            {
                "event": "invoke",
                "tool_name": name,
                "args": list(args),
                "kwargs": kwargs,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "success": outcome.success,
                "error": outcome.error,
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
            base_path = _resolve_registry_path(
                tool_registry_path
                or os.path.join(
                    os.getcwd(), "generated_tools"
                )
            )
            print(f"[ToolRegistry] Creating new registry at '{base_path}'")
            _REGISTRY_INSTANCE = ToolRegistry(base_path)
        elif tool_registry_path and os.path.abspath(_resolve_registry_path(tool_registry_path)) != _REGISTRY_INSTANCE.base_path:
            resolved_path = _resolve_registry_path(tool_registry_path)
            print(
                "[ToolRegistry] Switching registry base path to '"
                f"{resolved_path}'"
            )
            _REGISTRY_INSTANCE = ToolRegistry(resolved_path)
    return _REGISTRY_INSTANCE
