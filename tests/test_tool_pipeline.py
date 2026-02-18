from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tool_registry_mod = _load_module(
    "tool_registry", ROOT / "src" / "self_evolving_agent" / "tool_registry.py"
)
tool_validation_mod = _load_module(
    "tool_validation", ROOT / "src" / "self_evolving_agent" / "tool_validation.py"
)

ToolRegistry = tool_registry_mod.ToolRegistry
validate_tool_code = tool_validation_mod.validate_tool_code


TOOL_CODE = """\
\"\"\"
Echo payload tool.
\"\"\"
from __future__ import annotations


def run(payload: dict) -> dict:
    \"\"\"
    Example: run({\"foo\": \"bar\"})
    \"\"\"
    try:
        if not isinstance(payload, dict):
            return {\"error\": \"payload must be dict\"}
        return {\"echo\": payload}
    except Exception as exc:
        return {\"error\": str(exc)}


def self_test() -> bool:
    good = run({\"foo\": \"bar\"})
    assert good.get(\"echo\") == {\"foo\": \"bar\"}
    bad = run(\"nope\")
    assert \"error\" in bad
    return True
"""


def test_tool_pipeline_end_to_end() -> None:
    result = validate_tool_code(TOOL_CODE)
    assert result.success, result.error
    assert isinstance(result.smoke_output, dict)

    input_schema = {
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ToolRegistry(tmpdir)
        metadata = registry.register_tool(
            name="echo_payload",
            code=TOOL_CODE,
            signature="run(payload: dict) -> dict",
            description="Echo the payload dict.",
            tool_type="utility",
            tool_category="parser",
            input_schema=input_schema,
            capabilities=["echo", "passthrough"],
        )
        assert metadata is not None
        outcome = registry.invoke_tool(metadata.name, {"foo": "bar"})
        assert outcome.success, outcome.error
        assert outcome.output == {"echo": {"foo": "bar"}}
