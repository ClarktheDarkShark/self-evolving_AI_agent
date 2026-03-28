from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.self_evolving_agent.tool_registry import ToolRegistry
from src.self_evolving_agent.tool_validation import validate_tool_code

TOOL_CODE = """\
\"\"\"
Echo payload tool.

# INVOKE_WITH: run(payload)
# RUN_PAYLOAD_REQUIRED: foo
# RUN_PAYLOAD_OPTIONAL:
\"\"\"
from __future__ import annotations


def run(payload: dict) -> dict:
    \"\"\"
    Example: run({\"foo\": \"bar\"})
    \"\"\"
    try:
        if not isinstance(payload, dict):
            return {\"status\": \"ERROR\", \"final_variable\": None, \"observation\": \"payload must be dict\"}
        foo = payload.get(\"foo\", \"\")
        return {\"status\": \"SUCCESS\", \"final_variable\": foo, \"observation\": f\"echo:{foo}\"}
    except Exception as exc:
        return {\"status\": \"ERROR\", \"final_variable\": None, \"observation\": str(exc)}


def self_test() -> bool:
    good = run({\"foo\": \"bar\"})
    assert good.get(\"status\") == \"SUCCESS\"
    assert good.get(\"final_variable\") == \"bar\"
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
        assert isinstance(outcome.output, dict)
        assert outcome.output.get("status") == "SUCCESS"
        assert outcome.output.get("final_variable") == "bar"
