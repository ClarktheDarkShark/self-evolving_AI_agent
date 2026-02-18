import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from src.self_evolving_agent.toolgen_contracts import validate_toolgen_output
from src.self_evolving_agent.tool_registry import ToolRegistry


VALID_TOOL = (
    "###TOOL_START\n"
    "\"\"\"\n"
    "INPUT_SCHEMA: required=task_text,asked_for,trace,actions_spec,run_id,state_dir; "
    "optional=constraints,output_contract,draft_response,candidate_output,env_observation >\n"
    "\"\"\"\n"
    "import json\n"
    "import os\n"
    "\n"
    "# INVOKE_WITH: {\"args\":[<RUN_PAYLOAD>], \"kwargs\":{}}\n"
    "# RUN_PAYLOAD_REQUIRED: [\"task_text\",\"asked_for\",\"trace\",\"actions_spec\",\"run_id\",\"state_dir\"]\n"
    "# RUN_PAYLOAD_OPTIONAL: [\"constraints\",\"output_contract\",\"draft_response\",\"candidate_output\",\"env_observation\"]\n"
    "# INVOKE_EXAMPLE: {\"args\":[{\"task_text\":\"...\",\"asked_for\":\"...\",\"trace\":[],\"actions_spec\":{},\"run_id\":\"r1\",\"state_dir\":\"./state\"}], \"kwargs\":{}}\n"
    "# Example: run({'task_text':'...','asked_for':'...','trace':[],'actions_spec':{},'run_id':'r1','state_dir':'./state'})\n"
    "# input_schema_required: task_text, asked_for, trace, actions_spec, run_id, state_dir\n"
    "# input_schema_optional: constraints, output_contract, draft_response, candidate_output, env_observation\n"
    "\n"
    "def run(payload: dict) -> dict:\n"
    "    try:\n"
    "        return {\n"
    "            \"status\": \"need_step\",\n"
    "            \"next_action\": None,\n"
    "            \"answer_recommendation\": None,\n"
    "            \"final_answer_line\": None,\n"
    "            \"plan\": [],\n"
    "            \"validation\": {},\n"
    "            \"rationale\": [],\n"
    "            \"errors\": [],\n"
    "            \"warnings\": [],\n"
    "        }\n"
    "    except Exception as e:\n"
    "        return {\n"
    "            \"status\": \"error\",\n"
    "            \"next_action\": None,\n"
    "            \"answer_recommendation\": None,\n"
    "            \"final_answer_line\": None,\n"
    "            \"plan\": [],\n"
    "            \"validation\": {},\n"
    "            \"rationale\": [str(e)],\n"
    "            \"errors\": [str(e)],\n"
    "            \"warnings\": [],\n"
    "        }\n"
    "\n"
    "def self_test() -> bool:\n"
    "    return True\n"
    "###TOOL_END"
)


def test_toolgen_contract_valid_code_passes():
    result = validate_toolgen_output(VALID_TOOL)
    assert result.ok, result.errors


def test_toolgen_contract_missing_markers():
    bad = VALID_TOOL.replace("###TOOL_START\n", "")
    result = validate_toolgen_output(bad)
    assert not result.ok
    assert any(err.startswith("A:") for err in result.errors)


def test_toolgen_contract_extra_triple_quotes():
    bad = VALID_TOOL.replace(
        "def self_test() -> bool:",
        '"""extra"""\n\ndef self_test() -> bool:',
    )
    result = validate_toolgen_output(bad)
    assert not result.ok
    assert "B:docstring_triple_quotes_invalid" in result.errors


def test_toolgen_contract_wrong_signatures():
    bad = VALID_TOOL.replace("def run(payload: dict) -> dict:", "def run(x):")
    result = validate_toolgen_output(bad)
    assert not result.ok
    assert "D:run_signature_missing" in result.errors


def test_toolgen_contract_forbidden_substring():
    bad = VALID_TOOL.replace("return {", "return {  # eval(")
    result = validate_toolgen_output(bad)
    assert not result.ok
    assert "E:forbidden_substring:eval(" in result.errors


def test_toolgen_contract_sha_stable():
    result = validate_toolgen_output(VALID_TOOL)
    assert result.ok
    lines = result.normalized_code.splitlines()
    joined = "\n".join(lines).rstrip() + "\n"
    assert hashlib.sha256(joined.encode("utf-8")).hexdigest() == result.code_sha256


def test_toolgen_register_invoke_sha_stable():
    result = validate_toolgen_output(VALID_TOOL)
    assert result.ok
    with TemporaryDirectory() as td:
        registry = ToolRegistry(td)
        meta = registry.register_tool(
            name="contract_test_generated_tool",
            code=result.normalized_code,
            signature="run(payload: dict) -> dict",
            description="test",
            tool_type="utility",
            tool_category="utility",
            input_schema=None,
            capabilities=None,
            environment=None,
        )
        assert meta is not None
        tool_path = registry._get_tool_path(meta.name, environment=None)
        data = Path(tool_path).read_bytes()
        sha_before = hashlib.sha256(data).hexdigest()
        registry.invoke_tool(meta.name, {"task_text": "t", "asked_for": "a", "trace": [], "actions_spec": {}})
        sha_after = hashlib.sha256(Path(tool_path).read_bytes()).hexdigest()
        assert sha_before == sha_after == result.code_sha256
