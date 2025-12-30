from __future__ import annotations

from src.self_evolving_agent.tool_spec import ToolSpec
from src.self_evolving_agent.tool_validation import validate_tool_code
from src.self_evolving_agent.tool_registry import get_registry
from src.self_evolving_agent.tool_retrieval import retrieve_tools


def main() -> int:
    registry = get_registry("outputs/tool_library", force_reset=True)

    spec = ToolSpec(
        name="echo_helper",
        description="Return the provided text uppercased.",
        signature="run(text: str) -> str",
        code_lines=[
            '"""',
            "Module: echo_helper",
            "Purpose: Return the provided text uppercased.",
            '"""',
            "",
            "def run(text: str) -> str:",
            '    """Return the text uppercased."""',
            "    return str(text).upper()",
        ],
    )

    code = "\n".join(spec.code_lines).rstrip() + "\n"
    validation = validate_tool_code(code)
    if not validation.success:
        print(f"[smoke] validation failed: {validation.error}")
        return 1

    meta = registry.register_tool(
        name=spec.name,
        code=code,
        signature=spec.signature,
        description=spec.description,
        tool_type=spec.tool_type,
        input_schema=spec.input_schema,
        capabilities=spec.capabilities,
    )
    if meta is None:
        print("[smoke] register_tool failed")
        return 1

    registry.record_validation_result(meta.name, True)
    tools = (
        registry.list_latest_tools()
        if hasattr(registry, "list_latest_tools")
        else registry.list_tools()
    )
    retrieved = retrieve_tools("uppercase the text", tools, top_k=5, min_reliability=0.2)
    if not retrieved:
        print("[smoke] retrieval failed")
        return 1

    tool_name = retrieved[0].tool.name
    first = registry.invoke_tool(
        tool_name,
        "hello",
        invocation_context={"environment": "smoke"},
    )
    second = registry.invoke_tool(
        tool_name,
        "hello again",
        invocation_context={"environment": "smoke"},
    )

    print("[smoke] tool:", tool_name)
    print("[smoke] first:", first.output, "success=", first.success)
    print("[smoke] second:", second.output, "success=", second.success)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
