import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_evolving_agent.tool_registry import ToolRegistry


def test_tool_registry_persistence_and_invocation(tmp_path) -> None:
    registry = ToolRegistry(str(tmp_path))
    metadata = registry.register_tool(
        name="adder_tool",
        code="def run(x, y):\n    return x + y\n",
        signature="run(x, y)",
        description="Adds two numbers.",
    )
    assert metadata.name == "adder_tool"
    tool_file = tmp_path / "generated_tools" / "adder_tool.py"
    assert tool_file.exists()

    reloaded = ToolRegistry(str(tmp_path))
    tools = reloaded.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "adder_tool"

    result = reloaded.invoke_tool("adder_tool", 1, 2)
    assert result.success
    assert result.output == 3

    # usage count should persist
    persisted = ToolRegistry(str(tmp_path))
    assert persisted.list_tools()[0].usage_count == 1
