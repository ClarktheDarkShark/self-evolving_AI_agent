from src.tasks.instance.os_interaction.task import OSInteraction
from src.tasks.instance.os_interaction.typings import AgentAction


def test_parse_act_bash():
    text = "Act: bash\n```bash\nls -la\n```"
    result = OSInteraction._parse_agent_response(text)
    assert result.action == AgentAction.EXECUTE
    assert result.content == "ls -la"


def test_parse_action_bash():
    text = "Action: bash\n```bash\npwd\n```"
    result = OSInteraction._parse_agent_response(text)
    assert result.action == AgentAction.EXECUTE
    assert result.content == "pwd"


def test_parse_finish():
    text = "Action: finish"
    result = OSInteraction._parse_agent_response(text)
    assert result.action == AgentAction.FINISH
    assert result.content is None
