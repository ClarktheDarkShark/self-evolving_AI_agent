from __future__ import annotations

import json
import re
from urllib.error import URLError

import src.agents.instance.sage_agent_controller as sage_controller_module
from src.agents.instance.sage_agent_controller import SAGEAgentController
from src.sage.invoker import SAGEInvocationResult, invoke_sage_program
from src.sage.parser import parse_sage_code
from src.typings import ChatHistory, ChatHistoryItem, Role


SAGE_CODE = """
from SPARQLWrapper import SPARQLWrapper, JSON

def solve():
    sparql = SPARQLWrapper("http://localhost:9999/blazegraph/namespace/kb/sparql")
    query = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT ?entity WHERE {
      ?entity fb:type.object.name "Test Entity"@en .
    } LIMIT 50
    \"\"\"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
""".strip()


class _FakeQueryResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def convert(self) -> dict:
        return self._payload


class _SuccessSPARQLWrapper:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.query_text = None
        self.return_format = None

    def setQuery(self, query_text: str) -> None:
        self.query_text = query_text

    def setReturnFormat(self, return_format: str) -> None:
        self.return_format = return_format

    def query(self) -> _FakeQueryResponse:
        return _FakeQueryResponse(
            {
                "head": {"vars": ["entity"]},
                "results": {
                    "bindings": [
                        {
                            "entity": {
                                "type": "uri",
                                "value": "http://rdf.freebase.com/ns/m.test_entity",
                            }
                        }
                    ]
                },
            }
        )


class _FailureSPARQLWrapper:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.query_text = None
        self.return_format = None

    def setQuery(self, query_text: str) -> None:
        self.query_text = query_text

    def setReturnFormat(self, return_format: str) -> None:
        self.return_format = return_format

    def query(self) -> _FakeQueryResponse:
        raise URLError(ConnectionRefusedError(61, "Connection refused"))


class _SuccessSPARQLModule:
    SPARQLWrapper = _SuccessSPARQLWrapper
    JSON = "JSON"


class _FailureSPARQLModule:
    SPARQLWrapper = _FailureSPARQLWrapper
    JSON = "JSON"


class _FakeLanguageModel:
    role_dict = {
        Role.USER: "user",
        Role.AGENT: "assistant",
    }

    def inference(self, histories, inference_config_dict, system_prompt):
        if "routing mechanism" in system_prompt:
            content = '{"action":"generate_tool"}'
        elif "expert knowledge graph query generator" in system_prompt:
            content = "###QUERY_START\n" + SAGE_CODE + "\n###QUERY_END"
        else:
            content = "Final Answer: ignored"
        return [ChatHistoryItem(role=Role.AGENT, content=content)]


def _extract_bridge_payload(action_line: str) -> dict:
    match = re.match(r'^Action: execute_macro\("([^"]+)", (.*)\)$', action_line)
    if match is None:
        raise AssertionError(f"Unexpected action line: {action_line}")
    return json.loads(match.group(2))


def main() -> None:
    parsed_program = parse_sage_code(SAGE_CODE)

    success_result = invoke_sage_program(
        parsed_program,
        endpoint_url="http://127.0.0.1:3001/kb/sparql",
        sparqlwrapper_module=_SuccessSPARQLModule(),
    )
    print("SUCCESS_CASE")
    print(json.dumps(
        {
            "success": success_result.success,
            "failure_kind": success_result.failure_kind,
            "payload": success_result.payload,
        },
        indent=2,
        sort_keys=True,
    ))

    failure_result = invoke_sage_program(
        parsed_program,
        endpoint_url="http://127.0.0.1:9/kb/sparql",
        sparqlwrapper_module=_FailureSPARQLModule(),
    )
    print("FAILURE_CASE")
    print(json.dumps(
        {
            "success": failure_result.success,
            "failure_kind": failure_result.failure_kind,
            "error": failure_result.error,
            "diagnostics": failure_result.diagnostics,
        },
        indent=2,
        sort_keys=True,
    ))

    def run_controller_case(invocation_result: SAGEInvocationResult) -> tuple[str, dict, list[dict]]:
        original_execute = sage_controller_module.execute_sage_code_with_result
        sage_controller_module.execute_sage_code_with_result = lambda code: invocation_result
        try:
            controller = SAGEAgentController(language_model=_FakeLanguageModel())
            events: list[dict] = []
            controller._emit_generated_tools_event = lambda payload: events.append(dict(payload))

            chat_history = ChatHistory()
            chat_history.inject(
                ChatHistoryItem(
                    role=Role.USER,
                    content="Question: which test entity should be returned?, Entities: ['Test Entity']",
                )
            )
            response = controller._inference(chat_history)
            bridge_payload = _extract_bridge_payload(response.content)
            return response.content, bridge_payload, events
        finally:
            sage_controller_module.execute_sage_code_with_result = original_execute

    controller_success_output, controller_success_bridge_payload, controller_success_events = run_controller_case(
        SAGEInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["entity"]},
                "results": {
                    "bindings": [
                        {
                            "entity": {
                                "type": "uri",
                                "value": "http://rdf.freebase.com/ns/m.test_entity",
                            }
                        }
                    ]
                },
            },
        )
    )
    print("CONTROLLER_SUCCESS")
    print(json.dumps(
        {
            "agent_output": controller_success_output,
            "bridge_payload": controller_success_bridge_payload,
            "events": [
                event
                for event in controller_success_events
                if event.get("event")
                in {
                    "toolgen_smoke_test",
                    "sage_adapter_artifact_classified",
                    "sage_adapter_materialization_selected",
                    "sage_adapter_bridge_emitted",
                }
            ],
        },
        indent=2,
        sort_keys=True,
    ))

    response_content, bridge_payload, events = run_controller_case(
        SAGEInvocationResult(
            success=False,
            error="<urlopen error [Errno 61] Connection refused>",
            failure_kind="endpoint_unavailable",
            diagnostics={
                "endpoint_url": "http://127.0.0.1:9/kb/sparql",
                "errno": 61,
            },
        )
    )

    selected_events = [
        event
        for event in events
        if event.get("event")
        in {
            "toolgen_smoke_test",
            "sage_invoker_endpoint_unavailable",
            "sage_controller_execution_fallback_used",
            "sage_adapter_artifact_classified",
            "sage_adapter_materialization_selected",
            "sage_adapter_bridge_emitted",
        }
    ]
    print("CONTROLLER_FAILURE_FALLBACK")
    print(json.dumps(
        {
            "agent_output": response_content,
            "bridge_payload": bridge_payload,
            "events": selected_events,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
