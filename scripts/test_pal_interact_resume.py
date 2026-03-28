import json
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from src.factories.chat_history_item import ChatHistoryItemFactory
from src.tasks.instance.knowledge_graph.task import KnowledgeGraph
from src.tasks.server import TaskServer
from src.typings import Role, SampleStatus, Session, TaskName


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = str(REPO_ROOT / "outputs" / "pal_runtime" / "tool_state")


def make_task() -> KnowledgeGraph:
    factory = ChatHistoryItemFactory(
        chat_history_item_dict_path=str(
            REPO_ROOT / "chat_history_items" / "standard" / "knowledge_graph.json"
        )
    )
    return KnowledgeGraph(
        task_name=TaskName.KNOWLEDGE_GRAPH,
        chat_history_item_factory=factory,
        sparql_url="http://127.0.0.1:3001/kb/sparql",
        ontology_dir_path=str(REPO_ROOT / "data" / "v0121" / "knowledge_graph" / "ontology"),
        data_file_path=str(
            REPO_ROOT
            / "data"
            / "v0303"
            / "knowledge_graph"
            / "processed"
            / "hf_export"
            / "entry_dict.json"
        ),
        max_round=15,
        data_source="local",
    )


def bootstrap_session(sample_index: str) -> Session:
    task = make_task()
    session = Session(task_name=TaskName.KNOWLEDGE_GRAPH, sample_index=sample_index)
    task.reset(session)
    return session


def make_client() -> TestClient:
    app = FastAPI()
    router = APIRouter()
    TaskServer(router, make_task())
    app.include_router(router, prefix="/api")
    return TestClient(app)


def to_jsonable(session: Session) -> dict:
    return json.loads(session.model_dump_json())


def build_empty_bridge_session(sample_index: str) -> Session:
    seed = bootstrap_session(sample_index)
    session = Session(
        task_name=TaskName.KNOWLEDGE_GRAPH,
        sample_index=sample_index,
        sample_status=SampleStatus.RUNNING,
    )
    for idx in range(seed.chat_history.get_value_length()):
        session.chat_history.inject(seed.chat_history.get_item_deep_copy(idx))
    session.chat_history.inject(
        {
            "role": Role.AGENT,
            "content": (
                'Action: execute_macro("pal_benchmark_bridge_macro", '
                '{"pal_artifact_type": "empty", "pal_artifact_value": [], '
                '"pal_artifact_source": "raw_execution", '
                '"pal_artifact_diagnostics": {"binding_count": 0}, '
                '"run_id": "knowledge_graph_3", '
                f'"state_dir": "{STATE_DIR}"'
                "})"
            ),
        }
    )
    session.tool_invoked = ["pal_benchmark_bridge_macro"]
    session.tool_invoked_any = True
    return session


def build_entity_id_final_answer_session(sample_index: str) -> Session:
    seed = bootstrap_session(sample_index)
    session = Session(
        task_name=TaskName.KNOWLEDGE_GRAPH,
        sample_index=sample_index,
        sample_status=SampleStatus.RUNNING,
    )
    for idx in range(seed.chat_history.get_value_length()):
        session.chat_history.inject(seed.chat_history.get_item_deep_copy(idx))
    session.chat_history.inject(
        {
            "role": Role.AGENT,
            "content": (
                'Action: execute_macro("pal_benchmark_bridge_macro", '
                '{"pal_artifact_type": "entity_id", '
                '"pal_artifact_value": "m.02h8b9t", '
                '"pal_artifact_source": "raw_execution", '
                '"pal_artifact_diagnostics": {"binding_count": 1}, '
                '"run_id": "knowledge_graph_0", '
                f'"state_dir": "{STATE_DIR}"'
                "})"
            ),
        }
    )
    session.chat_history.inject(
        {
            "role": Role.USER,
            "content": (
                "Macro result: pal_benchmark_bridge_macro -> SUCCESS. "
                "Final variable: #0. Observation: PAL benchmark bridge "
                "materialized entity_id into a benchmark variable."
            ),
        }
    )
    session.chat_history.inject(
        {
            "role": Role.AGENT,
            "content": "Final Answer: #0",
        }
    )
    session.tool_invoked = ["pal_benchmark_bridge_macro"]
    session.tool_invoked_any = True
    return session


def main() -> None:
    client = make_client()

    empty_session = build_empty_bridge_session("3")
    empty_response = client.post(
        "/api/interact",
        json={"session": to_jsonable(empty_session)},
    )
    empty_response.raise_for_status()
    returned_empty = Session.model_validate(empty_response.json()["session"])
    print("CASE empty pending bridge")
    print("status_code", empty_response.status_code)
    print("sample_status", returned_empty.sample_status)
    print("last_user", returned_empty.chat_history.get_item_deep_copy(-1).content)
    print()

    entity_session = build_entity_id_final_answer_session("0")
    entity_response = client.post(
        "/api/interact",
        json={"session": to_jsonable(entity_session)},
    )
    entity_response.raise_for_status()
    returned_entity = Session.model_validate(entity_response.json()["session"])
    print("CASE entity_id final answer")
    print("status_code", entity_response.status_code)
    print("sample_status", returned_entity.sample_status)
    print("task_output", returned_entity.task_output)
    print("finish_reason", returned_entity.finish_reason)


if __name__ == "__main__":
    main()
