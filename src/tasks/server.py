from fastapi import FastAPI, APIRouter, HTTPException
from datetime import datetime, timezone
import time
import uvicorn

from .task import Task, DatasetItem
from src.typings import TaskRequest, TaskResponse, Role
from src.utils import Server, SafeLogger


class TaskServer(Server):
    def __init__(self, router: APIRouter, task: Task[DatasetItem]) -> None:
        Server.__init__(self, router, task)
        self.task = task
        self.router.post("/get_sample_index_list")(self.get_sample_index_list)
        self.router.post("/reset")(self.reset)
        self.router.post("/interact")(self.interact)
        self.router.post("/complete")(self.complete)
        self.router.post("/release")(self.release)
        self.router.post("/calculate_metric")(self.calculate_metric)
        self.router.post("/evaluate_generated_macro")(self.evaluate_generated_macro)

    def get_sample_index_list(self) -> TaskResponse.GetSampleIndexList:
        sample_index_list = self.task.get_sample_index_list()
        return TaskResponse.GetSampleIndexList(sample_index_list=sample_index_list)

    def reset(self, data: TaskRequest.Reset) -> TaskResponse.Reset:
        try:
            self.task.reset(data.session)
            return TaskResponse.Reset(session=data.session)
        except Exception as exc:
            SafeLogger.error("Task reset failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def interact(self, data: TaskRequest.Interact) -> TaskResponse.Interact:
        start_ts = datetime.now(timezone.utc).isoformat()
        start_time = time.monotonic()
        last_agent = None
        try:
            items = data.session.chat_history
            for idx in range(items.get_value_length() - 1, -1, -1):
                item = items.get_item_deep_copy(idx)
                if item.role == Role.AGENT:
                    last_agent = item.content
                    break
        except Exception:
            last_agent = None
        SafeLogger.info(
            "Task interact start task=%s sample=%s chat_len=%s start_ts=%s last_agent=%s",
            getattr(data.session, "task_name", None),
            getattr(data.session, "sample_index", None),
            data.session.chat_history.get_value_length(),
            start_ts,
            last_agent,
        )
        try:
            self.task.interact(data.session)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            SafeLogger.error(
                "Task interact failed task=%s sample=%s elapsed_ms=%.1f error=%s",
                getattr(data.session, "task_name", None),
                getattr(data.session, "sample_index", None),
                elapsed_ms,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        end_ts = datetime.now(timezone.utc).isoformat()
        SafeLogger.info(
            "Task interact end task=%s sample=%s elapsed_ms=%.1f end_ts=%s",
            getattr(data.session, "task_name", None),
            getattr(data.session, "sample_index", None),
            elapsed_ms,
            end_ts,
        )
        return TaskResponse.Interact(session=data.session)

    def complete(self, data: TaskRequest.Complete) -> TaskResponse.Complete:
        self.task.complete(data.session)
        return TaskResponse.Complete(session=data.session)

    def release(self) -> None:
        self.task.release()
        return

    def calculate_metric(
        self, data: TaskRequest.CalculateMetric
    ) -> TaskResponse.CalculateMetric:
        metric = self.task.calculate_metric(data.session_partial_list)
        return TaskResponse.CalculateMetric(metric=metric)

    def evaluate_generated_macro(
        self, data: TaskRequest.EvaluateGeneratedMacro
    ) -> TaskResponse.EvaluateGeneratedMacro:
        """Run a generated macro tool against the live KG environment.

        Only tasks that implement evaluate_generated_macro() support this
        endpoint. For other task types the endpoint returns an error JSON.
        """
        fn = getattr(self.task, "evaluate_generated_macro", None)
        if not callable(fn):
            result_json = '{"status": "error", "error": "task_does_not_support_evaluate_generated_macro"}'
        else:
            try:
                result_json = fn(data.tool_code, data.execution_payload_json)
            except Exception as exc:
                import traceback as _tb
                result_json = '{"status": "error", "error": ' + repr(str(exc)) + '}'
                SafeLogger.error(
                    "evaluate_generated_macro failed: %s\n%s", exc, _tb.format_exc()
                )
        return TaskResponse.EvaluateGeneratedMacro(result_json=result_json)

    def shutdown(self) -> None:
        self.release()

    @staticmethod
    def start_server(task: Task[DatasetItem], port: int, prefix: str) -> None:
        app = FastAPI()
        router = APIRouter()
        # Create an instance to access the shutdown method
        server_instance = TaskServer(router, task)
        app.include_router(router, prefix=prefix)
        # Add the shutdown event handler using lifespan events
        # https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated
        app.add_event_handler("shutdown", server_instance.shutdown)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_config=None,
        )
