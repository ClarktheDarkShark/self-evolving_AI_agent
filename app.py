import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from src.self_evolving_agent import SelfEvolvingController
from src.typings import ChatHistory, ChatHistoryItem, Role


ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
INDEX_PATH = TEMPLATES_DIR / "index.html"

app = FastAPI()


def _safe_preview(value: Any, max_len: int = 220) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


_FLOW_LABELS = {
    "orchestrator_input": "Orchestrator",
    "orchestrator_output": "Orchestrator",
    "tool_agent_input": "Tool Invoker",
    "tool_agent_output": "Tool Invoker",
    "solver_input": "Solver",
    "solver_output": "Solver",
    "solver_metrics": "Solver",
    "final_response": "Solver",
    "tool_advisory_compare": "Advisor",
    "run_scope": "Runner",
}


def _format_flow_event(event: Dict[str, Any]) -> str:
    name = str(event.get("event") or "event")
    label = _FLOW_LABELS.get(name, "Trace")
    parts = [name.replace("_", " ")]
    tool_name = event.get("tool_name")
    if tool_name:
        parts.append(f"tool={tool_name}")
    if name == "orchestrator_output":
        reason = event.get("reason") or event.get("output")
        if reason:
            parts.append(_safe_preview(reason))
    if name == "tool_agent_input" and event.get("tool_args"):
        parts.append("tool args prepared")
    if name == "tool_agent_output":
        success = event.get("success")
        if success is not None:
            parts.append(f"success={success}")
    if name == "solver_input":
        stage = (event.get("payload") or {}).get("stage")
        if stage:
            parts.append(f"stage={stage}")
    if name == "final_response":
        parts.append("final response ready")
    return f"[{label}] " + " | ".join(p for p in parts if p)


def _format_trace_event(event: Dict[str, Any]) -> Optional[str]:
    trace_type = event.get("trace_type")
    if trace_type == "flow_event":
        return _format_flow_event(event)
    if trace_type == "generated_tool":
        name = str(event.get("event") or "generated_tool")
        tool_name = event.get("tool_name")
        suffix = f" {tool_name}" if tool_name else ""
        return f"[ToolGen] {name}{suffix}"
    if trace_type == "loop_log":
        message = event.get("message")
        if message:
            return f"[Loop] {_safe_preview(message)}"
    return None


def _build_language_model() -> Any:
    provider = os.getenv("LATM_MODEL_PROVIDER", "openai").strip().lower()
    role_dict = {"user": "user", "agent": "assistant"}
    if provider in {"openai", "oai"}:
        from src.language_models.instance.openai_language_model import OpenaiLanguageModel

        model_name = (
            os.getenv("LATM_MODEL_NAME")
            or os.getenv("OPENAI_MODEL_NAME")
            or "gpt-5-mini"
        )
        return OpenaiLanguageModel(
            model_name=model_name,
            role_dict=role_dict,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if provider in {"huggingface", "hf"}:
        from src.language_models.instance.huggingface_language_model import (
            HuggingfaceLanguageModel,
        )

        model_path = os.getenv("LATM_MODEL_NAME") or os.getenv(
            "HF_MODEL_NAME_OR_PATH"
        )
        if not model_path:
            raise RuntimeError(
                "Missing Hugging Face model path. Set LATM_MODEL_NAME or "
                "HF_MODEL_NAME_OR_PATH."
            )
        dtype_raw = os.getenv("HF_MODEL_DTYPE", "bfloat16").strip()
        device_map = os.getenv("HF_MODEL_DEVICE_MAP", "auto").strip()
        try:
            import torch

            dtype = getattr(torch, dtype_raw, dtype_raw)
        except Exception:
            dtype = dtype_raw
        return HuggingfaceLanguageModel(
            model_name_or_path=model_path,
            role_dict=role_dict,
            dtype=dtype,
            device_map=device_map,
        )
    raise RuntimeError(f"Unsupported LATM_MODEL_PROVIDER: {provider}")


def _load_inference_config() -> Dict[str, Any]:
    raw = os.getenv("LATM_INFERENCE_CONFIG", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LATM_INFERENCE_CONFIG must be valid JSON.") from exc
    return parsed if isinstance(parsed, dict) else {}


def _run_agent(task_text: str, trace_hook) -> str:
    language_model = _build_language_model()
    inference_config = _load_inference_config()
    controller = SelfEvolvingController(
        language_model=language_model,
        tool_registry_path=os.getenv("TOOL_REGISTRY_PATH", "outputs/tool_registry"),
        max_generated_tools_per_run=int(os.getenv("LATM_MAX_GENERATED_TOOLS", "3")),
        inference_config_dict=inference_config,
        system_prompt=os.getenv("LATM_SYSTEM_PROMPT", ""),
        force_tool_generation_if_missing=os.getenv(
            "LATM_FORCE_TOOLGEN", "1"
        ).strip()
        == "1",
        tool_match_min_score=float(os.getenv("LATM_TOOL_MATCH_MIN_SCORE", "0.7")),
        include_registry_in_prompt=os.getenv(
            "LATM_INCLUDE_REGISTRY", "1"
        ).strip()
        == "1",
        environment_label=os.getenv("LATM_ENV_LABEL", "webui"),
        retrieval_top_k=int(os.getenv("LATM_RETRIEVAL_TOP_K", "5")),
        reuse_top_k=int(os.getenv("LATM_REUSE_TOP_K", "3")),
        reuse_similarity_threshold=(
            float(os.getenv("LATM_REUSE_SIM_THRESHOLD"))
            if os.getenv("LATM_REUSE_SIM_THRESHOLD")
            else None
        ),
        reuse_min_reliability=float(os.getenv("LATM_REUSE_MIN_RELIABILITY", "0.0")),
        canonical_tool_naming=os.getenv(
            "LATM_CANONICAL_TOOL_NAMING", "1"
        ).strip()
        == "1",
    )
    controller.set_trace_hook(trace_hook)

    chat_history = ChatHistory()
    chat_history.inject(ChatHistoryItem(role=Role.USER, content=task_text))
    result = controller._inference(chat_history)
    return result.content or ""


@app.get("/")
async def index() -> HTMLResponse:
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return HTMLResponse(
        "<h1>Missing templates/index.html</h1>",
        status_code=500,
    )


@app.websocket("/ws/task")
async def ws_task(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
    except Exception:
        await websocket.send_json(
            {"type": "error", "message": "Invalid JSON payload."}
        )
        await websocket.close()
        return

    task_text = str(payload.get("task_text") or "").strip()
    if not task_text:
        await websocket.send_json(
            {"type": "error", "message": "task_text is required."}
        )
        await websocket.close()
        return

    await websocket.send_json(
        {"type": "ack", "message": "Task accepted. Streaming execution trace."}
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    def trace_hook(event: Dict[str, Any]) -> None:
        message = _format_trace_event(event)
        if not message:
            return
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "trace",
                    "message": message,
                    "event": event.get("event"),
                    "trace_type": event.get("trace_type"),
                    "ts": time.time(),
                },
            )
        except RuntimeError:
            return

    async def sender() -> None:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                await websocket.send_json(item)
        except WebSocketDisconnect:
            return

    sender_task = asyncio.create_task(sender())
    await queue.put(
        {
            "type": "trace",
            "message": "[System] Run started",
            "trace_type": "system",
            "ts": time.time(),
        }
    )

    try:
        answer = await asyncio.to_thread(_run_agent, task_text, trace_hook)
        await queue.put(
            {
                "type": "complete",
                "answer": answer,
                "ts": time.time(),
            }
        )
    except Exception as exc:
        await queue.put(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "ts": time.time(),
            }
        )
    finally:
        await queue.put(None)
        await sender_task
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
