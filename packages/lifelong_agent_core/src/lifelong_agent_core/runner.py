from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .adapters import EnvironmentAdapter, EnvStepResult
from .config import AgentConfig
from .controller import SelfEvolvingController
from .exceptions import AgentError
from .model import ModelAdapter
from .results import AgentResult, GeneratedToolRecord, ToolCallRecord
from .tool_registry import get_registry
from .types import ChatHistory, ChatHistoryItem, Message, Role


def _to_chat_history(messages: Iterable[Message]) -> ChatHistory:
    history = ChatHistory()
    for msg in messages:
        normalized = msg.role.strip().lower()
        if normalized == "user":
            role = Role.USER
        elif normalized in ("agent", "assistant"):
            role = Role.AGENT
        else:
            raise ValueError(f"Unsupported message role: {msg.role}")
        history.inject(ChatHistoryItem(role=role, content=msg.content))
    return history


def run_episode(
    task_input: Any,
    env_adapter: EnvironmentAdapter,
    config: AgentConfig,
) -> AgentResult:
    if config.model_adapter is None:
        raise ValueError("AgentConfig.model_adapter must be provided.")

    model_adapter: ModelAdapter = config.model_adapter
    initial_messages = env_adapter.initialize(task_input)
    chat_history = _to_chat_history(initial_messages)

    if chat_history.get_value_length() == 0:
        raise ValueError("EnvironmentAdapter.initialize returned no messages.")
    if chat_history.get_item_deep_copy(-1).role != Role.USER:
        raise ValueError("Chat history must end with a user message.")

    tool_calls: list[ToolCallRecord] = []
    generated_tools: list[GeneratedToolRecord] = []

    def _listener(payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "invoke":
            tool_calls.append(
                ToolCallRecord(
                    tool_name=str(payload.get("tool_name")),
                    success=bool(payload.get("success")),
                    args=list(payload.get("args") or []),
                    kwargs=dict(payload.get("kwargs") or {}),
                    error=payload.get("error"),
                    metadata={k: v for k, v in payload.items() if k not in {"event", "args", "kwargs"}},
                )
            )
        elif event == "register":
            generated_tools.append(
                GeneratedToolRecord(
                    tool_name=str(payload.get("tool_name")),
                    fingerprint=payload.get("fingerprint"),
                    metadata={k: v for k, v in payload.items() if k != "event"},
                )
            )
        if config.event_listener:
            try:
                config.event_listener(payload)
            except Exception:
                pass

    registry = get_registry(config.tool_registry_path)
    registry.add_event_listener(_listener)

    controller = SelfEvolvingController(
        language_model=model_adapter,
        tool_registry_path=config.tool_registry_path,
        max_generated_tools_per_run=config.max_generated_tools_per_run,
        inference_config_dict=config.inference_config,
        system_prompt=config.system_prompt,
        force_tool_generation_if_missing=config.force_tool_generation_if_missing,
        tool_match_min_score=config.tool_match_min_score,
        include_registry_in_prompt=config.include_registry_in_prompt,
        environment_label=config.environment_label,
        retrieval_top_k=config.retrieval_top_k,
        reuse_top_k=config.reuse_top_k,
        reuse_similarity_threshold=config.reuse_similarity_threshold,
        reuse_min_reliability=config.reuse_min_reliability,
        canonical_tool_naming=config.canonical_tool_naming,
    )

    steps = 0
    termination_reason = "max_steps"
    final_answer = ""
    raw_output: str | None = None

    while steps < config.max_steps:
        steps += 1
        try:
            agent_item = controller._inference(chat_history)
        except AgentError as exc:
            termination_reason = f"agent_error:{exc}"
            raw_output = ""
            break

        raw_output = agent_item.content
        chat_history.inject(agent_item)

        env_result: EnvStepResult = env_adapter.step(raw_output)
        if env_result.observation is not None:
            chat_history.inject(
                ChatHistoryItem(role=Role.USER, content=env_result.observation)
            )
        elif not env_result.done:
            # Ensure roles alternate even if environment emits no observation.
            chat_history.inject(ChatHistoryItem(role=Role.USER, content=""))

        if env_result.done:
            final_answer = env_result.final_answer or raw_output
            termination_reason = env_result.termination_reason or "done"
            break

    return AgentResult(
        final_answer=final_answer,
        termination_reason=termination_reason,
        tool_calls=tool_calls,
        generated_tools=generated_tools,
        steps=steps,
        raw_agent_output=raw_output,
        metadata={"config": asdict(config)},
    )
