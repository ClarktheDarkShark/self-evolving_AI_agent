import json
import re
from typing import Any, Mapping, Optional, Sequence
from typing_extensions import override

from src.agents.agent import Agent
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.language_models import LanguageModel
from src.typings import (
    AgentContextLimitException,
    AgentOutOfMemoryException,
    AgentUnknownException,
    ChatHistory,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    LanguageModelOutOfMemoryException,
    LanguageModelUnknownException,
    Role,
)

from .tool_registry import ToolMetadata, get_registry


class SelfEvolvingController(Agent):
    """
    Controller that wraps a base LanguageModelAgent but adds lightweight support
    for self-generated tools. The controller expects the model to surface
    tool-related intents through `<action name="...">...</action>` wrappers.

    - `<action name="create_tool">{"name": "...", "description": "...", "signature": "...", "code": "..."}</action>`
      registers a new tool and persists it on disk.
    - `<action name="<tool_name>">{...args...}</action>` is forwarded to the task
      side; environments can decide whether to execute a generated tool or
      fallback to their native toolchains.

    The default `plan_and_generate_tool` implementation simply persists the tool
    provided by the model, but the method is intentionally kept small so users
    can extend it with more elaborate planning or safety checks.
    """

    _ACTION_PATTERN = re.compile(
        r"<action\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</action>", re.MULTILINE
    )

    def __init__(
        self,
        language_model: LanguageModel,
        tool_registry_path: str,
        max_generated_tools_per_run: int = 3,
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        bootstrap_tools: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: str = (
            "You are a helpful assistant that can emit <action> blocks to either "
            "call previously generated tools or request new ones. Always keep task-"
            "specific output formats (e.g., Action: Operation) intact."
        ),
    ):
        self._language_model_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=system_prompt,
            inference_config_dict=inference_config_dict,
        )
        self._registry = get_registry(tool_registry_path)
        self._max_generated_tools_per_run = max_generated_tools_per_run
        self._generated_tool_counter = 0
        self._bootstrap_tools(bootstrap_tools or [])

    def _handle_creation_block(
        self, payload: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            print(
                "[SelfEvolvingController] Reached generated tool limit; skipping creation request."
            )
            return None
        try:
            creation_request = json.loads(payload)
            if not isinstance(creation_request, Mapping):
                raise ValueError("Tool creation payload must be a JSON object.")
            tool_name = str(creation_request.get("name") or "generated_tool")
            description = str(creation_request.get("description") or "")
            signature = str(creation_request.get("signature") or "run(*args, **kwargs)")
            code = str(creation_request.get("code") or "")
        except Exception:
            # Invalid payloads are ignored; keep the agent output unchanged.
            print(
                "[SelfEvolvingController] Failed to parse tool creation payload; ignoring."
            )
            return None
        print(
            "[SelfEvolvingController] Parsed tool creation request for '"
            f"{tool_name}' with signature '{signature}'."
        )
        metadata = self.plan_and_generate_tool(
            tool_name=tool_name,
            description=description,
            signature=signature,
            code=code,
            chat_history=chat_history,
        )
        if metadata:
            print(
                f"[SelfEvolvingController] Successfully registered tool '{metadata.name}'."
            )
            self._generated_tool_counter += 1
        else:
            print(
                "[SelfEvolvingController] Tool generation skipped or failed for request."
            )
        return metadata

    def _process_actions(
        self, chat_history: ChatHistory, content: str
    ) -> tuple[str, list[ToolMetadata]]:
        created_tools: list[ToolMetadata] = []
        for match in self._ACTION_PATTERN.finditer(content):
            action_name = match.group("name")
            body = match.group("body").strip()
            if action_name.lower() in {"create_tool", "generate_tool"}:
                if metadata := self._handle_creation_block(body, chat_history):
                    created_tools.append(metadata)
        return content, created_tools

    def plan_and_generate_tool(
        self,
        *,
        tool_name: str,
        description: str,
        signature: str,
        code: str,
        chat_history: ChatHistory,
    ) -> Optional[ToolMetadata]:
        """
        Decide whether and how to persist a new tool.

        The default behavior trusts the provided payload and writes it to disk.
        Advanced users can override or monkey-patch this method to inject
        alignment checks, compile-time validation, or cross-session reuse logic.
        """
        print(
            "[SelfEvolvingController] Attempting to persist generated tool '"
            f"{tool_name}' to registry."
        )
        return self._registry.register_tool(
            name=tool_name,
            code=code,
            signature=signature,
            description=description,
        )

    def _bootstrap_tools(self, bootstrap_tools: Sequence[Mapping[str, Any]]) -> None:
        if not bootstrap_tools:
            return
        print(
            f"[SelfEvolvingController] Bootstrapping {len(bootstrap_tools)} tool(s) from config."
        )
        for index, tool in enumerate(bootstrap_tools):
            if not isinstance(tool, Mapping):
                print(
                    "[SelfEvolvingController] Skipping non-mapping bootstrap entry at "
                    f"index {index}."
                )
                continue
            name = str(tool.get("name") or f"bootstrap_tool_{index}")
            description = str(tool.get("description") or "")
            signature = str(tool.get("signature") or "run(*args, **kwargs)")
            code = str(tool.get("code") or "")
            metadata = self._registry.register_tool(
                name=name,
                code=code,
                signature=signature,
                description=description,
            )
            self._generated_tool_counter += 1
            print(
                f"[SelfEvolvingController] Bootstrapped tool '{metadata.name}' with signature "
                f"'{metadata.signature}'."
            )

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        try:
            model_response = self._language_model_agent._inference(chat_history)
        except LanguageModelContextLimitException as e:
            raise AgentContextLimitException(str(e)) from e
        except LanguageModelOutOfMemoryException as e:
            raise AgentOutOfMemoryException(str(e)) from e
        except LanguageModelUnknownException as e:
            raise AgentUnknownException(str(e)) from e
        processed_content, _ = self._process_actions(
            chat_history, model_response.content
        )
        return ChatHistoryItem(role=model_response.role, content=processed_content)

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        return self._language_model_agent.get_role_dict()
