from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .types import Message


@dataclass
class EnvStepResult:
    observation: str | None
    done: bool
    final_answer: str | None = None
    termination_reason: str | None = None
    metadata: dict[str, Any] | None = None


class EnvironmentAdapter(Protocol):
    """Minimal interface for benchmark-specific environment glue."""

    def initialize(self, task_input: Any) -> Sequence[Message]:
        """Return initial conversation messages for the episode."""

    def step(self, agent_output: str) -> EnvStepResult:
        """Apply the agent output to the environment and return the next observation."""
