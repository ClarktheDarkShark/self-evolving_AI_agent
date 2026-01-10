# Lifelong Agent Core

Reusable self-evolving agent core with tool generation, registry, and a benchmark-agnostic runner.
This package is extracted from LifelongAgentBench to provide a stable entrypoint for running
single episodes with a custom environment adapter.

## What This Package Provides

- A self-evolving agent loop that can generate, validate, store, retrieve, and reuse tools.
- A persistent tool registry with dedupe and canonical naming.
- A benchmark-agnostic entrypoint (`run_episode`) for running one sample/episode.
- Minimal interfaces for plugging in environment-specific logic (adapter) and model backends.

This package intentionally excludes dataset loading, evaluation harnesses, and benchmark-specific
environment logic.

## Installation (local)

```bash
pip install -e packages/lifelong_agent_core
```

## Quick Start

Implement a small adapter and model wrapper, then call `run_episode`.

```python
from lifelong_agent_core import AgentConfig, EnvStepResult, run_episode
from lifelong_agent_core.model import ModelResponse
from lifelong_agent_core.types import Message


class DummyEnvAdapter:
    def initialize(self, task_input):
        return [Message(role="user", content=str(task_input))]

    def step(self, agent_output: str) -> EnvStepResult:
        return EnvStepResult(observation=None, done=True, final_answer=agent_output)


class DummyModelAdapter:
    def complete(self, messages, *, system_prompt: str, inference_config=None):
        return ModelResponse(content="Action: Answer\nFinal Answer: ok")


config = AgentConfig(
    model_adapter=DummyModelAdapter(),
    tool_registry_path="outputs/tool_library",
    system_prompt="You are a helpful assistant.",
)

result = run_episode("Ping", DummyEnvAdapter(), config)
print(result.final_answer)
```

## Entry Method

`run_episode(task_input, env_adapter, config)` returns an `AgentResult`:

- `final_answer`: the final answer emitted or environment-provided
- `termination_reason`: why the run ended (done, max_steps, agent_error, etc.)
- `tool_calls`: list of tool invocation records
- `generated_tools`: list of tools created during the run
- `steps`: number of agent steps taken

## Core Interfaces

### EnvironmentAdapter

You provide the benchmark glue here.

```python
class EnvironmentAdapter(Protocol):
    def initialize(self, task_input) -> Sequence[Message]:
        ...
    def step(self, agent_output: str) -> EnvStepResult:
        ...
```

### ModelAdapter

Wrap your LLM backend in a minimal API:

```python
class ModelAdapter(Protocol):
    def complete(self, messages, *, system_prompt: str, inference_config=None) -> ModelResponse:
        ...
```

## Tool Lifecycle (inside the package)

1. Propose tool (LLM toolgen prompt)
2. Validate tool (compile + smoke test + optional self_test)
3. Register tool (persistent registry, dedupe, canonical naming)
4. Retrieve tool (lexical retrieval)
5. Invoke tool (internal; not an environment action)
6. Reuse tool (reuse gate avoids duplicates)

## Configuration Notes

`AgentConfig` includes:
- `tool_registry_path`: persistent storage for tools
- `max_generated_tools_per_run`: cap on new tools
- `force_tool_generation_if_missing`: allow generation when no tool matches
- `tool_match_min_score`: reuse threshold for retrieval
- `include_registry_in_prompt`: include tool registry in solver prompt
- `environment_label`: label for logging/metrics
- `reuse_top_k`, `reuse_similarity_threshold`, `reuse_min_reliability`
- `canonical_tool_naming`: stable name from fingerprint
- `max_steps`: max agent steps per episode

## Logging and Metrics

The package collects tool events via the registry event listener and returns them in
`AgentResult` (tool calls + generated tools). Use `AgentConfig.event_listener` to
tap into raw registry events.

## Keeping Legacy Behavior

In LifelongAgentBench, packaged behavior is gated by a boolean. The default is
legacy behavior unless `use_packaged_agent` is set to true in configuration.
