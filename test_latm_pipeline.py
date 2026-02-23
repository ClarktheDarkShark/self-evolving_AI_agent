#!/usr/bin/env python3
"""
test_latm_pipeline.py — Integration test for the LATM escape hatch pipeline.

Verifies the end-to-end flow:
  Orchestrator detects failure → triggers request_new_tool escape hatch →
  ToolGen generates + validates a new tool → Orchestrator resumes with new tool.

Three scenarios:
  1. Node Explosion (SPARQL timeout on high-degree entities)
  2. Logic Gap (no math/aggregation tool available)
  3. Infinite Loop (state explosion from repeated intersections)

Prerequisites:
  - Project dependencies installed (pip install -r requirements.txt)
  - OPENAI_API_KEY set (required)
  - OPENAI_BASE_URL set (optional, for custom endpoints)

Usage:
  OPENAI_API_KEY=sk-... python test_latm_pipeline.py
  MODEL_NAME=gpt-4o python test_latm_pipeline.py
"""

from __future__ import annotations

# ── Environment variable overrides (MUST come before any project imports) ─────
import os

_SAVED_ENV: dict[str, str | None] = {}


def _set_env(key: str, value: str) -> None:
    _SAVED_ENV.setdefault(key, os.environ.get(key))
    os.environ[key] = value


# Safeguard: 2-second SPARQL timeout if any real query is attempted
_set_env("LIFELONG_SPARQL_TIMEOUT_S", "2")
# Enable ToolGen (default is OFF)
_set_env("TOOLGEN_OFF", "0")
# Increase ToolGen timeout for tests (seconds)
_set_env("LIFELONG_TOOLGEN_TIMEOUT_S", "180")
# Lower ToolGen validator threshold for tests
_set_env("LIFELONG_TOOLGEN_MIN_GRADE", "6")
# Respect environment override for pipeline
PIPELINE_MODE = os.getenv("TOOLGEN_PIPELINE", "none")
_set_env("TOOLGEN_PIPELINE", PIPELINE_MODE)

# ── Standard library ─────────────────────────────────────────────────────────
import json
import contextlib
import io
import pathlib
import shutil
import sys
import tempfile
import traceback

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# KG ENVIRONMENT CONTRACT (injected as user[0] in every scenario)
# ═══════════════════════════════════════════════════════════════════════════════

KG_ENV_CONTRACT = (
    "You are an agent that answers complex questions by querying a Knowledge Graph. "
    "You have the following tools:\n"
    "  get_relations(entity_or_variable) — returns outgoing relations\n"
    "  get_neighbors(variable, relation) — returns connected entities\n"
    "  intersection(var1, var2) — AND two variables of the same type\n"
    "  count(variable) — count entities in a variable\n"
    "  get_attributes(variable) — returns attributes for a variable\n"
    "  argmax(variable, attribute) — entity with maximum attribute value\n"
    "  argmin(variable, attribute) — entity with minimum attribute value\n\n"
    "Variables are created automatically and referenced as #0, #1, #2, etc.\n"
    "When you have found the answer, respond with: Final Answer: #<variable_number>\n\n"
    "Respond in the format:\n"
    "Thought: <reasoning>\n"
    "Action: <tool_name>(<arguments>)"
)

AGENT_ACK = (
    "Thought: I understand the task. I will use the Knowledge Graph tools to find "
    "the answer step by step."
)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════


def _inject(chat_history, role_str: str, content: str) -> None:
    """Helper: inject a message into ChatHistory with the right Role."""
    from src.typings import Role

    role = Role.USER if role_str == "user" else Role.AGENT
    chat_history.inject({"role": role, "content": content})


def build_scenario_1(chat_history) -> None:
    """Node Explosion — SPARQL timeout errors on high-degree entities."""
    _inject(chat_history, "user", KG_ENV_CONTRACT)
    _inject(chat_history, "agent", AGENT_ACK)
    _inject(
        chat_history,
        "user",
        "Question: Find all movies that share the generic genre 'Drama', "
        "the color 'Blue' in their poster, and were released in the 'United States'.\n"
        "Entities: [Drama, Blue, United States]",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: I need to find movies with genre Drama. Let me get the relations.\n"
        "Action: get_relations(Drama)",
    )
    _inject(
        chat_history,
        "user",
        "Observation: Error - 10-second timeout reached. "
        "Node explosion on 'Drama'.",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: Drama timed out. Let me try United States instead.\n"
        "Action: get_relations(United States)",
    )
    _inject(
        chat_history,
        "user",
        "Error: SPARQL query timed out after 2 seconds. "
        "The entity 'United States' has too many connected triples (node explosion). "
        "The baseline atomic tools cannot handle high-degree entities.",
    )


def build_scenario_2(chat_history) -> None:
    """Logic Gap — successful KG queries but no math/percentage tool."""
    countries = (
        "Albania, Andorra, Austria, Belarus, Belgium, Bosnia_and_Herzegovina, "
        "Bulgaria, Croatia, Cyprus, Czech_Republic, Denmark, Estonia, Finland, "
        "France, Germany, Greece, Hungary, Iceland, Ireland, Italy, Latvia, "
        "Liechtenstein, Lithuania, Luxembourg, Malta, Moldova, Monaco, "
        "Montenegro, Netherlands, North_Macedonia, Norway, Poland, Portugal, "
        "Romania, Russia, San_Marino, Serbia, Slovakia, Slovenia, Spain, "
        "Sweden, Switzerland, Ukraine, United_Kingdom, Vatican_City"
    )
    _inject(chat_history, "user", KG_ENV_CONTRACT)
    _inject(chat_history, "agent", AGENT_ACK)
    _inject(
        chat_history,
        "user",
        "Question: Calculate the percentage of European countries that have a "
        "population strictly greater than 50 million. Return just the percentage.\n"
        "Entities: [Europe]",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: First I need to find European countries.\n"
        "Action: get_relations(Europe)",
    )
    _inject(
        chat_history,
        "user",
        "Observation: Variable #0 created for entity Europe. "
        "Available relations: [geography.continent.countries, "
        "location.continent.area, location.continent.first_level_divisions]",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: Get all European countries via the countries relation.\n"
        "Action: get_neighbors(#0, geography.continent.countries)",
    )
    _inject(
        chat_history,
        "user",
        f"Observation: Variable #1 created. Type: location.country. "
        f"Found [{countries}]. 44 entities total. "
        f"Note: You now need to filter by population > 50 million and then "
        f"calculate a percentage, but none of the available tools can perform "
        f"mathematical division or percentage calculations.",
    )


def build_scenario_3(chat_history) -> None:
    """Infinite Loop — repeated intersection calls losing track of state."""
    _inject(chat_history, "user", KG_ENV_CONTRACT)
    _inject(chat_history, "agent", AGENT_ACK)
    _inject(
        chat_history,
        "user",
        "Question: Find the single actor who has co-starred with exactly all "
        "of these 5 specific actors: Tom_Hanks, Leonardo_DiCaprio, Brad_Pitt, "
        "Morgan_Freeman, and Matt_Damon.\n"
        "Entities: [Tom_Hanks, Leonardo_DiCaprio, Brad_Pitt, Morgan_Freeman, Matt_Damon]",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: Find co-stars of Tom Hanks first.\n"
        "Action: get_neighbors(Tom_Hanks, film.actor.film)",
    )
    _inject(
        chat_history,
        "user",
        "Observation: Variable #0 created. Type: film.actor. "
        "[m.a01, m.a02, m.a03, m.a04, m.a05, m.a06, m.a07, m.a08, m.a09, "
        "m.a10, m.a11, m.a12, m.a13, m.a14, m.a15, m.a16, m.a17, m.a18, "
        "m.a19, m.a20, m.a21, m.a22, m.a23, m.a24, m.a25]. 25 entities.",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: Now find co-stars of Leonardo DiCaprio to intersect.\n"
        "Action: get_neighbors(Leonardo_DiCaprio, film.actor.film)",
    )
    _inject(
        chat_history,
        "user",
        "Observation: Variable #1 created. Type: film.actor. "
        "[m.b01, m.b02, m.b03, m.b04, m.b05, m.b06, m.b07, m.b08, m.b09, "
        "m.b10, m.b11, m.b12, m.b13, m.b14, m.b15, m.b16, m.b17, m.b18, "
        "m.b19, m.b20, m.b21, m.b22, m.b23, m.b24, m.b25, m.b26, m.b27, "
        "m.b28, m.b29, m.b30]. 30 entities.",
    )
    _inject(
        chat_history,
        "agent",
        "Thought: Intersect #0 and #1 to find common co-stars.\n"
        "Action: intersection(#0, #1)",
    )
    _inject(
        chat_history,
        "user",
        "Observation: Variable #2 created. Type: film.actor. "
        "[m.a01, m.a02, m.a03, m.a04, m.a05, m.a06, m.a07, m.a08]. 8 entities. "
        "WARNING: You still need to intersect with 3 more actors (Brad_Pitt, "
        "Morgan_Freeman, Matt_Damon). At the current rate you are creating too "
        "many intermediate variables and risk losing track of state. "
        "Consider requesting a batch multi-intersection tool.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    ("NODE EXPLOSION (Timeout)", build_scenario_1),
    ("LOGIC GAP (Missing Math)", build_scenario_2),
    ("INFINITE LOOP (State Explosion)", build_scenario_3),
]


def _read_registry_metadata_names(registry_dir: str, env: str) -> set[str]:
    """Read metadata.json directly from disk to list tool names for env."""
    try:
        path = pathlib.Path(registry_dir) / "metadata.json"
        if not path.exists():
            return set()
        data = json.loads(path.read_text(encoding="utf-8"))
        names = set()
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            if env and entry.get("environment") != env:
                continue
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
        return names
    except Exception:
        return set()


def _get_registry_tool_names(registry, env: str, registry_dir: str | None = None) -> set[str]:
    """Return set of tool names in the registry for the given environment."""
    if registry_dir:
        names = _read_registry_metadata_names(registry_dir, env)
        if names:
            return names
    try:
        if hasattr(registry, "list_latest_tools"):
            tools = registry.list_latest_tools(environment=env)
        else:
            tools = registry.list_tools(environment=env)
        return {t.name for t in tools}
    except Exception:
        return set()


def run_scenario(
    controller,
    registry,
    scenario_index: int,
    scenario_name: str,
    build_fn,
) -> dict:
    """Run one scenario and return a results dict."""
    from src.typings import Role, TaskName, SampleStatus
    from src.typings.session import Session

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO {scenario_index + 1}: {scenario_name}")
    print(f"{'=' * 70}")

    # Pre-state
    registry_dir = getattr(controller, "_toolgen_registry_dir", None)
    tools_before = _get_registry_tool_names(
        registry, "knowledge_graph", registry_dir=registry_dir
    )
    counter_before = getattr(controller, "_generated_tool_counter", 0)

    # Build session + chat history
    session = Session(
        task_name=TaskName.KNOWLEDGE_GRAPH,
        sample_index=str(scenario_index),
    )
    session.sample_status = SampleStatus.RUNNING
    build_fn(session.chat_history)
    initial_len = session.chat_history.get_value_length()

    # Run inference with stdout capture
    captured = io.StringIO()
    error_text = None
    try:
        with contextlib.redirect_stdout(captured):
            controller.inference(session)
    except Exception:
        error_text = traceback.format_exc()

    stdout_text = captured.getvalue()

    trace_lines = [
        line for line in stdout_text.splitlines() if "[PROMPT_TRACE]" in line
    ]
    if trace_lines:
        print("  PROMPT TRACE:")
        for line in trace_lines:
            print(f"  {line}")

    last_turn = {}
    if hasattr(controller, "get_last_turn_data"):
        try:
            last_turn = controller.get_last_turn_data() or {}
        except Exception:
            last_turn = {}

    decisions = last_turn.get("orchestrator_decisions")
    if isinstance(decisions, list) and decisions:
        for idx, decision in enumerate(decisions):
            action = decision.get("action")
            tool_name = decision.get("tool_name", "N/A")
            stage = decision.get("_stage", "decision")
            print(f"  [TURN {idx}] Decision ({stage}): {action} | Tool: {tool_name}")
            if action == "request_new_tool" or tool_name == "request_new_tool":
                print(
                    f"  [!] ESCAPE HATCH ACTIVATED: {decision.get('reasoning')}"
                )
    elif isinstance(last_turn.get("orchestrator_decision"), dict):
        decision = last_turn.get("orchestrator_decision", {})
        action = decision.get("action")
        tool_name = decision.get("tool_name", "N/A")
        print(f"  [TURN 0] Decision: {action} | Tool: {tool_name}")
        if action == "request_new_tool" or tool_name == "request_new_tool":
            print(f"  [!] ESCAPE HATCH ACTIVATED: {decision.get('reasoning')}")

    # Post-state
    tools_after = _get_registry_tool_names(
        registry, "knowledge_graph", registry_dir=registry_dir
    )
    new_tools = tools_after - tools_before
    counter_after = getattr(controller, "_generated_tool_counter", 0)
    final_len = session.chat_history.get_value_length()

    # Assertions
    results = {}

    # 1. Escape hatch triggered?
    escape_triggered = "[!] ESCAPE HATCH TRIGGERED" in stdout_text
    if not escape_triggered:
        decisions = last_turn.get("orchestrator_decisions")
        if isinstance(decisions, list):
            for decision in decisions:
                if (
                    decision.get("action") == "request_new_tool"
                    or decision.get("tool_name") == "request_new_tool"
                ):
                    escape_triggered = True
                    break
        elif isinstance(last_turn.get("orchestrator_decision"), dict):
            decision = last_turn.get("orchestrator_decision", {})
            if (
                decision.get("action") == "request_new_tool"
                or decision.get("tool_name") == "request_new_tool"
            ):
                escape_triggered = True
    results["escape_hatch"] = escape_triggered
    _print_check("Escape hatch triggered", escape_triggered)

    # 2. Tool generated by ToolGen?
    tool_generated = counter_after > counter_before
    results["tool_generated"] = tool_generated
    _print_check(
        f"ToolGen generated a tool (counter {counter_before} → {counter_after})",
        tool_generated,
    )

    # 3. New tool in registry?
    tool_registered = len(new_tools) > 0
    results["tool_registered"] = tool_registered
    if new_tools:
        _print_check(f"Tool registered in metadata.json: {new_tools}", True)
    else:
        _print_check("Tool registered in metadata.json", False)

    # 4. No crash?
    no_crash = (
        error_text is None
        and session.sample_status != SampleStatus.AGENT_UNKNOWN_ERROR
    )
    results["no_crash"] = no_crash
    _print_check(f"No crash (status: {session.sample_status})", no_crash)

    # 5. Agent produced a response?
    produced_response = final_len > initial_len
    results["response"] = produced_response
    _print_check(
        f"Agent produced response (history {initial_len} → {final_len})",
        produced_response,
    )

    # Extra: show agent's final message if available
    if final_len > initial_len:
        try:
            last = session.chat_history.get_item_deep_copy(final_len - 1)
            preview = last.content[:200].replace("\n", " ")
            print(f"  [INFO] Last message ({last.role}): {preview}...")
        except Exception:
            pass

    if error_text:
        print(f"  [ERROR] Exception during inference:\n{error_text[:500]}")

    # Print captured orchestrator output (condensed)
    orch_lines = [
        line
        for line in stdout_text.splitlines()
        if any(
            kw in line
            for kw in [
                "[ORCHESTRATOR]",
                "ESCAPE HATCH",
                "[ToolGen]",
                "escape_hatch",
                "request_new_tool",
            ]
        )
    ]
    if orch_lines:
        print("  [LOG] Key orchestrator events:")
        for line in orch_lines[:10]:
            print(f"    {line.strip()}")

    results["new_tool_names"] = list(new_tools)
    return results


def _print_check(label: str, passed: bool) -> None:
    tag = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
    print(f"  {tag} {label}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # ── Dependency check ──────────────────────────────────────────────────
    try:
        import pydantic  # noqa: F401
        import openai  # noqa: F401
    except ImportError as e:
        print(
            f"ERROR: Missing dependency: {e}\n"
            "Install project deps first: pip install -r requirements.txt"
        )
        sys.exit(1)

    # ── Preflight checks ──────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set. Cannot run integration test.")
        sys.exit(1)

    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")

    # ── Create isolated temp directory ────────────────────────────────────
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="latm_test_"))
    output_dir = tmp_dir / "output"
    output_dir.mkdir(parents=True)
    _set_env("LIFELONG_OUTPUT_DIR", str(output_dir))
    _set_env("TOOL_REGISTRY_ROOT", str(output_dir / "tool_library"))

    print("=" * 70)
    print("  LATM ESCAPE HATCH INTEGRATION TEST")
    print("=" * 70)
    print(f"  Model:      {model_name}")
    print(f"  Base URL:   {base_url or '(default OpenAI)'}")
    print(f"  Output dir: {output_dir}")
    print(f"  TOOLGEN_OFF={os.environ.get('TOOLGEN_OFF')}")
    print(f"  TOOLGEN_PIPELINE={PIPELINE_MODE}")

    try:
        # ── Construct LLM and controller ──────────────────────────────────
        from src.language_models.instance.openai_language_model import (
            OpenaiLanguageModel,
        )
        from src.self_evolving_agent.controller import SelfEvolvingController
        from src.self_evolving_agent.tool_registry import get_registry

        lm = OpenaiLanguageModel(
            model_name=model_name,
            role_dict={"user": "user", "agent": "assistant"},
            api_key=api_key,
            base_url=base_url,
        )

        # The controller derives registry_path from LIFELONG_OUTPUT_DIR
        controller = SelfEvolvingController(
            language_model=lm,
            tool_registry_path=str(output_dir / "tool_library"),
            max_generated_tools_per_run=5,
            inference_config_dict=None,
            use_orchestrator=True,
            environment_label="knowledge_graph",
            use_packaged_agent=False,
        )

        # Get the actual registry the controller is using
        registry = controller._registry
        registry_dir = getattr(controller, "_toolgen_registry_dir", "unknown")
        print(f"  Registry:   {registry_dir}")
        print()

        # ── Run scenarios ─────────────────────────────────────────────────
        all_results: list[dict] = []
        for idx, (name, build_fn) in enumerate(SCENARIOS):
            result = run_scenario(controller, registry, idx, name, build_fn)
            all_results.append(result)

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("  SUMMARY")
        print(f"{'=' * 70}")

        total_pass = 0
        total_checks = 0
        for idx, (name, _) in enumerate(SCENARIOS):
            r = all_results[idx]
            checks = [
                r["escape_hatch"],
                r["tool_generated"],
                r["tool_registered"],
                r["no_crash"],
                r["response"],
            ]
            passed = sum(checks)
            total_pass += passed
            total_checks += len(checks)
            print(f"  Scenario {idx + 1} ({name}): {passed}/{len(checks)} passed")
            if r["new_tool_names"]:
                print(f"    New tools: {r['new_tool_names']}")

        # Final registry state
        registry_dir = getattr(controller, "_toolgen_registry_dir", None)
        all_tools = _get_registry_tool_names(
            registry, "knowledge_graph", registry_dir=registry_dir
        )
        print(f"\n  Total tools in registry: {len(all_tools)}")
        if all_tools:
            print(f"    {all_tools}")
        print(f"\n  Overall: {total_pass}/{total_checks} checks passed")
        print()

    finally:
        # ── Teardown ──────────────────────────────────────────────────────
        # Restore original env vars
        for key, original in _SAVED_ENV.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

        # Force-reset the global registry singleton
        try:
            from src.self_evolving_agent.tool_registry import get_registry

            get_registry(str(tmp_dir / "_cleanup"), force_reset=True)
        except Exception:
            pass

        # Clean up temp directory
        try:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            print(f"[cleanup] Removed temp dir: {tmp_dir}")
        except Exception:
            print(f"[cleanup] WARNING: Could not remove temp dir: {tmp_dir}")


if __name__ == "__main__":
    main()
