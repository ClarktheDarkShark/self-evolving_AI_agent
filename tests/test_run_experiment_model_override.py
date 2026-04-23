from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.run_experiment as run_experiment


def _base_raw_config() -> dict[str, object]:
    return {
        "language_model_dict": {
            "gpt-5": {"module": "gpt5"},
            "gpt-5-mini": {"module": "mini"},
            "gpt-5-nano": {"module": "nano"},
        },
        "assignment_config": {
            "language_model_list": [{"name": "gpt-5-mini"}],
            "agent": {
                "name": "language_model_agent",
                "custom_parameters": {
                    "language_model": "gpt-5-mini",
                },
            },
        },
    }


def test_language_model_override_switches_assignment_and_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIFELONG_MODEL", "gpt-5-nano")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    updated = run_experiment._maybe_override_language_model(_base_raw_config())

    assert updated["assignment_config"]["language_model_list"] == [
        {"name": "gpt-5-nano"}
    ]
    assert (
        updated["assignment_config"]["agent"]["custom_parameters"]["language_model"]
        == "gpt-5-nano"
    )


def test_language_model_override_accepts_openai_model_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIFELONG_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5")

    updated = run_experiment._maybe_override_language_model(_base_raw_config())

    assert updated["assignment_config"]["language_model_list"] == [{"name": "gpt-5"}]
