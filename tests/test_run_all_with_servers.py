from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import scripts.run_all_with_servers as run_all_with_servers
from scripts.run_all_with_servers import (
    _build_inline_trigger_baseline_summary,
    _build_trusted_success_bank_context,
    _record_trusted_family_success,
    _inline_family_evolution_allowed_for,
    _latest_selected_family_for_sample,
    _run_one_with_sample_boundary_family_evolution,
)
from src.pal.family_policy_evolution import build_family_policy_store
from src.pal.reusable_tool_families import get_baseline_reusable_family_policy_bundles


def test_enable_pal_agent_flag_is_defined() -> None:
    assert hasattr(run_all_with_servers, "ENABLE_PAL_AGENT")
    assert isinstance(run_all_with_servers.ENABLE_PAL_AGENT, bool)


def test_latest_selected_family_for_sample_falls_back_to_attempt_decision(
    tmp_path: pathlib.Path,
) -> None:
    log_path = tmp_path / "generated_tools.log"
    rows = [
        {
            "event": "pal_attempt_decision",
            "sample_index": "15",
            "selected_family": "count_over_direct_relation",
        },
        {
            "event": "pal_attempt_decision",
            "sample_index": "11",
            "selected_family": "single_anchor_lookup",
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert _latest_selected_family_for_sample(tmp_path, "15") == "count_over_direct_relation"


def test_latest_selected_family_for_sample_prefers_finalized_decision(
    tmp_path: pathlib.Path,
) -> None:
    log_path = tmp_path / "generated_tools.log"
    rows = [
        {
            "event": "pal_attempt_decision",
            "sample_index": "15",
            "selected_family": "count_over_direct_relation",
        },
        {
            "event": "pal_attempt_decision_finalized",
            "sample_index": "15",
            "selected_family": "count_over_joined_set",
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert _latest_selected_family_for_sample(tmp_path, "15") == "count_over_joined_set"


def test_inline_family_evolution_defaults_to_all_families(monkeypatch) -> None:
    monkeypatch.delenv("PAL_INLINE_FAMILY_EVOLUTION_FAMILIES", raising=False)

    assert _inline_family_evolution_allowed_for("count_over_direct_relation") is True
    assert _inline_family_evolution_allowed_for("multi_anchor_intersection") is True


def test_record_trusted_family_success_keeps_one_regression_guard(tmp_path: pathlib.Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generated_tools.log"
    rows = [
        {
            "event": "pal_attempt_decision_finalized",
            "sample_index": "11",
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
            "dangerous_overreach": False,
            "trust_contract": {"materialization_allowed": True, "dangerous_overreach": False},
        },
        {
            "event": "pal_attempt_decision_finalized",
            "sample_index": "241",
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
            "dangerous_overreach": False,
            "trust_contract": {"materialization_allowed": True, "dangerous_overreach": False},
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    store_path = tmp_path / "store"
    bank = _record_trusted_family_success(
        output_dir=output_dir,
        store_path=store_path,
        family_name="count_over_direct_relation",
        sample_index="11",
        active_version="2026-03-31",
        progress_log_path=log_path,
    )
    assert bank == ["11"]

    bank = _record_trusted_family_success(
        output_dir=output_dir,
        store_path=store_path,
        family_name="count_over_direct_relation",
        sample_index="241",
        active_version="2026-03-31",
        progress_log_path=log_path,
    )
    assert bank == ["241"]

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    assert store.get_trusted_success_bank("count_over_direct_relation") == ["241"]
    assert store.get_trusted_success_bank_metadata("count_over_direct_relation")[
        "evaluation_context"
    ] == _build_trusted_success_bank_context(
        family_name="count_over_direct_relation",
        active_version="2026-03-31",
    )


def test_inline_trigger_baseline_summary_includes_run_dir(tmp_path: pathlib.Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_inline_trigger_baseline_summary(
        output_dir=output_dir,
        sample_index="3",
        session_record={
            "sample_status": "completed",
            "evaluation_record": {"outcome": "incorrect"},
        },
    )

    assert summary["sample_index"] == "3"
    assert summary["sample_status"] == "completed"
    assert summary["evaluation_outcome"] == "incorrect"
    assert summary["run_dir"] == str(output_dir)


def test_boundary_runner_revisits_pending_candidate_after_family_warms(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    repo_root = PROJECT_ROOT
    config_path = "dummy_inline_boundary.yaml"
    combined_dir = tmp_path / "outputs"
    store_path = combined_dir / "knowledge_graph" / "dummy_inline_boundary" / "family_policy_store"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate = store.create_candidate_update(
        family_name="count_over_direct_relation",
        scaffold_signature="count_over_direct_relation|species|fictional_universe.fictional_setting.universe|fictional_universe.fictional_universe.species",
        relation_names=[
            "fictional_universe.fictional_setting.universe",
            "fictional_universe.fictional_universe.species",
        ],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "2"},
    )
    assert candidate is not None

    monkeypatch.setattr(run_all_with_servers, "_load_sample_order_from_config", lambda _: ["11", "241"])
    monkeypatch.setattr(run_all_with_servers, "_extract_task_name", lambda _: "knowledge_graph")

    def fake_write_single_sample_config(*, source_config_path, sample_index, stem):
        path = repo_root / f"{stem}.yaml"
        path.write_text("assignment_config:\n  sample_order:\n    - \"0\"\n", encoding="utf-8")
        return path

    monkeypatch.setattr(run_all_with_servers, "_write_single_sample_config", fake_write_single_sample_config)
    monkeypatch.setattr(run_all_with_servers, "_run_one", lambda *args, **kwargs: 0)

    def fake_load_session_for_sample(output_dir, sample_index):
        sample_index = str(sample_index)
        if sample_index == "2":
            return {
                "sample_index": "2",
                "sample_status": "agent_unknown_error",
                "evaluation_record": {"outcome": "incorrect"},
            }
        return {
            "sample_index": sample_index,
            "sample_status": "completed",
            "evaluation_record": {"outcome": "correct"},
        }

    monkeypatch.setattr(run_all_with_servers, "_load_session_for_sample", fake_load_session_for_sample)
    monkeypatch.setattr(
        run_all_with_servers,
        "_latest_selected_family_for_sample",
        lambda output_dir, sample_index: "count_over_direct_relation",
    )

    def fake_latest_attempt_decision(output_dir, sample_index):
        sample_index = str(sample_index)
        if sample_index in {"11", "241"}:
            return {
                "selected_family": "count_over_direct_relation",
                "family_bundle_version": "2026-03-31",
                "dangerous_overreach": False,
                "materialization_allowed": True,
                "trust_contract": {
                    "materialization_allowed": True,
                    "dangerous_overreach": False,
                },
            }
        return {}

    monkeypatch.setattr(run_all_with_servers, "_latest_attempt_decision_for_sample", fake_latest_attempt_decision)

    calls: list[dict[str, str]] = []

    def fake_run_between_sample_family_evolution(**kwargs):
        calls.append(
            {
                "family_name": str(kwargs["family_name"]),
                "sample_index": str(kwargs["sample_index"]),
                "trigger_source": str(kwargs["trigger_source"]),
                "boundary_sample_index": str(kwargs["boundary_sample_index"]),
                "parent_output_dir": str(kwargs["parent_output_dir"]),
            }
        )
        evolving_store = build_family_policy_store(
            baseline_bundles=get_baseline_reusable_family_policy_bundles(),
            store_path=store_path,
        )
        pending = evolving_store.get_pending_candidates("count_over_direct_relation")
        assert pending
        evolving_store.reject_candidate(
            "count_over_direct_relation",
            candidate_version=str(pending[0]["candidate_version"]),
            evaluation_results={"event": "inline_test_rejected"},
            rejection_reason="inline_gate_failed",
        )
        return 0

    monkeypatch.setattr(
        run_all_with_servers,
        "_run_between_sample_family_evolution",
        fake_run_between_sample_family_evolution,
    )

    rc = _run_one_with_sample_boundary_family_evolution(config_path, combined_dir)
    assert rc == 0

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    assert store.get_trusted_success_bank("count_over_direct_relation") == ["241"]
    assert calls == [
        {
            "family_name": "count_over_direct_relation",
            "sample_index": "2",
            "trigger_source": "pending_candidate_after_trusted_success",
            "boundary_sample_index": "11",
            "parent_output_dir": str(
                combined_dir
                / "knowledge_graph"
                / "dummy_inline_boundary"
                / "family_policy_inline_runs"
            ),
        }
    ]


def test_boundary_runner_prioritizes_current_bad_sample_over_older_pending_candidate(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    repo_root = PROJECT_ROOT
    config_path = "dummy_inline_boundary_current_failure.yaml"
    combined_dir = tmp_path / "outputs"
    store_path = (
        combined_dir
        / "knowledge_graph"
        / "dummy_inline_boundary_current_failure"
        / "family_policy_store"
    )
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate = store.create_candidate_update(
        family_name="count_over_direct_relation",
        scaffold_signature="count_over_direct_relation|species|fictional_universe.fictional_setting.universe|fictional_universe.fictional_universe.species",
        relation_names=[
            "fictional_universe.fictional_setting.universe",
            "fictional_universe.fictional_universe.species",
        ],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "2"},
    )
    assert candidate is not None

    monkeypatch.setattr(
        run_all_with_servers,
        "_load_sample_order_from_config",
        lambda _: ["14"],
    )
    monkeypatch.setattr(
        run_all_with_servers,
        "_extract_task_name",
        lambda _: "knowledge_graph",
    )

    def fake_write_single_sample_config(*, source_config_path, sample_index, stem):
        path = repo_root / f"{stem}.yaml"
        path.write_text(
            "assignment_config:\n  sample_order:\n    - \"0\"\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(
        run_all_with_servers,
        "_write_single_sample_config",
        fake_write_single_sample_config,
    )
    monkeypatch.setattr(run_all_with_servers, "_run_one", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        run_all_with_servers,
        "_load_session_for_sample",
        lambda output_dir, sample_index: {
            "sample_index": str(sample_index),
            "sample_status": "completed",
            "evaluation_record": {"outcome": "incorrect"},
        },
    )
    monkeypatch.setattr(
        run_all_with_servers,
        "_latest_selected_family_for_sample",
        lambda output_dir, sample_index: "count_over_direct_relation",
    )
    monkeypatch.setattr(
        run_all_with_servers,
        "_latest_attempt_decision_for_sample",
        lambda output_dir, sample_index: {
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
            "dangerous_overreach": False,
            "materialization_allowed": True,
            "trust_contract": {
                "materialization_allowed": True,
                "dangerous_overreach": False,
            },
        },
    )

    calls: list[dict[str, str]] = []

    def fake_run_between_sample_family_evolution(**kwargs):
        calls.append(
            {
                "family_name": str(kwargs["family_name"]),
                "sample_index": str(kwargs["sample_index"]),
                "trigger_source": str(kwargs["trigger_source"]),
                "boundary_sample_index": str(kwargs["boundary_sample_index"]),
            }
        )
        return 0

    monkeypatch.setattr(
        run_all_with_servers,
        "_run_between_sample_family_evolution",
        fake_run_between_sample_family_evolution,
    )

    rc = _run_one_with_sample_boundary_family_evolution(config_path, combined_dir)
    assert rc == 0
    assert calls == [
        {
            "family_name": "count_over_direct_relation",
            "sample_index": "14",
            "trigger_source": "sample_failure_or_wrong_completion",
            "boundary_sample_index": "14",
        }
    ]
