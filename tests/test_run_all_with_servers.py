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
    _latest_attempt_decision_for_sample,
    _record_tool_evolution_signal,
    _record_trusted_family_success,
    _inline_family_evolution_allowed_for,
    _latest_selected_family_for_sample,
    _resolve_family_policy_store_path,
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


def test_latest_attempt_decision_for_sample_falls_back_to_attempt_decision(
    tmp_path: pathlib.Path,
) -> None:
    log_path = tmp_path / "generated_tools.log"
    rows = [
        {
            "event": "pal_attempt_decision",
            "sample_index": "15",
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
        }
    ]
    log_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    latest = _latest_attempt_decision_for_sample(tmp_path, "15")
    assert latest["selected_family"] == "count_over_direct_relation"
    assert latest["family_bundle_version"] == "2026-03-31"


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
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        },
        {
            "event": "pal_macro_solver_review",
            "sample_index": "11",
            "accepted_final": True,
        },
        {
            "event": "pal_attempt_decision_finalized",
            "sample_index": "241",
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
            "dangerous_overreach": False,
            "trust_contract": {"materialization_allowed": True, "dangerous_overreach": False},
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        },
        {
            "event": "pal_macro_solver_review",
            "sample_index": "241",
            "accepted_final": True,
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


def test_record_trusted_family_success_ignores_partial_or_untrusted_tool_output(
    tmp_path: pathlib.Path,
) -> None:
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
            "tool_result_status": "partial",
            "tool_result_solves_task": False,
            "tool_result_trusted_for_materialization": False,
        }
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    bank = _record_trusted_family_success(
        output_dir=output_dir,
        store_path=tmp_path / "store",
        family_name="count_over_direct_relation",
        sample_index="11",
        active_version="2026-03-31",
        progress_log_path=log_path,
    )

    assert bank == []


def test_record_trusted_family_success_requires_solver_acceptance_when_review_exists(
    tmp_path: pathlib.Path,
) -> None:
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
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        },
        {
            "event": "pal_macro_solver_review",
            "sample_index": "11",
            "accepted_final": False,
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    bank = _record_trusted_family_success(
        output_dir=output_dir,
        store_path=tmp_path / "store",
        family_name="count_over_direct_relation",
        sample_index="11",
        active_version="2026-03-31",
        progress_log_path=log_path,
    )

    assert bank == []


def test_record_trusted_family_success_allows_direct_pal_success_without_solver_review(
    tmp_path: pathlib.Path,
) -> None:
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
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        }
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    bank = _record_trusted_family_success(
        output_dir=output_dir,
        store_path=tmp_path / "store",
        family_name="count_over_direct_relation",
        sample_index="11",
        active_version="2026-03-31",
        progress_log_path=log_path,
    )

    assert bank == ["11"]


def test_record_trusted_family_success_persists_reusable_run_summary(
    tmp_path: pathlib.Path,
) -> None:
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
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        }
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    bank = _record_trusted_family_success(
        output_dir=output_dir,
        store_path=tmp_path / "store",
        family_name="count_over_direct_relation",
        sample_index="11",
        active_version="2026-03-31",
        session_record={
            "sample_status": "completed",
            "evaluation_record": {"outcome": "correct"},
        },
        progress_log_path=log_path,
    )

    assert bank == ["11"]
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    metadata = store.get_trusted_success_bank_metadata("count_over_direct_relation")
    assert metadata["evaluation_results"]["run_summary"] == {
        "sample_index": "11",
        "sample_status": "completed",
        "evaluation_outcome": "correct",
        "dangerous_overreach_count": 0,
        "run_dir": str(output_dir),
    }


def test_record_tool_evolution_signal_tracks_trusted_success_and_clean_failure(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAL_ENABLE_FAMILY_POLICY_EVOLUTION", "1")
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "pal_query_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plan_path = artifacts_dir / "sample.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "query_shape": "count_over_direct_relation",
                "answer_mode": "count",
                "anchored_entities": [
                    {
                        "surface": "Percussionist",
                        "chosen_alias": "Percussionist",
                        "resolved_entity_id": "m.02h66l4",
                        "role": "anchor",
                    }
                ],
                "normalized_aliases": [
                    {
                        "surface": "Percussionist",
                        "chosen_alias": "Percussionist",
                        "reason": "explicit_entity:attribute_value.profession",
                    }
                ],
                "join_structure": {"type": "count", "anchor_constraints": []},
                "relation_paths": [
                    {
                        "relation": "people.profession.people_with_this_profession",
                        "from_role": "anchor",
                        "to_role": "count_set",
                        "grounding_source": "dynamic_probe",
                    }
                ],
                "candidate_set_variable": "person",
                "count_set_variable": "person",
                "shared_answer_variable": "person",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "generated_tools.log"
    rows = [
        {
            "event": "pal_query_plan_generated",
            "sample_index": "11",
            "plan_artifact_path": str(plan_path),
        },
        {
            "event": "pal_attempt_decision_finalized",
            "sample_index": "11",
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
            "dangerous_overreach": False,
            "trust_contract": {"materialization_allowed": True, "dangerous_overreach": False},
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        },
        {
            "event": "pal_macro_solver_review",
            "sample_index": "11",
            "accepted_final": True,
        },
        {
            "event": "pal_query_plan_generated",
            "sample_index": "15",
            "plan_artifact_path": str(plan_path),
        },
        {
            "event": "pal_attempt_decision_finalized",
            "sample_index": "15",
            "selected_family": "count_over_direct_relation",
            "family_bundle_version": "2026-03-31",
            "dangerous_overreach": False,
            "trust_contract": {"materialization_allowed": False, "dangerous_overreach": False},
            "tool_result_status": "failed",
            "tool_result_solves_task": False,
            "tool_result_trusted_for_materialization": False,
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    store_path = tmp_path / "store"
    _record_tool_evolution_signal(
        output_dir=output_dir,
        store_path=store_path,
        family_name="count_over_direct_relation",
        sample_index="11",
        active_version="2026-03-31",
        session_record={
            "sample_status": "completed",
            "evaluation_record": {"outcome": "correct"},
        },
        progress_log_path=log_path,
    )
    _record_tool_evolution_signal(
        output_dir=output_dir,
        store_path=store_path,
        family_name="count_over_direct_relation",
        sample_index="15",
        active_version="2026-03-31",
        session_record={
            "sample_status": "agent_unknown_error",
            "finish_reason": "[AgentUnknownException] pal_tool_failure_bypassed:pal_query_not_accepted:repairable_bad_count_set",
            "evaluation_record": {"outcome": "incorrect"},
        },
        progress_log_path=log_path,
    )

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    context = store.get_tool_evolution_context("count_over_direct_relation")
    assert context["source_version"] == "2026-03-31"
    assert len(context["preferred_patterns"]) == 1
    assert len(context["avoid_patterns"]) == 1
    assert "repairable_bad_count_set" in context["avoid_patterns"][0]["failure_labels"]


def test_record_tool_evolution_signal_works_in_standard_family_evolution_mode(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PAL_ENABLE_FAMILY_POLICY_EVOLUTION", raising=False)
    monkeypatch.setenv("PAL_ENABLE_STANDARD_FAMILY_EVOLUTION", "1")
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "pal_query_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plan_path = artifacts_dir / "sample.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "query_shape": "count_over_direct_relation",
                "answer_mode": "count",
                "anchored_entities": [{"surface": "bourbon whisky", "chosen_alias": "bourbon whisky", "role": "anchor"}],
                "normalized_aliases": [],
                "join_structure": {"type": "count", "anchor_constraints": []},
                "relation_paths": [
                    {
                        "relation": "distilled_spirits.distilled_spirit.spirit_type",
                        "from_role": "count_set",
                        "to_role": "anchor",
                        "grounding_source": "dynamic_probe",
                    }
                ],
                "candidate_set_variable": "candidate",
                "count_set_variable": "candidate",
                "shared_answer_variable": "candidate",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "generated_tools.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "pal_query_plan_generated",
                        "sample_index": "241",
                        "plan_artifact_path": str(plan_path),
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_attempt_decision_finalized",
                        "sample_index": "241",
                        "selected_family": "count_over_direct_relation",
                        "family_bundle_version": "2026-03-31",
                        "dangerous_overreach": False,
                        "trust_contract": {"materialization_allowed": True, "dangerous_overreach": False},
                        "tool_result_status": "success",
                        "tool_result_solves_task": True,
                        "tool_result_trusted_for_materialization": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store_path = tmp_path / "store"
    _record_tool_evolution_signal(
        output_dir=output_dir,
        store_path=store_path,
        family_name="count_over_direct_relation",
        sample_index="241",
        active_version="2026-03-31",
        session_record={
            "sample_status": "completed",
            "evaluation_record": {"outcome": "correct"},
        },
        progress_log_path=log_path,
    )

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    context = store.get_tool_evolution_context("count_over_direct_relation")
    assert context["source_version"] == "2026-03-31"
    assert len(context["preferred_patterns"]) == 1


def test_resolve_family_policy_store_path_defaults_to_persistent_store_for_evolution(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAL_ENABLE_STANDARD_FAMILY_EVOLUTION", "1")
    monkeypatch.delenv("PAL_FAMILY_POLICY_STORE_PATH", raising=False)
    monkeypatch.delenv("PAL_PERSISTENT_FAMILY_POLICY_STORE", raising=False)

    store_path = _resolve_family_policy_store_path(
        repo_root=tmp_path,
        aggregate_output_dir=tmp_path / "outputs" / "run_a",
        task_name="knowledge_graph",
    )

    assert store_path == (
        tmp_path / "outputs" / "persistent_family_policy_store" / "knowledge_graph"
    ).resolve()


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
                "tool_result_status": "success",
                "tool_result_solves_task": True,
                "tool_result_trusted_for_materialization": True,
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
        lambda _: ["14", "15"],
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
            "tool_result_status": "success",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
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
        },
        {
            "family_name": "count_over_direct_relation",
            "sample_index": "15",
            "trigger_source": "sample_failure_or_wrong_completion",
            "boundary_sample_index": "15",
        }
    ]


def test_boundary_runner_uses_single_sample_config_directly(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    config_path = "dummy_single_sample_boundary.yaml"
    combined_dir = tmp_path / "outputs"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_all_with_servers,
        "_load_sample_order_from_config",
        lambda _: ["43"],
    )
    monkeypatch.setattr(
        run_all_with_servers,
        "_extract_task_name",
        lambda _: "knowledge_graph",
    )

    def fake_run_one(config_path_arg, combined_dir_arg, *, output_dir_override=None, extra_env=None):
        calls.append(
            {
                "config_path": config_path_arg,
                "combined_dir": combined_dir_arg,
                "output_dir_override": output_dir_override,
                "extra_env": dict(extra_env or {}),
            }
        )
        return 0

    monkeypatch.setattr(run_all_with_servers, "_run_one", fake_run_one)

    def fail_write_single_sample_config(**kwargs):
        raise AssertionError("single-sample configs should not be rewritten")

    monkeypatch.setattr(
        run_all_with_servers,
        "_write_single_sample_config",
        fail_write_single_sample_config,
    )

    rc = _run_one_with_sample_boundary_family_evolution(config_path, combined_dir)

    assert rc == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["config_path"] == config_path
    assert call["output_dir_override"] == (
        combined_dir / "knowledge_graph" / "dummy_single_sample_boundary"
    )
    assert call["extra_env"]["PAL_ENABLE_FAMILY_POLICY_EVOLUTION"] == "1"
    assert call["extra_env"]["PAL_ENABLE_FAMILY_POLICY_PROMOTION"] == "0"


def test_main_activates_standard_family_evolution_without_extra_family_flag(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv(
        "LIFELONG_CONFIG_PATHS",
        "configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml",
    )
    monkeypatch.delenv("PAL_ENABLE_FAMILY_POLICY_EVOLUTION", raising=False)
    monkeypatch.setattr(run_all_with_servers, "ENABLE_STANDARD_FAMILY_EVOLUTION", True)
    monkeypatch.setattr(run_all_with_servers, "ENABLE_PAL_AGENT", True)
    monkeypatch.setattr(
        run_all_with_servers,
        "_run_one_with_sample_boundary_family_evolution",
        lambda config_path, combined_dir: calls.append(("boundary", config_path)) or 0,
    )
    monkeypatch.setattr(
        run_all_with_servers,
        "_run_one",
        lambda config_path, combined_dir, **kwargs: calls.append(("plain", config_path)) or 0,
    )

    rc = run_all_with_servers.main()

    assert rc == 0
    assert calls == [
        (
            "boundary",
            "configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml",
        )
    ]
