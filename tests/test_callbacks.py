from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.callbacks.instance.consecutive_abnormal_agent_inference_process_handling_callback import (
    ConsecutiveAbnormalAgentInferenceProcessHandlingCallback,
)


def test_consecutive_abnormal_callback_restore_state_defaults_when_files_missing(
    tmp_path: pathlib.Path,
) -> None:
    callback = ConsecutiveAbnormalAgentInferenceProcessHandlingCallback(
        tolerance_count=2
    )
    callback.set_state_dir(str(tmp_path / "callback_state"))

    callback.restore_state()

    assert callback.consecutive_abnormality_count == 0
    assert callback.aborted_sample_index_list == []


def test_consecutive_abnormal_callback_restore_state_reads_saved_state(
    tmp_path: pathlib.Path,
) -> None:
    callback = ConsecutiveAbnormalAgentInferenceProcessHandlingCallback(
        tolerance_count=2
    )
    state_dir = tmp_path / "callback_state"
    callback.set_state_dir(str(state_dir))
    (state_dir / "consecutive_abnormality_count.json").write_text(
        json.dumps({"consecutive_abnormality_count": 3})
    )
    (state_dir / "aborted_sample_index_list.json").write_text(
        json.dumps(["14", "15"])
    )

    callback.restore_state()

    assert callback.consecutive_abnormality_count == 3
    assert callback.aborted_sample_index_list == ["14", "15"]
