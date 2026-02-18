from __future__ import annotations

import importlib.resources as resources
from typing import Dict


_PROMPT_FILES: Dict[str, str] = {
    "phase1_skeleton_contracts": "phase1_skeleton_contracts.txt",
    "phase2_trace_normalization": "phase2_trace_normalization.txt",
    "phase3_next_action": "phase3_next_action.txt",
    "phase4_answer_gates": "phase4_answer_gates.txt",
    "phase5_self_tests": "phase5_self_tests.txt",
    "phase6_auditor": "phase6_auditor.txt",
    "integrator": "integrator.txt",
}


def load_prompt(name: str) -> str:
    filename = _PROMPT_FILES.get(name)
    if not filename:
        raise ValueError(f"Unknown prompt name: {name}")
    package = __package__ or "src.toolgen_staged.prompts"
    return resources.files(package).joinpath(filename).read_text(encoding="utf-8")


__all__ = [
    "load_prompt",
]
