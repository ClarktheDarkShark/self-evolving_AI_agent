from __future__ import annotations

import re
from typing import Iterable


_SECTION_MARKERS = (
    "available actions",
    "output contract",
    "output rules",
    "final answer format",
)
_KEEP_KEYWORDS = (
    "one action per turn",
    "only one action per turn",
    "action:",
    "final answer",
)


def _condense_contract(text: str, *, max_lines: int = 80, max_chars: int = 1800) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    kept: list[str] = []
    seen: set[str] = set()
    capture = False
    capture_count = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(marker in low for marker in _SECTION_MARKERS):
            capture = True
            capture_count = 0
        if capture:
            if line.startswith("###") and not any(marker in low for marker in _SECTION_MARKERS):
                capture = False
            else:
                capture_count += 1
                if capture_count <= 25:
                    _append_unique(line, kept, seen)
        if any(keyword in low for keyword in _KEEP_KEYWORDS):
            _append_unique(line, kept, seen)
        if line.startswith("Action:") or line.startswith("Final Answer:"):
            _append_unique(line, kept, seen)
        if len(kept) >= max_lines:
            break
    condensed = "\n".join(kept)
    if len(condensed) > max_chars:
        condensed = condensed[: max_chars - 3] + "..."
    return condensed


def _append_unique(line: str, kept: list[str], seen: set[str]) -> None:
    if line in seen:
        return
    kept.append(line)
    seen.add(line)


def _render_tasks(tasks: Iterable[str]) -> str:
    chunks: list[str] = []
    for idx, task in enumerate(tasks, start=1):
        cleaned = (task or "").strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        chunks.append(f"Task {idx}:\n{cleaned}")
    return "\n\n".join(chunks)


def build_task_pack(env_name: str, env_contract_full: str, tasks: list[str]) -> str:
    condensed_contract = _condense_contract(env_contract_full)
    task_block = _render_tasks(tasks)
    header = "CONDENSED CONTRACT (key rules only):\n" + condensed_contract
    footer = (
        "AGGREGATE NOTE:\n"
        + "You have received 10 tasks from the same environment. Build a tool that best supports general task completion similar to these in a multi-step environment.\n"
        + "Support JSON state with run_id/state_dir/cursor."
    )
    return (
        f"ENVIRONMENT: {env_name}\n\n"
        # + header
        + "\n\nTASKS:\n"
        + task_block
        + "\n\n"
        + footer
    )


def build_multi_env_task_pack(
    env_sections: Iterable[tuple[str, str, list[str]]],
) -> str:
    rendered_sections: list[str] = []
    for env_name, env_contract, tasks in env_sections:
        env_instructions = (env_contract or "").strip()
        task_block = _render_tasks(tasks)
        rendered_sections.append(
            f"ENVIRONMENT: {env_name}\n"
            + "ENV INSTRUCTIONS:\n"
            + env_instructions
            + "\n\nTASKS:\n"
            + task_block
        )
    footer = (
        "AGGREGATE NOTE:\n"
        + "You have received 10 tasks from the same environment. Build a tool that best supports general task completion similar to these in a multi-step environment.\n"
        + "Support JSON state with run_id/state_dir/cursor."
    )
    return "\n\n".join(rendered_sections) + "\n\n" + footer
