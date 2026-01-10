from __future__ import annotations

from dataclasses import dataclass
from threading import local
from typing import Optional


@dataclass
class InferenceContext:
    task_name: Optional[str] = None
    sample_index: Optional[str] = None
    chat_history_len: Optional[int] = None


_context_state = local()


def set_inference_context(
    *,
    task_name: Optional[str],
    sample_index: Optional[str],
    chat_history_len: Optional[int],
) -> None:
    _context_state.value = InferenceContext(
        task_name=task_name,
        sample_index=sample_index,
        chat_history_len=chat_history_len,
    )


def get_inference_context() -> InferenceContext:
    return getattr(_context_state, "value", InferenceContext())


def clear_inference_context() -> None:
    if hasattr(_context_state, "value"):
        delattr(_context_state, "value")
