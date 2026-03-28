from __future__ import annotations

from src.self_evolving_agent.controller_prompts import (
    TOOLGEN_USER_APPENDIX,
)


TOOLGEN_SYSTEM_PROMPT_BASELINE = TOOLGEN_USER_APPENDIX

def get_toolgen_system_prompt(pipeline: str, env_name: str | None = None) -> str:
    return TOOLGEN_SYSTEM_PROMPT_BASELINE
