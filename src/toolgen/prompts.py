from __future__ import annotations

from src.self_evolving_agent.controller_prompts import (
    TOOLGEN_USER_APPENDIX,
    AGG_TOOLGEN_USER_DB,
    AGG_TOOLGEN_USER_KG,
    AGG_TOOLGEN_USER_OS,
)


TOOLGEN_SYSTEM_PROMPT_BASELINE = TOOLGEN_USER_APPENDIX

def get_toolgen_system_prompt(pipeline: str, env_name: str | None = None) -> str:
    if pipeline == "aggregate3":
        if env_name == "knowledge_graph":
            return AGG_TOOLGEN_USER_KG
        if env_name == "os_interaction":
            return AGG_TOOLGEN_USER_OS
        if env_name == "db_bench":
            return AGG_TOOLGEN_USER_DB
        return AGG_TOOLGEN_USER_KG
    return TOOLGEN_SYSTEM_PROMPT_BASELINE
