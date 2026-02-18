from __future__ import annotations

import os
from typing import Literal

ToolgenMode = Literal["staged", "legacy"]


def get_toolgen_mode() -> ToolgenMode:
    raw = os.getenv("AGG_TOOLGEN_MODE", "staged")
    value = (raw or "staged").strip().lower()
    if value in ("staged", ""):
        return "staged"
    if value == "legacy":
        return "legacy"
    print(f"[ToolGen] WARNING: invalid AGG_TOOLGEN_MODE='{raw}', defaulting to staged")
    return "staged"
