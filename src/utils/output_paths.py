import os
import re
from typing import Optional

_OUTPUT_DIR_ENV = "LIFELONG_OUTPUT_DIR"
_OUTPUT_TAG_ENV = "LIFELONG_OUTPUT_TAG"


def get_output_dir_override() -> Optional[str]:
    value = os.getenv(_OUTPUT_DIR_ENV, "").strip()
    return value or None


def _sanitize_tag(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")
    return safe


def get_output_tag() -> str:
    tag = os.getenv(_OUTPUT_TAG_ENV, "").strip()
    return _sanitize_tag(tag) if tag else ""


def get_output_prefix() -> str:
    tag = get_output_tag()
    if not tag:
        return ""
    return f"{tag}_"


def prefix_filename(filename: str) -> str:
    prefix = get_output_prefix()
    if not prefix:
        return filename
    return f"{prefix}{filename}"
