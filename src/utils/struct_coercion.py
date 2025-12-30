from __future__ import annotations

import ast
import json
from typing import Any


def coerce_struct(value: Any) -> Any:
    """
    Convert stringified dict/list (JSON or Python literal) into real dict/list.
    If already a dict/list, returns as-is.
    """
    if isinstance(value, (dict, list)) or value is None:
        return value
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return value

    # Try JSON first.
    try:
        if (s[0] in "{[" and s[-1] in "}]"):
            return json.loads(s)
    except Exception:
        pass

    # Try Python literal: "{'a': 1}", "['x','y']".
    try:
        return ast.literal_eval(s)
    except Exception:
        return value
