from __future__ import annotations

import datetime

_TIMESTAMP_ANCHOR = datetime.datetime.now()


def get_predefined_timestamp_structure() -> dict[str, str]:
    return {
        "TIMESTAMP": _TIMESTAMP_ANCHOR.strftime("%Y-%m-%d-%H-%M-%S"),
        "TIMESTAMP_DATE": _TIMESTAMP_ANCHOR.strftime("%Y-%m-%d"),
        "TIMESTAMP_TIME": _TIMESTAMP_ANCHOR.strftime("%H-%M-%S"),
    }
