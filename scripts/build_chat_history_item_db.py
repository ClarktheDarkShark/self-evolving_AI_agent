# scripts/build_chat_history_item_db_fixed.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

USER_FIELD_CANDIDATES = [
    # common names across benchmarks
    "prompt",
    "query",
    "question",
    "instruction",
    "input",
    "task",
    "context",
    "desc",
    "description",
    "user",
    "user_prompt",
]

AGENT_FIELD_CANDIDATES = [
    "ans",
    "answer",
    "response",
    "output",
    "assistant",
    "assistant_response",
]


def first_present(row: dict[str, Any], candidates: list[str]) -> Optional[str]:
    for k in candidates:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def main() -> None:
    ds = load_dataset("csyq/LifelongAgentBench", data_dir="db_bench", split="train")

    value: dict[str, dict[str, str]] = {}
    missing_user = 0
    missing_agent = 0

    for row in ds:
        sample_index = row.get("sample_index")
        if sample_index is None:
            raise ValueError("Row missing sample_index")

        user_txt = first_present(row, USER_FIELD_CANDIDATES)
        agent_txt = first_present(row, AGENT_FIELD_CANDIDATES)

        # Fallbacks for agent: sometimes "ans" exists but is not a string (rare)
        if agent_txt is None and "ans" in row:
            agent_txt = str(row["ans"])

        if user_txt is None:
            missing_user += 1
            # Put *something* rather than crash so you can see what's going on
            user_txt = json.dumps(
                {k: row[k] for k in row.keys() if k not in ("ans",)},
                ensure_ascii=False,
            )

        if agent_txt is None:
            missing_agent += 1
            agent_txt = ""  # or str(row) if you prefer

        # IMPORTANT: ChatHistoryItem is ONLY {role, content}
        # ChatHistoryItemDict expects a single ChatHistoryItem per index, not a whole conversation.
        # We'll store the user prompt by default; you can switch to agent if that's what server wants.
        value[str(sample_index)] = {
            "role": "user",
            "content": user_txt,
        }

        # If you think the server wants the *agent* as the chat history item, swap above for:
        # value[str(sample_index)] = {"role": "agent", "content": agent_txt}

    out = {"value": value}
    out_path = Path("chat_history_items/standard/db_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(value)} items -> {out_path.resolve()}")
    print(f"Missing user field: {missing_user}")
    print(f"Missing agent field: {missing_agent}")
    print("Note: output stores role='user' by default. Change to 'agent' if needed.")


if __name__ == "__main__":
    main()
