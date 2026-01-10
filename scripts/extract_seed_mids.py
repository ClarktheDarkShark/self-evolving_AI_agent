#!/usr/bin/env python3
import re, sys
from pathlib import Path

# Matches common Freebase MID forms:
#  - m.0d06, g.123abc
#  - /m/0d06, /g/123abc
#  - http://rdf.freebase.com/ns/m.0d06 (optionally with <...>)
MID_RE = re.compile(
    r"(?:<)?(?:https?://rdf\.freebase\.com/ns/)?(?:(?:/)?)([mg]\.[0-9A-Za-z_]+)(?:>)?"
)
MID_SLASH_RE = re.compile(r"(?:/)([mg])/(?:)([0-9A-Za-z_]+)")

def normalize(found: str) -> str:
    return found.strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: extract_seed_mids.py <entry_dict.json> <out_mids.txt>", file=sys.stderr)
        sys.exit(2)

    src = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve()

    text = src.read_text(encoding="utf-8", errors="replace")

    mids = set()

    # m.xxx / g.xxx / full URI forms
    for m in MID_RE.finditer(text):
        mids.add(normalize(m.group(1)))

    # /m/xxx and /g/xxx forms
    for m in MID_SLASH_RE.finditer(text):
        mids.add(f"{m.group(1)}.{m.group(2)}")

    mids = sorted(mids)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(mids) + ("\n" if mids else ""), encoding="utf-8")

    print(f"[seed] source={src}")
    print(f"[seed] wrote={len(mids):,} mids -> {out}")
    if len(mids) == 0:
        print("[seed] ERROR: extracted 0 mids. Your dataset may not contain Freebase IDs in any standard form.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
