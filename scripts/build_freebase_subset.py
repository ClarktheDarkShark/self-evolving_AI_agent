#!/usr/bin/env python3
import os, sys, subprocess, shutil
from pathlib import Path

def pick_decompress_cmd(dump_path: Path):
    # Prefer pigz (multi-thread), then gzip
    if shutil.which("pigz"):
        return ["pigz", "-dc", str(dump_path)]
    if shutil.which("gzip"):
        return ["gzip", "-dc", str(dump_path)]
    raise RuntimeError("Need pigz or gzip available on PATH to stream .gz")

def mid_to_uri(mid: str) -> str:
    # Freebase RDF URIs typically look like:
    # <http://rdf.freebase.com/ns/m.0d06>
    return f"<http://rdf.freebase.com/ns/{mid}>"

def load_seed_uris(seed_mids_path: Path) -> set[str]:
    mids = [ln.strip() for ln in seed_mids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return set(mid_to_uri(m) for m in mids)

def parse_s_o(line: str):
    # N-Triples allows arbitrary whitespace (spaces, tabs) between tokens.
    line = line.strip()
    if not line:
        return None, None
    parts = line.split(None, 3)  # split on ANY whitespace, max 3 splits
    if len(parts) < 3:
        return None, None
    return parts[0], parts[2]


def is_entity_uri(tok: str) -> bool:
    return tok.startswith("<http://rdf.freebase.com/ns/") and tok.endswith(">")

def build_subset_pass(dump_path: Path, keep_set: set[str], out_path: Path, collect_neighbors: bool):
    dec = pick_decompress_cmd(dump_path)
    proc = subprocess.Popen(dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1024*1024)

    neighbors = set()
    kept = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line or line[0] != "<":
                continue
            s, o = parse_s_o(line)
            if not s:
                continue

            # Keep if subject or object is in keep_set
            keep = (s in keep_set) or (o in keep_set)
            if keep:
                out.write(line)
                kept += 1
                if collect_neighbors:
                    # expand: if a kept triple connects to another entity, record it
                    if s in keep_set and is_entity_uri(o):
                        neighbors.add(o)
                    if o in keep_set and is_entity_uri(s):
                        neighbors.add(s)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Decompress command failed: {' '.join(dec)} (exit {proc.returncode})")

    return kept, neighbors

def main():
    if len(sys.argv) < 5:
        print("Usage: build_freebase_subset.py <freebase.nt.gz> <seed_mids.txt> <out_dir> <hops:0|1>")
        sys.exit(2)

    dump_path = Path(sys.argv[1]).expanduser().resolve()
    seed_mids = Path(sys.argv[2]).expanduser().resolve()
    out_dir = Path(sys.argv[3]).expanduser().resolve()
    hops = int(sys.argv[4])

    seed_uris = load_seed_uris(seed_mids)

    pass1_out = out_dir / "subset_pass1.nt"
    kept1, neighbors = build_subset_pass(dump_path, seed_uris, pass1_out, collect_neighbors=(hops >= 1))
    print(f"PASS1 kept {kept1:,} triples -> {pass1_out}")

    if hops == 0:
        final_out = out_dir / "freebase_subset.nt"
        pass1_out.rename(final_out)
        print(f"FINAL -> {final_out}")
        return

    expanded = set(seed_uris) | neighbors
    pass2_out = out_dir / "freebase_subset.nt"
    kept2, _ = build_subset_pass(dump_path, expanded, pass2_out, collect_neighbors=False)
    print(f"PASS2 kept {kept2:,} triples -> {pass2_out}")
    print(f"Expanded entity URIs: {len(expanded):,} (seed {len(seed_uris):,} + neighbors {len(neighbors):,})")
    print(f"FINAL -> {pass2_out}")

if __name__ == "__main__":
    main()
