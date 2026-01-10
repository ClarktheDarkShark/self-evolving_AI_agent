#kg_sqarql_server.py

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import platform


KG_DUMP_ENV = "LIFELONG_KG_DUMP_PATH"
KG_DATA_DIR_ENV = "LIFELONG_KG_DATA_DIR"
KG_CONTAINER_ENV = "LIFELONG_KG_CONTAINER_NAME"
KG_LOADER_IMAGE_ENV = "LIFELONG_KG_LOADER_IMAGE"
KG_DOCKER_PLATFORM_ENV = "LIFELONG_KG_DOCKER_PLATFORM"


def _append_log(log_path: Optional[Path], text: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _run(cmd: list[str], *, log_path: Optional[Path] = None) -> subprocess.CompletedProcess:
    _append_log(log_path, f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        _append_log(log_path, result.stdout)
    if result.stderr:
        _append_log(log_path, result.stderr)
    return result


def _resolve_dump_path(dump_path: Optional[str]) -> Path:
    candidate = dump_path or os.getenv(KG_DUMP_ENV, "")
    if not candidate:
        raise FileNotFoundError(
            "KG dump not configured. Provide a Freebase RDF dump in N-Triples (.nt/.nt.gz) or TTL (.ttl/.ttl.gz).\n"
            f"Set {KG_DUMP_ENV}=<path-to-dump> to continue."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"KG dump not found at {path}.\n"
            f"Set {KG_DUMP_ENV} to a valid dump file (Freebase RDF .nt/.nt.gz/.ttl/.ttl.gz)."
        )
    return path


def _resolve_data_dir(data_dir: Optional[str]) -> Path:
    candidate = data_dir or os.getenv(KG_DATA_DIR_ENV, "")
    if candidate:
        return Path(candidate).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "data" / "knowledge_graph" / "sparql" / "fuseki").resolve()


def _container_name(name: Optional[str]) -> str:
    return name or os.getenv(KG_CONTAINER_ENV, "lifelong_fuseki")


def _loader_image() -> str:
    return os.getenv(KG_LOADER_IMAGE_ENV, "stain/jena:latest")


def _docker_platform_candidates(log_path: Optional[Path] = None) -> list[str]:
    platform_env = os.getenv(KG_DOCKER_PLATFORM_ENV, "").strip()
    if platform_env:
        return [platform_env]
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        _append_log(
            log_path,
            "Defaulting Docker platform to linux/amd64 on ARM host. "
            f"Override with {KG_DOCKER_PLATFORM_ENV} if needed.",
        )
        return ["linux/amd64"]
    return [""]


def _dataset_dir(data_dir: Path, dataset: str) -> Path:
    return data_dir / "databases" / dataset


def _dataset_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.iterdir())


def start_fuseki(
    *,
    endpoint_url: str,
    dump_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    dataset: str = "kb",
    port: int = 3001,
    container_name: Optional[str] = None,
    force_reload: bool = False,
    log_path: Optional[Path] = None,
) -> None:
    dump = _resolve_dump_path(dump_path)
    data_root = _resolve_data_dir(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    container = _container_name(container_name)
    dataset_path = _dataset_dir(data_root, dataset)

    # stop any prior container
    _run(["docker", "rm", "-f", container], log_path=log_path)

    platform_candidates = _docker_platform_candidates(log_path)

    # ----------------------------
    # 1) LOAD FIRST (if needed)
    # ----------------------------
    if force_reload or not _dataset_has_data(dataset_path):
        dataset_path.mkdir(parents=True, exist_ok=True)

        loader_image = _loader_image()  # default stain/jena:latest
        last_err = ""

        for platform_name in platform_candidates + ["linux/amd64"]:
            load_cmd = ["docker", "run", "--rm"]
            if platform_name:
                load_cmd += ["--platform", platform_name]
            load_cmd += [
                "-v", f"{data_root}:/fuseki/databases",
                "-v", f"{dump}:/data/kg_dump:ro",
                loader_image,
                "/jena/bin/tdb2.tdbloader",
                "--loc", f"/fuseki/databases/{dataset}",
                "/data/kg_dump",
                ]
            result = _run(load_cmd, log_path=log_path)
            if result.returncode == 0:
                last_err = ""
                break
            last_err = result.stderr or result.stdout or "unknown loader error"

        if last_err:
            raise RuntimeError(f"Failed to load KG dump with loader image '{loader_image}': {last_err.strip()}")

    # ----------------------------
    # 2) START FUSEKI (pointing at loaded DB)
    #    IMPORTANT: command must NOT start with '--'
    # ----------------------------
    result = None
    for platform_name in platform_candidates + ["linux/amd64"]:
        run_cmd = [
            "docker", "run",
            "-d",
            "--rm",
            "--name", container,
        ]
        if platform_name:
            run_cmd += ["--platform", platform_name]

        run_cmd += [
            "-p", f"{port}:3030",
            "-e", "ADMIN_PASSWORD=admin",
            # Optional but recommended for big loads:
            "-e", "JVM_ARGS=-Xmx12g",
            "-v", f"{data_root}:/fuseki",
            "stain/jena-fuseki:latest",
            "fuseki-server",          # <-- key fix (command first)
            "--tdb2",
            "--loc", f"/fuseki/databases/{dataset}",
            f"/{dataset}",
        ]

        result = _run(run_cmd, log_path=log_path)
        if result.returncode == 0:
            break

    if result is None or result.returncode != 0:
        raise RuntimeError("Failed to start Fuseki container.")

    _append_log(log_path, f"SPARQL endpoint: {endpoint_url}")



@dataclass
class SparqlHealth:
    reachable: bool
    has_data: bool
    error: Optional[str] = None


def sparql_health(endpoint_url: str, timeout_s: float = 5.0) -> SparqlHealth:
    query = "ASK { ?s ?p ?o }"
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    try:
        req = urllib.request.Request(
            endpoint_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
        obj = json.loads(payload)
        has_data = bool(obj.get("boolean"))
        return SparqlHealth(reachable=True, has_data=has_data)
    except Exception as exc:
        return SparqlHealth(reachable=False, has_data=False, error=str(exc))


def wait_for_sparql_ready(endpoint_url: str, timeout_s: int = 120) -> SparqlHealth:
    start = time.time()
    last_health = SparqlHealth(reachable=False, has_data=False, error="not checked")
    while time.time() - start < timeout_s:
        last_health = sparql_health(endpoint_url)
        if last_health.reachable and last_health.has_data:
            return last_health
        time.sleep(2)
    return last_health


def stop_fuseki(container_name: Optional[str] = None, *, log_path: Optional[Path] = None) -> None:
    container = _container_name(container_name)
    _run(["docker", "rm", "-f", container], log_path=log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage local KG SPARQL server via Docker.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start Fuseki and load KG dump")
    start.add_argument("--endpoint", default="http://127.0.0.1:3001/kb/sparql")
    start.add_argument("--dump", default=None)
    start.add_argument("--data-dir", default=None)
    start.add_argument("--dataset", default="kb")
    start.add_argument("--port", type=int, default=3001)
    start.add_argument("--container", default=None)
    start.add_argument("--force-reload", action="store_true")
    start.add_argument("--log", default=None)

    stop = sub.add_parser("stop", help="Stop Fuseki container")
    stop.add_argument("--container", default=None)
    stop.add_argument("--log", default=None)

    health = sub.add_parser("health", help="Check SPARQL readiness")
    health.add_argument("--endpoint", default="http://127.0.0.1:3001/kb/sparql")

    args = parser.parse_args()

    log_path = Path(args.log).resolve() if getattr(args, "log", None) else None

    if args.command == "start":
        start_fuseki(
            endpoint_url=args.endpoint,
            dump_path=args.dump,
            data_dir=args.data_dir,
            dataset=args.dataset,
            port=args.port,
            container_name=args.container,
            force_reload=args.force_reload,
            log_path=log_path,
        )
        return 0
    if args.command == "stop":
        stop_fuseki(container_name=args.container, log_path=log_path)
        return 0
    if args.command == "health":
        health_status = sparql_health(args.endpoint)
        print(json.dumps(health_status.__dict__, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
