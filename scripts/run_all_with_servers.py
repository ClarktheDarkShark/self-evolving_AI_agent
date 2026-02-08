# scripts/run_all_with_servers.py
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import kg_sparql_server

CONFIG_PATHS = [
    # "configs/assignments/experiments/llama_31_8b_instruct/instance/os_interaction/instance/standard.yaml",
    "configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml",
    # "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml",
    
]

SPARQL_ENDPOINT = "http://127.0.0.1:3001/kb/sparql"

# Fuseki server (serve-only; no dump loading)
FUSEKI_CONTAINER = os.getenv("LIFELONG_KG_CONTAINER_NAME", "lifelong_fuseki")
FUSEKI_IMAGE = os.getenv("LIFELONG_FUSEKI_IMAGE", "stain/jena-fuseki:latest")
FUSEKI_PLATFORM = os.getenv("LIFELONG_KG_DOCKER_PLATFORM", "linux/amd64")  # ARM host typically needs amd64

FUSEKI_HOST_PORT = int(os.getenv("LIFELONG_FUSEKI_PORT", "3001"))
FUSEKI_DATASET = os.getenv("LIFELONG_FUSEKI_DATASET", "kb")  # path /kb and databases/kb
KG_DATA_DIR_ENV = "LIFELONG_KG_DATA_DIR"


def _append_log(log_path: Path | None, text: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _log_json_preview(
    *,
    source: str,
    status_code: str,
    content_type: str,
    body: str,
) -> None:
    preview = (body or "").replace("\n", "\\n")[:200]
    print(
        f"[json-parse] source={source} status={status_code} "
        f"content_type={content_type} body_head={preview}"
    )


def _run(cmd: list[str], *, log_path: Path | None = None) -> subprocess.CompletedProcess:
    _append_log(log_path, f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        _append_log(log_path, result.stdout)
    if result.stderr:
        _append_log(log_path, result.stderr)
    return result


def sparql_probe(endpoint: str, timeout_s: int = 5) -> tuple[bool, bool, str | None]:
    query = "ASK WHERE { ?s ?p ?o }"
    body = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/sparql-results+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            _log_json_preview(
                source=endpoint,
                status_code=str(resp.status),
                content_type=str(resp.headers.get("Content-Type", "")),
                body=raw,
            )
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "boolean" in payload:
                    return True, bool(payload["boolean"]), None
            except Exception:
                pass
            return True, False, f"Unexpected response (first 200 chars): {raw[:200]}"
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return True, False, f"HTTPError {e.code}: {detail[:200]}"
    except Exception as e:
        return False, False, str(e)


def _wait_for_server(url: str, timeout_s: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 405:
                try:
                    with urllib.request.urlopen(url, timeout=2) as resp:
                        if 200 <= resp.status < 300:
                            return True
                except (urllib.error.URLError, TimeoutError):
                    pass
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    return False


def _tail_file(path: Path, n_lines: int = 200) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception as exc:
        return f"[run_all_with_servers] Unable to read log file: {exc}\n"


def _sanitize_log_name(config_path: str) -> str:
    return config_path.replace("/", "_").replace(".yaml", "") + ".log"


def _extract_task_name(config_path: str) -> str:
    parts = config_path.split("/")
    if "instance" in parts:
        idx = parts.index("instance")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(config_path).stem


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_docker_containers_on_port(port: int) -> None:
    ps = subprocess.run(
        ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}"],
        capture_output=True, text=True, check=False,
    )
    ids = [x.strip() for x in (ps.stdout or "").splitlines() if x.strip()]
    for cid in ids:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, text=True, check=False)



def _kill_port(port: int) -> None:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    pids = [int(pid) for pid in result.stdout.split() if pid.strip().isdigit()]
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 3
    while time.time() < deadline:
        if not any(_pid_exists(pid) for pid in pids):
            return
        time.sleep(0.2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _preflight_kill_ports(ports: list[int]) -> None:
    for port in ports:
        _kill_port(port)
    time.sleep(1)


def _list_output_dirs(outputs_root: Path) -> list[Path]:
    if not outputs_root.exists():
        return []
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
    return [p for p in outputs_root.iterdir() if p.is_dir() and pattern.match(p.name)]


def _pick_output_dir(before: list[Path], after: list[Path]) -> Path | None:
    before_set = {p.resolve() for p in before}
    new_dirs = [p for p in after if p.resolve() not in before_set]
    if new_dirs:
        new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return new_dirs[0]
    if after:
        after.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return after[0]
    return None


def _merge_runs(output_dir: Path, combined_dir: Path) -> None:
    runs_path = output_dir / "runs.json"
    if not runs_path.exists():
        return
    combined_path = combined_dir / "runs.json"
    try:
        raw = runs_path.read_text(encoding="utf-8")
        _log_json_preview(
            source=str(runs_path),
            status_code="file",
            content_type="application/json",
            body=raw,
        )
        new_runs = json.loads(raw)
    except Exception:
        return
    if not isinstance(new_runs, list):
        return
    existing: list[dict[str, object]] = []
    if combined_path.exists():
        try:
            raw_existing = combined_path.read_text(encoding="utf-8")
            _log_json_preview(
                source=str(combined_path),
                status_code="file",
                content_type="application/json",
                body=raw_existing,
            )
            existing = json.loads(raw_existing)
        except Exception:
            existing = []
    if not isinstance(existing, list):
        existing = []
    combined_path.write_text(
        json.dumps(existing + new_runs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _merge_metrics(output_dir: Path, combined_dir: Path) -> None:
    metric_path = output_dir / "metric.json"
    runs_path = output_dir / "runs.json"
    if not metric_path.exists() or not runs_path.exists():
        return
    try:
        raw_metric = metric_path.read_text(encoding="utf-8")
        _log_json_preview(
            source=str(metric_path),
            status_code="file",
            content_type="application/json",
            body=raw_metric,
        )
        metric_data = json.loads(raw_metric)
        raw_runs = runs_path.read_text(encoding="utf-8")
        _log_json_preview(
            source=str(runs_path),
            status_code="file",
            content_type="application/json",
            body=raw_runs,
        )
        runs_data = json.loads(raw_runs)
    except Exception:
        return
    task_name = None
    if isinstance(runs_data, list) and runs_data:
        task_name = runs_data[0].get("task_name")
    if not task_name:
        task_name = output_dir.name
    combined_path = combined_dir / "metric.json"
    combined_metric = {}
    if combined_path.exists():
        try:
            raw_combined = combined_path.read_text(encoding="utf-8")
            _log_json_preview(
                source=str(combined_path),
                status_code="file",
                content_type="application/json",
                body=raw_combined,
            )
            combined_metric = json.loads(raw_combined)
        except Exception:
            combined_metric = {}
    if not isinstance(combined_metric, dict):
        combined_metric = {}
    combined_metric.setdefault(task_name, []).append(metric_data)
    combined_path.write_text(
        json.dumps(combined_metric, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _copy_aux_files(output_dir: Path, combined_dir: Path, tag: str) -> None:
    for filename in ("config.yaml", "exception.txt", "singleton_logger_client.log", "singleton_logger_server.log"):
        src = output_dir / filename
        if not src.exists():
            continue
        dest = combined_dir / f"{tag}_{filename}"
        try:
            shutil.copy2(src, dest)
        except Exception:
            continue


def _resolve_fuseki_data_root(repo_root: Path) -> Path:
    env_dir = os.getenv(KG_DATA_DIR_ENV, "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (repo_root / "data" / "knowledge_graph" / "fuseki_db").resolve()


def _docker_container_running(name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode == 0 and name in (r.stdout or "")


def _start_fuseki_serve_only(repo_root: Path, fuseki_log_path: Path) -> bool:
    """
    Start Fuseki from existing TDB2 directory only (no loading).
    """
    data_root = _resolve_fuseki_data_root(repo_root)
    dataset_dir = data_root / "databases" / FUSEKI_DATASET

    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print("[run_all_with_servers] KG dataset directory missing/empty.")
        print(f"[run_all_with_servers] Expected non-empty TDB2 dir: {dataset_dir}")
        return False

    _append_log(fuseki_log_path, "Starting Fuseki (serve-only from existing TDB2).")
    _run(["docker", "rm", "-f", FUSEKI_CONTAINER], log_path=fuseki_log_path)

    # IMPORTANT on macOS:
    # lsof won't reliably reveal Docker's port forwarder. If any container publishes this port,
    # docker run will fail with "port is already allocated". Kill those containers first.
    _stop_docker_containers_on_port(FUSEKI_HOST_PORT)

    # Also try to kill any normal local processes using the port (non-Docker)
    _kill_port(FUSEKI_HOST_PORT)

    run_cmd = [
        "docker", "run", "-d",
        "--name", FUSEKI_CONTAINER,
        "--platform", FUSEKI_PLATFORM,
        "-p", f"{FUSEKI_HOST_PORT}:{FUSEKI_HOST_PORT}",
        "-e", "ADMIN_PASSWORD=admin",
        "-v", f"{data_root}:/fuseki",
        FUSEKI_IMAGE,
        "./fuseki-server",
        "--port", str(FUSEKI_HOST_PORT),
        "--tdb2", "--loc", f"/fuseki/databases/{FUSEKI_DATASET}",
        f"/{FUSEKI_DATASET}",
    ]

    res = _run(run_cmd, log_path=fuseki_log_path)
    if res.returncode != 0:
        print("[run_all_with_servers] docker run failed starting Fuseki.")

        # # Print the REAL docker error immediately
        # if res.stderr:
        print("[run_all_with_servers] docker stderr:")
        print(res.stderr.strip())

        # if res.stdout:
        print("[run_all_with_servers] docker stdout:")
        print(res.stdout.strip())

        print(f"[run_all_with_servers] Fuseki log: {fuseki_log_path}")
        print(_tail_file(fuseki_log_path))
        return False

    # Show log tail right after start (helps immediately)
    # print("Log: ", _tail_file(fuseki_log_path))

    # Print last container logs on start to make failures obvious
    logs = subprocess.run(
        ["docker", "logs", "--tail", "80", FUSEKI_CONTAINER],
        capture_output=True,
        text=True,
        check=False,
    )
    if logs.stdout:
        _append_log(fuseki_log_path, "--- docker logs (tail 80) ---\n" + logs.stdout)

    return True



def _ensure_fuseki_ready(repo_root: Path, fuseki_log_path: Path) -> tuple[bool, bool]:
    """
    Returns: (started_by_script, ok)
    """
    # First probe
    reachable, has_data, err = sparql_probe(SPARQL_ENDPOINT)
    print(f"[KG preflight] endpoint={SPARQL_ENDPOINT}")
    print(f"[KG preflight] reachable={reachable}, has_data={has_data}")
    if err:
        print(f"[KG preflight] error={err}")
        pass

    if reachable and has_data:
        return False, True

    # If it isn't healthy, start it (serve-only)
    started = True
    ok = _start_fuseki_serve_only(repo_root, fuseki_log_path)
    if not ok:
        return started, False

    # Now wait for readiness
    health = kg_sparql_server.wait_for_sparql_ready(SPARQL_ENDPOINT, timeout_s=120)
    if not (health.reachable and health.has_data):
        print("[run_all_with_servers] SPARQL readiness failed after starting Fuseki.")
        print(f"[run_all_with_servers] Expected endpoint: {SPARQL_ENDPOINT}")
        print(f"[run_all_with_servers] Reachable: {health.reachable}")
        print(f"[run_all_with_servers] KB loaded: {health.has_data}")
        if health.error:
            print(f"[run_all_with_servers] Error: {health.error}")
            pass

        # surface container status/logs to console
        ps = subprocess.run(["docker", "ps", "-a", "--filter", f"name={FUSEKI_CONTAINER}"],
                            capture_output=True, text=True, check=False)
        if ps.stdout:
            print(ps.stdout.strip())
            pass
        logs = subprocess.run(["docker", "logs", "--tail", "80", FUSEKI_CONTAINER],
                              capture_output=True, text=True, check=False)
        if logs.stdout:
            print(logs.stdout.strip())

        print(f"[run_all_with_servers] Fuseki log: {fuseki_log_path}")
        return started, False

    return started, True


def _run_one(config_path: str, combined_dir: Path) -> int:
    is_kg = "knowledge_graph" in config_path

    repo_root = Path(__file__).resolve().parents[1]
    full_path = repo_root / config_path
    if not full_path.exists():
        print(f"[run_all_with_servers] Missing config: {full_path}")
        return 1

    logs_dir = repo_root / "logs" / "run_all_with_servers"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / _sanitize_log_name(config_path)
    fuseki_log_path = log_path.with_suffix(".fuseki.log")
    task_name = _extract_task_name(config_path)
    config_name = Path(config_path).stem

    # SELF-CONTAINED: ensure Fuseki is up BEFORE anything else for KG
    started_fuseki = False
    if is_kg:
        started_fuseki, ok = _ensure_fuseki_ready(repo_root, fuseki_log_path)
        if not ok:
            return 1

    server_cmd = [
        sys.executable,
        "src/distributed_deployment_utils/start_server.py",
        "--config_path",
        config_path,
    ]
    client_cmd = [
        sys.executable,
        "src/run_experiment.py",
        "--config_path",
        config_path,
    ]

    env = os.environ.copy()
    env_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env_py_path}" if env_py_path else str(repo_root)
    env["LIFELONG_OUTPUT_DIR"] = str(combined_dir / task_name / config_name)
    env["LIFELONG_OUTPUT_TAG"] = ""
    if is_kg:
        ontology_dir = str(repo_root / "data" / "knowledge_graph" / "ontology")
        if not env.get("KG_ONTOLOGY_DIR"):
            env["KG_ONTOLOGY_DIR"] = ontology_dir
        if not env.get("LIFELONG_KG_ONTOLOGY_DIR"):
            env["LIFELONG_KG_ONTOLOGY_DIR"] = ontology_dir

    print(f"[run_all_with_servers] Starting server: {config_path}")
    _preflight_kill_ports([8000, 8001])

    log_fp = log_path.open("w", encoding="utf-8")
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=repo_root,
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        if not _wait_for_server("http://127.0.0.1:8000/api/ping"):
            print("[run_all_with_servers] Task server did not become ready on 8000.")
            print(f"[run_all_with_servers] Server log: {log_path}")
            print(_tail_file(log_path))
            return 1

        if not _wait_for_server("http://127.0.0.1:8001/api/ping"):
            print("[run_all_with_servers] ChatHistoryItemFactory server did not become ready on 8001.")
            print(f"[run_all_with_servers] Server log: {log_path}")
            print(_tail_file(log_path))
            return 1

        print(f"[run_all_with_servers] Running client: {config_path}")
        result = subprocess.run(client_cmd, cwd=repo_root, env=env, check=False)
        if result.returncode != 0:
            print(f"[run_all_with_servers] Client failed for {config_path} (exit={result.returncode})")
            print(f"[run_all_with_servers] Server log: {log_path}")
            print(_tail_file(log_path))
            return result.returncode


    finally:
        if server_proc.poll() is None:
            try:
                os.killpg(server_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2)
        try:
            log_fp.close()
        except Exception:
            pass

        # Stop Fuseki only if THIS script started it.
        if is_kg and started_fuseki:
            try:
                _run(["docker", "rm", "-f", FUSEKI_CONTAINER], log_path=fuseki_log_path)
            except Exception:
                pass

    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    combined_dir = repo_root / "outputs" / f"run_all_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    combined_dir.mkdir(parents=True, exist_ok=True)

    for config_path in CONFIG_PATHS:
        code = _run_one(config_path, combined_dir)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
