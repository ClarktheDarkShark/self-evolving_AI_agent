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
import urllib.request
from datetime import datetime
from pathlib import Path


CONFIG_PATHS = [
    "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml",
    "configs/assignments/experiments/llama_31_8b_instruct/instance/os_interaction/instance/standard.yaml",
    "configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml",
]


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


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
    return [
        p for p in outputs_root.iterdir()
        if p.is_dir() and pattern.match(p.name)
    ]


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
        new_runs = json.loads(runs_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(new_runs, list):
        return
    existing: list[dict[str, object]] = []
    if combined_path.exists():
        try:
            existing = json.loads(combined_path.read_text(encoding="utf-8"))
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
        metric_data = json.loads(metric_path.read_text(encoding="utf-8"))
        runs_data = json.loads(runs_path.read_text(encoding="utf-8"))
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
            combined_metric = json.loads(combined_path.read_text(encoding="utf-8"))
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


def _run_one(config_path: str, combined_dir: Path) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    full_path = repo_root / config_path
    if not full_path.exists():
        print(f"[run_all_with_servers] Missing config: {full_path}")
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
    if env_py_path:
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env_py_path}"
    else:
        env["PYTHONPATH"] = str(repo_root)

    logs_dir = repo_root / "logs" / "run_all_with_servers"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / _sanitize_log_name(config_path)

    outputs_root = repo_root / "outputs"
    before_outputs = _list_output_dirs(outputs_root)

    print(f"[run_all_with_servers] Starting server: {config_path}")
    _preflight_kill_ports([8000, 8001])
    with log_path.open("w", encoding="utf-8") as log_fp:
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
        after_outputs = _list_output_dirs(outputs_root)
        output_dir = _pick_output_dir(before_outputs, after_outputs)
        if output_dir is None:
            print("[run_all_with_servers] Unable to locate output directory for merge.")
        else:
            _merge_runs(output_dir, combined_dir)
            _merge_metrics(output_dir, combined_dir)
            tag = output_dir.name
            _copy_aux_files(output_dir, combined_dir, tag)
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
