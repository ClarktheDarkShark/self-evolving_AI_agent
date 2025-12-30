from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


def _wait_for_server(url: str, timeout_s: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = requests.post(url, json={}, timeout=2)
            if 200 <= response.status_code < 300:
                return True
            if response.status_code == 405:
                response = requests.get(url, timeout=2)
                if 200 <= response.status_code < 300:
                    return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=(
            "configs/assignments/experiments/llama_31_8b_instruct/"
            "instance/db_bench/instance/standard.yaml"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.typings import Session, TaskName, SampleStatus

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}".rstrip(os.pathsep)
    )

    server_cmd = [
        sys.executable,
        "src/distributed_deployment_utils/start_server.py",
        "--config_path",
        args.config_path,
    ]
    proc = subprocess.Popen(
        server_cmd,
        cwd=repo_root,
        env=env,
        start_new_session=True,
    )
    try:
        if not _wait_for_server("http://127.0.0.1:8000/api/ping"):
            print("Task server did not become ready on 8000.")
            return 1
        if not _wait_for_server("http://127.0.0.1:8001/api/ping"):
            print("ChatHistoryItemFactory server did not become ready on 8001.")
            return 1

        session = Session(
            task_name=TaskName.DB_BENCH,
            sample_index="0",
            sample_status=SampleStatus.INITIAL,
        )
        response = requests.post(
            "http://127.0.0.1:8000/api/reset",
            json={"session": session.model_dump(mode="json")},
            timeout=10,
        )
        if not response.ok:
            print("Reset failed:", response.status_code, response.text)
            return 1
        print("Reset OK:", response.json())
        return 0
    finally:
        _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
