from __future__ import annotations

import sys
from typing import List

import requests

BASE_URL = "http://localhost:7860"
TASKS = ["easy", "medium", "hard"]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_health() -> None:
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    if resp.status_code != 200:
        fail("Health endpoint is not reachable")


def check_tasks() -> None:
    for task in TASKS:
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"task": task, "seed": 1}, timeout=10)
        if reset_resp.status_code != 200:
            fail(f"Reset failed for task={task}")

        payload = requests.post(
            f"{BASE_URL}/step",
            json={
                "route": "technical",
                "priority": "medium",
                "next_action": "ask_for_info",
                "response": "We are reviewing this and will follow up.",
            },
            timeout=10,
        )
        if payload.status_code != 200:
            fail(f"Step failed for task={task}")

        step_data = payload.json()
        score = step_data["info"]["latest_score"]
        if not (0.0 <= score <= 1.0):
            fail(f"Score out of range for task={task}: {score}")


def check_state() -> None:
    resp = requests.get(f"{BASE_URL}/state", timeout=10)
    if resp.status_code != 200:
        fail("State endpoint failed")
    data = resp.json()
    required_keys: List[str] = ["initialized", "done", "task", "step_index", "best_score", "trajectory"]
    for key in required_keys:
        if key not in data:
            fail(f"Missing key in state response: {key}")


def main() -> None:
    check_health()
    check_tasks()
    check_state()
    print("PASS: basic submission checks succeeded")


if __name__ == "__main__":
    main()
