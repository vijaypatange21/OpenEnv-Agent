from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

OPENENV_URL = os.getenv("OPENENV_URL", "https://vijaypatange21-meta.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional when the pipeline uses from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TIMEOUT_SECONDS = 20

TASKS = ["easy", "medium", "hard"]


def heuristic_action(message: str) -> Dict[str, str]:
    text = message.lower()

    if any(token in text for token in ["charged", "invoice", "subscription", "refund"]):
        route = "billing"
    elif any(token in text for token in ["login", "two-factor", "account", "recover"]):
        route = "account"
    elif any(token in text for token in ["delivered", "carrier", "shipment", "order"]):
        route = "shipping"
    elif any(token in text for token in ["security", "phishing", "private", "compliance"]):
        route = "safety"
    else:
        route = "technical"

    if any(token in text for token in ["urgent", "immediate", "two hours", "blocks"]):
        priority = "urgent"
    elif any(token in text for token in ["today", "second time", "high impact", "violate"]):
        priority = "high"
    elif any(token in text for token in ["soon", "week"]):
        priority = "medium"
    else:
        priority = "medium"

    if route == "billing":
        next_action = "refund"
    elif route in {"technical", "safety", "shipping"}:
        next_action = "escalate"
    else:
        next_action = "ask_for_info"

    response = (
        "Sorry for the trouble. We will verify details, route this to the correct team, "
        "and send an update within 24 hours."
    )

    return {
        "route": route,
        "priority": priority,
        "next_action": next_action,
        "response": response,
    }


def llm_action(client: OpenAI, task: str, observation: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not MODEL_NAME:
        return None

    prompt = (
        "You are a customer support triage assistant. Return only JSON with keys "
        "route, priority, next_action, response.\n"
        "Allowed route: billing, technical, account, shipping, safety, other.\n"
        "Allowed priority: low, medium, high, urgent.\n"
        "Allowed next_action: resolve, ask_for_info, escalate, refund, monitor.\n"
        "Task: {task}\n"
        "Customer message: {message}\n"
        "History: {history}\n"
        "Feedback: {feedback}\n"
    ).format(
        task=task,
        message=observation["customer_message"],
        history=observation.get("conversation_history", []),
        feedback=observation.get("feedback", ""),
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        if not content:
            return None
        payload = json.loads(content)
        return {
            "route": payload.get("route", "technical"),
            "priority": payload.get("priority", "medium"),
            "next_action": payload.get("next_action", "ask_for_info"),
            "response": payload.get("response", "We are investigating and will update you."),
        }
    except Exception:
        return None


def run_episode(client: OpenAI, task: str) -> Dict[str, Any]:
    print(json.dumps({"event": "START", "task": task}, sort_keys=True))

    reset_resp = requests.post(
        f"{OPENENV_URL}/reset",
        json={"task": task, "seed": 7},
        timeout=TIMEOUT_SECONDS,
    )
    reset_resp.raise_for_status()
    observation = reset_resp.json()

    done = False
    steps = 0
    last_info: Dict[str, Any] = {}

    while not done and steps < 8:
        action = llm_action(client, task, observation)
        if action is None:
            action = heuristic_action(observation["customer_message"])

        step_resp = requests.post(
            f"{OPENENV_URL}/step",
            json=action,
            timeout=TIMEOUT_SECONDS,
        )
        step_resp.raise_for_status()

        payload = step_resp.json()
        observation = payload["observation"]
        done = payload["done"]
        last_info = payload["info"]
        steps += 1

        print(
            json.dumps(
                {
                    "event": "STEP",
                    "task": task,
                    "step": steps,
                    "reward": payload["reward"]["value"],
                    "latest_score": payload["info"]["latest_score"],
                    "best_score": payload["info"]["best_score"],
                    "done": done,
                },
                sort_keys=True,
            )
        )

    state_resp = requests.get(f"{OPENENV_URL}/state", timeout=TIMEOUT_SECONDS)
    state_resp.raise_for_status()
    state = state_resp.json()

    result = {
        "task": task,
        "steps": steps,
        "best_score": state["best_score"],
        "success": bool(last_info.get("success", False)),
    }

    print(json.dumps({"event": "END", **result}, sort_keys=True))
    return result


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required but not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = []
    for task in TASKS:
        result = run_episode(client, task)
        results.append(result)
        print(f"Task={task} steps={result['steps']} score={result['best_score']:.4f} success={result['success']}")

    mean_score = sum(item["best_score"] for item in results) / len(results)
    print(f"Mean baseline score: {mean_score:.4f}")


if __name__ == "__main__":
    main()
