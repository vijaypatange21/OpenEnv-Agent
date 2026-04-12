"""Baseline inference script for Scaler Env.

The script evaluates all built-in support-triage tasks by default. If OpenAI
credentials are available, it uses the OpenAI client to choose actions.
Otherwise it falls back to a deterministic heuristic policy so the repository
remains locally runnable.
"""

from __future__ import annotations

import json
import importlib
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

try:
    from scaler_env import ScalerAction, ScalerEnvironment
    from scaler_env.server.tasks import TASK_BY_NAME
except ModuleNotFoundError:  # pragma: no cover
    from models import ScalerAction
    from server.scaler_env_environment import ScalerEnvironment
    from server.tasks import TASK_BY_NAME

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME") or os.getenv("SCALER_TASK_NAME") or "all"
MAX_STEPS = 6
TEMPERATURE = 0.0
MAX_TOKENS = 220
SUCCESS_SCORE_THRESHOLD = 0.75

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer-support operations agent.
    You must choose one structured action in JSON.

    Available operations:
    - classify: assign the right category and priority
    - respond: write the customer-facing reply
    - request_info: ask for missing verification or transaction details
    - escalate: hand the case to the correct internal team
    - resolve: mark the workflow as resolved
    - close: close the ticket after the work is complete

    Return exactly one JSON object with these keys:
    operation, category, priority, team, response, confidence, notes

    Keep the action aligned with the current stage, ticket text, and prior history.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _compact_json(payload: Dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_json_action(content: str) -> Dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
    return json.loads(content)


def _heuristic_action(task_name: str, stage: str, observation) -> ScalerAction:
    if task_name == "password_reset_easy":
        if stage == "classify":
            return ScalerAction(operation="classify", category="account_access", priority="medium", confidence=0.95)
        if stage == "respond":
            return ScalerAction(
                operation="respond",
                response=(
                    "Please complete verification and then use the password reset link in the login screen. "
                    "If you still cannot sign in, reply with any error message you see."
                ),
                confidence=0.90,
            )
        return ScalerAction(operation="close", confidence=0.95)

    if task_name == "duplicate_charge_medium":
        if stage == "classify":
            return ScalerAction(operation="classify", category="billing", priority="high", confidence=0.95)
        if stage == "request_info":
            return ScalerAction(
                operation="request_info",
                response="Please share the transaction date, the last four digits of the card, and the receipt or statement screenshot.",
                confidence=0.90,
            )
        if stage == "respond":
            return ScalerAction(
                operation="respond",
                response=(
                    "Thanks. I will review the duplicate charge, reference the statement, and explain the refund process once I confirm the transaction details."
                ),
                confidence=0.88,
            )
        return ScalerAction(operation="close", confidence=0.95)

    if task_name == "privacy_deletion_hard":
        if stage == "classify":
            return ScalerAction(operation="classify", category="privacy_request", priority="urgent", confidence=0.95)
        if stage == "request_info":
            return ScalerAction(
                operation="request_info",
                response="Please verify your identity and confirm account ownership so we can proceed safely.",
                confidence=0.92,
            )
        if stage == "escalate":
            return ScalerAction(operation="escalate", team="privacy", notes="Escalate for deletion review", confidence=0.95)
        if stage == "respond":
            return ScalerAction(
                operation="respond",
                response=(
                    "I cannot confirm deletion until the privacy team reviews the request and verification is complete."
                ),
                confidence=0.88,
            )
        return ScalerAction(operation="close", confidence=0.95)

    return ScalerAction(operation="respond", response="Acknowledged.", confidence=0.5)


def _build_user_prompt(task_name: str, obs, history: List[Dict]) -> str:
    history_text = json.dumps(history[-4:], ensure_ascii=False)
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Difficulty: {obs.difficulty}
        Current stage: {obs.current_stage}
        Allowed operations: {obs.allowed_operations}
        Remaining steps: {obs.remaining_steps}
        Task info: {obs.task_info}
        Ticket subject: {obs.subject}
        Customer message: {obs.customer_message}
        Customer tier: {obs.customer_tier}
        Last feedback: {obs.last_feedback}
        History: {history_text}

        Return one JSON object with keys:
        operation, category, priority, team, response, confidence, notes
        """
    ).strip()


def _get_llm_action(client: Any, task_name: str, obs, history: List[Dict]) -> Tuple[ScalerAction, Optional[str]]:
    user_prompt = _build_user_prompt(task_name, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        data = _parse_json_action(content)
        action = ScalerAction.model_validate(data) if hasattr(ScalerAction, "model_validate") else ScalerAction.parse_obj(data)
        return action, None
    except Exception as exc:  # pragma: no cover - model/network dependent
        fallback = _heuristic_action(task_name, obs.current_stage, obs)
        return fallback, str(exc)


def _run_task(task_name: str, client: Optional[Any]) -> None:
    env = ScalerEnvironment()
    obs = env.reset(task_name=task_name)
    rewards: List[float] = []
    history: List[Dict] = []
    success = False

    log_start(task_name, "scaler_env", MODEL_NAME if client else "heuristic-fallback")

    for step_idx in range(1, MAX_STEPS + 1):
        if client is None:
            action = _heuristic_action(task_name, obs.current_stage, obs)
            error = None
        else:
            action, error = _get_llm_action(client, task_name, obs, history)

        action_payload = action.model_dump(exclude_none=True, mode="json") if hasattr(action, "model_dump") else action.dict(exclude_none=True)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
        history.append({"action": action_payload, "reward": float(obs.reward or 0.0), "done": obs.done})
        log_step(step_idx, _compact_json(action_payload), float(obs.reward or 0.0), bool(obs.done), error)
        if obs.done:
            break

    score = float(env.state.final_score)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success, env.state.step_count, score, rewards)


def main() -> None:
    task_names = list(TASK_BY_NAME.keys()) if TASK_NAME == "all" else [TASK_NAME]
    client = None
    if API_KEY:
        try:
            openai_module = importlib.import_module("openai")
            client = openai_module.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception:  # pragma: no cover - optional local dependency
            client = None

    for task_name in task_names:
        _run_task(task_name, client)


if __name__ == "__main__":
    main()
