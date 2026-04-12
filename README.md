---
title: Scaler Env Support Triage
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - support
  - triage
---

# Scaler Env

Scaler Env is a real-world OpenEnv environment that simulates customer-support triage.
An agent receives a ticket, classifies the issue, selects the next workflow step,
and resolves the case while respecting policy constraints.

## Why this environment exists

Customer-support operations are a practical agent-evaluation domain:

- humans do this work every day
- actions are structured and verifiable
- the task has natural partial progress signals
- policy mistakes can be penalized deterministically

## Environment API

The environment follows the standard OpenEnv pattern:

- `reset(seed=None, episode_id=None, task_name=None, **kwargs)`
- `step(action)`
- `state`

### Action space

`ScalerAction` is a structured ticket-management action with the following fields:

- `operation`: one of `classify`, `respond`, `request_info`, `escalate`, `resolve`, `close`
- `category`: predicted ticket category such as `billing` or `privacy_request`
- `priority`: predicted urgency such as `low`, `medium`, `high`, or `urgent`
- `team`: escalation target, for example `privacy`
- `response`: customer-facing text for responses or follow-ups
- `confidence`: optional confidence score in `[0, 1]`
- `notes`: optional internal notes

### Observation space

`ScalerObservation` returns the public ticket context and feedback:

- ticket metadata: `task_name`, `difficulty`, `ticket_id`, `subject`, `customer_message`, `customer_tier`
- progress data: `current_stage`, `allowed_operations`, `step_count`, `remaining_steps`, `progress`
- feedback data: `last_action_summary`, `last_feedback`, `reward_breakdown`, `history_excerpt`
- extra structured `metadata`

### State

`ScalerState` exposes episode metadata:

- `episode_id`
- `step_count`
- `task_name`
- `difficulty`
- `current_stage`
- `progress`
- `resolved`
- `closed`
- `final_score`
- `last_operation`
- `history_length`
- `last_feedback`

## Tasks

Three deterministic tasks are built in.

| Task | Difficulty | Goal |
|---|---:|---|
| `password_reset_easy` | easy | Classify an account-access issue and send correct password-reset guidance |
| `duplicate_charge_medium` | medium | Handle a billing dispute, collect the right details, and explain the refund flow |
| `privacy_deletion_hard` | hard | Handle a privacy deletion request with verification, escalation, and safe wording |

Each task has a programmatic grader that returns a normalized score in `[0.0, 1.0]`.

## Reward design

The step reward is shaped to reflect partial progress:

- correct operations add progress
- field-level matches increase partial credit
- unsafe or premature actions are penalized
- the final score is normalized to `[0, 1]`

This makes the environment learnable without being trivial.

## Setup

Install dependencies in the project root:

```bash
uv sync
```

Run the server locally:

```bash
uv run server
```

The web UI and API will be available on port `8000`.

## Using the client

```python
from scaler_env import ScalerAction, ScalerEnv

env = ScalerEnv(base_url="http://localhost:8000")
result = env.reset(task_name="password_reset_easy")

action = ScalerAction(
    operation="classify",
    category="account_access",
    priority="medium",
)
result = env.step(action)
print(result.observation.reward)
```

## Baseline inference

The project includes a root-level `inference.py` script that:

- uses the OpenAI client for model calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- prints structured `[START]`, `[STEP]`, and `[END]` logs
- evaluates all three tasks reproducibly

## Example baseline scores

Current deterministic fallback baseline scores:

- `password_reset_easy`: 1.000
- `duplicate_charge_medium`: 1.000
- `privacy_deletion_hard`: 1.000

Mean score: 1.000

The `inference.py` script prints one score per task and can switch to an OpenAI-backed policy when `HF_TOKEN` and `API_BASE_URL` are provided.

## Docker / Hugging Face Spaces

The repository includes a container-ready `Dockerfile` and OpenEnv manifest.
This project is intended to deploy as a Hugging Face Space tagged `openenv`.

## Repository layout

```text
scaler_env/
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── inference.py
├── Dockerfile
└── server/
    ├── app.py
    ├── scaler_env_environment.py
    └── tasks.py
```
