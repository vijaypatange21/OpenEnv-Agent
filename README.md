---
title: OpenEnv Support Triage
emoji: "🚑"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# OpenEnv Support Triage (FastAPI)

This project implements a complete OpenEnv-style environment around a real-world workflow: customer support ticket triage.

Agents interact through the standard API pattern:
- `reset()` starts an episode.
- `step(action)` evaluates one triage decision.
- `state()` returns the current environment state and trajectory.

The environment is built for learning, not just pass/fail checks. Reward is shaped over time and graders provide deterministic scores in `[0.0, 1.0]`.

## Why this is real-world

Production support organizations must reliably:
- Route requests to the correct team.
- Prioritize incidents based on impact and urgency.
- Pick immediate operational actions.
- Draft policy-aligned responses for risky incidents.

This environment simulates those decisions with deterministic grading logic.

## OpenEnv Spec Coverage

Implemented assets:
- Typed Pydantic models in `app/models.py`.
- API endpoints in `app/main.py`.
- Environment engine in `app/env.py`.
- OpenEnv metadata in `openenv.yaml`.

Endpoints:
- `POST /reset` -> `Observation`
- `POST /step` -> `StepResult`
- `GET /state` -> `EnvState`

## Task Set and Graders

Three graded tasks with increasing difficulty:

1. `easy` (Ticket Routing)  
2. `medium` (Routing + Action Planning)  
3. `hard` (Incident Triage + Response Draft)

All graders are deterministic and return scores in `[0.0, 1.0]`.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860