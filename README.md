---
<<<<<<< HEAD
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
- `POST /step` -> `StepResult` containing `observation`, `reward`, `done`, `info`
- `GET /state` -> `EnvState`

## Task Set and Graders

Three graded tasks with increasing difficulty:

1. `easy` (Ticket Routing)
- Goal: correct route + priority.
- Score weights: route 0.7, priority 0.3.

2. `medium` (Routing + Action Planning)
- Goal: route + priority + next action.
- Score weights: route 0.5, priority 0.2, next_action 0.3.

3. `hard` (Incident Triage + Response Draft)
- Goal: route + priority + next action + response quality.
- Score weights: route 0.35, priority 0.15, next_action 0.2, response_quality 0.3.
- Response quality is deterministic and based on required/forbidden keywords.

All graders are deterministic and return scores in `[0.0, 1.0]`.

## Reward Function (Meaningful and Shaped)

The reward in each `step()` is trajectory-aware:
- Positive for improvement over prior best score.
- Penalty for repeated non-improving attempts.
- Time penalty that increases with step index.
- Terminal success bonus; terminal failure penalty.

This provides partial progress signals and discourages loops.

## Action and Observation Spaces

### Action

`Action` fields:
- `route`: one of billing, technical, account, shipping, safety, other
- `priority`: one of low, medium, high, urgent
- `next_action`: one of resolve, ask_for_info, escalate, refund, monitor
- `response`: free text draft response (10 to 1200 chars)

### Observation

`Observation` fields:
- `task`, `scenario_id`
- `step_index`, `max_steps`
- `customer_message`
- `conversation_history` (attempt summaries)
- `feedback` (grader guidance)
- `score_so_far`

## Repository Structure

```
.
├── app/
│   ├── __init__.py
│   ├── env.py
│   ├── grading.py
│   ├── main.py
│   ├── models.py
│   └── tasks.py
├── Dockerfile
├── inference.py
├── openenv.yaml
├── requirements.txt
└── validate_submission.py
```

## Local Setup

1. Create env and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

3. Optional quick check:

```bash
python validate_submission.py
```

## Baseline Inference Script

The script required by submission is at root as `inference.py`.

It:
- Uses OpenAI client for LLM action generation.
- Reads model endpoint credentials from env vars.
- Runs all 3 tasks and prints reproducible scores.

Expected env vars:
- `OPENENV_URL` (default `http://localhost:7860`)
- `API_BASE_URL` (LLM endpoint)
- `MODEL_NAME` (LLM model id)
- `OPENAI_API_KEY` or `HF_TOKEN` (auth)

Run:

```bash
python inference.py
```

If LLM config is missing or unavailable, the script falls back to deterministic heuristics so the pipeline remains runnable.

## Docker and Hugging Face Spaces

Build and run locally:

```bash
docker build -t openenv-support-triage .
docker run --rm -p 7860:7860 openenv-support-triage
```

This repository is configured for Docker Spaces (see YAML frontmatter at top of this README).

## API Examples

Reset:

```bash
curl -X POST http://localhost:7860/reset \
	-H "content-type: application/json" \
	-d '{"task":"easy","seed":7}'
```

Step:

```bash
curl -X POST http://localhost:7860/step \
	-H "content-type: application/json" \
	-d '{
		"route":"billing",
		"priority":"high",
		"next_action":"refund",
		"response":"Sorry for this issue. We will process the refund and update you within 24 hours."
	}'
```

State:

```bash
curl http://localhost:7860/state
```

## Notes on Determinism

- Graders are rule-based and deterministic.
- Scenario selection is deterministic by cursor unless a seed is provided.
- Baseline script uses fixed temperature and fixed reset seed.
=======
title: Meta
emoji: 🦀
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
short_description: 'for meta hackathon '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 427a3233cc511595c87233605ed0d1b20444c685
