from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .env import SupportTriageEnv
from .models import Action, EnvState, Observation, ResetRequest, StepResult

app = FastAPI(
    title="OpenEnv Support Triage Environment",
    version="0.1.0",
    description=(
        "A real-world customer support triage environment with deterministic graders, "
        "trajectory rewards, and OpenEnv-compatible reset/step/state APIs."
    ),
)

env = SupportTriageEnv()


@app.get("/")
def root() -> dict:
    return {
        "name": "openenv-support-triage",
        "status": "ok",
        "routes": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()) -> Observation:
    try:
        return env.reset(req)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.get("/reset", response_model=Observation)
def reset_get(task: str = "easy") -> Observation:
    try:
        req = ResetRequest(task=task)
        return env.reset(req)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return env.step(action)
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    return env.state()
