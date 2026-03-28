from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TaskLevel = Literal["easy", "medium", "hard"]
RouteLabel = Literal["billing", "technical", "account", "shipping", "safety", "other"]
PriorityLabel = Literal["low", "medium", "high", "urgent"]
NextActionLabel = Literal["resolve", "ask_for_info", "escalate", "refund", "monitor"]


class Action(BaseModel):
    route: RouteLabel = Field(..., description="Team that should own the ticket.")
    priority: PriorityLabel = Field(..., description="Service urgency level.")
    next_action: NextActionLabel = Field(..., description="Immediate follow-up action.")
    response: str = Field(..., min_length=10, max_length=1200)


class Observation(BaseModel):
    task: TaskLevel
    scenario_id: str
    step_index: int
    max_steps: int
    customer_message: str
    conversation_history: List[str]
    feedback: str
    score_so_far: float = Field(..., ge=0.0, le=1.0)


class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    components: Dict[str, float]


class StepInfo(BaseModel):
    latest_score: float = Field(..., ge=0.0, le=1.0)
    best_score: float = Field(..., ge=0.0, le=1.0)
    component_scores: Dict[str, float]
    success: bool
    attempts_remaining: int
    feedback: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: StepInfo


class ResetRequest(BaseModel):
    task: TaskLevel = "easy"
    scenario_id: Optional[str] = None
    seed: Optional[int] = None


class EnvState(BaseModel):
    initialized: bool
    done: bool
    task: Optional[TaskLevel]
    scenario_id: Optional[str]
    step_index: int
    max_steps: int
    best_score: float = Field(..., ge=0.0, le=1.0)
    trajectory: List[Dict[str, object]]
