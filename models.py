# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the customer-support triage environment."""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel
from pydantic import Field


class TaskDifficulty(str, Enum):
    """Difficulty tier for the built-in tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SupportOperation(str, Enum):
    """High-level action types available to the agent."""

    CLASSIFY = "classify"
    RESPOND = "respond"
    REQUEST_INFO = "request_info"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    CLOSE = "close"


class ScalerAction(Action):
    """Structured action for triaging and resolving a support ticket."""

    operation: SupportOperation = Field(..., description="The high-level action to take")
    category: Optional[str] = Field(default=None, description="Predicted ticket category")
    priority: Optional[str] = Field(default=None, description="Predicted ticket priority")
    team: Optional[str] = Field(default=None, description="Escalation target team")
    response: Optional[str] = Field(default=None, description="Customer-facing response or follow-up message")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Agent confidence in the action")
    notes: Optional[str] = Field(default=None, description="Optional internal notes")
    message: Optional[str] = Field(default=None, description="Legacy free-form text action, used as response fallback")


class ScalerReward(BaseModel):
    """Typed reward breakdown returned inside the observation."""

    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized progress contribution")
    policy: float = Field(default=0.0, ge=0.0, le=1.0, description="Policy-compliance contribution")
    quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Communication quality contribution")
    penalty: float = Field(default=0.0, ge=0.0, le=1.0, description="Penalty applied to the action")
    total: float = Field(default=0.0, ge=0.0, le=1.0, description="Final shaped reward for the step")


class ScalerObservation(Observation):
    """Public environment observation for a support ticket episode."""

    task_name: str = Field(default="", description="Active task identifier")
    difficulty: TaskDifficulty = Field(default=TaskDifficulty.EASY, description="Task difficulty tier")
    ticket_id: str = Field(default="", description="Public ticket identifier")
    subject: str = Field(default="", description="Ticket subject line")
    customer_message: str = Field(default="", description="Customer-provided text")
    customer_tier: str = Field(default="", description="Customer segment or account tier")
    current_stage: str = Field(default="triage", description="Current milestone stage")
    allowed_operations: List[str] = Field(default_factory=list, description="Operations that make sense next")
    step_count: int = Field(default=0, ge=0, description="Current step count")
    remaining_steps: int = Field(default=0, ge=0, description="Remaining steps before timeout")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Cumulative normalized progress")
    last_action_summary: str = Field(default="", description="Summary of the most recent action")
    last_feedback: str = Field(default="", description="Short grader feedback for the last action")
    reward_breakdown: ScalerReward = Field(default_factory=ScalerReward, description="Reward breakdown")
    history_excerpt: List[str] = Field(default_factory=list, description="Recent action history")
    task_info: Dict[str, Any] = Field(default_factory=dict, description="Task metadata that survives serialization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional structured metadata")


class ScalerState(State):
    """Episode state exposed through the state endpoint."""

    task_name: str = Field(default="", description="Active task identifier")
    difficulty: TaskDifficulty = Field(default=TaskDifficulty.EASY, description="Task difficulty tier")
    step_count: int = Field(default=0, ge=0, description="Number of executed steps")
    current_stage: str = Field(default="triage", description="Current milestone stage")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized cumulative progress")
    resolved: bool = Field(default=False, description="Whether the case has been resolved")
    closed: bool = Field(default=False, description="Whether the ticket has been closed")
    final_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Final normalized score so far")
    last_operation: Optional[SupportOperation] = Field(default=None, description="Most recent operation")
    history_length: int = Field(default=0, ge=0, description="Number of recorded actions")
    last_feedback: str = Field(default="", description="Last grader message")
