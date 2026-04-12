"""Support-ticket triage environment for OpenEnv."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ScalerAction,
        ScalerObservation,
        ScalerReward,
        ScalerState,
        SupportOperation,
    )
    from .tasks import Milestone, SupportTaskSpec, default_task_name, get_task
except ImportError:  # pragma: no cover
    from models import (
        ScalerAction,
        ScalerObservation,
        ScalerReward,
        ScalerState,
        SupportOperation,
    )
    from server.tasks import Milestone, SupportTaskSpec, default_task_name, get_task


class ScalerEnvironment(Environment[ScalerAction, ScalerObservation, ScalerState]):
    """A realistic customer-support triage environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: Optional[str] = None):
        super().__init__()
        self._default_task_name = task_name or os.getenv("SCALER_TASK_NAME") or default_task_name()
        self._task: SupportTaskSpec = get_task(self._default_task_name)
        self._state = self._new_state()
        self._history: List[str] = []
        self._best_fraction_by_stage: List[float] = []
        self._milestone_index = 0
        self._completed_weight = 0.0
        self._penalty_total = 0.0
        self._last_action_summary = ""
        self._last_feedback = ""
        self._terminated_reason = ""
        self._seed: Optional[int] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> ScalerObservation:
        self._seed = seed
        self._task = get_task(task_name or self._default_task_name)
        self._state = self._new_state(episode_id=episode_id)
        self._history = []
        self._best_fraction_by_stage = [0.0 for _ in self._task.milestones]
        self._milestone_index = 0
        self._completed_weight = 0.0
        self._penalty_total = 0.0
        self._last_action_summary = ""
        self._last_feedback = ""
        self._terminated_reason = ""

        return self._build_observation(
            reward=0.0,
            done=False,
            reward_breakdown=ScalerReward(),
            info={"event": "reset", **kwargs},
        )

    def step(self, action: ScalerAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ScalerObservation:  # type: ignore[override]
        if self._state.closed or self._state.resolved:
            return self._build_observation(
                reward=0.0,
                done=True,
                reward_breakdown=ScalerReward(),
                info={"event": "already_terminated", "reason": self._terminated_reason, **kwargs},
            )

        previous_visible_progress = self._visible_progress()
        self._state.step_count += 1
        self._state.last_operation = action.operation

        milestone = self._current_milestone()
        summary = self._summarize_action(action)
        self._last_action_summary = summary

        fraction, feedback, penalty, unsafe = self._score_milestone(action, milestone)
        self._history.append(summary)
        self._last_feedback = feedback
        self._state.last_feedback = feedback

        if unsafe:
            self._terminated_reason = feedback
            self._state.closed = True
            self._penalty_total = min(1.0, self._penalty_total + penalty)
            self._state.history_length = len(self._history)
            self._state.progress = self._visible_progress()
            self._state.final_score = self._final_score()
            return self._build_observation(
                reward=0.0,
                done=True,
                reward_breakdown=ScalerReward(progress=0.0, policy=0.0, quality=0.0, penalty=penalty, total=0.0),
                info={"event": "unsafe_action", "reason": feedback, **kwargs},
            )

        if action.operation == SupportOperation.CLOSE and self._milestone_index < len(self._task.milestones) - 1:
            penalty += 0.15
            self._terminated_reason = "premature_close"
            self._state.closed = True
            self._penalty_total = min(1.0, self._penalty_total + penalty)
            self._state.history_length = len(self._history)
            self._state.progress = self._visible_progress()
            self._state.final_score = self._final_score()
            return self._build_observation(
                reward=0.0,
                done=True,
                reward_breakdown=ScalerReward(progress=0.0, policy=0.0, quality=0.0, penalty=penalty, total=0.0),
                info={"event": "premature_close", "reason": feedback, **kwargs},
            )

        self._best_fraction_by_stage[self._milestone_index] = max(
            self._best_fraction_by_stage[self._milestone_index], fraction
        )
        if fraction >= 1.0 and action.operation == milestone.operation:
            self._completed_weight = min(1.0, self._completed_weight + milestone.weight)
            self._milestone_index += 1
            if self._milestone_index >= len(self._task.milestones):
                self._state.resolved = True
                self._state.closed = True
                self._terminated_reason = self._terminated_reason or "all_milestones_completed"

        current_visible_progress = self._visible_progress()
        delta_progress = max(0.0, current_visible_progress - previous_visible_progress)
        self._penalty_total = min(1.0, self._penalty_total + penalty)
        step_reward = max(0.0, min(1.0, delta_progress - penalty))

        reward_breakdown = ScalerReward(
            progress=delta_progress,
            policy=1.0 if action.operation == milestone.operation else 0.0,
            quality=self._quality_score(action, milestone),
            penalty=penalty,
            total=step_reward,
        )

        self._state.current_stage = self._current_stage_name()
        self._state.history_length = len(self._history)
        self._state.progress = current_visible_progress
        self._state.final_score = self._final_score()

        done = self._state.closed or self._state.resolved or self._state.step_count >= self._task.max_steps
        if self._state.step_count >= self._task.max_steps and not done:
            self._terminated_reason = self._terminated_reason or "step_limit_reached"
            self._state.closed = True
            self._state.final_score = self._final_score(extra_penalty=0.05)
            reward_breakdown.total = max(0.0, min(1.0, reward_breakdown.total - 0.05))
            done = True

        return self._build_observation(
            reward=reward_breakdown.total,
            done=done,
            reward_breakdown=reward_breakdown,
            info={
                "event": "step",
                "milestone_index": self._milestone_index,
                "completed_weight": self._completed_weight,
                "visible_progress": self._state.progress,
                "terminated_reason": self._terminated_reason,
                **kwargs,
            },
        )

    @property
    def state(self) -> ScalerState:
        return self._state

    def _new_state(self, episode_id: Optional[str] = None) -> ScalerState:
        return ScalerState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task.task_name,
            difficulty=self._task.difficulty,
            current_stage=self._task.milestones[0].operation.value if self._task.milestones else "triage",
            progress=0.0,
            resolved=False,
            closed=False,
            final_score=0.0,
            last_operation=None,
            history_length=0,
            last_feedback="",
        )

    def _current_milestone(self) -> Milestone:
        if not self._task.milestones:
            raise RuntimeError("Task has no milestones configured.")
        index = min(self._milestone_index, len(self._task.milestones) - 1)
        return self._task.milestones[index]

    def _visible_progress(self) -> float:
        if not self._task.milestones:
            return 0.0
        if self._milestone_index >= len(self._task.milestones):
            return 1.0
        current_weight = self._task.milestones[self._milestone_index].weight
        current_fraction = self._best_fraction_by_stage[self._milestone_index]
        return max(0.0, min(1.0, self._completed_weight + current_weight * current_fraction))

    def _final_score(self, extra_penalty: float = 0.0) -> float:
        score = self._visible_progress() - min(1.0, self._penalty_total + extra_penalty)
        return max(0.0, min(1.0, score))

    def _current_stage_name(self) -> str:
        if self._state.resolved:
            return "resolved"
        if self._state.closed:
            return "closed"
        if not self._task.milestones:
            return "triage"
        index = min(self._milestone_index, len(self._task.milestones) - 1)
        return self._task.milestones[index].operation.value

    def _score_milestone(self, action: ScalerAction, milestone: Milestone) -> Tuple[float, str, float, bool]:
        response = (action.response or action.message or "").strip().lower()
        penalty = 0.0
        unsafe = False
        feedback = milestone.feedback or "Action processed."

        if self._contains_unsafe_text(response):
            penalty += 0.35
            unsafe = True
            return 0.0, "Unsafe or destructive wording detected.", penalty, unsafe

        if action.operation != milestone.operation:
            penalty += 0.08
            return 0.0, f"Expected {milestone.operation.value} at this stage.", penalty, unsafe

        checks: List[float] = [1.0]
        if milestone.category is not None:
            checks.append(1.0 if self._normalize(action.category) == self._normalize(milestone.category) else 0.0)
        if milestone.priority is not None:
            checks.append(1.0 if self._normalize(action.priority) == self._normalize(milestone.priority) else 0.0)
        if milestone.team is not None:
            checks.append(1.0 if self._normalize(action.team) == self._normalize(milestone.team) else 0.0)
        if milestone.response_keywords:
            checks.append(self._keyword_fraction(response, milestone.response_keywords))
        if milestone.min_response_length:
            checks.append(min(len(response) / float(milestone.min_response_length), 1.0))

        fraction = max(0.0, min(1.0, sum(checks) / len(checks)))
        if fraction >= 1.0:
            feedback = milestone.feedback or feedback
        elif fraction >= 0.5:
            feedback = f"Partial progress: {feedback}"
        else:
            feedback = f"Low confidence on this milestone: {feedback}"
        return fraction, feedback, penalty, unsafe

    def _quality_score(self, action: ScalerAction, milestone: Milestone) -> float:
        response = (action.response or action.message or "").strip()
        if not response or milestone.min_response_length <= 0:
            return 0.0
        return max(0.0, min(1.0, min(len(response) / float(milestone.min_response_length), 1.0) * 0.15))

    def _contains_unsafe_text(self, response: str) -> bool:
        lowered = response.lower()
        all_bad = tuple(self._task.unsafe_keywords) + tuple(
            keyword for milestone in self._task.milestones for keyword in milestone.forbidden_keywords
        )
        return any(keyword in lowered for keyword in all_bad)

    @staticmethod
    def _keyword_fraction(text: str, keywords: Tuple[str, ...]) -> float:
        if not keywords:
            return 1.0
        matches = sum(1 for keyword in keywords if keyword.lower() in text)
        return matches / float(len(keywords))

    @staticmethod
    def _normalize(value: Optional[str]) -> str:
        return (value or "").strip().lower().replace(" ", "_")

    def _summarize_action(self, action: ScalerAction) -> str:
        payload = action.model_dump(exclude_none=True)
        if "response" not in payload and "message" in payload:
            payload["response"] = payload["message"]
        return str(payload)

    def _build_observation(
        self,
        reward: float,
        done: bool,
        reward_breakdown: ScalerReward,
        info: Dict[str, Any],
    ) -> ScalerObservation:
        remaining_steps = max(0, self._task.max_steps - self._state.step_count)
        return ScalerObservation(
            done=done,
            reward=reward,
            task_name=self._task.task_name,
            difficulty=self._task.difficulty,
            ticket_id=self._task.ticket_id,
            subject=self._task.subject,
            customer_message=self._task.customer_message,
            customer_tier=self._task.customer_tier,
            current_stage=self._current_stage_name(),
            allowed_operations=self._task.allowed_operations,
            step_count=self._state.step_count,
            remaining_steps=remaining_steps,
            progress=self._state.progress,
            last_action_summary=self._last_action_summary,
            last_feedback=self._last_feedback,
            reward_breakdown=reward_breakdown,
            history_excerpt=self._history[-4:],
            task_info={
                "task_description": self._task.description,
                "policy_hint": self._task.policy_hint,
                "expected_category": self._task.expected_category,
                "expected_priority": self._task.expected_priority,
                "max_steps": self._task.max_steps,
                "terminating_reason": self._terminated_reason,
                "seed": self._seed,
                **info,
            },
            metadata={
                "task_description": self._task.description,
                "policy_hint": self._task.policy_hint,
                "expected_category": self._task.expected_category,
                "expected_priority": self._task.expected_priority,
                "max_steps": self._task.max_steps,
                "terminating_reason": self._terminated_reason,
                "seed": self._seed,
                **info,
            },
        )
