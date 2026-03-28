from __future__ import annotations

import random
from typing import Dict, List, Optional

from .grading import grade_action
from .models import Action, EnvState, Observation, ResetRequest, Reward, StepInfo, StepResult
from .tasks import SCENARIOS, TASK_CONFIGS, Scenario


class SupportTriageEnv:
    """Deterministic support-triage environment with shaped rewards over attempts."""

    def __init__(self) -> None:
        self._task_cursor: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        self.initialized = False
        self.done = False
        self.task: Optional[str] = None
        self.scenario: Optional[Scenario] = None
        self.step_index = 0
        self.max_steps = 0
        self.best_score = 0.0
        self.feedback = "Call reset() to start a task episode."
        self.trajectory: List[Dict[str, object]] = []

    def _select_scenario(self, req: ResetRequest) -> Scenario:
        candidates = SCENARIOS[req.task]
        if req.scenario_id:
            for item in candidates:
                if item.scenario_id == req.scenario_id:
                    return item
            valid_ids = ", ".join(s.scenario_id for s in candidates)
            raise ValueError(f"Unknown scenario_id '{req.scenario_id}'. Valid values: {valid_ids}")

        if req.seed is not None:
            rng = random.Random(req.seed)
            return rng.choice(candidates)

        cursor = self._task_cursor[req.task] % len(candidates)
        self._task_cursor[req.task] += 1
        return candidates[cursor]

    def _observation(self) -> Observation:
        if self.scenario is None or self.task is None:
            raise RuntimeError("Environment is not initialized.")

        history_lines = [
            (
                f"Attempt {item['attempt']}: route={item['route']}, priority={item['priority']}, "
                f"next_action={item['next_action']}, score={item['score']:.3f}"
            )
            for item in self.trajectory
        ]

        return Observation(
            task=self.task,
            scenario_id=self.scenario.scenario_id,
            step_index=self.step_index,
            max_steps=self.max_steps,
            customer_message=self.scenario.customer_message,
            conversation_history=history_lines,
            feedback=self.feedback,
            score_so_far=round(self.best_score, 4),
        )

    def reset(self, req: ResetRequest) -> Observation:
        scenario = self._select_scenario(req)
        config = TASK_CONFIGS[req.task]

        self.initialized = True
        self.done = False
        self.task = req.task
        self.scenario = scenario
        self.step_index = 0
        self.max_steps = config.max_steps
        self.best_score = 0.0
        self.feedback = "Episode started. Submit an action with route, priority, next_action, and response."
        self.trajectory = []
        return self._observation()

    def step(self, action: Action) -> StepResult:
        if not self.initialized or self.scenario is None or self.task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.done:
            info = StepInfo(
                latest_score=round(self.best_score, 4),
                best_score=round(self.best_score, 4),
                component_scores={},
                success=self.best_score >= TASK_CONFIGS[self.task].success_threshold,
                attempts_remaining=0,
                feedback="Episode is already finished. Call reset() to start again.",
            )
            return StepResult(observation=self._observation(), reward=Reward(value=0.0, components={}), done=True, info=info)

        config = TASK_CONFIGS[self.task]
        grade = grade_action(config, self.scenario, action)

        previous_best = self.best_score
        self.best_score = max(self.best_score, grade.score)

        progress_gain = max(0.0, grade.score - previous_best)
        repeated_miss_penalty = -0.02 if grade.score <= previous_best and self.step_index > 0 else 0.0
        time_penalty = 0.03 * self.step_index

        reward_value = progress_gain + repeated_miss_penalty - time_penalty

        self.step_index += 1
        success = self.best_score >= config.success_threshold
        attempts_remaining = max(0, self.max_steps - self.step_index)
        self.done = success or self.step_index >= self.max_steps

        if self.done and success:
            reward_value += 0.1
        elif self.done and not success:
            reward_value -= 0.05

        reward_value = max(-1.0, min(1.0, round(reward_value, 4)))
        self.feedback = grade.feedback

        self.trajectory.append(
            {
                "attempt": self.step_index,
                "route": action.route,
                "priority": action.priority,
                "next_action": action.next_action,
                "score": round(grade.score, 4),
                "reward": reward_value,
            }
        )

        observation = self._observation()
        reward = Reward(
            value=reward_value,
            components={
                "progress_gain": round(progress_gain, 4),
                "time_penalty": round(-time_penalty, 4),
                "repeated_miss_penalty": round(repeated_miss_penalty, 4),
            },
        )
        info = StepInfo(
            latest_score=round(grade.score, 4),
            best_score=round(self.best_score, 4),
            component_scores=grade.components,
            success=success,
            attempts_remaining=attempts_remaining,
            feedback=grade.feedback,
        )

        return StepResult(observation=observation, reward=reward, done=self.done, info=info)

    def state(self) -> EnvState:
        return EnvState(
            initialized=self.initialized,
            done=self.done,
            task=self.task,
            scenario_id=self.scenario.scenario_id if self.scenario else None,
            step_index=self.step_index,
            max_steps=self.max_steps,
            best_score=round(self.best_score, 4),
            trajectory=self.trajectory,
        )
