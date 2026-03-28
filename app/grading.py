from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import Action
from .tasks import Scenario, TaskConfig


@dataclass(frozen=True)
class GradeResult:
    score: float
    components: Dict[str, float]
    feedback: str


def _safe_text(text: str) -> str:
    return " ".join(text.lower().split())


def _keyword_fraction(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    text_l = _safe_text(text)
    found = sum(1 for kw in keywords if kw.lower() in text_l)
    return found / float(len(keywords))


def grade_action(config: TaskConfig, scenario: Scenario, action: Action) -> GradeResult:
    components: Dict[str, float] = {}
    feedback_parts: List[str] = []

    route_score = 1.0 if action.route == scenario.target_route else 0.0
    components["route"] = route_score
    if route_score < 1.0:
        feedback_parts.append("Route does not match the owning team.")

    priority_score = 1.0 if action.priority == scenario.target_priority else 0.0
    components["priority"] = priority_score
    if priority_score < 1.0:
        feedback_parts.append("Priority level is off for the reported impact.")

    next_action_score = 1.0 if action.next_action == scenario.target_next_action else 0.0
    if "next_action" in config.component_weights:
        components["next_action"] = next_action_score
        if next_action_score < 1.0:
            feedback_parts.append("Next action should better match operational policy.")

    if "response_quality" in config.component_weights:
        required_hit_rate = _keyword_fraction(action.response, scenario.required_keywords)
        forbidden_rate = _keyword_fraction(action.response, scenario.forbidden_keywords)
        response_quality = max(0.0, required_hit_rate - forbidden_rate)
        components["response_quality"] = round(response_quality, 4)

        if required_hit_rate < 1.0:
            feedback_parts.append("Response misses key policy or assurance details.")
        if forbidden_rate > 0.0:
            feedback_parts.append("Response contains unsafe or non-compliant wording.")

    weighted_score = 0.0
    for name, weight in config.component_weights.items():
        weighted_score += components.get(name, 0.0) * weight

    weighted_score = max(0.0, min(1.0, round(weighted_score, 4)))
    if not feedback_parts:
        feedback_parts.append("Strong triage decision. Continue with the same policy alignment.")

    return GradeResult(
        score=weighted_score,
        components=components,
        feedback=" ".join(feedback_parts),
    )
