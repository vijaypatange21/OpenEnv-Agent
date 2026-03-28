from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import TaskLevel


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    customer_message: str
    target_route: str
    target_priority: str
    target_next_action: str
    required_keywords: List[str]
    forbidden_keywords: List[str]


@dataclass(frozen=True)
class TaskConfig:
    level: TaskLevel
    title: str
    description: str
    max_steps: int
    success_threshold: float
    component_weights: Dict[str, float]


SCENARIOS: Dict[TaskLevel, List[Scenario]] = {
    "easy": [
        Scenario(
            scenario_id="easy-1",
            customer_message=(
                "I was charged twice for order #71821. I can see both transactions on my card. "
                "Please fix this today because I need that money for rent."
            ),
            target_route="billing",
            target_priority="high",
            target_next_action="refund",
            required_keywords=["sorry", "refund", "24 hours"],
            forbidden_keywords=["ignore", "no issue"],
        ),
        Scenario(
            scenario_id="easy-2",
            customer_message=(
                "I moved to a new city and cannot log in because the two-factor code still goes to "
                "my old phone. Can you help me recover access?"
            ),
            target_route="account",
            target_priority="medium",
            target_next_action="ask_for_info",
            required_keywords=["verify", "identity", "help"],
            forbidden_keywords=["share password"],
        ),
        Scenario(
            scenario_id="easy-3",
            customer_message=(
                "My order says delivered but nothing arrived at my address. This is the second time "
                "this month."
            ),
            target_route="shipping",
            target_priority="high",
            target_next_action="escalate",
            required_keywords=["carrier", "investigate", "update"],
            forbidden_keywords=["not our problem"],
        ),
    ],
    "medium": [
        Scenario(
            scenario_id="medium-1",
            customer_message=(
                "Our team cannot export invoices from your dashboard after yesterday's release. "
                "Accounting closes in two hours and this blocks payroll."
            ),
            target_route="technical",
            target_priority="urgent",
            target_next_action="escalate",
            required_keywords=["incident", "engineer", "eta"],
            forbidden_keywords=["wait a week"],
        ),
        Scenario(
            scenario_id="medium-2",
            customer_message=(
                "I need to cancel a subscription that renewed this morning. I had already emailed "
                "support last week, so I expect a refund."
            ),
            target_route="billing",
            target_priority="medium",
            target_next_action="refund",
            required_keywords=["policy", "refund", "confirmation"],
            forbidden_keywords=["cannot help"],
        ),
        Scenario(
            scenario_id="medium-3",
            customer_message=(
                "The app crashes whenever we upload a CSV larger than 20MB. We are on the enterprise "
                "plan and this blocks onboarding for 40 users."
            ),
            target_route="technical",
            target_priority="high",
            target_next_action="escalate",
            required_keywords=["reproduce", "logs", "priority"],
            forbidden_keywords=["downgrade"],
        ),
    ],
    "hard": [
        Scenario(
            scenario_id="hard-1",
            customer_message=(
                "A user reported that private documents from another account briefly appeared in their "
                "search results. We disabled sharing links, but we need immediate guidance and an audit trail."
            ),
            target_route="safety",
            target_priority="urgent",
            target_next_action="escalate",
            required_keywords=["security", "contain", "audit", "timeline"],
            forbidden_keywords=["public forum", "screenshot online"],
        ),
        Scenario(
            scenario_id="hard-2",
            customer_message=(
                "I got an email from your domain asking for my credit card details to keep my account "
                "active. Is this real? If not, what should I do now?"
            ),
            target_route="safety",
            target_priority="high",
            target_next_action="monitor",
            required_keywords=["phishing", "do not share", "report", "secure"],
            forbidden_keywords=["send card", "it is safe"],
        ),
        Scenario(
            scenario_id="hard-3",
            customer_message=(
                "We operate in healthcare and noticed patient records exported without mandatory fields. "
                "This could violate compliance obligations. We need corrective action and communication steps."
            ),
            target_route="technical",
            target_priority="urgent",
            target_next_action="escalate",
            required_keywords=["compliance", "incident", "root cause", "next update"],
            forbidden_keywords=["ignore", "no risk"],
        ),
    ],
}


TASK_CONFIGS: Dict[TaskLevel, TaskConfig] = {
    "easy": TaskConfig(
        level="easy",
        title="Ticket Routing",
        description="Pick correct route and priority for a single customer ticket.",
        max_steps=3,
        success_threshold=0.95,
        component_weights={
            "route": 0.7,
            "priority": 0.3,
        },
    ),
    "medium": TaskConfig(
        level="medium",
        title="Routing + Action Planning",
        description="Pick route, urgency, and immediate operational action.",
        max_steps=4,
        success_threshold=0.95,
        component_weights={
            "route": 0.5,
            "priority": 0.2,
            "next_action": 0.3,
        },
    ),
    "hard": TaskConfig(
        level="hard",
        title="Incident Triage + Response Draft",
        description="Handle high-risk support incidents with compliant communication.",
        max_steps=5,
        success_threshold=0.9,
        component_weights={
            "route": 0.35,
            "priority": 0.15,
            "next_action": 0.2,
            "response_quality": 0.3,
        },
    ),
}
