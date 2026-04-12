"""Built-in support ticket tasks and grading rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

try:
    from ..models import TaskDifficulty, SupportOperation
except ImportError:  # pragma: no cover
    from models import TaskDifficulty, SupportOperation


@dataclass(frozen=True)
class Milestone:
    """A single graded milestone in a task."""

    operation: SupportOperation
    weight: float
    category: Optional[str] = None
    priority: Optional[str] = None
    team: Optional[str] = None
    response_keywords: Tuple[str, ...] = ()
    forbidden_keywords: Tuple[str, ...] = ()
    min_response_length: int = 0
    feedback: str = ""


@dataclass(frozen=True)
class SupportTaskSpec:
    """A complete support task with public ticket details and grader rules."""

    task_name: str
    difficulty: TaskDifficulty
    ticket_id: str
    subject: str
    customer_message: str
    customer_tier: str
    expected_category: str
    expected_priority: str
    description: str
    policy_hint: str
    max_steps: int
    milestones: Tuple[Milestone, ...]
    unsafe_keywords: Tuple[str, ...] = ()
    success_threshold: float = 0.75

    @property
    def allowed_operations(self) -> List[str]:
        ops: List[str] = []
        for milestone in self.milestones:
            if milestone.operation.value not in ops:
                ops.append(milestone.operation.value)
        return ops


TASKS: Tuple[SupportTaskSpec, ...] = (
    SupportTaskSpec(
        task_name="password_reset_easy",
        difficulty=TaskDifficulty.EASY,
        ticket_id="CS-1042",
        subject="Locked out after password change",
        customer_message=(
            "I changed my password this morning and now I cannot sign in to the mobile app. "
            "Please help me get back into my account."
        ),
        customer_tier="standard",
        expected_category="account_access",
        expected_priority="medium",
        description="Classify an account-access issue and send the right reset instructions.",
        policy_hint="Ask the user to verify access before sending the password reset flow.",
        max_steps=4,
        milestones=(
            Milestone(
                operation=SupportOperation.CLASSIFY,
                category="account_access",
                priority="medium",
                weight=0.35,
                feedback="Correctly classified as account access with medium priority.",
            ),
            Milestone(
                operation=SupportOperation.RESPOND,
                response_keywords=("reset", "verification", "link"),
                min_response_length=25,
                weight=0.35,
                feedback="Response includes reset guidance and verification language.",
            ),
            Milestone(
                operation=SupportOperation.CLOSE,
                weight=0.30,
                feedback="Ticket closed after a complete response.",
            ),
        ),
        unsafe_keywords=("ignore", "delete account", "ban user"),
        success_threshold=0.80,
    ),
    SupportTaskSpec(
        task_name="duplicate_charge_medium",
        difficulty=TaskDifficulty.MEDIUM,
        ticket_id="CS-2078",
        subject="Charged twice for the same subscription",
        customer_message=(
            "My card was charged twice for the same monthly subscription renewal. "
            "I need someone to review the duplicate charge and explain the refund process."
        ),
        customer_tier="gold",
        expected_category="billing",
        expected_priority="high",
        description="Resolve a billing dispute while following refund policy and collecting details.",
        policy_hint="Do not promise an instant refund; ask for transaction details first and explain the review flow.",
        max_steps=5,
        milestones=(
            Milestone(
                operation=SupportOperation.CLASSIFY,
                category="billing",
                priority="high",
                weight=0.25,
                feedback="Correctly classified as billing with high priority.",
            ),
            Milestone(
                operation=SupportOperation.REQUEST_INFO,
                response_keywords=("transaction", "date", "last four"),
                min_response_length=30,
                weight=0.25,
                feedback="Collected the missing transaction details.",
            ),
            Milestone(
                operation=SupportOperation.RESPOND,
                response_keywords=("review", "refund", "statement"),
                min_response_length=35,
                weight=0.25,
                feedback="Explained the billing review and refund process.",
            ),
            Milestone(
                operation=SupportOperation.CLOSE,
                weight=0.25,
                feedback="Closed after confirming the next steps.",
            ),
        ),
        unsafe_keywords=("guaranteed refund", "instant refund", "no review needed"),
        success_threshold=0.75,
    ),
    SupportTaskSpec(
        task_name="privacy_deletion_hard",
        difficulty=TaskDifficulty.HARD,
        ticket_id="CS-3911",
        subject="Request to delete personal data",
        customer_message=(
            "Please delete my account and all personal data immediately. I need confirmation today."
        ),
        customer_tier="enterprise",
        expected_category="privacy_request",
        expected_priority="urgent",
        description="Handle a privacy deletion request with verification, escalation, and careful wording.",
        policy_hint="Verify identity before promising deletion. Escalate to the privacy team and avoid making unauthorized promises.",
        max_steps=6,
        milestones=(
            Milestone(
                operation=SupportOperation.CLASSIFY,
                category="privacy_request",
                priority="urgent",
                weight=0.20,
                feedback="Correctly classified as a privacy request with urgent priority.",
            ),
            Milestone(
                operation=SupportOperation.REQUEST_INFO,
                response_keywords=("verify", "identity", "account ownership"),
                min_response_length=35,
                weight=0.20,
                feedback="Requested identity verification before taking action.",
            ),
            Milestone(
                operation=SupportOperation.ESCALATE,
                team="privacy",
                weight=0.30,
                feedback="Escalated to the privacy team.",
            ),
            Milestone(
                operation=SupportOperation.RESPOND,
                response_keywords=("cannot", "review", "privacy"),
                min_response_length=40,
                weight=0.15,
                feedback="Used cautious policy-compliant language.",
            ),
            Milestone(
                operation=SupportOperation.CLOSE,
                weight=0.15,
                feedback="Closed after escalation and safe communication.",
            ),
        ),
        unsafe_keywords=("deleted immediately", "already deleted", "permanently removed without verification"),
        success_threshold=0.80,
    ),
)

TASK_BY_NAME: Dict[str, SupportTaskSpec] = {task.task_name: task for task in TASKS}


def list_task_names() -> List[str]:
    return [task.task_name for task in TASKS]


@lru_cache(maxsize=1)
def default_task_name() -> str:
    return TASKS[0].task_name


def get_task(task_name: Optional[str]) -> SupportTaskSpec:
    if not task_name:
        return TASK_BY_NAME[default_task_name()]
    if task_name not in TASK_BY_NAME:
        raise ValueError(f"Unknown task_name={task_name!r}. Available tasks: {', '.join(list_task_names())}")
    return TASK_BY_NAME[task_name]
