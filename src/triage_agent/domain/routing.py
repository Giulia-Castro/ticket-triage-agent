"""Deterministic routing rules applied on top of the model's classification.

The LLM interprets free text; the business rules that decide queue, final
severity and SLA stay in plain, testable code.
"""

from __future__ import annotations

from dataclasses import dataclass

from triage_agent.domain.enums import AffectedScope, Category, Queue, Severity, TriageStatus
from triage_agent.domain.model import Classification, TriageDecision

_QUEUE_BY_CATEGORY: dict[Category, Queue] = {
    Category.ACCESS: Queue.SERVICE_DESK,
    Category.BILLING: Queue.FINANCE,
    Category.BUG: Queue.ENGINEERING,
    Category.PERFORMANCE: Queue.SRE,
    Category.INTEGRATION: Queue.INTEGRATIONS,
    Category.FEATURE_REQUEST: Queue.PRODUCT,
}

_SLA_HOURS: dict[Severity, int] = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 8,
    Severity.MEDIUM: 24,
    Severity.LOW: 72,
}

_ESCALATE_ON_WIDE_IMPACT = frozenset(
    {Category.ACCESS, Category.BUG, Category.PERFORMANCE, Category.INTEGRATION}
)


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    confidence_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

    def decide(self, classification: Classification) -> TriageDecision:
        severity = self._final_severity(classification)
        if classification.confidence < self.confidence_threshold:
            status, queue = TriageStatus.NEEDS_REVIEW, Queue.HUMAN_REVIEW
        else:
            status, queue = TriageStatus.ROUTED, _QUEUE_BY_CATEGORY[classification.category]
        return TriageDecision(
            status=status, queue=queue, severity=severity, sla_hours=_SLA_HOURS[severity]
        )

    @staticmethod
    def _final_severity(classification: Classification) -> Severity:
        if classification.category is Category.FEATURE_REQUEST:
            return Severity.LOW
        if (
            classification.affected_scope is AffectedScope.ALL_USERS
            and classification.category in _ESCALATE_ON_WIDE_IMPACT
        ):
            return Severity.highest(classification.severity, Severity.HIGH)
        return classification.severity
