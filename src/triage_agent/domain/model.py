"""Entities and value objects of the triage domain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from triage_agent.domain.enums import AffectedScope, Category, Queue, Severity, TriageStatus
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.domain.result import Err, Ok, Result

MAX_SUBJECT_LENGTH = 200
MAX_BODY_LENGTH = 8000


@dataclass(frozen=True, slots=True)
class Ticket:
    """A support ticket as received. Build it with `Ticket.create` to enforce invariants."""

    subject: str
    body: str

    @classmethod
    def create(cls, subject: str, body: str) -> Result[Ticket, DomainError]:
        subject = (subject or "").strip()
        body = (body or "").strip()
        if not subject:
            return Err(DomainError(ErrorCode.TICKET_SUBJECT_REQUIRED))
        if len(subject) > MAX_SUBJECT_LENGTH:
            return Err(DomainError(ErrorCode.TICKET_SUBJECT_TOO_LONG, {"max": MAX_SUBJECT_LENGTH}))
        if not body:
            return Err(DomainError(ErrorCode.TICKET_BODY_REQUIRED))
        if len(body) > MAX_BODY_LENGTH:
            return Err(DomainError(ErrorCode.TICKET_BODY_TOO_LONG, {"max": MAX_BODY_LENGTH}))
        return Ok(cls(subject=subject, body=body))

    def with_text(self, subject: str, body: str) -> Ticket:
        return replace(self, subject=subject, body=body)


@dataclass(frozen=True, slots=True)
class Classification:
    """What the model (or any classifier) concluded about a ticket."""

    category: Category
    severity: Severity
    affected_scope: AffectedScope
    product_area: str
    summary: str
    suggested_reply: str
    confidence: float
    kb_article_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TriageDecision:
    """Deterministic routing outcome produced by the RoutingPolicy."""

    status: TriageStatus
    queue: Queue
    severity: Severity
    sla_hours: int


@dataclass(frozen=True, slots=True)
class LlmUsage:
    model_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: int


@dataclass(frozen=True, slots=True)
class Feedback:
    """Human correction used to measure real-world accuracy."""

    category: Category
    severity: Severity
    given_by: str
    given_at: datetime


@dataclass(slots=True)
class Triage:
    """Aggregate root. `ticket` holds text already masked for personal data."""

    id: str
    ticket: Ticket
    classification: Classification
    decision: TriageDecision
    usage: LlmUsage
    created_at: datetime
    created_by: str
    updated_at: datetime | None = None
    updated_by: str | None = None
    feedback: Feedback | None = None
    version: int = 1

    def apply_feedback(
        self,
        category: Category,
        severity: Severity,
        expected_version: int,
        actor: str,
        now: datetime,
    ) -> Result[Triage, DomainError]:
        if expected_version != self.version:
            return Err(
                DomainError(
                    ErrorCode.TRIAGE_VERSION_CONFLICT,
                    {"expected": expected_version, "current": self.version},
                )
            )
        self.feedback = Feedback(category=category, severity=severity, given_by=actor, given_at=now)
        self.updated_at = now
        self.updated_by = actor
        return Ok(self)

    @property
    def category_was_correct(self) -> bool | None:
        if self.feedback is None:
            return None
        return self.feedback.category == self.classification.category
