"""Ports: what the use cases need from the outside world."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

from triage_agent.domain.enums import Category, Severity
from triage_agent.domain.errors import DomainError
from triage_agent.domain.model import Classification, LlmUsage, Ticket, Triage
from triage_agent.domain.result import Result


@dataclass(frozen=True, slots=True)
class AgentResult:
    classification: Classification
    usage: LlmUsage


class TriageAgent(Protocol):
    def classify(self, ticket: Ticket) -> Result[AgentResult, DomainError]:
        """Classify an already-masked ticket. Raises TechnicalError on infrastructure failure."""
        ...


@dataclass(frozen=True, slots=True)
class KbArticle:
    id: str
    title: str
    summary: str


class KnowledgeBase(Protocol):
    def search(self, query: str, limit: int) -> list[KbArticle]: ...

    def exists(self, article_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class SimilarTicket:
    triage_id: str
    category: Category
    severity: Severity
    summary: str


class SimilarTicketFinder(Protocol):
    def find_similar(self, text: str, limit: int) -> list[SimilarTicket]: ...


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    total: int
    routed: int
    needs_review: int
    by_category: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_cost_usd: Decimal = Decimal("0")
    avg_cost_usd: Decimal = Decimal("0")
    feedback_count: int = 0
    category_accuracy: float | None = None
    severity_accuracy: float | None = None


class TriageRepository(SimilarTicketFinder, Protocol):
    def add(self, triage: Triage) -> Triage: ...

    def get(self, triage_id: str) -> Triage | None: ...

    def save_feedback(self, triage: Triage) -> Result[Triage, DomainError]: ...

    def list_recent(self, limit: int) -> list[Triage]: ...

    def metrics_summary(self) -> MetricsSummary: ...
