"""HTTP DTOs. Responses never echo the ticket text (data minimization)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from triage_agent.application.ports import MetricsSummary
from triage_agent.domain.enums import AffectedScope, Category, Queue, Severity, TriageStatus
from triage_agent.domain.model import Triage

# Transport limits protect the server; business limits live in the domain.
MAX_TRANSPORT_SUBJECT = 1_000
MAX_TRANSPORT_BODY = 20_000


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TriageRequest(_CamelModel):
    subject: str = Field(
        max_length=MAX_TRANSPORT_SUBJECT, examples=["Não consigo entrar no sistema"]
    )
    body: str = Field(
        max_length=MAX_TRANSPORT_BODY,
        examples=["Desde hoje cedo aparece 'senha inválida', mas a senha está correta."],
    )


class FeedbackRequest(_CamelModel):
    category: Category
    severity: Severity
    version: int = Field(ge=1, description="Current version of the triage (optimistic lock).")


class UsageResponse(_CamelModel):
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: int


class FeedbackResponse(_CamelModel):
    category: Category
    severity: Severity
    given_at: datetime


class TriageResponse(_CamelModel):
    id: str
    status: TriageStatus
    queue: Queue
    category: Category
    severity: Severity
    model_severity: Severity
    affected_scope: AffectedScope
    sla_hours: int
    product_area: str
    summary: str
    suggested_reply: str
    confidence: float
    kb_article_ids: list[str]
    usage: UsageResponse
    feedback: FeedbackResponse | None
    version: int
    created_at: datetime

    @classmethod
    def from_domain(cls, triage: Triage) -> TriageResponse:
        c, d, u = triage.classification, triage.decision, triage.usage
        return cls(
            id=triage.id,
            status=d.status,
            queue=d.queue,
            category=c.category,
            severity=d.severity,
            model_severity=c.severity,
            affected_scope=c.affected_scope,
            sla_hours=d.sla_hours,
            product_area=c.product_area,
            summary=c.summary,
            suggested_reply=c.suggested_reply,
            confidence=c.confidence,
            kb_article_ids=list(c.kb_article_ids),
            usage=UsageResponse(
                model=u.model_name,
                prompt_version=u.prompt_version,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cost_usd=u.cost_usd,
                latency_ms=u.latency_ms,
            ),
            feedback=(
                FeedbackResponse(
                    category=triage.feedback.category,
                    severity=triage.feedback.severity,
                    given_at=triage.feedback.given_at,
                )
                if triage.feedback
                else None
            ),
            version=triage.version,
            created_at=triage.created_at,
        )


class MetricsResponse(_CamelModel):
    total: int
    routed: int
    needs_review: int
    by_category: dict[str, int]
    avg_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: Decimal
    avg_cost_usd: Decimal
    feedback_count: int
    category_accuracy: float | None
    severity_accuracy: float | None

    @classmethod
    def from_summary(cls, summary: MetricsSummary) -> MetricsResponse:
        return cls(
            total=summary.total,
            routed=summary.routed,
            needs_review=summary.needs_review,
            by_category=summary.by_category,
            avg_latency_ms=summary.avg_latency_ms,
            p95_latency_ms=summary.p95_latency_ms,
            total_cost_usd=summary.total_cost_usd,
            avg_cost_usd=summary.avg_cost_usd,
            feedback_count=summary.feedback_count,
            category_accuracy=summary.category_accuracy,
            severity_accuracy=summary.severity_accuracy,
        )
