"""SQLAlchemy implementation of the TriageRepository port."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import StaleDataError

from triage_agent.application.ports import MetricsSummary, SimilarTicket
from triage_agent.domain.enums import AffectedScope, Category, Queue, Severity, TriageStatus
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.domain.model import (
    Classification,
    Feedback,
    LlmUsage,
    Ticket,
    Triage,
    TriageDecision,
)
from triage_agent.domain.result import Err, Ok, Result
from triage_agent.infrastructure.persistence.orm import TriageRow
from triage_agent.infrastructure.text_similarity import overlap_score, tokenize

SIMILARITY_WINDOW = 500
LATENCY_WINDOW = 1000


def _to_row(triage: Triage) -> TriageRow:
    c, d, u = triage.classification, triage.decision, triage.usage
    return TriageRow(
        id=triage.id,
        subject_masked=triage.ticket.subject,
        body_masked=triage.ticket.body,
        category=c.category.value,
        model_severity=c.severity.value,
        affected_scope=c.affected_scope.value,
        product_area=c.product_area[:100],
        summary=c.summary[:500],
        suggested_reply=c.suggested_reply,
        confidence=Decimal(str(round(c.confidence, 3))),
        kb_article_ids=",".join(c.kb_article_ids)[:500],
        triage_status=d.status.value,
        queue_name=d.queue.value,
        final_severity=d.severity.value,
        sla_hours=d.sla_hours,
        model_name=u.model_name[:60],
        prompt_version=u.prompt_version,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cost_usd=u.cost_usd,
        latency_ms=u.latency_ms,
        createdat=triage.created_at,
        createdby=triage.created_by,
    )


def _to_domain(row: TriageRow) -> Triage:
    feedback = None
    if row.feedback_category and row.feedback_severity and row.updatedat and row.updatedby:
        feedback = Feedback(
            category=Category(row.feedback_category),
            severity=Severity(row.feedback_severity),
            given_by=row.updatedby,
            given_at=row.updatedat,
        )
    return Triage(
        id=row.id,
        ticket=Ticket(subject=row.subject_masked, body=row.body_masked),
        classification=Classification(
            category=Category(row.category),
            severity=Severity(row.model_severity),
            affected_scope=AffectedScope(row.affected_scope),
            product_area=row.product_area,
            summary=row.summary,
            suggested_reply=row.suggested_reply,
            confidence=float(row.confidence),
            kb_article_ids=tuple(i for i in row.kb_article_ids.split(",") if i),
        ),
        decision=TriageDecision(
            status=TriageStatus(row.triage_status),
            queue=Queue(row.queue_name),
            severity=Severity(row.final_severity),
            sla_hours=row.sla_hours,
        ),
        usage=LlmUsage(
            model_name=row.model_name,
            prompt_version=row.prompt_version,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cost_usd=Decimal(row.cost_usd),
            latency_ms=row.latency_ms,
        ),
        created_at=row.createdat,
        created_by=row.createdby,
        updated_at=row.updatedat,
        updated_by=row.updatedby,
        feedback=feedback,
        version=row.rowversion,
    )


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return float(ordered[index])


class SqlTriageRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def add(self, triage: Triage) -> Triage:
        row = _to_row(triage)
        with self._sessions.begin() as session:
            session.add(row)
            session.flush()
            triage.version = row.rowversion
        return triage

    def get(self, triage_id: str) -> Triage | None:
        with self._sessions() as session:
            row = session.get(TriageRow, triage_id)
            return _to_domain(row) if row else None

    def save_feedback(self, triage: Triage) -> Result[Triage, DomainError]:
        if triage.feedback is None:
            raise ValueError("triage has no feedback to save")
        conflict = Err(DomainError(ErrorCode.TRIAGE_VERSION_CONFLICT, {"expected": triage.version}))
        try:
            with self._sessions.begin() as session:
                row = session.get(TriageRow, triage.id)
                if row is None:
                    return Err(DomainError(ErrorCode.TRIAGE_NOT_FOUND, {"id": triage.id}))
                if row.rowversion != triage.version:
                    return conflict
                row.feedback_category = triage.feedback.category.value
                row.feedback_severity = triage.feedback.severity.value
                row.updatedat = triage.updated_at
                row.updatedby = triage.updated_by
                session.flush()
                triage.version = row.rowversion
        except StaleDataError:
            return conflict
        return Ok(triage)

    def list_recent(self, limit: int) -> list[Triage]:
        with self._sessions() as session:
            rows = session.scalars(
                select(TriageRow).order_by(TriageRow.createdat.desc()).limit(limit)
            ).all()
            return [_to_domain(row) for row in rows]

    def find_similar(self, text: str, limit: int) -> list[SimilarTicket]:
        query_tokens = tokenize(text)
        if not query_tokens:
            return []
        with self._sessions() as session:
            rows = session.execute(
                select(
                    TriageRow.id,
                    TriageRow.subject_masked,
                    TriageRow.summary,
                    TriageRow.category,
                    TriageRow.final_severity,
                )
                .order_by(TriageRow.createdat.desc())
                .limit(SIMILARITY_WINDOW)
            ).all()
        scored = [
            (overlap_score(query_tokens, tokenize(f"{r.subject_masked} {r.summary}")), r)
            for r in rows
        ]
        best = sorted((item for item in scored if item[0] > 0), key=lambda item: -item[0])
        return [
            SimilarTicket(
                triage_id=r.id,
                category=Category(r.category),
                severity=Severity(r.final_severity),
                summary=r.summary,
            )
            for _, r in best[:limit]
        ]

    def metrics_summary(self) -> MetricsSummary:
        with self._sessions() as session:
            totals = session.execute(
                select(
                    func.count(TriageRow.id),
                    func.sum(
                        case((TriageRow.triage_status == TriageStatus.ROUTED.value, 1), else_=0)
                    ),
                    func.avg(TriageRow.latency_ms),
                    func.sum(TriageRow.cost_usd),
                    func.count(TriageRow.feedback_category),
                    func.sum(case((TriageRow.feedback_category == TriageRow.category, 1), else_=0)),
                    func.sum(
                        case((TriageRow.feedback_severity == TriageRow.final_severity, 1), else_=0)
                    ),
                )
            ).one()
            by_category = dict(
                session.execute(
                    select(TriageRow.category, func.count(TriageRow.id)).group_by(
                        TriageRow.category
                    )
                ).all()
            )
            latencies = list(
                session.scalars(
                    select(TriageRow.latency_ms)
                    .order_by(TriageRow.createdat.desc())
                    .limit(LATENCY_WINDOW)
                ).all()
            )

        total, routed, avg_latency, total_cost, feedback_count, cat_ok, sev_ok = totals
        total = int(total or 0)
        total_cost = Decimal(total_cost or 0).quantize(Decimal("0.000001"))
        feedback_count = int(feedback_count or 0)
        return MetricsSummary(
            total=total,
            routed=int(routed or 0),
            needs_review=total - int(routed or 0),
            by_category={str(k): int(v) for k, v in by_category.items()},
            avg_latency_ms=round(float(avg_latency or 0), 1),
            p95_latency_ms=_percentile(latencies, 95),
            total_cost_usd=total_cost,
            avg_cost_usd=(total_cost / total).quantize(Decimal("0.000001"))
            if total
            else Decimal("0"),
            feedback_count=feedback_count,
            category_accuracy=round(int(cat_ok or 0) / feedback_count, 4)
            if feedback_count
            else None,
            severity_accuracy=round(int(sev_ok or 0) / feedback_count, 4)
            if feedback_count
            else None,
        )


def utc_now() -> datetime:
    """Naive UTC timestamp: portable across PostgreSQL, Oracle and SQL Server."""
    return datetime.now(UTC).replace(tzinfo=None)
