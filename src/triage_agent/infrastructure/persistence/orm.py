"""ORM mapping.

Portability rules: lowercase identifiers (SQLAlchemy never quotes them, so they
fold to the database's default case), names up to 30 characters, no reserved
words, enums stored as strings, audit columns and an optimistic-lock version.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TriageRow(Base):
    __tablename__ = "ticket_triage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_masked: Mapped[str] = mapped_column(String(200))
    body_masked: Mapped[str] = mapped_column(Text)

    category: Mapped[str] = mapped_column(String(30))
    model_severity: Mapped[str] = mapped_column(String(20))
    affected_scope: Mapped[str] = mapped_column(String(20))
    product_area: Mapped[str] = mapped_column(String(100))
    summary: Mapped[str] = mapped_column(String(500))
    suggested_reply: Mapped[str] = mapped_column(Text)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3))
    kb_article_ids: Mapped[str] = mapped_column(String(500))

    triage_status: Mapped[str] = mapped_column(String(20))
    queue_name: Mapped[str] = mapped_column(String(30))
    final_severity: Mapped[str] = mapped_column(String(20))
    sla_hours: Mapped[int] = mapped_column(Integer)

    model_name: Mapped[str] = mapped_column(String(60))
    prompt_version: Mapped[str] = mapped_column(String(20))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int] = mapped_column(Integer)

    feedback_category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    feedback_severity: Mapped[str | None] = mapped_column(String(20), nullable=True)

    createdat: Mapped[datetime] = mapped_column(DateTime)
    createdby: Mapped[str] = mapped_column(String(60))
    updatedat: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updatedby: Mapped[str | None] = mapped_column(String(60), nullable=True)
    rowversion: Mapped[int] = mapped_column(Integer, nullable=False)

    __mapper_args__ = {"version_id_col": rowversion}
    __table_args__ = (
        Index("ix_ticket_triage_createdat", "createdat"),
        Index("ix_ticket_triage_category", "category"),
    )
