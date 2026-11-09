from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from triage_agent.application.ports import AgentResult
from triage_agent.domain.enums import AffectedScope, Category, Severity
from triage_agent.domain.errors import DomainError
from triage_agent.domain.model import Classification, LlmUsage, Ticket
from triage_agent.domain.result import Ok, Result
from triage_agent.infrastructure.persistence.database import build_engine, build_session_factory
from triage_agent.infrastructure.persistence.orm import Base
from triage_agent.infrastructure.persistence.repository import SqlTriageRepository

ROOT = Path(__file__).resolve().parent.parent
FIXED_NOW = datetime(2026, 11, 8, 12, 0, 0)


def make_classification(**overrides: object) -> Classification:
    values: dict[str, object] = {
        "category": Category.ACCESS,
        "severity": Severity.MEDIUM,
        "affected_scope": AffectedScope.SINGLE_USER,
        "product_area": "login",
        "summary": "User cannot log in",
        "suggested_reply": "We are looking into it.",
        "confidence": 0.9,
        "kb_article_ids": ("KB-001",),
    }
    values.update(overrides)
    return Classification(**values)  # type: ignore[arg-type]


def make_usage(**overrides: object) -> LlmUsage:
    values: dict[str, object] = {
        "model_name": "fake-model",
        "prompt_version": "v2",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost_usd": Decimal("0.008000"),
        "latency_ms": 1500,
    }
    values.update(overrides)
    return LlmUsage(**values)  # type: ignore[arg-type]


class FakeAgent:
    """Returns a preset outcome and records the tickets it saw."""

    def __init__(self, outcome: Result[AgentResult, DomainError] | None = None) -> None:
        self.outcome = outcome or Ok(AgentResult(make_classification(), make_usage()))
        self.seen: list[Ticket] = []
        self.error: Exception | None = None

    def classify(self, ticket: Ticket) -> Result[AgentResult, DomainError]:
        self.seen.append(ticket)
        if self.error:
            raise self.error
        return self.outcome


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = build_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    yield build_session_factory(engine)
    engine.dispose()


@pytest.fixture
def repository(session_factory: sessionmaker[Session]) -> SqlTriageRepository:
    return SqlTriageRepository(session_factory)
