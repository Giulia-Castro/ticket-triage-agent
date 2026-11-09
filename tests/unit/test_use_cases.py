from __future__ import annotations

from tests.conftest import FIXED_NOW, FakeAgent
from triage_agent.application.use_cases import (
    FeedbackCommand,
    GetMetricsUseCase,
    GetTriageUseCase,
    ListTriagesUseCase,
    SubmitFeedbackUseCase,
    TriageCommand,
    TriageTicketUseCase,
)
from triage_agent.domain.enums import Category, Queue, Severity
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.domain.result import Err, Ok
from triage_agent.domain.routing import RoutingPolicy
from triage_agent.infrastructure.persistence.repository import SqlTriageRepository


def _use_case(agent: FakeAgent, repository: SqlTriageRepository) -> TriageTicketUseCase:
    return TriageTicketUseCase(
        agent=agent,
        repository=repository,
        policy=RoutingPolicy(),
        clock=lambda: FIXED_NOW,
        id_factory=lambda: "fixed-id",
    )


def test_triage_masks_personal_data_before_the_agent(repository: SqlTriageRepository) -> None:
    agent = FakeAgent()
    result = _use_case(agent, repository).execute(
        TriageCommand("Login de joao@exemplo.com", "CPF 123.456.789-09 bloqueado", "tester")
    )
    assert isinstance(result, Ok)
    seen = agent.seen[0]
    assert "joao@exemplo.com" not in seen.subject
    assert "[CPF]" in seen.body
    stored = repository.get("fixed-id")
    assert stored is not None
    assert "[EMAIL]" in stored.ticket.subject
    assert stored.decision.queue is Queue.SERVICE_DESK
    assert stored.created_by == "tester"


def test_triage_returns_validation_error_without_calling_agent(
    repository: SqlTriageRepository,
) -> None:
    agent = FakeAgent()
    result = _use_case(agent, repository).execute(TriageCommand("", "body", "tester"))
    assert isinstance(result, Err)
    assert result.error.code is ErrorCode.TICKET_SUBJECT_REQUIRED
    assert agent.seen == []


def test_triage_propagates_model_refusal(repository: SqlTriageRepository) -> None:
    agent = FakeAgent(outcome=Err(DomainError(ErrorCode.TRIAGE_REFUSED_BY_MODEL)))
    result = _use_case(agent, repository).execute(TriageCommand("s", "b", "tester"))
    assert isinstance(result, Err)
    assert result.error.code is ErrorCode.TRIAGE_REFUSED_BY_MODEL
    assert repository.list_recent(10) == []


def test_get_and_list(repository: SqlTriageRepository) -> None:
    _use_case(FakeAgent(), repository).execute(TriageCommand("s", "b", "tester"))
    assert isinstance(GetTriageUseCase(repository).execute("fixed-id", "tester"), Ok)
    missing = GetTriageUseCase(repository).execute("nope", "tester")
    assert isinstance(missing, Err)
    assert missing.error.code is ErrorCode.TRIAGE_NOT_FOUND
    assert len(ListTriagesUseCase(repository).execute(10, "tester")) == 1


def test_feedback_flow_updates_metrics_and_version(repository: SqlTriageRepository) -> None:
    _use_case(FakeAgent(), repository).execute(TriageCommand("s", "b", "tester"))
    feedback = SubmitFeedbackUseCase(repository, lambda: FIXED_NOW)

    first = feedback.execute(FeedbackCommand("fixed-id", Category.ACCESS, Severity.HIGH, 1, "rev"))
    assert isinstance(first, Ok)
    assert first.value.version == 2

    stale = feedback.execute(FeedbackCommand("fixed-id", Category.BUG, Severity.HIGH, 1, "rev"))
    assert isinstance(stale, Err)
    assert stale.error.code is ErrorCode.TRIAGE_VERSION_CONFLICT

    missing = feedback.execute(FeedbackCommand("nope", Category.BUG, Severity.HIGH, 1, "rev"))
    assert isinstance(missing, Err)

    metrics = GetMetricsUseCase(repository).execute()
    assert metrics.total == 1
    assert metrics.feedback_count == 1
    assert metrics.category_accuracy == 1.0
    assert metrics.severity_accuracy == 0.0
