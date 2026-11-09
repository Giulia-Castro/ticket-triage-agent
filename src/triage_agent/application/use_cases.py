"""Use cases orchestrating domain rules and ports."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from triage_agent.application.ports import MetricsSummary, TriageAgent, TriageRepository
from triage_agent.domain.enums import Category, Severity
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.domain.masking import mask_personal_data
from triage_agent.domain.model import Ticket, Triage
from triage_agent.domain.result import Err, Ok, Result
from triage_agent.domain.routing import RoutingPolicy

logger = logging.getLogger(__name__)

type Clock = Callable[[], datetime]
type IdFactory = Callable[[], str]


@dataclass(frozen=True, slots=True)
class TriageCommand:
    subject: str
    body: str
    actor: str


class TriageTicketUseCase:
    def __init__(
        self,
        agent: TriageAgent,
        repository: TriageRepository,
        policy: RoutingPolicy,
        clock: Clock,
        id_factory: IdFactory,
    ) -> None:
        self._agent = agent
        self._repository = repository
        self._policy = policy
        self._clock = clock
        self._id_factory = id_factory

    def execute(self, command: TriageCommand) -> Result[Triage, DomainError]:
        match Ticket.create(command.subject, command.body):
            case Err() as error:
                return error
            case Ok(ticket):
                pass

        masked_subject = mask_personal_data(ticket.subject)
        masked_body = mask_personal_data(ticket.body)
        masked_ticket = ticket.with_text(masked_subject.text, masked_body.text)

        match self._agent.classify(masked_ticket):
            case Err() as error:
                logger.warning("triage.refused", extra={"context": {"code": error.error.code}})
                return error
            case Ok(agent_result):
                pass

        triage = Triage(
            id=self._id_factory(),
            ticket=masked_ticket,
            classification=agent_result.classification,
            decision=self._policy.decide(agent_result.classification),
            usage=agent_result.usage,
            created_at=self._clock(),
            created_by=command.actor,
        )
        saved = self._repository.add(triage)
        logger.info(
            "triage.created",
            extra={
                "context": {
                    "triageId": saved.id,
                    "category": saved.classification.category,
                    "queue": saved.decision.queue,
                    "status": saved.decision.status,
                    "maskedFields": masked_subject.total + masked_body.total,
                    "latencyMs": saved.usage.latency_ms,
                    "costUsd": str(saved.usage.cost_usd),
                }
            },
        )
        return Ok(saved)


class GetTriageUseCase:
    def __init__(self, repository: TriageRepository) -> None:
        self._repository = repository

    def execute(self, triage_id: str, actor: str) -> Result[Triage, DomainError]:
        triage = self._repository.get(triage_id)
        if triage is None:
            return Err(DomainError(ErrorCode.TRIAGE_NOT_FOUND, {"id": triage_id}))
        # Access audit: who read which ticket (text is masked, but access is still traceable).
        logger.info("triage.read", extra={"context": {"triageId": triage_id, "actor": actor}})
        return Ok(triage)


class ListTriagesUseCase:
    def __init__(self, repository: TriageRepository) -> None:
        self._repository = repository

    def execute(self, limit: int, actor: str) -> list[Triage]:
        logger.info("triage.list", extra={"context": {"limit": limit, "actor": actor}})
        return self._repository.list_recent(limit)


@dataclass(frozen=True, slots=True)
class FeedbackCommand:
    triage_id: str
    category: Category
    severity: Severity
    expected_version: int
    actor: str


class SubmitFeedbackUseCase:
    def __init__(self, repository: TriageRepository, clock: Clock) -> None:
        self._repository = repository
        self._clock = clock

    def execute(self, command: FeedbackCommand) -> Result[Triage, DomainError]:
        triage = self._repository.get(command.triage_id)
        if triage is None:
            return Err(DomainError(ErrorCode.TRIAGE_NOT_FOUND, {"id": command.triage_id}))
        match triage.apply_feedback(
            command.category,
            command.severity,
            command.expected_version,
            command.actor,
            self._clock(),
        ):
            case Err() as error:
                return error
            case Ok(updated):
                result = self._repository.save_feedback(updated)
        if result.is_ok():
            logger.info(
                "triage.feedback",
                extra={
                    "context": {
                        "triageId": command.triage_id,
                        "categoryCorrect": updated.category_was_correct,
                        "actor": command.actor,
                    }
                },
            )
        return result


class GetMetricsUseCase:
    def __init__(self, repository: TriageRepository) -> None:
        self._repository = repository

    def execute(self) -> MetricsSummary:
        return self._repository.metrics_summary()
