from __future__ import annotations

import pytest

from tests.conftest import FIXED_NOW, make_classification, make_usage
from triage_agent.domain.enums import AffectedScope, Category, Queue, Severity, TriageStatus
from triage_agent.domain.errors import ErrorCode
from triage_agent.domain.masking import mask_personal_data
from triage_agent.domain.model import MAX_BODY_LENGTH, MAX_SUBJECT_LENGTH, Ticket, Triage
from triage_agent.domain.result import Err, Ok
from triage_agent.domain.routing import RoutingPolicy

# ---------------------------------------------------------------- Ticket


def test_ticket_create_trims_text() -> None:
    result = Ticket.create("  Login  ", "  cannot log in  ")
    assert result == Ok(Ticket(subject="Login", body="cannot log in"))


@pytest.mark.parametrize(
    ("subject", "body", "code"),
    [
        ("", "body", ErrorCode.TICKET_SUBJECT_REQUIRED),
        ("   ", "body", ErrorCode.TICKET_SUBJECT_REQUIRED),
        ("x" * (MAX_SUBJECT_LENGTH + 1), "body", ErrorCode.TICKET_SUBJECT_TOO_LONG),
        ("subject", "", ErrorCode.TICKET_BODY_REQUIRED),
        ("subject", "x" * (MAX_BODY_LENGTH + 1), ErrorCode.TICKET_BODY_TOO_LONG),
    ],
)
def test_ticket_create_rejects_invalid_input(subject: str, body: str, code: ErrorCode) -> None:
    result = Ticket.create(subject, body)
    assert isinstance(result, Err)
    assert result.error.code is code
    assert not result.is_ok()


def test_classification_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError):
        make_classification(confidence=1.5)


# ---------------------------------------------------------------- Severity


def test_severity_rank_and_highest() -> None:
    assert Severity.LOW.rank < Severity.CRITICAL.rank
    assert Severity.highest(Severity.MEDIUM, Severity.HIGH) is Severity.HIGH
    assert Severity.highest(Severity.CRITICAL, Severity.HIGH) is Severity.CRITICAL


# ---------------------------------------------------------------- Masking


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("write to maria.souza@exemplo.com please", "EMAIL"),
        ("CPF 123.456.789-09", "CPF"),
        ("CPF 12345678909", "CPF"),
        ("CNPJ 12.345.678/0001-90", "CNPJ"),
        ("card 4111 1111 1111 1111", "CARD"),
        ("call (11) 98765-4321", "PHONE"),
        ("call +55 11 98765-4321", "PHONE"),
        ("plate ABC1D23", "PLATE"),
        ("plate ABC-1234", "PLATE"),
    ],
)
def test_masking_replaces_personal_data(text: str, label: str) -> None:
    result = mask_personal_data(text)
    assert f"[{label}]" in result.text
    assert result.counts.get(label) == 1
    assert result.total == 1


def test_masking_keeps_ordinary_text() -> None:
    text = "Error 500 on order 4471 since 08:00, report has 120h"
    assert mask_personal_data(text).text == text
    assert mask_personal_data(text).total == 0


# ---------------------------------------------------------------- Routing


def test_routing_sends_confident_ticket_to_category_queue() -> None:
    decision = RoutingPolicy().decide(make_classification(category=Category.BILLING))
    assert decision.status is TriageStatus.ROUTED
    assert decision.queue is Queue.FINANCE
    assert decision.sla_hours == 24


def test_routing_sends_low_confidence_to_human_review() -> None:
    decision = RoutingPolicy(0.7).decide(make_classification(confidence=0.5))
    assert decision.status is TriageStatus.NEEDS_REVIEW
    assert decision.queue is Queue.HUMAN_REVIEW


def test_routing_escalates_wide_impact_to_at_least_high() -> None:
    classification = make_classification(
        category=Category.PERFORMANCE,
        severity=Severity.MEDIUM,
        affected_scope=AffectedScope.ALL_USERS,
    )
    decision = RoutingPolicy().decide(classification)
    assert decision.severity is Severity.HIGH
    assert decision.sla_hours == 8


def test_routing_never_lowers_critical() -> None:
    classification = make_classification(
        category=Category.BUG, severity=Severity.CRITICAL, affected_scope=AffectedScope.ALL_USERS
    )
    assert RoutingPolicy().decide(classification).severity is Severity.CRITICAL


def test_routing_does_not_escalate_billing() -> None:
    classification = make_classification(
        category=Category.BILLING, severity=Severity.LOW, affected_scope=AffectedScope.ALL_USERS
    )
    assert RoutingPolicy().decide(classification).severity is Severity.LOW


def test_routing_caps_feature_requests_at_low() -> None:
    classification = make_classification(
        category=Category.FEATURE_REQUEST, severity=Severity.CRITICAL
    )
    decision = RoutingPolicy().decide(classification)
    assert decision.severity is Severity.LOW
    assert decision.queue is Queue.PRODUCT
    assert decision.sla_hours == 72


def test_routing_policy_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        RoutingPolicy(1.2)


# ---------------------------------------------------------------- Feedback


def _triage() -> Triage:
    classification = make_classification()
    return Triage(
        id="t-1",
        ticket=Ticket("s", "b"),
        classification=classification,
        decision=RoutingPolicy().decide(classification),
        usage=make_usage(),
        created_at=FIXED_NOW,
        created_by="tester",
    )


def test_feedback_is_applied_with_matching_version() -> None:
    triage = _triage()
    assert triage.category_was_correct is None
    result = triage.apply_feedback(Category.BUG, Severity.HIGH, 1, "reviewer", FIXED_NOW)
    assert isinstance(result, Ok)
    assert triage.feedback is not None
    assert triage.updated_by == "reviewer"
    assert triage.category_was_correct is False


def test_feedback_conflicts_on_stale_version() -> None:
    result = _triage().apply_feedback(Category.BUG, Severity.HIGH, 7, "reviewer", FIXED_NOW)
    assert isinstance(result, Err)
    assert result.error.code is ErrorCode.TRIAGE_VERSION_CONFLICT
