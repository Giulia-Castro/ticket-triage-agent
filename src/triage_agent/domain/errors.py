"""Business errors returned through Result. Technical failures are exceptions elsewhere."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCode(StrEnum):
    TICKET_SUBJECT_REQUIRED = "TICKET_SUBJECT_REQUIRED"
    TICKET_SUBJECT_TOO_LONG = "TICKET_SUBJECT_TOO_LONG"
    TICKET_BODY_REQUIRED = "TICKET_BODY_REQUIRED"
    TICKET_BODY_TOO_LONG = "TICKET_BODY_TOO_LONG"
    TRIAGE_NOT_FOUND = "TRIAGE_NOT_FOUND"
    TRIAGE_VERSION_CONFLICT = "TRIAGE_VERSION_CONFLICT"
    TRIAGE_REFUSED_BY_MODEL = "TRIAGE_REFUSED_BY_MODEL"


@dataclass(frozen=True, slots=True)
class DomainError:
    code: ErrorCode
    params: Mapping[str, str | int] = field(default_factory=dict)
