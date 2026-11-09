"""Fixed business codes. Persisted as strings, never as ordinals."""

from __future__ import annotations

from enum import StrEnum


class Category(StrEnum):
    ACCESS = "ACCESS"
    BILLING = "BILLING"
    BUG = "BUG"
    PERFORMANCE = "PERFORMANCE"
    INTEGRATION = "INTEGRATION"
    FEATURE_REQUEST = "FEATURE_REQUEST"


class Severity(StrEnum):
    """Declaration order defines rank: LOW < MEDIUM < HIGH < CRITICAL."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)

    @staticmethod
    def highest(first: Severity, second: Severity) -> Severity:
        return first if first.rank >= second.rank else second


class AffectedScope(StrEnum):
    SINGLE_USER = "SINGLE_USER"
    TEAM = "TEAM"
    ALL_USERS = "ALL_USERS"
    UNKNOWN = "UNKNOWN"


class Queue(StrEnum):
    SERVICE_DESK = "SERVICE_DESK"
    FINANCE = "FINANCE"
    ENGINEERING = "ENGINEERING"
    SRE = "SRE"
    INTEGRATIONS = "INTEGRATIONS"
    PRODUCT = "PRODUCT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class TriageStatus(StrEnum):
    ROUTED = "ROUTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
