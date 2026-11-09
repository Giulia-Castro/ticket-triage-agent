"""RFC 7807 Problem Details responses with localized title and detail."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from triage_agent.api.i18n import resolve_locale, translate
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.infrastructure.logging import correlation_id_var

PROBLEM_CONTENT_TYPE = "application/problem+json"


class ApiErrorCode(StrEnum):
    VALIDATION_FAILED = "VALIDATION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_DOMAIN_STATUS: dict[ErrorCode, int] = {
    ErrorCode.TICKET_SUBJECT_REQUIRED: 422,
    ErrorCode.TICKET_SUBJECT_TOO_LONG: 422,
    ErrorCode.TICKET_BODY_REQUIRED: 422,
    ErrorCode.TICKET_BODY_TOO_LONG: 422,
    ErrorCode.TRIAGE_NOT_FOUND: 404,
    ErrorCode.TRIAGE_VERSION_CONFLICT: 409,
    ErrorCode.TRIAGE_REFUSED_BY_MODEL: 422,
}


def problem_response(
    request: Request,
    status: int,
    code: str,
    params: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    locale = resolve_locale(request.headers.get("accept-language"))
    body: dict[str, Any] = {
        "type": f"urn:triage-agent:problem:{code.lower().replace('_', '-')}",
        "title": translate(locale, code, "title", params),
        "status": status,
        "detail": translate(locale, code, "detail", params),
        "instance": request.url.path,
        "code": code,
        "correlationId": correlation_id_var.get(),
    }
    if errors:
        body["errors"] = errors
    return JSONResponse(
        status_code=status,
        content=body,
        media_type=PROBLEM_CONTENT_TYPE,
        headers={"Content-Language": locale, **(headers or {})},
    )


def domain_problem(request: Request, error: DomainError) -> JSONResponse:
    return problem_response(
        request, _DOMAIN_STATUS[error.code], error.code.value, dict(error.params)
    )
