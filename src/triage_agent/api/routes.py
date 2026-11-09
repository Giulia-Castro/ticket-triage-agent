"""HTTP endpoints. Each one translates a Result into a response or a problem."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text

from triage_agent.api.dependencies import ActorDep, ContainerDep
from triage_agent.api.problems import PROBLEM_CONTENT_TYPE, domain_problem
from triage_agent.api.schemas import (
    FeedbackRequest,
    MetricsResponse,
    TriageRequest,
    TriageResponse,
)
from triage_agent.application.use_cases import FeedbackCommand, TriageCommand
from triage_agent.domain.result import Err, Ok

_PROBLEM: dict[int | str, dict[str, Any]] = {
    status: {"description": "Problem Details (RFC 7807)", "content": {PROBLEM_CONTENT_TYPE: {}}}
    for status in (401, 404, 409, 422, 503)
}

router = APIRouter(prefix="/v1", tags=["triage"])
health_router = APIRouter(tags=["health"])


@router.post("/triages", status_code=201, response_model=TriageResponse, responses=_PROBLEM)
def create_triage(
    payload: TriageRequest,
    request: Request,
    response: Response,
    container: ContainerDep,
    actor: ActorDep,
) -> TriageResponse | JSONResponse:
    command = TriageCommand(subject=payload.subject, body=payload.body, actor=actor)
    match container.triage_ticket.execute(command):
        case Err(error):
            return domain_problem(request, error)
        case Ok(triage):
            response.headers["Location"] = f"/v1/triages/{triage.id}"
            return TriageResponse.from_domain(triage)


@router.get("/triages", response_model=list[TriageResponse], responses=_PROBLEM)
def list_triages(
    container: ContainerDep,
    actor: ActorDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[TriageResponse]:
    return [TriageResponse.from_domain(t) for t in container.list_triages.execute(limit, actor)]


@router.get("/triages/{triage_id}", response_model=TriageResponse, responses=_PROBLEM)
def get_triage(
    triage_id: str, request: Request, container: ContainerDep, actor: ActorDep
) -> TriageResponse | JSONResponse:
    match container.get_triage.execute(triage_id, actor):
        case Err(error):
            return domain_problem(request, error)
        case Ok(triage):
            return TriageResponse.from_domain(triage)


@router.patch("/triages/{triage_id}/feedback", response_model=TriageResponse, responses=_PROBLEM)
def submit_feedback(
    triage_id: str,
    payload: FeedbackRequest,
    request: Request,
    container: ContainerDep,
    actor: ActorDep,
) -> TriageResponse | JSONResponse:
    command = FeedbackCommand(
        triage_id=triage_id,
        category=payload.category,
        severity=payload.severity,
        expected_version=payload.version,
        actor=actor,
    )
    match container.submit_feedback.execute(command):
        case Err(error):
            return domain_problem(request, error)
        case Ok(triage):
            return TriageResponse.from_domain(triage)


@router.get("/metrics/summary", response_model=MetricsResponse, responses=_PROBLEM)
def metrics_summary(container: ContainerDep, actor: ActorDep) -> MetricsResponse:
    return MetricsResponse.from_summary(container.get_metrics.execute())


@health_router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@health_router.get("/health/ready")
def ready(container: ContainerDep) -> dict[str, str]:
    with container.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}
