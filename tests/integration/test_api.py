"""End-to-end HTTP tests: real FastAPI app, real SQLite database, fake LLM agent."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ROOT, FakeAgent
from triage_agent.api.main import create_app
from triage_agent.application.errors import LlmUnavailableError
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.domain.result import Err
from triage_agent.infrastructure.container import build_container
from triage_agent.infrastructure.persistence.orm import Base
from triage_agent.infrastructure.settings import Settings

API_KEY = "test-key-0123456789abcdef"
AUTH = {"X-API-Key": API_KEY}
TICKET = {
    "subject": "Não consigo entrar",
    "body": "Senha inválida desde hoje. Meu e-mail é a@b.com",
}


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def client(tmp_path: Path, agent: FakeAgent) -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        api_keys=f"tester:{API_KEY}",
        knowledge_base_path=ROOT / "data" / "knowledge_base.json",
        prompts_dir=ROOT / "prompts" / "triage",
    )
    container = build_container(settings, agent=agent)
    Base.metadata.create_all(container.engine)
    with TestClient(create_app(container), raise_server_exceptions=False) as test_client:
        yield test_client
    container.engine.dispose()


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key-0123456789"}])
def test_requests_without_valid_key_are_rejected(client: TestClient, headers: dict) -> None:
    response = client.post("/v1/triages", json=TICKET, headers=headers)
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "UNAUTHORIZED"


def test_create_then_get_triage(client: TestClient) -> None:
    created = client.post("/v1/triages", json=TICKET, headers=AUTH)
    assert created.status_code == 201
    body = created.json()
    assert created.headers["Location"] == f"/v1/triages/{body['id']}"
    assert created.headers["X-Correlation-ID"]
    assert body["queue"] == "SERVICE_DESK"
    assert body["usage"]["promptVersion"] == "v2"
    assert body["version"] == 1
    # Data minimization: the ticket text is never echoed back.
    assert "a@b.com" not in created.text

    fetched = client.get(f"/v1/triages/{body['id']}", headers=AUTH)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]

    listed = client.get("/v1/triages?limit=5", headers=AUTH)
    assert [t["id"] for t in listed.json()] == [body["id"]]


def test_correlation_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Correlation-ID": "my-correlation-1"})
    assert response.headers["X-Correlation-ID"] == "my-correlation-1"


def test_not_found_problem_is_localized(client: TestClient) -> None:
    response = client.get("/v1/triages/missing", headers={**AUTH, "Accept-Language": "pt-BR"})
    assert response.status_code == 404
    problem = response.json()
    assert problem["code"] == "TRIAGE_NOT_FOUND"
    assert problem["title"] == "Triagem não encontrada"
    assert "missing" in problem["detail"]
    assert response.headers["Content-Language"] == "pt-BR"


def test_unknown_route_is_a_problem(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"


def test_domain_validation_error(client: TestClient) -> None:
    response = client.post("/v1/triages", json={"subject": " ", "body": "x"}, headers=AUTH)
    assert response.status_code == 422
    assert response.json()["code"] == "TICKET_SUBJECT_REQUIRED"


def test_transport_validation_error_lists_fields(client: TestClient) -> None:
    response = client.post("/v1/triages", json={"subject": "only subject"}, headers=AUTH)
    assert response.status_code == 422
    problem = response.json()
    assert problem["code"] == "VALIDATION_FAILED"
    assert problem["errors"][0]["field"] == "body"


def test_feedback_with_optimistic_lock(client: TestClient) -> None:
    triage_id = client.post("/v1/triages", json=TICKET, headers=AUTH).json()["id"]
    url = f"/v1/triages/{triage_id}/feedback"

    ok = client.patch(
        url, json={"category": "ACCESS", "severity": "HIGH", "version": 1}, headers=AUTH
    )
    assert ok.status_code == 200
    assert ok.json()["version"] == 2
    assert ok.json()["feedback"]["severity"] == "HIGH"

    stale = client.patch(
        url, json={"category": "BUG", "severity": "LOW", "version": 1}, headers=AUTH
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "TRIAGE_VERSION_CONFLICT"

    metrics = client.get("/v1/metrics/summary", headers=AUTH).json()
    assert metrics["total"] == 1
    assert metrics["feedbackCount"] == 1
    assert metrics["categoryAccuracy"] == 1.0
    assert metrics["byCategory"] == {"ACCESS": 1}


def test_llm_outage_returns_503(client: TestClient, agent: FakeAgent) -> None:
    agent.error = LlmUnavailableError("down")
    response = client.post("/v1/triages", json=TICKET, headers=AUTH)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "10"
    assert response.json()["code"] == "SERVICE_UNAVAILABLE"


def test_model_refusal_returns_422(client: TestClient, agent: FakeAgent) -> None:
    agent.outcome = Err(DomainError(ErrorCode.TRIAGE_REFUSED_BY_MODEL))
    response = client.post("/v1/triages", json=TICKET, headers=AUTH)
    assert response.status_code == 422
    assert response.json()["code"] == "TRIAGE_REFUSED_BY_MODEL"


def test_unexpected_error_hides_details(client: TestClient, agent: FakeAgent) -> None:
    agent.error = RuntimeError("secret internal detail")
    response = client.post("/v1/triages", json=TICKET, headers=AUTH)
    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["code"] == "INTERNAL_ERROR"
