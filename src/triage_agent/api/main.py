"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from triage_agent import __version__
from triage_agent.api.dependencies import UnauthorizedError
from triage_agent.api.problems import ApiErrorCode, problem_response
from triage_agent.api.routes import health_router, router
from triage_agent.application.errors import TechnicalError
from triage_agent.infrastructure.container import Container, build_container
from triage_agent.infrastructure.logging import configure_logging, correlation_id_var
from triage_agent.infrastructure.settings import get_settings

logger = logging.getLogger("triage_agent.http")
CORRELATION_HEADER = "X-Correlation-ID"


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "container", None) is None:
            settings = get_settings()
            configure_logging(settings.log_level)
            app.state.container = build_container(settings)
        yield

    app = FastAPI(
        title="Ticket Triage Agent",
        version=__version__,
        description="Classifies and routes support tickets with an LLM agent.",
        lifespan=lifespan,
    )
    app.state.container = container
    app.include_router(health_router)
    app.include_router(router)
    _register_middleware(app)
    _register_exception_handlers(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlation_and_access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(CORRELATION_HEADER, "")
        correlation_id = incoming if 8 <= len(incoming) <= 64 else str(uuid.uuid4())
        token = correlation_id_var.set(correlation_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers[CORRELATION_HEADER] = correlation_id
            logger.info(
                "http.request",
                extra={
                    "context": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "durationMs": round((time.perf_counter() - started) * 1000, 1),
                    }
                },
            )
            return response
        finally:
            correlation_id_var.reset(token)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> Response:
        errors = [
            {"field": ".".join(str(p) for p in err["loc"][1:]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return problem_response(request, 422, ApiErrorCode.VALIDATION_FAILED, errors=errors)

    @app.exception_handler(UnauthorizedError)
    async def on_unauthorized(request: Request, exc: UnauthorizedError) -> Response:
        return problem_response(
            request, 401, ApiErrorCode.UNAUTHORIZED, headers={"WWW-Authenticate": "ApiKey"}
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == 404:
            return problem_response(request, 404, ApiErrorCode.RESOURCE_NOT_FOUND)
        return problem_response(request, exc.status_code, ApiErrorCode.INTERNAL_ERROR)

    @app.exception_handler(TechnicalError)
    async def on_technical(request: Request, exc: TechnicalError) -> Response:
        logger.error("dependency.unavailable", exc_info=exc)
        return problem_response(
            request, 503, ApiErrorCode.SERVICE_UNAVAILABLE, headers={"Retry-After": "10"}
        )

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception) -> Response:
        logger.error("unhandled.exception", exc_info=exc)
        return problem_response(request, 500, ApiErrorCode.INTERNAL_ERROR)


app = create_app()
