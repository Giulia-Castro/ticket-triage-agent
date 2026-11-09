"""Composition root: wires adapters into use cases from settings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import anthropic
from sqlalchemy import Engine

from triage_agent.application.ports import SimilarTicketFinder, TriageAgent
from triage_agent.application.use_cases import (
    GetMetricsUseCase,
    GetTriageUseCase,
    ListTriagesUseCase,
    SubmitFeedbackUseCase,
    TriageTicketUseCase,
)
from triage_agent.domain.routing import RoutingPolicy
from triage_agent.infrastructure.claude_agent import ClaudeTriageAgent
from triage_agent.infrastructure.keyword_agent import KeywordTriageAgent
from triage_agent.infrastructure.knowledge_base import JsonKnowledgeBase
from triage_agent.infrastructure.persistence.database import build_engine, build_session_factory
from triage_agent.infrastructure.persistence.repository import SqlTriageRepository, utc_now
from triage_agent.infrastructure.prompts import load_prompt
from triage_agent.infrastructure.resilience import CircuitBreaker
from triage_agent.infrastructure.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    engine: Engine
    triage_ticket: TriageTicketUseCase
    get_triage: GetTriageUseCase
    list_triages: ListTriagesUseCase
    submit_feedback: SubmitFeedbackUseCase
    get_metrics: GetMetricsUseCase


def build_agent(
    settings: Settings, similar: SimilarTicketFinder, kb: JsonKnowledgeBase
) -> TriageAgent:
    if settings.triage_agent == "keyword":
        return KeywordTriageAgent(kb)
    client = anthropic.Anthropic(
        timeout=settings.claude_timeout_seconds,
        # Classification is read-only, so retrying is safe (idempotent).
        max_retries=settings.claude_max_retries,
    )
    return ClaudeTriageAgent(
        client=client,
        prompt=load_prompt(settings.prompts_dir, settings.prompt_version),
        knowledge_base=kb,
        similar_tickets=similar,
        circuit_breaker=CircuitBreaker(
            settings.circuit_failure_threshold, settings.circuit_reset_seconds
        ),
        model=settings.claude_model,
        effort=settings.claude_effort,
        max_tokens=settings.claude_max_tokens,
        max_tool_rounds=settings.agent_max_tool_rounds,
        price_input_per_mtok=settings.price_input_per_mtok,
        price_output_per_mtok=settings.price_output_per_mtok,
        fallbacks_enabled=settings.claude_fallbacks_enabled,
    )


def build_container(settings: Settings, agent: TriageAgent | None = None) -> Container:
    engine = build_engine(settings.database_url)
    repository = SqlTriageRepository(build_session_factory(engine))
    kb = JsonKnowledgeBase.from_file(settings.knowledge_base_path)
    agent = agent or build_agent(settings, repository, kb)
    return Container(
        settings=settings,
        engine=engine,
        triage_ticket=TriageTicketUseCase(
            agent=agent,
            repository=repository,
            policy=RoutingPolicy(settings.confidence_threshold),
            clock=utc_now,
            id_factory=lambda: str(uuid.uuid4()),
        ),
        get_triage=GetTriageUseCase(repository),
        list_triages=ListTriagesUseCase(repository),
        submit_feedback=SubmitFeedbackUseCase(repository, utc_now),
        get_metrics=GetMetricsUseCase(repository),
    )
