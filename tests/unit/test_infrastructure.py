from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.conftest import ROOT
from triage_agent.domain.enums import Category, Severity
from triage_agent.domain.model import Ticket
from triage_agent.domain.result import Ok
from triage_agent.infrastructure.keyword_agent import BASELINE_MODEL_NAME, KeywordTriageAgent
from triage_agent.infrastructure.knowledge_base import JsonKnowledgeBase
from triage_agent.infrastructure.logging import JsonFormatter, correlation_id_var
from triage_agent.infrastructure.prompts import load_prompt
from triage_agent.infrastructure.resilience import CircuitBreaker, CircuitState
from triage_agent.infrastructure.settings import Settings
from triage_agent.infrastructure.text_similarity import overlap_score, tokenize

KB_PATH = ROOT / "data" / "knowledge_base.json"


def test_tokenize_strips_accents_and_stopwords() -> None:
    assert tokenize("A cobrança da Fatura está errada") == {"cobranca", "fatura", "esta", "errada"}
    assert overlap_score(set(), {"a"}) == 0.0


def test_knowledge_base_search_ranks_relevant_articles() -> None:
    kb = JsonKnowledgeBase.from_file(KB_PATH)
    results = kb.search("webhook assinatura token", 3)
    assert results[0].id == "KB-008"
    assert kb.exists("KB-001")
    assert not kb.exists("KB-999")
    assert kb.search("", 3) == []


def test_prompts_load_with_version_check(tmp_path: Path) -> None:
    prompt = load_prompt(ROOT / "prompts" / "triage", "v2")
    assert prompt.version == "v2"
    assert "<ticket>" in prompt.text
    with pytest.raises(ValueError):
        load_prompt(ROOT / "prompts" / "triage", "../secrets")
    (tmp_path / "v9.md").write_text("no header", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt(tmp_path, "v9")
    (tmp_path / "v8.md").write_text("---\nversion: v7\n---\ntext", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt(tmp_path, "v8")


def test_circuit_breaker_half_opens_after_timeout() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10, clock=lambda: now[0])
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()
    now[0] = 11
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request()
    breaker.record_failure()  # trial call failed: open again
    assert breaker.state is CircuitState.OPEN
    now[0] = 30
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_keyword_agent_classifies_offline() -> None:
    agent = KeywordTriageAgent(JsonKnowledgeBase.from_file(KB_PATH))
    result = agent.classify(
        Ticket("Fatura cobrada em dobro", "O boleto veio com cobrança duplicada")
    )
    assert isinstance(result, Ok)
    assert result.value.classification.category is Category.BILLING
    assert result.value.usage.model_name == BASELINE_MODEL_NAME

    critical = agent.classify(Ticket("Sistema fora", "Ninguém da empresa acessa, tudo parado"))
    assert isinstance(critical, Ok)
    assert critical.value.classification.severity is Severity.CRITICAL

    empty = agent.classify(Ticket("???", "..."))
    assert isinstance(empty, Ok)
    assert empty.value.classification.confidence == 0.0


def test_json_formatter_masks_personal_data_and_adds_correlation_id() -> None:
    token = correlation_id_var.set("corr-123")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "user ana@x.com", None, None)
        record.context = {"cpf": "123.456.789-09", "count": 3}
        payload = json.loads(JsonFormatter().format(record))
    finally:
        correlation_id_var.reset(token)
    assert payload["message"] == "user [EMAIL]"
    assert payload["context"] == {"cpf": "[CPF]", "count": 3}
    assert payload["correlationId"] == "corr-123"


def test_settings_parse_api_keys_deny_malformed_entries() -> None:
    settings = Settings(
        api_keys="ops:abcdefghijklmnop0123, broken, short:abc, :nolabelkey123456789"
    )
    assert settings.parsed_api_keys() == {"abcdefghijklmnop0123": "ops"}
