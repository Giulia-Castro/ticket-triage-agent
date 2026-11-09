"""Rule-based baseline classifier.

Two purposes: (1) the baseline every LLM prompt version is compared against in
the evals, and (2) an offline mode to run the API without an API key.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal

from triage_agent.application.ports import AgentResult, KnowledgeBase
from triage_agent.domain.enums import AffectedScope, Category, Severity
from triage_agent.domain.errors import DomainError
from triage_agent.domain.model import Classification, LlmUsage, Ticket
from triage_agent.domain.result import Ok, Result
from triage_agent.infrastructure.text_similarity import tokenize

BASELINE_MODEL_NAME = "keyword-baseline"

_CATEGORY_KEYWORDS: dict[Category, frozenset[str]] = {
    Category.ACCESS: frozenset(
        [
            "login",
            "senha",
            "password",
            "acesso",
            "acessar",
            "entrar",
            "bloqueado",
            "bloqueada",
            "sso",
            "mfa",
            "autenticacao",
            "permissao",
            "perfil",
            "token",
            "expirou",
            "convite",
        ]
    ),
    Category.BILLING: frozenset(
        [
            "fatura",
            "boleto",
            "cobranca",
            "cobrado",
            "cobrada",
            "pagamento",
            "nota",
            "fiscal",
            "plano",
            "reembolso",
            "valor",
            "cartao",
            "desconto",
            "contrato",
        ]
    ),
    Category.BUG: frozenset(
        [
            "erro",
            "bug",
            "falha",
            "quebrado",
            "nao",
            "funciona",
            "travou",
            "exception",
            "botao",
            "incorreto",
            "errado",
            "duplicado",
            "sumiu",
            "crash",
        ]
    ),
    Category.PERFORMANCE: frozenset(
        [
            "lento",
            "lentidao",
            "demora",
            "demorando",
            "timeout",
            "carregando",
            "performance",
            "devagar",
            "travando",
            "minutos",
        ]
    ),
    Category.INTEGRATION: frozenset(
        [
            "api",
            "webhook",
            "integracao",
            "endpoint",
            "token",
            "json",
            "payload",
            "sincronizacao",
            "erp",
            "429",
            "401",
            "callback",
        ]
    ),
    Category.FEATURE_REQUEST: frozenset(
        [
            "sugestao",
            "gostaria",
            "seria",
            "possivel",
            "melhoria",
            "funcionalidade",
            "poderia",
            "permitir",
            "adicionar",
            "opcao",
        ]
    ),
}

_CRITICAL_HINTS = frozenset(
    ["parado", "parada", "fora", "indisponivel", "ninguem", "todos", "producao"]
)
_HIGH_HINTS = frozenset(
    ["urgente", "bloqueado", "bloqueada", "impede", "impedindo", "clientes", "equipe"]
)
_ALL_USERS_HINTS = frozenset(["todos", "ninguem", "empresa", "inteira"])
_TEAM_HINTS = frozenset(["equipe", "time", "setor"])


class KeywordTriageAgent:
    def __init__(
        self, knowledge_base: KnowledgeBase, timer: Callable[[], float] = time.perf_counter
    ) -> None:
        self._kb = knowledge_base
        self._timer = timer

    def classify(self, ticket: Ticket) -> Result[AgentResult, DomainError]:
        started = self._timer()
        tokens = tokenize(f"{ticket.subject} {ticket.body}")

        scores = {cat: len(tokens & words) for cat, words in _CATEGORY_KEYWORDS.items()}
        category = max(scores, key=lambda cat: scores[cat])
        total_hits = sum(scores.values())
        confidence = scores[category] / total_hits if total_hits else 0.0

        if tokens & _ALL_USERS_HINTS:
            scope = AffectedScope.ALL_USERS
        elif tokens & _TEAM_HINTS:
            scope = AffectedScope.TEAM
        else:
            scope = AffectedScope.SINGLE_USER

        if tokens & _CRITICAL_HINTS and scope is AffectedScope.ALL_USERS:
            severity = Severity.CRITICAL
        elif tokens & (_HIGH_HINTS | _CRITICAL_HINTS):
            severity = Severity.HIGH
        elif category is Category.FEATURE_REQUEST:
            severity = Severity.LOW
        else:
            severity = Severity.MEDIUM

        articles = self._kb.search(f"{ticket.subject} {ticket.body}", 2)
        classification = Classification(
            category=category,
            severity=severity,
            affected_scope=scope,
            product_area="unknown",
            summary=ticket.subject[:500],
            suggested_reply="",
            confidence=round(confidence, 3),
            kb_article_ids=tuple(article.id for article in articles),
        )
        usage = LlmUsage(
            model_name=BASELINE_MODEL_NAME,
            prompt_version="n/a",
            input_tokens=0,
            output_tokens=0,
            cost_usd=Decimal("0"),
            latency_ms=int((self._timer() - started) * 1000),
        )
        return Ok(AgentResult(classification=classification, usage=usage))
