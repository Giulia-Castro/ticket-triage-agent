"""Claude-backed triage agent.

The agent may call two read-only tools (knowledge base search and similar past
tickets) before answering. The final answer is constrained by a JSON schema
(structured outputs) and validated again with Pydantic before it becomes a
domain object.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import anthropic
from pydantic import BaseModel, Field, ValidationError, field_validator

from triage_agent.application.errors import LlmUnavailableError
from triage_agent.application.ports import AgentResult, KnowledgeBase, SimilarTicketFinder
from triage_agent.domain.enums import AffectedScope, Category, Severity
from triage_agent.domain.errors import DomainError, ErrorCode
from triage_agent.domain.model import Classification, LlmUsage, Ticket
from triage_agent.domain.result import Err, Ok, Result
from triage_agent.infrastructure.prompts import Prompt
from triage_agent.infrastructure.resilience import CircuitBreaker

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOOL_QUERY_LENGTH = 300
KB_RESULTS = 3
SIMILAR_RESULTS = 3

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal knowledge base for help articles related to the ticket. "
            "Returns up to 3 articles with id, title and summary. Use it to find the article "
            "ids to cite and to ground the suggested reply."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Short keyword query."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_similar_tickets",
        "description": (
            "Find up to 3 previously triaged tickets with similar wording, with their category "
            "and severity. Use it when the category is ambiguous or to check for an ongoing "
            "incident (many similar tickets at once)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Short keyword query."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "severity": {"type": "string", "enum": [s.value for s in Severity]},
        "affected_scope": {"type": "string", "enum": [a.value for a in AffectedScope]},
        "product_area": {"type": "string"},
        "summary": {"type": "string"},
        "suggested_reply": {"type": "string"},
        "confidence": {"type": "number"},
        "kb_article_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "category",
        "severity",
        "affected_scope",
        "product_area",
        "summary",
        "suggested_reply",
        "confidence",
        "kb_article_ids",
    ],
    "additionalProperties": False,
}


class _ClassificationPayload(BaseModel):
    category: Category
    severity: Severity
    affected_scope: AffectedScope
    product_area: str = Field(max_length=100)
    summary: str = Field(max_length=500)
    suggested_reply: str = Field(max_length=4000)
    confidence: float
    kb_article_ids: list[str]

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return min(max(value, 0.0), 1.0)


def render_ticket(ticket: Ticket) -> str:
    """Wrap ticket text in tags so the prompt can treat it strictly as data."""

    def neutralize(text: str) -> str:
        return text.replace("<ticket", "&lt;ticket").replace("</ticket", "&lt;/ticket")

    return (
        "Triage the support ticket below.\n\n"
        "<ticket>\n"
        f"<subject>{neutralize(ticket.subject)}</subject>\n"
        f"<body>\n{neutralize(ticket.body)}\n</body>\n"
        "</ticket>"
    )


class ClaudeTriageAgent:
    def __init__(
        self,
        client: anthropic.Anthropic,
        prompt: Prompt,
        knowledge_base: KnowledgeBase,
        similar_tickets: SimilarTicketFinder,
        circuit_breaker: CircuitBreaker,
        *,
        model: str,
        effort: str,
        max_tokens: int,
        max_tool_rounds: int,
        price_input_per_mtok: Decimal,
        price_output_per_mtok: Decimal,
        fallbacks_enabled: bool = True,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._kb = knowledge_base
        self._similar = similar_tickets
        self._breaker = circuit_breaker
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._price_in = price_input_per_mtok
        self._price_out = price_output_per_mtok
        self._fallbacks_enabled = fallbacks_enabled
        self._timer = timer

    def classify(self, ticket: Ticket) -> Result[AgentResult, DomainError]:
        if not self._breaker.allow_request():
            raise LlmUnavailableError("circuit breaker is open")
        try:
            result = self._run(ticket)
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            self._breaker.record_failure()
            raise LlmUnavailableError(type(exc).__name__) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                self._breaker.record_failure()
            raise LlmUnavailableError(f"LLM API returned {exc.status_code}") from exc
        self._breaker.record_success()
        return result

    # ------------------------------------------------------------------ internals

    def _request_kwargs(self, final_round: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": self._prompt.text,
            "tools": TOOLS,
            "output_config": {
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        }
        if final_round:
            # Out of tool budget: the model must answer now.
            kwargs["tool_choice"] = {"type": "none"}
        if self._fallbacks_enabled:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _run(self, ticket: Ticket) -> Result[AgentResult, DomainError]:
        started = self._timer()
        messages: list[dict[str, Any]] = [{"role": "user", "content": render_ticket(ticket)}]
        input_tokens = output_tokens = 0
        served_by = self._model
        tool_rounds = 0

        while True:
            final_round = tool_rounds >= self._max_tool_rounds
            response = self._client.beta.messages.create(
                messages=messages, **self._request_kwargs(final_round)
            )
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            served_by = response.model or served_by

            if response.stop_reason == "refusal":
                return Err(DomainError(ErrorCode.TRIAGE_REFUSED_BY_MODEL))
            if response.stop_reason == "max_tokens":
                raise LlmUnavailableError("model output truncated at max_tokens")
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            if response.stop_reason == "tool_use":
                if final_round:
                    raise LlmUnavailableError("model kept calling tools after the tool budget")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": self._run_tools(response.content)})
                tool_rounds += 1
                continue
            break

        payload = self._parse(response.content)
        latency_ms = int((self._timer() - started) * 1000)
        classification = Classification(
            category=payload.category,
            severity=payload.severity,
            affected_scope=payload.affected_scope,
            product_area=payload.product_area,
            summary=payload.summary,
            suggested_reply=payload.suggested_reply,
            confidence=payload.confidence,
            # Keep only ids that exist: the model must not invent articles.
            kb_article_ids=tuple(
                i for i in dict.fromkeys(payload.kb_article_ids) if self._kb.exists(i)
            ),
        )
        usage = LlmUsage(
            model_name=served_by,
            prompt_version=self._prompt.version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._cost(input_tokens, output_tokens),
            latency_ms=latency_ms,
        )
        return Ok(AgentResult(classification=classification, usage=usage))

    def _run_tools(self, content: list[Any]) -> list[dict[str, Any]]:
        """Execute every tool call of the turn and return all results in one user message."""
        results: list[dict[str, Any]] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            output, is_error = self._execute_tool(block.name, block.input)
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)
        return results

    def _execute_tool(self, name: str, tool_input: Any) -> tuple[str, bool]:
        query = ""
        if isinstance(tool_input, dict):
            query = str(tool_input.get("query", ""))[:MAX_TOOL_QUERY_LENGTH]
        logger.info("agent.tool_call", extra={"context": {"tool": name}})
        if name == "search_knowledge_base":
            articles = self._kb.search(query, KB_RESULTS)
            data: list[dict[str, str]] = [
                {"id": a.id, "title": a.title, "summary": a.summary} for a in articles
            ]
            return json.dumps(data, ensure_ascii=False), False
        if name == "find_similar_tickets":
            similar = self._similar.find_similar(query, SIMILAR_RESULTS)
            data = [
                {"category": s.category, "severity": s.severity, "summary": s.summary}
                for s in similar
            ]
            return json.dumps(data, ensure_ascii=False), False
        return json.dumps({"error": f"unknown tool: {name}"}), True

    @staticmethod
    def _parse(content: list[Any]) -> _ClassificationPayload:
        text = next((b.text for b in content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LlmUnavailableError("model returned no text block")
        try:
            return _ClassificationPayload.model_validate_json(text)
        except ValidationError as exc:
            raise LlmUnavailableError("model output failed schema validation") from exc

    def _cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        million = Decimal(1_000_000)
        cost = (
            Decimal(input_tokens) * self._price_in + Decimal(output_tokens) * self._price_out
        ) / million
        return cost.quantize(Decimal("0.000001"))
