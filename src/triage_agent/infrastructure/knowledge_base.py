"""Knowledge base backed by a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from triage_agent.application.ports import KbArticle
from triage_agent.infrastructure.text_similarity import overlap_score, tokenize


class JsonKnowledgeBase:
    def __init__(self, articles: list[KbArticle]) -> None:
        self._articles = {article.id: article for article in articles}
        self._tokens = {
            article.id: tokenize(f"{article.title} {article.summary}") for article in articles
        }

    @classmethod
    def from_file(cls, path: Path) -> JsonKnowledgeBase:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([KbArticle(id=a["id"], title=a["title"], summary=a["summary"]) for a in raw])

    def search(self, query: str, limit: int) -> list[KbArticle]:
        query_tokens = tokenize(query)
        scored = [
            (overlap_score(query_tokens, tokens), article_id)
            for article_id, tokens in self._tokens.items()
        ]
        ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: -item[0])
        return [self._articles[article_id] for _, article_id in ranked[:limit]]

    def exists(self, article_id: str) -> bool:
        return article_id in self._articles
