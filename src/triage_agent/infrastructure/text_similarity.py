"""Lightweight lexical similarity (no external service). Good enough for a few thousand rows."""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS = frozenset(
    [
        "a",
        "o",
        "e",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "para",
        "por",
        "com",
        "sem",
        "que",
        "se",
        "ao",
        "aos",
        "the",
        "and",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "not",
        "it",
        "this",
        "that",
        "from",
        "at",
        "be",
        "or",
        "as",
        "an",
        "by",
        "nao",
        "sim",
        "mais",
        "muito",
        "ja",
        "ele",
        "ela",
        "eles",
        "elas",
        "eu",
        "voce",
        "nos",
        "meu",
        "minha",
        "seu",
        "sua",
        "isso",
        "esse",
        "essa",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tokenize(text: str) -> set[str]:
    words = _TOKEN.findall(_strip_accents(text.lower()))
    return {word for word in words if word not in _STOPWORDS}


def overlap_score(query_tokens: set[str], document_tokens: set[str]) -> float:
    """Share of query tokens found in the document (0..1)."""
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)
