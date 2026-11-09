"""User-facing messages in en and pt-BR, selected by Accept-Language."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SUPPORTED_LOCALES = ("en", "pt-BR")
DEFAULT_LOCALE = "en"
_MESSAGES_DIR = Path(__file__).parent / "locales"


@lru_cache
def _catalog(locale: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_MESSAGES_DIR / f"{locale}.json").read_text("utf-8"))
    return data


def resolve_locale(accept_language: str | None) -> str:
    if not accept_language:
        return DEFAULT_LOCALE
    for part in accept_language.split(","):
        tag = part.split(";")[0].strip().lower()
        if tag.startswith("pt"):
            return "pt-BR"
        if tag.startswith("en"):
            return "en"
    return DEFAULT_LOCALE


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def translate(locale: str, code: str, field: str, params: dict[str, Any] | None = None) -> str:
    entry = _catalog(locale).get("errors", {}).get(code) or _catalog(DEFAULT_LOCALE)["errors"].get(
        code, {}
    )
    template = str(entry.get(field, code))
    return template.format_map(_SafeDict(params or {}))
