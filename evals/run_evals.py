"""Run a triage agent against the labeled dataset and write a report.

Examples:
    python -m evals.run_evals --agent keyword
    python -m evals.run_evals --agent claude --prompt-version v1 --yes
    python -m evals.run_evals --agent claude --prompt-version v2 --limit 10 --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import EvalItem, build_report, to_markdown
from triage_agent.application.errors import TechnicalError
from triage_agent.application.ports import SimilarTicket, TriageAgent
from triage_agent.domain.enums import Category, TriageStatus
from triage_agent.domain.masking import mask_personal_data
from triage_agent.domain.model import Ticket
from triage_agent.domain.result import Ok
from triage_agent.domain.routing import RoutingPolicy
from triage_agent.infrastructure.container import build_agent
from triage_agent.infrastructure.knowledge_base import JsonKnowledgeBase
from triage_agent.infrastructure.settings import Settings

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.jsonl"
RESULTS_DIR = ROOT / "results"


class _NoHistory:
    """Evals run each ticket in isolation: no similar-ticket leakage between items."""

    def find_similar(self, text: str, limit: int) -> list[SimilarTicket]:
        return []


def load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def evaluate(
    agent: TriageAgent, rows: list[dict[str, Any]], policy: RoutingPolicy
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for row in rows:
        ticket = Ticket(
            subject=mask_personal_data(row["subject"]).text,
            body=mask_personal_data(row["body"]).text,
        )
        try:
            outcome = agent.classify(ticket)
        except TechnicalError as exc:
            items.append(_failed(row, f"technical: {exc}"))
            continue
        if not isinstance(outcome, Ok):
            items.append(_failed(row, f"refused: {outcome.error.code}"))
            continue
        decision = policy.decide(outcome.value.classification)
        usage = outcome.value.usage
        items.append(
            EvalItem(
                ticket_id=row["id"],
                expected_category=row["expected_category"],
                expected_severity=row["expected_severity"],
                predicted_category=outcome.value.classification.category.value,
                predicted_severity=decision.severity.value,
                needs_review=decision.status is TriageStatus.NEEDS_REVIEW,
                latency_ms=usage.latency_ms,
                cost_usd=float(usage.cost_usd),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        )
        print(f"  {row['id']}: {items[-1].predicted_category}/{items[-1].predicted_severity}")
    return items


def _failed(row: dict[str, Any], error: str) -> EvalItem:
    print(f"  {row['id']}: ERROR {error}")
    return EvalItem(
        ticket_id=row["id"],
        expected_category=row["expected_category"],
        expected_severity=row["expected_severity"],
        predicted_category=None,
        predicted_severity=None,
        needs_review=True,
        latency_ms=0,
        cost_usd=0.0,
        error=error,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--agent", choices=["claude", "keyword"], default="keyword")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args(argv)

    settings = Settings(triage_agent=args.agent, prompt_version=args.prompt_version)
    rows = load_dataset(DATASET, args.limit)
    if args.agent == "claude" and not args.yes:
        answer = input(
            f"This calls the Claude API {len(rows)}+ times and costs real money. Go? [y/N] "
        )
        if answer.strip().lower() != "y":
            return 1

    kb = JsonKnowledgeBase.from_file(settings.knowledge_base_path)
    agent = build_agent(settings, _NoHistory(), kb)
    policy = RoutingPolicy(settings.confidence_threshold)

    print(f"Evaluating {args.agent} ({args.prompt_version}) on {len(rows)} tickets...")
    items = evaluate(agent, rows, policy)
    report = build_report(items, [c.value for c in Category])

    label = (
        "keyword-baseline"
        if args.agent == "keyword"
        else f"{settings.claude_model}-{args.prompt_version}"
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"{label}-{stamp}.json"
    output.write_text(
        json.dumps(
            {
                "label": label,
                "runAt": stamp,
                "dataset": DATASET.name,
                "report": asdict(report),
                "items": [asdict(item) for item in items],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print(to_markdown(report, label))
    print(f"\nSaved to {output.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
