"""Pure metric functions for the eval runner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

SEVERITY_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass(frozen=True, slots=True)
class EvalItem:
    ticket_id: str
    expected_category: str
    expected_severity: str
    predicted_category: str | None
    predicted_severity: str | None
    needs_review: bool
    latency_ms: int
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class EvalReport:
    total: int
    errors: int
    category_accuracy: float
    category_macro_f1: float
    severity_accuracy: float
    severity_within_one: float
    review_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_cost_usd: float
    avg_cost_usd: float
    per_category: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return float(ordered[index])


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_class_metrics(items: Sequence[EvalItem], labels: Sequence[str]) -> dict[str, ClassMetrics]:
    metrics: dict[str, ClassMetrics] = {}
    for label in labels:
        tp = sum(1 for i in items if i.expected_category == label and i.predicted_category == label)
        fp = sum(1 for i in items if i.expected_category != label and i.predicted_category == label)
        fn = sum(1 for i in items if i.expected_category == label and i.predicted_category != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = sum(1 for i in items if i.expected_category == label)
        metrics[label] = ClassMetrics(round(precision, 4), round(recall, 4), round(f1, 4), support)
    return metrics


def severity_distance(expected: str, predicted: str | None) -> int | None:
    if predicted not in SEVERITY_ORDER:
        return None
    return abs(SEVERITY_ORDER.index(expected) - SEVERITY_ORDER.index(predicted))


def build_report(items: Sequence[EvalItem], labels: Sequence[str]) -> EvalReport:
    total = len(items)
    per_category = per_class_metrics(items, labels)
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in [*labels, "ERROR"]} for expected in labels
    }
    for item in items:
        confusion[item.expected_category][item.predicted_category or "ERROR"] += 1

    distances = [severity_distance(i.expected_severity, i.predicted_severity) for i in items]
    latencies = [i.latency_ms for i in items if i.error is None]
    total_cost = sum(i.cost_usd for i in items)
    return EvalReport(
        total=total,
        errors=sum(1 for i in items if i.error),
        category_accuracy=round(
            _safe_div(sum(1 for i in items if i.predicted_category == i.expected_category), total),
            4,
        ),
        category_macro_f1=round(
            _safe_div(sum(m.f1 for m in per_category.values()), len(labels)), 4
        ),
        severity_accuracy=round(_safe_div(sum(1 for d in distances if d == 0), total), 4),
        severity_within_one=round(
            _safe_div(sum(1 for d in distances if d is not None and d <= 1), total), 4
        ),
        review_rate=round(_safe_div(sum(1 for i in items if i.needs_review), total), 4),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        total_cost_usd=round(total_cost, 6),
        avg_cost_usd=round(_safe_div(total_cost, total), 6),
        per_category=per_category,
        confusion=confusion,
    )


def to_markdown(report: EvalReport, title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tickets | {report.total} (errors: {report.errors}) |",
        f"| Category accuracy | {report.category_accuracy:.1%} |",
        f"| Category macro F1 | {report.category_macro_f1:.3f} |",
        f"| Severity exact | {report.severity_accuracy:.1%} |",
        f"| Severity within one level | {report.severity_within_one:.1%} |",
        f"| Sent to human review | {report.review_rate:.1%} |",
        f"| Latency p50 / p95 | {report.latency_p50_ms:.0f} ms / {report.latency_p95_ms:.0f} ms |",
        f"| Cost total / per ticket | US$ {report.total_cost_usd:.4f} / US$ {report.avg_cost_usd:.5f} |",
        "",
        "| Category | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label, m in report.per_category.items():
        lines.append(f"| {label} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} | {m.support} |")
    return "\n".join(lines)
