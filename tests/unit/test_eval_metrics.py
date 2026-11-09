from __future__ import annotations

from evals.metrics import EvalItem, build_report, percentile, severity_distance, to_markdown

LABELS = ["ACCESS", "BILLING"]


def _item(exp: str, pred: str | None, sev: str = "LOW", psev: str | None = "LOW", **kw) -> EvalItem:
    return EvalItem(
        ticket_id="x",
        expected_category=exp,
        expected_severity=sev,
        predicted_category=pred,
        predicted_severity=psev,
        needs_review=kw.get("review", False),
        latency_ms=kw.get("latency", 100),
        cost_usd=kw.get("cost", 0.01),
        error=kw.get("error"),
    )


def test_report_metrics() -> None:
    items = [
        _item("ACCESS", "ACCESS"),
        _item("ACCESS", "BILLING", sev="HIGH", psev="MEDIUM", review=True),
        _item("BILLING", "BILLING", sev="LOW", psev="CRITICAL", latency=300),
        _item("BILLING", None, psev=None, error="technical", cost=0.0),
    ]
    report = build_report(items, LABELS)
    assert report.total == 4
    assert report.errors == 1
    assert report.category_accuracy == 0.5
    assert report.severity_accuracy == 0.25
    assert report.severity_within_one == 0.5
    assert report.review_rate == 0.25
    assert report.per_category["ACCESS"].precision == 1.0
    assert report.per_category["ACCESS"].recall == 0.5
    assert report.confusion["BILLING"]["ERROR"] == 1
    assert report.total_cost_usd == 0.03
    assert "Category accuracy | 50.0%" in to_markdown(report, "test")


def test_helpers() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([10, 20, 30, 40], 50) == 20.0
    assert percentile([10, 20, 30, 40], 95) == 40.0
    assert severity_distance("LOW", "HIGH") == 2
    assert severity_distance("LOW", None) is None
    assert build_report([], LABELS).category_accuracy == 0.0
