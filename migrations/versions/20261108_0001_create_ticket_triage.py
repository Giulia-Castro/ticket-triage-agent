"""create ticket_triage

Revision ID: 0001
Revises:
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ticket_triage",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("subject_masked", sa.String(200), nullable=False),
        sa.Column("body_masked", sa.Text(), nullable=False),
        sa.Column("category", sa.String(30), nullable=False),
        sa.Column("model_severity", sa.String(20), nullable=False),
        sa.Column("affected_scope", sa.String(20), nullable=False),
        sa.Column("product_area", sa.String(100), nullable=False),
        sa.Column("summary", sa.String(500), nullable=False),
        sa.Column("suggested_reply", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("kb_article_ids", sa.String(500), nullable=False),
        sa.Column("triage_status", sa.String(20), nullable=False),
        sa.Column("queue_name", sa.String(30), nullable=False),
        sa.Column("final_severity", sa.String(20), nullable=False),
        sa.Column("sla_hours", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(60), nullable=False),
        sa.Column("prompt_version", sa.String(20), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("feedback_category", sa.String(30), nullable=True),
        sa.Column("feedback_severity", sa.String(20), nullable=True),
        sa.Column("createdat", sa.DateTime(), nullable=False),
        sa.Column("createdby", sa.String(60), nullable=False),
        sa.Column("updatedat", sa.DateTime(), nullable=True),
        sa.Column("updatedby", sa.String(60), nullable=True),
        sa.Column("rowversion", sa.Integer(), nullable=False),
    )
    op.create_index("ix_ticket_triage_createdat", "ticket_triage", ["createdat"])
    op.create_index("ix_ticket_triage_category", "ticket_triage", ["category"])


def downgrade() -> None:
    op.drop_index("ix_ticket_triage_category", table_name="ticket_triage")
    op.drop_index("ix_ticket_triage_createdat", table_name="ticket_triage")
    op.drop_table("ticket_triage")
