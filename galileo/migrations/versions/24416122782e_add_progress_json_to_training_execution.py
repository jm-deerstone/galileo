"""add progress_json to training_execution

Revision ID: 24416122782e
Revises: 8ec14381ca7c
Create Date: 2025-11-05 17:02:06.921317

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '24416122782e'
down_revision: Union[str, Sequence[str], None] = '8ec14381ca7c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "training_execution",
        sa.Column("progress_json", sa.VARCHAR(), nullable=True),
    )



def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_execution", "progress_json")
