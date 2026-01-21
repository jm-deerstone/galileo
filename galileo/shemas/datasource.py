from datetime import datetime
import os
from typing import List, Dict, Any

from pydantic import BaseModel, model_validator, validator, field_validator



# --- Pydantic Schemas ---
class DataSourceCreate(BaseModel):
    name: str

class DataSourceRead(BaseModel):
    id: str
    name: str
    class Config:
        orm_mode = True

class SnapshotRead(BaseModel):
    id: str
    path: str
    created_at: datetime
    size_bytes: int
    class Config:
        orm_mode = True


class DataSourceWithSnapshotRead(BaseModel):
    id: str
    name: str
    schema_json: str | None
    snapshots: list[SnapshotRead]
    active_snapshot_id: str | None

    class Config:
        orm_mode = True

class ColumnSummary(BaseModel):
    column: str
    type: str           # "numeric", "categorical", or "date"
    missing: int
    missing_pct: float
    unique: int
    stats: str

class SnapshotSummary(BaseModel):
    summary: list[ColumnSummary]


class ActiveSnapshotSet(BaseModel):
    snapshot_id: str

class RowsInsertRequest(BaseModel):
    rows: List[Dict[str, Any]]

    @field_validator("rows")
    def must_have_at_least_one(cls, v):
        if len(v) == 0:
            raise ValueError("Must supply at least one row")
        return v
