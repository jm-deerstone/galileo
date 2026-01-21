from datetime import datetime
import os
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, model_validator, validator, field_validator


# Pydantic schemas
class PreprocessCreate(BaseModel):
    name: str
    parent_ids: List[str]
    config: Dict[str, Any]   # e.g. {"steps":[{...}, ...]}

class PreprocessRead(BaseModel):
    id: str
    name: str
    parent_ids: List[str]
    child_id: str
    config: Dict[str, Any]

    class Config:
        orm_mode = True

class ExecuteRequest(BaseModel):
    snapshot_id: Optional[str] = None
    # for join steps, map from datasource_id → snapshot_id
    snapshots: Optional[Dict[str, str]] = None

class ExecutedRead(BaseModel):
    id: str
    preprocess_id: str
    input_snapshots: List[str]
    output_snapshot: str
    created_at: datetime
    details: List[Dict[str, Any]]

    class Config:
        orm_mode = True

class PreviewRequest(BaseModel):
    config: Dict[str, Any]
    # only used for non-join previews
    snapshot_id: Optional[str] = None

class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[list[str]]