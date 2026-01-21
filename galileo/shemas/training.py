from datetime import datetime
import os
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, model_validator, validator, field_validator


class TrainRequest(BaseModel):
    snapshot_id: str

class TrainingCreate(BaseModel):
    name: str
    datasource_id: str
    config: dict   # {"algorithm":"random_forest","params":{...},"features":[...],"target":"..."}

class TrainingRead(BaseModel):
    id: str
    name: str
    datasource_id: str
    config_json: dict    # or `str` if you want the raw JSON string
    input_schema_json: dict
    class Config_json:
        from_attributes = True

class TrainingExecutionRead(BaseModel):
    id: str
    training_id: str
    snapshot_id: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    metrics_json: str
    model_path: str
    class Config:
        from_attributes = True


class TrainingPreviewRequest(BaseModel):
    snapshot_id: str
    config: dict   # expects { algorithm, params, features, target }
