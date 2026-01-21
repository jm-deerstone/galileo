from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, model_validator, validator, field_validator


class DeploymentCreate(BaseModel):
    training_id: str

class DeploymentRead(BaseModel):
    id: str
    training_id: str

    class Config:
        orm_mode = True


class ModelDeploymentCreate(BaseModel):
    deployment_id: str
    training_execution_id: str

class ModelDeploymentRead(BaseModel):
    id: str
    deployment_id: str
    training_execution_id: str

class PredictRequest(BaseModel):
    # either map column→value, or a flat list of floats
    features: Union[Dict[str, Any], List[float]]

class PredictResponse(BaseModel):
    prediction: Any

class MonitorRead(BaseModel):
    model_deployment_id: str
    evaluated_on_snapshot: List[str]        # ← change here
    timestamp: datetime
    metrics: Dict[str, Optional[float]]