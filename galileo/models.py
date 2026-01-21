import uuid, os
from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
# backend/main.py
import io
import json
import os
import re
from datetime import datetime
from pydantic import BaseModel, model_validator
from sqlalchemy import (
    create_engine, Column, String, Table, ForeignKey, DateTime, Boolean
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

import uuid

from db import Base

# --- Association Tables ---
preprocess_parents = Table(
    'preprocess_parents', Base.metadata,
    Column('preprocess_id', String, ForeignKey('preprocess.id'), primary_key=True),
    Column('datasource_id', String, ForeignKey('datasource.id'), primary_key=True)
)
execution_snapshots = Table(
    'execution_snapshots', Base.metadata,
    Column('executed_preprocess_id', String, ForeignKey('executed_preprocess.id'), primary_key=True),
    Column('snapshot_id', String, ForeignKey('snapshot.id'), primary_key=True)
)

# --- ORM Models ---
class DataSource(Base):
    __tablename__ = 'datasource'
    id      = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name    = Column(String, nullable=False)
    schema_json = Column(String, nullable=True)  # new column to store schema info

    active_snapshot_id   = Column(String, ForeignKey('snapshot.id'), nullable=True)
    active_snapshot      = relationship(
        "Snapshot",
        foreign_keys=[active_snapshot_id],
        uselist=False
    )

    snapshots            = relationship(
        "Snapshot",
        back_populates="datasource",
        foreign_keys="[Snapshot.datasource_id]",    # ← here
        primaryjoin="DataSource.id == Snapshot.datasource_id"
    )

    preprocess_children = relationship("Preprocess", back_populates="datasource_child")
    training_children   = relationship("Training", back_populates="datasource")


class Preprocess(Base):
    __tablename__ = 'preprocess'
    id                  = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name                = Column(String, nullable=False)
    # JSON config specifying a list of {op, params}
    config              = Column(String, nullable=False)
    datasource_child_id = Column(String, ForeignKey('datasource.id'), nullable=False)
    datasource_child    = relationship("DataSource", back_populates="preprocess_children")
    datasource_parents  = relationship(
        "DataSource", secondary=preprocess_parents, backref="preprocess_parents"
    )
    executed_children   = relationship("ExecutedPreprocess", back_populates="preprocess")

class Snapshot(Base):
    __tablename__ = 'snapshot'
    id                   = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    path           = Column(String, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)

    datasource_id        = Column(String, ForeignKey('datasource.id'), nullable=False)
    datasource           = relationship(
        "DataSource",
        back_populates="snapshots",
        foreign_keys=[datasource_id]               # ← and here
    )

    executions           = relationship(
        "ExecutedPreprocess", secondary=execution_snapshots, back_populates="snapshots"
    )
    training_executions  = relationship("TrainingExecution", back_populates="snapshot")

    @property
    def size_bytes(self) -> int:
        """
        Compute the file size of self.path on disk. If the file doesn't exist, return 0.
        """
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

class ExecutedPreprocess(Base):
    __tablename__ = 'executed_preprocess'
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    preprocess_id  = Column(String, ForeignKey('preprocess.id'), nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)
    details_json = Column(JSON, nullable=True)
    preprocess     = relationship("Preprocess", back_populates="executed_children")
    snapshots      = relationship(
        "Snapshot",
        secondary=execution_snapshots,
        back_populates="executions"
    )

class Training(Base):
    __tablename__ = 'training'
    id                = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name              = Column(String, nullable=False)
    datasource_id     = Column(String, ForeignKey('datasource.id'), nullable=False)
    config_json       = Column(String, nullable=False)
    input_schema_json = Column(Text, nullable=False)
    created_at        = Column(DateTime, default=datetime.utcnow)

    automation_enabled = Column(Boolean, default=False)
    automation_schedule = Column(String, nullable=True)
    promotion_metrics = Column(String, nullable=True)  # Comma-separated string

    # one Training → one Deployment
    deployment        = relationship(
        "Deployment",
        back_populates="training",
        uselist=False
    )

    datasource        = relationship("DataSource", back_populates="training_children")
    executions        = relationship("TrainingExecution", back_populates="training")


class TrainingExecution(Base):
    __tablename__ = 'training_execution'
    id                 = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    training_id        = Column(String, ForeignKey('training.id'), nullable=False)
    snapshot_id        = Column(String, ForeignKey('snapshot.id'), nullable=False)
    status             = Column(String, default="running")
    progress_json = Column(String, default="{}")
    started_at         = Column(DateTime, default=datetime.utcnow)
    finished_at        = Column(DateTime, nullable=True)
    metrics_json       = Column(String, nullable=True)
    model_path         = Column(String, nullable=True)

    training           = relationship("Training", back_populates="executions")
    snapshot           = relationship("Snapshot", back_populates="training_executions")
    deployments        = relationship("ModelDeployment", back_populates="execution")


class Deployment(Base):
    __tablename__ = 'deployment'
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    training_id    = Column(String, ForeignKey('training.id'), nullable=False)

    # one Deployment → one parent Training
    training       = relationship(
        "Training",
        back_populates="deployment"
    )

    # one Deployment → many ModelDeployments
    deployments    = relationship(
        "ModelDeployment",
        back_populates="deployment"
    )


class ModelDeployment(Base):
    __tablename__ = 'model_deployment'
    id                      = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    deployment_id           = Column(String, ForeignKey('deployment.id'), nullable=False)
    training_execution_id   = Column(String, ForeignKey('training_execution.id'), nullable=False)
    promotion_type = Column(String, nullable=False, default="manual")  # "manual" or "metric"
    metric = Column(String, nullable=True)  # If promotion_type=="metric", the metric name
    locked = Column(Boolean, default=False)
    # back to Deployment
    deployment              = relationship(
        "Deployment",
        back_populates="deployments"
    )
    # back to TrainingExecution
    execution               = relationship(
        "TrainingExecution",
        back_populates="deployments"
    )