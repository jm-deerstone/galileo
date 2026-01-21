from __future__ import annotations
import json
from datetime import datetime
from typing import Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from db import DATABASE_URL
from models import DataSource, Snapshot, Preprocess, Training, TrainingExecution
from routers.preprocess import execute_preprocess
from shemas.preprocess import ExecuteRequest
from services.Trainings import run_training


def _root_active_snapshot_id(db: Session, ds_id: str) -> str:
    ds = db.get(DataSource, ds_id)
    if not ds or not ds.active_snapshot_id:
        raise HTTPException(400, f"No active snapshot for datasource {ds_id}")
    return ds.active_snapshot_id


def _preprocess_producing_child(db: Session, child_ds_id: str) -> Optional[Preprocess]:
    return (
        db.query(Preprocess)
        .filter(Preprocess.datasource_child_id == child_ds_id)
        .first()
    )


def _materialize_to_snapshot(db: Session, target_ds_id: str, memo: Dict[str, str]) -> str:
    """Return a snapshot_id representing the latest data for target_ds_id,
    materializing upstream preprocesses as needed."""
    if target_ds_id in memo:
        return memo[target_ds_id]

    pp = _preprocess_producing_child(db, target_ds_id)
    if not pp:
        snap_id = _root_active_snapshot_id(db, target_ds_id)
        memo[target_ds_id] = snap_id
        return snap_id

    parents = pp.datasource_parents
    if not parents:
        raise HTTPException(500, f"Preprocess {pp.id} has no parents")

    cfg = json.loads(pp.config)
    steps = cfg.get("steps", [])
    is_join = any(s["op"] == "join" for s in steps)

    if is_join:
        mapping = {p.id: _materialize_to_snapshot(db, p.id, memo) for p in parents}
        req = ExecuteRequest(snapshot_id=None, snapshots=mapping)
        exe = execute_preprocess(pp.id, req, db)  # ExecutedRead
        out_snap_id = exe.output_snapshot
    else:
        p0 = parents[0]
        parent_snap_id = _materialize_to_snapshot(db, p0.id, memo)
        req = ExecuteRequest(snapshot_id=parent_snap_id, snapshots=None)
        exe = execute_preprocess(pp.id, req, db)
        out_snap_id = exe.output_snapshot

    memo[target_ds_id] = out_snap_id
    return out_snap_id


def run_automation_for_training(db: Session, training_id: str) -> TrainingExecution:
    """Materialize entire pipeline (roots → preprocess chain) and run training once."""
    tr: Training = db.get(Training, training_id)
    if not tr:
        raise HTTPException(404, "Training not found")

    final_snap_id = _materialize_to_snapshot(db, tr.datasource_id, memo={})

    exec_rec = TrainingExecution(
        training_id=training_id,
        snapshot_id=final_snap_id,
        status="running",
        started_at=datetime.utcnow(),
        finished_at=None,
        metrics_json="",
        model_path="",
    )
    db.add(exec_rec)
    db.commit()
    db.refresh(exec_rec)

    # Run training synchronously; APScheduler runs jobs in its pool.
    run_training(execution_id=exec_rec.id, db_url=DATABASE_URL)

    return exec_rec
