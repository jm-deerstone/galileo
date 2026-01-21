from __future__ import annotations
import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services import preprocessors
from services.customCode import run_custom_join, run_custom_step
from db import get_db
from models import DataSource, Snapshot, Preprocess, ExecutedPreprocess
from config import SNAPSHOT_BASE

from shemas.preprocess import (
    PreprocessRead,
    PreprocessCreate,
    ExecutedRead,
    ExecuteRequest,
    PreviewResponse,
    PreviewRequest,
)

router = APIRouter()

# Operation registry (non-custom ops)
OP_REGISTRY = {
    "rename_column": preprocessors.rename_column,
    "drop_columns": preprocessors.drop_columns,
    "filter_rows": preprocessors.filter_rows,
    "filter_outliers": preprocessors.filter_outliers,
    "impute_missing": preprocessors.impute_missing,
    "label_encode": preprocessors.label_encode,
    "one_hot_encode": preprocessors.one_hot_encode,
    "scale_numeric": preprocessors.scale_numeric,
    "log_transform": preprocessors.log_transform,
    "extract_datetime_features": preprocessors.extract_datetime_features,
    "remove_duplicates": preprocessors.remove_duplicates,
    "bin_numeric": preprocessors.bin_numeric,
    "normalize_text": preprocessors.normalize_text,
    "cap_outliers": preprocessors.cap_outliers,
}


@router.post("/preprocesses/", response_model=PreprocessRead)
def create_preprocess(body: PreprocessCreate, db: Session = Depends(get_db)):
    # 1) Create the new child DataSource
    child = DataSource(name=f"{body.name}_output")
    db.add(child)
    db.flush()

    # 2) Instantiate Preprocess
    pp = Preprocess(
        name=body.name,
        config=json.dumps(body.config),
        datasource_child_id=child.id,
    )

    # 3) Link parent datasources
    if len(body.parent_ids) not in (1, 2):
        raise HTTPException(400, "Preprocess must have 1 or 2 parent datasources")
    parents = db.query(DataSource).filter(DataSource.id.in_(body.parent_ids)).all()
    if len(parents) != len(body.parent_ids):
        raise HTTPException(404, "One or more parent datasources not found")
    pp.datasource_parents = parents

    db.add(pp)
    db.commit()
    db.refresh(pp)

    return PreprocessRead(
        id=pp.id,
        name=pp.name,
        parent_ids=[d.id for d in parents],
        child_id=child.id,
        config=body.config,
    )


@router.post(
    "/preprocesses/{pp_id}/execute/",
    response_model=ExecutedRead,
    summary="Execute a preprocess (with copy-if-active) and return its execution record",
)
def execute_preprocess(
        pp_id: str,
        req: ExecuteRequest,
        db: Session = Depends(get_db),
):
    # 1) load the Preprocess
    pp: Preprocess = db.get(Preprocess, pp_id)
    if not pp:
        raise HTTPException(404, "Preprocess not found")

    # parents (1 for unary, >=2 for join)
    parent_ds_list = pp.datasource_parents
    if not parent_ds_list:
        raise HTTPException(500, "No parent datasource configured")

    # 2) parse config
    cfg = json.loads(pp.config)
    steps = cfg.get("steps", [])
    is_join = any(s["op"] == "join" for s in steps)

    # container for snapshots we consumed (for lineage)
    input_snapshots: List[Snapshot] = []

    # helper: if consuming an active snapshot, first copy it, otherwise use original
    def maybe_copy_snap(snap: Snapshot) -> Snapshot:
        ds = db.get(DataSource, snap.datasource_id)
        if ds and ds.active_snapshot_id == snap.id:
            dest = os.path.join(SNAPSHOT_BASE, ds.id)
            os.makedirs(dest, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            copy_path = os.path.join(dest, f"{ts}_input.csv")
            shutil.copy(snap.path, copy_path)
            new_snap = Snapshot(datasource_id=ds.id, path=copy_path)
            db.add(new_snap)
            db.flush()
            return new_snap
        return snap

    # 3) initialize DataFrame
    if not is_join:
        if not req.snapshot_id:
            raise HTTPException(400, "`snapshot_id` required for non-join execution")
        orig = db.get(Snapshot, req.snapshot_id)
        if not orig:
            raise HTTPException(404, "Snapshot not found")
        snap_to_use = maybe_copy_snap(orig)
        df = pd.read_csv(snap_to_use.path)
        input_snapshots.append(snap_to_use)
    else:
        df = None  # join bootstraps per step

    details: List[Dict] = []

    # 4) apply each step
    for step in steps:
        op = step["op"]
        params = step.get("params", {}).copy()

        if op == "join":
            # Resolve left/right snapshots in this order:
            # 1) req.snapshots mapping (datasource_id -> snapshot_id)
            # 2) parents' active_snapshot_id
            # 3) legacy config fields: params["left_snapshot_id"], params["right_snapshot_id"]
            left_snap_id = None
            right_snap_id = None

            if req.snapshots and len(parent_ds_list) >= 2:
                p0, p1 = parent_ds_list[0].id, parent_ds_list[1].id
                left_snap_id = req.snapshots.get(p0)
                right_snap_id = req.snapshots.get(p1)

            if not left_snap_id or not right_snap_id:
                try:
                    if not left_snap_id:
                        left_snap_id = parent_ds_list[0].active_snapshot_id
                    if not right_snap_id and len(parent_ds_list) > 1:
                        right_snap_id = parent_ds_list[1].active_snapshot_id
                except Exception:
                    pass

            # fallback to legacy explicit params
            left_snap_id = left_snap_id or params.get("left_snapshot_id")
            right_snap_id = right_snap_id or params.get("right_snapshot_id")

            if not left_snap_id or not right_snap_id:
                raise HTTPException(400, "Join requires two input snapshots")

            ls = db.get(Snapshot, left_snap_id)
            rs = db.get(Snapshot, right_snap_id)
            if not ls or not rs:
                raise HTTPException(404, "Join snapshot not found")

            # maybe copy active ones
            ls2 = maybe_copy_snap(ls)
            rs2 = maybe_copy_snap(rs)
            input_snapshots.extend([ls2, rs2])

            left_df = pd.read_csv(ls2.path)
            right_df = pd.read_csv(rs2.path)
            how = params.get("how", "inner")
            if how == "custom":
                df = run_custom_join(left_df, right_df, params.get("code"), params)
            else:
                df = preprocessors.join_step(left_df, right_df, params)

            details.append(
                {
                    "op": op,
                    "left_snapshot_id": ls2.id,
                    "right_snapshot_id": rs2.id,
                    "how": how,
                }
            )

        elif op == "rename_column":
            old, new = params["from"], params["to"]
            df = preprocessors.rename_column(df, params)
            details.append({"op": op, "column_renames": {old: new}})

        elif op == "impute_missing":
            col, strat = params["column"], params["strategy"]
            if strat == "mode":
                fill_val = df[col].mode().iloc[0]
            elif strat in ("mean", "median"):
                fill_val = getattr(df[col], strat)()
            else:
                fill_val = params.get("fill_value")
            df = preprocessors.impute_missing(df, params)
            details.append(
                {"op": op, "column": col, "strategy": strat, "imputed_value": fill_val}
            )

        elif op == "label_encode":
            col = params["column"]
            cats = df[col].astype("category")
            mapping = {cat: code for code, cat in enumerate(cats.cat.categories)}
            df = preprocessors.label_encode(df, params)
            details.append({"op": op, "column": col, "mapping": mapping})

        elif op == "one_hot_encode":
            col = params["column"]
            cats = sorted(df[col].dropna().unique().tolist())
            df = preprocessors.one_hot_encode(df, params)
            details.append({"op": op, "column": col, "categories": cats})

        else:
            func = getattr(preprocessors, op, None)
            if func:
                df = func(df, params)
                details.append({"op": op, "params": params})
            else:
                df = run_custom_step(df, params.get("code", ""), params)
                details.append({"op": op, "custom_python": True})

    # 5) recompute final schema
    schema = {"columns": []}
    for col in df.columns:
        series = df[col]
        dtype = pd.api.types.infer_dtype(series, skipna=True)
        schema["columns"].append(
            {"name": col, "dtype": dtype, "null_count": int(series.isna().sum())}
        )

    # 6) persist updated schema on the child datasource
    child_ds = db.get(DataSource, pp.datasource_child_id)
    if not child_ds:
        raise HTTPException(500, "Child datasource not found")
    child_ds.schema_json = json.dumps(schema)
    db.add(child_ds)
    db.commit()

    # 7) write new output snapshot
    out_dir = os.path.join(SNAPSHOT_BASE, child_ds.id)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    out_path = os.path.join(out_dir, f"{ts}_exec.csv")
    df.to_csv(out_path, index=False, header=True)

    new_out = Snapshot(datasource_id=child_ds.id, path=out_path)
    db.add(new_out)
    db.flush()

    # 8) record execution
    exec_rec = ExecutedPreprocess(preprocess_id=pp.id, details_json=details)
    # link inputs
    for s in input_snapshots:
        exec_rec.snapshots.append(s)
    # link output
    exec_rec.snapshots.append(new_out)

    db.add(exec_rec)
    db.commit()
    db.refresh(exec_rec)

    # 9) build response
    input_ids = [s.id for s in input_snapshots]
    return ExecutedRead(
        id=exec_rec.id,
        preprocess_id=pp.id,
        input_snapshots=input_ids,
        output_snapshot=new_out.id,
        created_at=exec_rec.created_at,
        details=details,
    )


@router.get("/preprocesses/", response_model=List[PreprocessRead])
def list_preprocesses(db: Session = Depends(get_db)):
    records = db.query(Preprocess).all()
    return [
        PreprocessRead(
            id=pp.id,
            name=pp.name,
            parent_ids=[d.id for d in pp.datasource_parents],
            child_id=pp.datasource_child_id,
            config=json.loads(pp.config),
        )
        for pp in records
    ]


@router.get(
    "/preprocesses/{pp_id}",
    response_model=PreprocessRead,
    summary="Retrieve a single preprocess definition by ID",
)
def read_preprocess(pp_id: str, db: Session = Depends(get_db)):
    pp = db.get(Preprocess, pp_id)
    if not pp:
        raise HTTPException(status_code=404, detail="Preprocess not found")
    return PreprocessRead(
        id=pp.id,
        name=pp.name,
        parent_ids=[d.id for d in pp.datasource_parents],
        child_id=pp.datasource_child_id,
        config=json.loads(pp.config),
    )


@router.get(
    "/preprocesses/{pp_id}/executions/",
    response_model=List[ExecutedRead],
    summary="List all executions for a given preprocess",
)
def list_executions(pp_id: str, db: Session = Depends(get_db)):
    pp = db.get(Preprocess, pp_id)
    if not pp:
        raise HTTPException(status_code=404, detail="Preprocess not found")

    parent_ids = {ds.id for ds in pp.datasource_parents}
    child_id = pp.datasource_child_id

    records = (
        db.query(ExecutedPreprocess)
        .filter(ExecutedPreprocess.preprocess_id == pp_id)
        .order_by(ExecutedPreprocess.created_at)
        .all()
    )

    out: List[ExecutedRead] = []
    for exe in records:
        input_ids = [snap.id for snap in exe.snapshots if snap.datasource_id in parent_ids]
        output = next((s.id for s in exe.snapshots if s.datasource_id == child_id), None)
        if not input_ids or output is None:
            continue

        raw = exe.details_json or []
        if isinstance(raw, str):
            try:
                details_list = json.loads(raw)
            except json.JSONDecodeError:
                details_list = []
        elif isinstance(raw, list):
            details_list = raw
        else:
            details_list = []

        out.append(
            ExecutedRead(
                id=exe.id,
                preprocess_id=exe.preprocess_id,
                input_snapshots=input_ids,
                output_snapshot=output,
                created_at=exe.created_at,
                details=details_list,
            )
        )

    return out


@router.post(
    "/preprocesses/preview/",
    response_model=PreviewResponse,
    summary="Preview a preprocess without saving or creating a snapshot",
)
def preview_preprocess(req: PreviewRequest, db: Session = Depends(get_db)):
    steps: List[Dict[str, Any]] = req.config.get("steps", [])
    df = None

    # 1) If there's a join step, bootstrap from LEFT snapshot
    if any(s["op"] == "join" for s in steps):
        join_step = next(s for s in steps if s["op"] == "join")
        params = join_step.get("params", {})
        left_snap_id = params.get("left_snapshot_id")
        if not left_snap_id:
            raise HTTPException(400, "Missing left_snapshot_id for join preview")
        left_snap = db.get(Snapshot, left_snap_id)
        if not left_snap:
            raise HTTPException(404, "Left snapshot not found")
        df = pd.read_csv(left_snap.path)
    else:
        if not req.snapshot_id:
            raise HTTPException(400, "snapshot_id is required for non-join preview")
        snap = db.get(Snapshot, req.snapshot_id)
        if not snap:
            raise HTTPException(404, "Snapshot not found")
        df = pd.read_csv(snap.path)

    # 3) Apply every step
    for step in steps:
        op_name = step["op"]
        params = step.get("params", {})
        if op_name == "join":
            ls = db.get(Snapshot, params["left_snapshot_id"])
            rs = db.get(Snapshot, params["right_snapshot_id"])
            left_df = pd.read_csv(ls.path)
            right_df = pd.read_csv(rs.path)
            if params.get("how") == "custom":
                df = run_custom_join(left_df, right_df, params.get("code"), params)
            else:
                df = preprocessors.join_step(left_df, right_df, params)
        elif op_name == "custom_python":
            code = params.get("code", "")
            df = run_custom_step(df, code, params)
        else:
            func = OP_REGISTRY.get(op_name)
            if not func:
                raise HTTPException(400, f"Unknown operation '{op_name}'")
            df = func(df, params)

    preview = df.head(5)
    return PreviewResponse(columns=preview.columns.tolist(), rows=preview.astype(str).values.tolist())
