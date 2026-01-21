# src/controllers/trainings.py  (or wherever this router lives)
from __future__ import annotations
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from services.Trainings import ALG_REGISTRY, CLASSIFIERS, REGRESSORS, run_training
from db import get_db, DATABASE_URL
from models import (
    DataSource,
    Snapshot,
    Training,
    TrainingExecution,
    ExecutedPreprocess,
    ModelDeployment,
    Deployment,
)
from services.automation_scheduler import TrainingAutomationScheduler
from shemas.training import (
    TrainingRead,
    TrainingCreate,
    TrainingPreviewRequest,
    TrainRequest,
    TrainingExecutionRead,
)
from services.automation import run_automation_for_training

logger = logging.getLogger("trainings")
logger.setLevel(logging.DEBUG)

router = APIRouter()

# Optional: Ray preview (kept from your code)
import ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# --------- SCHEDULER BOOTSTRAP ---------
try:
    scheduler  # type: ignore
except NameError:
    scheduler = TrainingAutomationScheduler(DATABASE_URL)
    scheduler.start()
    scheduler.warm_boot()
# ---------------------------------------


@router.post("/trainings/", response_model=TrainingRead)
def create_training(body: TrainingCreate, db: Session = Depends(get_db)):
    ds = db.get(DataSource, body.datasource_id)
    if not ds:
        raise HTTPException(404, "Datasource not found")

    full_schema = json.loads(ds.schema_json)
    wanted = set(body.config.get("features", []) + [body.config.get("target")])
    input_cols = [col for col in full_schema.get("columns", []) if col.get("name") in wanted]
    input_schema = {"columns": input_cols}

    tr = Training(
        name=body.name,
        datasource_id=body.datasource_id,
        config_json=json.dumps(body.config),
        input_schema_json=json.dumps(input_schema),
    )
    db.add(tr)
    db.commit()
    db.refresh(tr)

    return TrainingRead(
        id=tr.id,
        name=tr.name,
        datasource_id=tr.datasource_id,
        config_json=body.config,
        input_schema_json=input_schema,
        created_at=tr.created_at,
    )


@router.get("/trainings/", response_model=List[TrainingRead])
def list_trainings(db: Session = Depends(get_db)):
    trainings = db.query(Training).all()
    return [
        {
            "id": tr.id,
            "name": tr.name,
            "datasource_id": tr.datasource_id,
            "config_json": json.loads(tr.config_json),
            "input_schema_json": json.loads(tr.input_schema_json),
        }
        for tr in trainings
    ]


@router.get("/trainings/{tr_id}", response_model=TrainingRead)
def get_training(tr_id: str, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")
    return {
        "id": tr.id,
        "name": tr.name,
        "datasource_id": tr.datasource_id,
        "config_json": json.loads(tr.config_json),
        "input_schema_json": json.loads(tr.input_schema_json),
    }


@router.get("/trainings/{tr_id}/allowed_metrics/")
def get_allowed_metrics(tr_id: str, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")
    config = json.loads(tr.config_json)
    alg = config.get("algorithm")
    if alg in CLASSIFIERS:
        return ["accuracy"]
    elif alg in REGRESSORS:
        return ["r2", "mse", "rmse"]
    else:
        return []


@router.post("/trainings/preview/")
def preview_training(req: TrainingPreviewRequest, db: Session = Depends(get_db)):
    snap = db.get(Snapshot, req.snapshot_id)
    if not snap:
        return JSONResponse(status_code=404, content={"error": "Snapshot not found"})
    try:
        df = pd.read_csv(snap.path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read snapshot: {e}"})

    cfg = req.config or {}
    alg = cfg.get("algorithm")
    params = cfg.get("params", {}).copy()
    features = cfg.get("features", [])
    target = cfg.get("target")

    window_spec = params.pop("window_spec", None) or cfg.get("window_spec")
    ModelCls = ALG_REGISTRY.get(alg)
    if ModelCls is None:
        return JSONResponse(status_code=400, content={"error": f"Unsupported algorithm {alg}"})

    if window_spec:
        try:
            fw = [
                {"name": p["name"], "start_idx": int(p["start_idx"]), "end_idx": int(p["end_idx"])}
                for p in window_spec["features"]
            ]
            tw = {
                "name": window_spec["target"]["name"],
                "start_idx": int(window_spec["target"]["start_idx"]),
                "end_idx": int(window_spec["target"]["end_idx"]),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Bad window_spec: {e}"})

        X_rows, y_rows = [], []
        max_end = max(p["end_idx"] for p in fw + [tw])
        for i in range(max_end, len(df)):
            row_feats = []
            for p in fw:
                vals = df[p["name"]].iloc[i - p["end_idx"] : i - p["start_idx"] + 1]
                row_feats.extend(vals.tolist())
            X_rows.append(row_feats)
            tvals = df[tw["name"]].iloc[i - tw["end_idx"] : i - tw["start_idx"] + 1]
            y_rows.append(tvals.tolist() if len(tvals) > 1 else tvals.iloc[0])

        X_fit = pd.DataFrame(X_rows)
        y_df = pd.DataFrame(y_rows)
        y_fit = y_df.iloc[:, 0] if y_df.shape[1] == 1 else y_df
    else:
        if not features:
            return JSONResponse(status_code=400, content={"error": "No feature columns provided."})
        if not target:
            return JSONResponse(status_code=400, content={"error": "No target column provided."})

        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing columns: {missing}"})

        X_fit = df[features]
        y_fit = df[target]

    try:
        is_multitarget = isinstance(y_fit, pd.DataFrame) and y_fit.shape[1] > 1
        if alg in CLASSIFIERS:
            def check_series(s):
                if pd.api.types.is_float_dtype(s):
                    raise ValueError("Classification algorithms require discrete target values.")
            if is_multitarget:
                for col in y_fit.columns:
                    check_series(y_fit[col])
            else:
                check_series(y_fit)
        elif alg in REGRESSORS:
            def check_reg(s):
                if not pd.api.types.is_numeric_dtype(s):
                    raise ValueError("Regression algorithms require numeric target values.")
            if is_multitarget:
                for col in y_fit.columns:
                    check_reg(y_fit[col])
            else:
                check_reg(y_fit)

        X_fit = X_fit.iloc[:50]
        y_fit = y_fit.iloc[:50]

        ray_algos = {"xgboost_ray_cls", "xgboost_ray_reg", "lightgbm_ray_cls", "lightgbm_ray_reg"}
        is_ray_algo = alg in ray_algos
        if is_ray_algo:
            logger.info(f"Using Ray algorithm: {alg}")
            from xgboost_ray import RayDMatrix, train as xgb_ray_train, RayParams as XGBRayParams
            dtrain = RayDMatrix(X_fit, y_fit)
            ray_params = XGBRayParams(num_actors=1, cpus_per_actor=1)
            xgb_ray_train(
                params={"objective": "binary:logistic" if alg == "xgboost_ray_cls" else "reg:squarederror", **params},
                dtrain=dtrain,
                num_boost_round=10,
                ray_params=ray_params,
            )
        else:
            logger.info(f"Using sklearn algorithm: {alg}")
            base = ModelCls(**params)
            if is_multitarget:
                wrapper = MultiOutputClassifier if alg in CLASSIFIERS else MultiOutputRegressor
                model = wrapper(base)
            else:
                model = base
            model.fit(X_fit, y_fit)
    except TypeError as e:
        if "getaddrinfo" in str(e):
            logger.warning("Ray/XGBoost tracker bug detected in preview! Falling back to classic XGBoost for preview.")
            import xgboost as xgb
            model = xgb.XGBRegressor(**params)
            model.fit(X_fit, y_fit)
            return {"ok": True, "warning": "Ray preview failed (tracker bug), used classic XGBoost."}
        logger.error("EXCEPTION in /trainings/preview/:", exc_info=True)
        return JSONResponse(status_code=400, content={"error": f"{type(e).__name__}: {e}"})
    except Exception as e:
        logger.error("EXCEPTION in /trainings/preview/:", exc_info=True)
        return JSONResponse(status_code=400, content={"error": f"{type(e).__name__}: {e}"})

    return {"ok": True}


@router.post("/trainings/{tr_id}/execute/", response_model=TrainingExecutionRead, status_code=202)
def execute_training(
        tr_id: str,
        req: TrainRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")

    exec_rec = TrainingExecution(
        training_id=tr_id,
        snapshot_id=req.snapshot_id,
        status="running",
        started_at=datetime.utcnow(),
        finished_at=None,
        metrics_json="",
        model_path="",
    )
    db.add(exec_rec)
    db.commit()
    db.refresh(exec_rec)

    background_tasks.add_task(run_training, execution_id=exec_rec.id, db_url=DATABASE_URL)
    return exec_rec


@router.get("/trainings/{tr_id}/executions/", response_model=List[TrainingExecutionRead])
def list_training_executions(tr_id: str, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")
    return db.query(TrainingExecution).filter_by(training_id=tr_id).all()


@router.get(
    "/trainings/{training_id}/executions/{exec_id}/preprocess_steps/",
    summary="List all preprocess‐step details for a given training execution",
    response_model=List[Dict[str, Any]],
)
def list_preprocess_steps(training_id: str, exec_id: str, db: Session = Depends(get_db)):
    te = db.get(TrainingExecution, exec_id)
    if not te or te.training_id != training_id:
        raise HTTPException(404, "Training execution not found")

    final_snap_id = te.snapshot_id
    all_exes: List[ExecutedPreprocess] = db.query(ExecutedPreprocess).all()
    ordered_details: List[Dict[str, Any]] = []

    def recurse(snap_id: str):
        for exe in all_exes:
            child_ds = exe.preprocess.datasource_child_id
            produced = next(
                (s for s in exe.snapshots if s.id == snap_id and s.datasource_id == child_ds),
                None,
            )
            if not produced:
                continue

            parent_ids = {ds.id for ds in exe.preprocess.datasource_parents}
            for inp in exe.snapshots:
                if inp.datasource_id in parent_ids:
                    recurse(inp.id)

            details = exe.details_json or []
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except json.JSONDecodeError:
                    details = []
            if isinstance(details, list):
                ordered_details.extend(details)
            return

    recurse(final_snap_id)
    return ordered_details


# ------------- AUTOMATION API -------------
@router.get("/trainings/{tr_id}/automation_config/")
def get_automation_config(tr_id: str, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")
    return {
        "automation_enabled": tr.automation_enabled,
        "automation_schedule": tr.automation_schedule,
        "promotion_metrics": tr.promotion_metrics.split(",") if tr.promotion_metrics else [],
    }


@router.put("/trainings/{tr_id}/automation_config/")
def set_automation_config(tr_id: str, config: dict, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")

    tr.automation_enabled = bool(config.get("automation_enabled", False))
    tr.automation_schedule = config.get("automation_schedule") or ""
    metrics = config.get("promotion_metrics", []) or []
    tr.promotion_metrics = ",".join(metrics)
    db.commit()

    # keep scheduler in sync
    try:
        scheduler.add_or_update_training(tr)
    except Exception as e:
        logger.warning("Scheduler add_or_update_training failed for %s: %s", tr_id, e)
    return {"ok": True}


@router.delete("/trainings/{tr_id}/automation_config/", status_code=200)
def delete_automation_config(tr_id: str, db: Session = Depends(get_db)):
    """
    Clear automation settings for a training and remove its scheduled job.
    """
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")

    tr.automation_enabled = False
    tr.automation_schedule = ""
    tr.promotion_metrics = ""
    db.commit()

    # remove schedule
    try:
        scheduler.remove_training(tr_id)
    except Exception as e:
        logger.warning("Scheduler remove_training failed for %s: %s", tr_id, e)

    return {"ok": True}


@router.post("/trainings/{tr_id}/automation/run_now", response_model=TrainingExecutionRead)
def automation_run_now(tr_id: str, db: Session = Depends(get_db)):
    exec_rec = run_automation_for_training(db, tr_id)
    return exec_rec
# ------------- END AUTOMATION API -------------


@router.get("/training_executions/{exec_id}/progress/")
def get_training_progress(exec_id: str, db: Session = Depends(get_db)):
    exec_rec = db.get(TrainingExecution, exec_id)
    if not exec_rec or not exec_rec.progress_json:
        return {"progress": 0, "phase": "Starting", "detail": ""}
    return json.loads(exec_rec.progress_json)


@router.delete("/trainings/{tr_id}/delete", status_code=204)
def delete_training(tr_id: str, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")

    # IMPORTANT: unschedule before deleting records
    try:
        scheduler.remove_training(tr_id)
    except Exception as e:
        logger.warning("Scheduler remove_training failed for %s during training delete: %s", tr_id, e)

    executions = db.query(TrainingExecution).filter_by(training_id=tr_id).all()
    for exec_rec in executions:
        db.delete(exec_rec)

    db.query(ModelDeployment).filter(
        ModelDeployment.deployment_id.in_(db.query(Deployment.id).filter_by(training_id=tr_id))
    ).delete(synchronize_session=False)
    db.query(Deployment).filter_by(training_id=tr_id).delete(synchronize_session=False)

    db.delete(tr)
    db.commit()

    import shutil
    from services.Trainings import MODELS_DIR
    models_path = os.path.join(MODELS_DIR, tr_id)
    if os.path.exists(models_path):
        shutil.rmtree(models_path)

    return {"ok": True}
