from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from services import preprocessors
from services.customCode import run_custom_join, run_custom_step
from db import get_db
from models import DataSource, Snapshot, Preprocess, Training, Deployment, TrainingExecution, ModelDeployment

from shemas.deployment import DeploymentRead, DeploymentCreate, ModelDeploymentRead, ModelDeploymentCreate, \
    PredictResponse, PredictRequest, MonitorRead
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
router = APIRouter()


@router.post(
    "/deployments/",
    response_model=DeploymentRead,
)
def create_deployment(
        body: DeploymentCreate,
        db: Session = Depends(get_db),
):
    # ensure the training exists
    training = db.get(Training, body.training_id)
    if not training:
        raise HTTPException(
            404,
            detail="Training not found",
        )

    dep = Deployment(training_id=body.training_id)
    db.add(dep)
    db.commit()
    db.refresh(dep)
    return dep


@router.get(
    "/deployments/",
    response_model=List[DeploymentRead],
)
def list_deployments(db: Session = Depends(get_db)):
    return db.query(Deployment).all()


@router.get(
    "/deployments/{dep_id}",
    response_model=DeploymentRead,
)
def get_deployment(dep_id: str, db: Session = Depends(get_db)):
    dep = db.get(Deployment, dep_id)
    if not dep:
        raise HTTPException(
            status_code=404,
            detail="Deployment not found",
        )
    return dep

_loaded_models: Dict[str, Any] = {}

@router.get("/deployments/by_training/{training_id}/")
def get_deployments_for_training(training_id: str, db: Session = Depends(get_db)):
    dep = db.query(Deployment).filter_by(training_id=training_id).first()
    if not dep:
        return []
    # All deployments (manual and auto, per metric)
    deployments = (
        db.query(ModelDeployment)
        .filter_by(deployment_id=dep.id)
        .all()
    )
    return [
        {
            "id": md.id,
            "promotion_type": md.promotion_type,
            "metric": md.metric,
            "locked": md.locked,
            "training_execution_id": md.training_execution_id,
            # ...add other fields as needed...
        }
        for md in deployments
    ]

@router.post("/trainings/{tr_id}/executions/{exec_id}/promote_manual/")
def promote_manual(tr_id: str, exec_id: str, db: Session = Depends(get_db)):
    tr = db.get(Training, tr_id)
    if not tr:
        raise HTTPException(404, "Training not found")
    exec_rec = db.get(TrainingExecution, exec_id)
    if not exec_rec or exec_rec.status != "success":
        raise HTTPException(400, "Only successful executions can be promoted")
    dep = db.query(Deployment).filter_by(training_id=tr.id).first()
    if not dep:
        dep = Deployment(training_id=tr.id)
        db.add(dep)
        db.commit()
        db.refresh(dep)
    # Remove previous locked manual (optional, or just let them coexist)
    db.query(ModelDeployment).filter_by(
        deployment_id=dep.id, promotion_type="manual", locked=True
    ).delete()
    # Create new locked deployment
    md = ModelDeployment(
        deployment_id=dep.id,
        training_execution_id=exec_id,
        promotion_type="manual",
        locked=True,
        metric=None,
    )
    db.add(md)
    db.commit()
    return {"ok": True}

@router.get(
    "/deployments/{deployment_id}/model_deployments/",
    response_model=List[ModelDeploymentRead],
)
def list_model_deployments(deployment_id: str, db: Session = Depends(get_db)):
    dep = db.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(404, "Deployment not found")
    # thanks to the relationship on Deployment.deployments
    return dep.deployments

@router.post("/model_deployments/", response_model=ModelDeploymentRead)
def create_model_deployment(
        body: ModelDeploymentCreate,
        db: Session = Depends(get_db),
):
    # 1) verify the underlying Deployment
    dep = db.get(Deployment, body.deployment_id)
    if not dep:
        raise HTTPException(404, "Deployment not found")

    # 2) verify the execution exists and succeeded
    exec_rec = db.get(TrainingExecution, body.training_execution_id)
    if not exec_rec or exec_rec.status != "success":
        raise HTTPException(400, "Can only deploy a successful training execution")

    # 3) create the ModelDeployment row
    md = ModelDeployment(
        deployment_id=body.deployment_id,
        training_execution_id=body.training_execution_id,
    )
    db.add(md)
    db.commit()
    db.refresh(md)

    # 4) load the model file into RAM under md.id
    try:
        with open(exec_rec.model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    _loaded_models[md.id] = model

    return ModelDeploymentRead(
        id=md.id,
        deployment_id=md.deployment_id,
        training_execution_id=md.training_execution_id,
    )


# ─── /model_deployments/{id}/predict ─────────────────────────────────────────────

from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
import os, pickle, json, pandas as pd

# … your imports, _loaded_models, get_db, etc.
@router.post(
    "/model_deployments/{model_deployment_id}/predict",
    response_model=PredictResponse,
)
def predict(
        model_deployment_id: str,
        req: PredictRequest,
        db: Session = Depends(get_db),
):
    # 0) Load ModelDeployment record
    md = db.get(ModelDeployment, model_deployment_id)
    if md is None:
        raise HTTPException(404, "Model deployment not found")

    # 1) Load or cache the fitted model
    model = _loaded_models.get(model_deployment_id)
    if model is None:
        exec_rec = md.execution
        path = exec_rec.model_path
        if not path or not os.path.exists(path):
            raise HTTPException(500, "Model file not found on disk")
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise HTTPException(500, f"Failed to load model: {e}")
        _loaded_models[model_deployment_id] = model

    # 2) Fetch the Training to know what the user originally asked for
    tr = db.get(Training, md.execution.training_id)
    if tr is None:
        raise HTTPException(500, "Parent training not found")

    cfg = {}
    try:
        cfg = json.loads(tr.config_json)
    except Exception:
        raise HTTPException(500, "Invalid training config")

    # 3) Prepare a DataFrame for model.predict()
    # If they passed a flat list, assume it's already in the right order.
    if isinstance(req.features, list):
        X = pd.DataFrame([req.features])
    else:
        # classic dict mode: enforce exact feature set & order
        expected = cfg.get("features")
        if not isinstance(expected, list):
            raise HTTPException(500, "This model expects a sliding-window payload")
        missing = [f for f in expected if f not in req.features]
        if missing:
            raise HTTPException(400, f"Missing feature(s): {missing}")
        X = pd.DataFrame([req.features])[expected]

    # 4) Run prediction
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(400, f"Prediction error: {e}")

    out = pred.tolist() if hasattr(pred, "tolist") else pred
    return PredictResponse(prediction=out)

@router.post(
    "/model_deployments/{md_id}/monitor/",
    response_model=MonitorRead,
    summary="In-memory live evaluation against current active root snapshots",
)
def monitor_model(md_id: str, db: Session = Depends(get_db)) -> MonitorRead:
    # 1) Load the deployed model + its successful execution record
    md = db.get(ModelDeployment, md_id)
    if not md:
        raise HTTPException(404, "ModelDeployment not found")

    te = db.get(TrainingExecution, md.training_execution_id)
    if not te or te.status != "success":
        raise HTTPException(400, "Training execution not successful")

    try:
        with open(te.model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(500, f"Could not load trained model: {e}")

    # 2) Load the original training config
    tr = db.get(Training, te.training_id)
    cfg = json.loads(tr.config_json)
    alg = cfg.get("algorithm")
    # sliding‐window spec may live at top‐level or under params
    window_spec = cfg.get("window_spec") or cfg.get("params", {}).get("window_spec")
    # for classic
    features = cfg.get("features", []) or []
    target   = cfg.get("target")
    # remove window_spec from params if present
    params = cfg.get("params", {}).copy()
    params.pop("window_spec", None)

    # 3) Recursively apply every preprocess back to the true roots
    def build_df(dsid: str) -> pd.DataFrame:
        # find all preprocesses whose child is this dsid
        pps: List[Preprocess] = (
            db.query(Preprocess)
            .filter(Preprocess.datasource_child_id == dsid)
            .all()
        )
        if not pps:
            # leaf ⇒ this is a _root_ datasource
            ds = db.get(DataSource, dsid)
            snap_id = ds.active_snapshot_id
            if not snap_id:
                raise HTTPException(400, f"No active snapshot for datasource {dsid}")
            snap = db.get(Snapshot, snap_id)
            if not snap:
                raise HTTPException(404, "Snapshot not found")
            return pd.read_csv(snap.path)

        # otherwise, chain them in insertion order
        df = None
        for pp in pps:
            # load upstream data
            if any(s["op"] == "join" for s in json.loads(pp.config).get("steps", [])):
                # join: get each parent DataFrame
                parent_dfs = {
                    pds.id: build_df(pds.id)
                    for pds in pp.datasource_parents
                }
            else:
                # single‐parent
                parent = pp.datasource_parents[0]
                df = build_df(parent.id)

            # now apply each step
            for step in json.loads(pp.config)["steps"]:
                op = step["op"]
                p  = step.get("params", {})

                if op == "join":
                    left_id  = pp.datasource_parents[0].active_snapshot_id
                    right_id = pp.datasource_parents[1].active_snapshot_id
                    left_df  = pd.read_csv(db.get(Snapshot, left_id).path)
                    right_df = pd.read_csv(db.get(Snapshot, right_id).path)
                    if p.get("how") == "custom":
                        df = run_custom_join(left_df, right_df, p.get("code"), p)
                    else:
                        df = preprocessors.join_step(left_df, right_df, p)
                elif op == "rename_column":
                    df = preprocessors.rename_column(df, p)
                elif op == "impute_missing":
                    df = preprocessors.impute_missing(df, p)
                elif op == "label_encode":
                    df = preprocessors.label_encode(df, p)
                elif op == "one_hot_encode":
                    df = preprocessors.one_hot_encode(df, p)
                else:
                    fn = fn = getattr(preprocessors, op, None)
                    if fn:
                        df = fn(df, p)
                    else:
                        df = run_custom_step(df, step.get("code",""), p)

        return df

    # 4) Build the final DataFrame
    final_df = build_df(tr.datasource_id)

    # 5) Split into X & y
    if window_spec:
        fw = window_spec["features"]
        tw = window_spec["target"]
        X_rows, y_rows = [], []
        max_end = max(p["end_idx"] for p in fw + [tw])
        for i in range(max_end, len(final_df)):
            row = []
            for p in fw:
                vals = final_df[p["name"]].iloc[i - p["end_idx"]: i - p["start_idx"] + 1]
                row.extend(vals.tolist())
            X_rows.append(row)
            # target may be a window or single
            tvals = final_df[tw["name"]].iloc[i - tw["end_idx"]: i - tw["start_idx"] + 1]
            y_rows.append(tvals.iloc[0] if len(tvals)==1 else tvals.tolist())
        X = pd.DataFrame(X_rows)
        y_true = y_rows
    else:
        missing = [c for c in features + ([target] if target else []) if c not in final_df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns for evaluation: {missing}")
        X = final_df[features]
        y_true = final_df[target].tolist()

    # 6) Predict & compute metrics
    y_pred = model.predict(X)
    metrics: Dict[str, Optional[float]] = {}
    if alg in {"random_forest", "logistic_regression", "svm", "knn"}:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    else:
        mse = mean_squared_error(y_true, y_pred)
        metrics["mse"]  = mse
        metrics["rmse"] = mse ** 0.5
        metrics["r2"]   = r2_score(y_true, y_pred)

    # 7) Gather all ultimate root snapshot IDs
    root_ids: List[str] = []
    seen = set()
    stack = [tr.datasource_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        parents = (
            db.query(Preprocess)
            .filter(Preprocess.datasource_child_id == cur)
            .all()
        )
        if not parents:
            ds = db.get(DataSource, cur)
            if ds.active_snapshot_id:
                root_ids.append(ds.active_snapshot_id)
        else:
            for pp in parents:
                for pds in pp.datasource_parents:
                    stack.append(pds.id)

    return MonitorRead(
        model_deployment_id=md_id,
        evaluated_on_snapshot=root_ids,
        timestamp=datetime.utcnow(),
        metrics=metrics,
    )