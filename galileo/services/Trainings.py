import os
import json
import pickle
import logging
import inspect
from datetime import datetime

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    train_test_split
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Your models, logger, etc.
from logger import logger

from sklearn.metrics import (
    accuracy_score, r2_score
)


from models import Snapshot, Training, TrainingExecution, Deployment, ModelDeployment
from services.modelLandscape.data import _build_sliding_window
from services.modelLandscape.hpo import run_local_hpo, local_evolutionary_search
from services.modelLandscape.metrics import compute_classification_metrics, compute_regression_metrics, \
    compute_validation_curve, compute_learning_curve
from services.modelLandscape.modelconfig import ALG_REGISTRY, HPO_PARAM_DISTS, CLASSIFIERS, REGRESSORS
from services.modelLandscape.util import to_python_types

logger.info("Logging is working! You should see this in your console.")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def _instantiate(cls, **kwargs):
    sig = inspect.signature(cls)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid)

def are_params_complete(params, hpo_param_dists):
    """Check if all HPO keys are present in params."""
    if not hpo_param_dists:
        return True
    for k in hpo_param_dists:
        if k not in params:
            return False
    return True



def update_progress(db, exec_rec, progress: dict):
    exec_rec.progress_json = json.dumps(progress)
    db.add(exec_rec)
    db.commit()






def train_local_model(
        ModelCls,
        params,
        hpo_strategy,
        hpo_param_dists,
        X_train,
        y_train,
        X_val,
        y_val,
        is_classification,
        scoring,
        db=None,
        exec_rec=None,
        user_hpo_pop=None,
        user_hpo_gen=None,
        cv=3,
        skip_fit=False,   # <-- ADDED!
):

    base = _instantiate(ModelCls, random_state=42, **params)
    if is_classification and len(y_train.shape) > 1:
        model = MultiOutputClassifier(base)
    elif not is_classification and len(y_train.shape) > 1:
        model = MultiOutputRegressor(base)
    else:
        model = base

    best_params = {}
    hpo_curves = None
    if hpo_strategy and hpo_param_dists:
        if hpo_strategy == "evolutionary":
            def factory(**params):
                return _instantiate(ModelCls, random_state=42, **params)
            score_func = accuracy_score if is_classification else r2_score

            def progress_cb(prog):
                if db is not None and exec_rec is not None:
                    update_progress(db, exec_rec, prog)

            best_params, hpo_curves = local_evolutionary_search(
                factory, hpo_param_dists, X_train, y_train,
                X_val=X_val, y_val=y_val,
                pop_size=user_hpo_pop or 8, n_gens=user_hpo_gen or 10, score_func=score_func,
                progress_cb=progress_cb if db and exec_rec else None
            )
            # Final progress: indicate refit starting
            if db is not None and exec_rec is not None:
                update_progress(db, exec_rec, {
                    "stage": "refit",
                    "progress": 1.0
                })
            model = factory(**best_params)
        else:
            model, best_params = run_local_hpo(
                model, hpo_strategy, hpo_param_dists, X_train, y_train, scoring, cv=cv
            )
    # Only fit if not skipping (so, only fit during HPO!)
    if not skip_fit:

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
    else:
        y_train_pred = None
        y_val_pred = None
    return model, y_train_pred, y_val_pred, best_params, hpo_curves


def run_training(execution_id: str, db_url: str):
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        exec_rec = db.get(TrainingExecution, execution_id)
        tr = db.get(Training, exec_rec.training_id)
        cfg = json.loads(tr.config_json)
        algo_key = cfg["algorithm"]
        ModelCls = ALG_REGISTRY[algo_key]
        params = cfg.get("params", {}).copy()
        snap = db.get(Snapshot, exec_rec.snapshot_id)
        df = pd.read_csv(snap.path)
        window_spec = cfg.get("window_spec")
        if window_spec and isinstance(window_spec, dict) and window_spec.get("features") and window_spec.get("target"):
            X_fit, y_fit = _build_sliding_window(df, window_spec)
        else:
            X_fit = df[cfg["features"]]
            y_fit = df[cfg["target"]]
        test_ratio = float(cfg.get("split_ratio", 0.15))
        val_ratio = float(cfg.get("val_ratio", 0.15))
        X_temp, X_test, y_temp, y_test = train_test_split(X_fit, y_fit, test_size=test_ratio, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio / (1 - test_ratio), random_state=42)
        hpo_params = HPO_PARAM_DISTS.get(algo_key, {})
        hpo_strategy = cfg.get("hpo_strategy", None)
        use_hpo = (
                bool(hpo_params)
                and hpo_strategy
                and hpo_strategy != "manual"
                and not are_params_complete(params, hpo_params)
        )
        is_classification = algo_key in CLASSIFIERS
        scoring = "accuracy" if is_classification else "r2"

        # GET user overrides from config:
        user_hpo_pop = cfg.get("hpo_pop") or cfg.get("hpo_population") or None
        user_hpo_gen = cfg.get("hpo_gen") or cfg.get("hpo_generations") or None
        user_cv = int(cfg.get("cv", 3))

        # PATCH: Only fit in train_local_model if using HPO or missing params
        model, y_train_pred, y_val_pred, best_params, hpo_curves = train_local_model(
            ModelCls, params, hpo_strategy if use_hpo else None, hpo_params if use_hpo else None,
            X_train, y_train, X_val, y_val, is_classification, scoring,
            db=db, exec_rec=exec_rec,
            user_hpo_pop=user_hpo_pop, user_hpo_gen=user_hpo_gen,
            cv=user_cv,
            skip_fit=not use_hpo   # <--- THIS LINE!
        )

        if use_hpo and best_params:
            params.update(best_params)
            cfg["params"] = params
            tr.config_json = json.dumps(cfg)
            db.add(tr)
            db.commit()
            db.refresh(tr)
        # Final progress: refit still running
        if use_hpo and db is not None and exec_rec is not None:
            update_progress(db, exec_rec, {
                "stage": "refit",
                "progress": 1.0
            })

        # Always fit ONCE here on train+val (for both metrics and to save model)
        X_final = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_final = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        model.fit(X_final, y_final)

        y_train_pred = model.predict(X_final)
        y_test_pred = model.predict(X_test)
        # Metrics
        def flatten_pred(y):
            if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] == 1:
                return y.ravel()
            return y
        y_train_pred = flatten_pred(y_train_pred)
        y_test_pred  = flatten_pred(y_test_pred)
        y_final_np   = flatten_pred(y_final)
        y_test_np    = flatten_pred(y_test)
        end = datetime.utcnow()
        duration = (end - (exec_rec.started_at or end)).total_seconds()
        metrics = {"train_time_s": duration, "n_samples": len(y_final_np)}
        # Learning Curve
        try:
            metrics['learning_curve'] = compute_learning_curve(
                model, X_final, y_final, scoring=scoring
            )
            metrics['learning_curve_desc'] = (
                "Shows model performance (train/validation) as the training sample size increases. "
                "Helps diagnose underfitting/overfitting and data sufficiency."
            )
        except Exception as ex:
            logger.warning(f"Learning curve computation failed: {ex}")

        validation_curves = {}
        try:
            if hasattr(model, "get_params"):
                param_grid = HPO_PARAM_DISTS.get(algo_key, {})
                params = model.get_params()
                for param_name, param_info in param_grid.items():
                    # Get param_range
                    if isinstance(param_info, dict):
                        rng = param_info.get("range")
                        typ = param_info.get("type", "int")
                        if rng is None:
                            continue
                        if typ == "int":
                            param_range = np.linspace(rng[0], rng[1], 8, dtype=int)
                        elif typ == "float":
                            param_range = np.logspace(np.log10(rng[0]), np.log10(rng[1]), 8)
                        elif typ in ("int_or_none", "float_or_none"):
                            # Drop None for validation_curve (scikit-learn expects valid values)
                            if typ == "int_or_none":
                                param_range = np.linspace(rng[0], rng[1], 8, dtype=int)
                            else:
                                param_range = np.logspace(np.log10(rng[0]), np.log10(rng[1]), 8)
                        elif typ == "mlp_hidden_layer_sizes":
                            # Skip: can't sweep this in validation_curve
                            continue
                        else:
                            continue
                    elif isinstance(param_info, (list, tuple)):
                        param_range = param_info
                    else:
                        continue

                    # Only try if param is in model.get_params()
                    if param_name not in params:
                        continue

                    curve = compute_validation_curve(
                        model, X_final, y_final,
                        param_name=param_name,
                        param_range=param_range,
                        scoring=scoring
                    )
                    # Add description and save
                    desc = (
                        f"Model train/validation performance as '{param_name}' is varied. "
                        "Helps choose good hyperparameters and spot overfitting."
                    )
                    validation_curves[param_name] = {
                        "curve": curve,
                        "desc": desc,
                        "param_range": param_range.tolist() if hasattr(param_range, 'tolist') else list(param_range)
                    }

            if validation_curves:
                metrics['validation_curves'] = validation_curves

        except Exception as ex:
            logger.warning(f"Multi-param validation curve computation failed: {ex}")

        # Loss curve (neural nets, etc)
        if hasattr(model, 'loss_curve_'):
            metrics['loss_curve'] = list(getattr(model, 'loss_curve_'))
            metrics['loss_curve_desc'] = (
                "Training loss after each epoch/iteration. "
                "Monitors convergence and learning dynamics."
            )
        elif hasattr(model, 'loss_curve'):
            metrics['loss_curve'] = list(getattr(model, 'loss_curve'))
            metrics['loss_curve_desc'] = (
                "Training loss after each epoch/iteration. "
                "Monitors convergence and learning dynamics."
            )

        # Staged prediction curve (tree ensembles, boosting, etc.)
        def staged_metric_curve(model, X, y, metric_func, metric_name):
            if hasattr(model, 'staged_predict'):
                try:
                    staged = list(model.staged_predict(X))
                    scores = [metric_func(y, yhat) for yhat in staged]
                    return scores
                except Exception:
                    return None
            return None

        if is_classification:
            staged = staged_metric_curve(model, X_final, y_final, accuracy_score, "accuracy")
            if staged:
                metrics['staged_accuracy_curve'] = staged
                metrics['staged_accuracy_curve_desc'] = (
                    "Accuracy as the ensemble grows (after each boosting/bagging iteration). "
                    "Helps see if more estimators would help."
                )
        else:
            staged = staged_metric_curve(model, X_final, y_final, r2_score, "r2")
            if staged:
                metrics['staged_r2_curve'] = staged
                metrics['staged_r2_curve_desc'] = (
                    "R² score as the ensemble grows (after each boosting/bagging iteration). "
                    "Shows how model fit evolves."
                )

                # Save learning/loss curves if available
        if hpo_curves:
            metrics['evo_best_curve'] = hpo_curves.get('best_curve', [])
            metrics['evo_mean_curve'] = hpo_curves.get('mean_curve', [])

        if is_classification:
            metrics.update(
                compute_classification_metrics(model, X_final, y_final_np, X_test, y_test_np)
            )
        elif algo_key in REGRESSORS:
            metrics.update(
                compute_regression_metrics(model, X_final, y_final_np, X_test, y_test_np)
            )
        ts = end.strftime("%Y%m%dT%H%M%SZ")
        dpath = os.path.join(MODELS_DIR, exec_rec.training_id)
        os.makedirs(dpath, exist_ok=True)
        mpath = os.path.join(dpath, f"{ts}.pkl")
        with open(mpath, "wb") as f:
            pickle.dump(model, f)
        exec_rec.status = "success"
        exec_rec.finished_at = end
        exec_rec.metrics_json = json.dumps(to_python_types(metrics))
        exec_rec.model_path = mpath
        # Final done progress
        update_progress(db, exec_rec, {
            "stage": "done",
            "progress": 1.0
        })
        db.add(exec_rec)
        db.commit()
    except Exception as e:
        logger.exception("Training failed")
        db.rollback()
        now = datetime.utcnow()
        exec_rec.status = "failed"
        exec_rec.finished_at = now
        exec_rec.metrics_json = json.dumps({"error": str(e)})
        update_progress(db, exec_rec, {"stage": "failed", "status": "failed"})
        db.add(exec_rec)
        db.commit()
    finally:
        db.close()


def evaluate_and_promote(tr: Training, db_url: str):
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    dep = session.query(Deployment).filter_by(training_id=tr.id).first()
    if not dep:
        dep = Deployment(training_id=tr.id)
        session.add(dep)
        session.commit()
        session.refresh(dep)
    metrics = (tr.promotion_metrics or "").split(",")
    all_execs = (
        session.query(TrainingExecution)
        .filter_by(training_id=tr.id, status="success")
        .order_by(TrainingExecution.finished_at.desc())
        .all()
    )
    if not all_execs or not metrics or metrics == [""]:
        return
    for metric in metrics:
        current_auto = (
            session.query(ModelDeployment)
            .filter_by(deployment_id=dep.id, promotion_type="auto", metric=metric, locked=False)
            .order_by(ModelDeployment.id.desc())
            .first()
        )
        def get_metric(e, m):
            try:
                d = json.loads(e.metrics_json)
                return d.get(m)
            except Exception:
                return None
        best_exec = None
        best_value = None
        for exec_rec in all_execs:
            val = get_metric(exec_rec, metric)
            if val is None:
                continue
            if best_value is None or val > best_value:  # > for "higher is better", use < for loss
                best_value = val
                best_exec = exec_rec
        if not current_auto or best_exec.id != current_auto.training_execution_id:
            session.query(ModelDeployment).filter_by(
                deployment_id=dep.id, metric=metric, promotion_type="auto", locked=False
            ).delete()
            md = ModelDeployment(
                deployment_id=dep.id,
                training_execution_id=best_exec.id,
                promotion_type="auto",
                locked=False,
                metric=metric
            )
            session.add(md)
            session.commit()
