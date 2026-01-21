# services/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss,
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import learning_curve, validation_curve

def compute_learning_curve(model, X, y, scoring, n_jobs=-1):
    # This uses cross-validation to compute train/val scores at different train sizes
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, scoring=scoring, n_jobs=n_jobs, shuffle=True, random_state=42
    )
    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores": train_scores.mean(axis=1).tolist(),
        "valid_scores": valid_scores.mean(axis=1).tolist(),
    }

def compute_validation_curve(model, X, y, param_name, param_range, scoring, n_jobs=-1):
    # Evaluates model for a range of hyperparameter values
    train_scores, valid_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, scoring=scoring, n_jobs=n_jobs, cv=3
    )
    return {
        "param_range": list(param_range),
        "train_scores": train_scores.mean(axis=1).tolist(),
        "valid_scores": valid_scores.mean(axis=1).tolist(),
    }

def flatten_pred(y):
    if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] == 1:
        return y.ravel()
    return y

def compute_classification_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = flatten_pred(model.predict(X_train))
    y_test_pred = flatten_pred(model.predict(X_test))
    metrics = {}

    metrics.update({
        "accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_confusion_matrix": confusion_matrix(y_train, y_train_pred).tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "train_classification_report": classification_report(y_train, y_train_pred, output_dict=True),
        "test_classification_report": classification_report(y_test, y_test_pred, output_dict=True),
        "train_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
    })

    try:
        if hasattr(model, "predict_proba"):
            train_probs = model.predict_proba(X_train)
            test_probs = model.predict_proba(X_test)
            metrics["train_log_loss"] = log_loss(y_train, train_probs)
            metrics["test_log_loss"] = log_loss(y_test, test_probs)
    except Exception:
        pass

    return metrics

def compute_regression_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = flatten_pred(model.predict(X_train))
    y_test_pred = flatten_pred(model.predict(X_test))
    metrics = {}

    try:
        mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        mape = mean_absolute_percentage_error(y_train, y_train_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        evs = explained_variance_score(y_train, y_train_pred)
        test_evs = explained_variance_score(y_test, y_test_pred)
        metrics.update({
            "mse": mse, "rmse": mse ** 0.5, "r2": r2,
            "test_mse": test_mse, "test_rmse": test_mse ** 0.5, "test_r2": test_r2,
            "mae": mae, "test_mae": test_mae,
            "mape": mape, "test_mape": test_mape,
            "explained_variance": evs, "test_explained_variance": test_evs,
        })
    except Exception:
        pass

    return metrics

def get_metric(metrics_dict, metric_name):
    """Utility for extracting a metric value from a metrics dict."""
    return metrics_dict.get(metric_name)

# Add these two new curve helpers (for loss/staged curves):
def get_loss_curve(model):
    if hasattr(model, 'loss_curve_'):
        return list(getattr(model, 'loss_curve_'))
    elif hasattr(model, 'loss_curve'):
        return list(getattr(model, 'loss_curve'))
    return None

def staged_metric_curve(model, X, y, metric_func):
    """For staged_predict models (boosting, etc), get score as model evolves."""
    if hasattr(model, 'staged_predict'):
        try:
            staged = list(model.staged_predict(X))
            scores = [metric_func(y, yhat) for yhat in staged]
            return scores
        except Exception:
            return None
    return None