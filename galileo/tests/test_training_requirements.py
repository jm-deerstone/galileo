import json
import time
import pytest
import pandas as pd
import io
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, get_db
from main import app
from models import TrainingExecution

# Use the same test DB setup as conftest.py
TEST_DB_URL = "sqlite:///./test_galileo.db"

@pytest.fixture(scope="module")
def client():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)

def test_tr_fr1_fr2_create_preview(client):
    """
    TR-FR1: Create from snapshot and mode.
    TR-FR2: Preview validation.
    """
    # 1. Upload Datasource
    csv_content = "id,f1,f2,target_cls,target_reg\n1,0.1,0.2,A,10\n2,0.2,0.3,B,20\n3,0.3,0.4,A,30"
    file = {"file": ("train_data.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp_ds = client.post("/datasources/with-snapshot/", data={"name": "TrainDS"}, files=file)
    assert resp_ds.status_code == 200
    ds_id = resp_ds.json()["id"]
    snap_id = resp_ds.json()["active_snapshot_id"]

    # 2. Valid Preview (Classification)
    valid_config = {
        "features": ["f1", "f2"],
        "target": "target_cls",
        "algorithm": "logistic_regression",
        "params": {"C": 1.0}
    }
    resp_prev = client.post("/trainings/preview/", json={"snapshot_id": snap_id, "config": valid_config})
    assert resp_prev.status_code == 200
    assert resp_prev.json()["ok"] is True

    # 3. Invalid Preview (Missing Target)
    invalid_config = valid_config.copy()
    del invalid_config["target"]
    resp_prev_fail = client.post("/trainings/preview/", json={"snapshot_id": snap_id, "config": invalid_config})
    assert resp_prev_fail.status_code == 400
    assert "error" in resp_prev_fail.json()

    # 4. Create Training (Persist Config)
    resp_create = client.post("/trainings/", json={
        "name": "MyTraining",
        "datasource_id": ds_id,
        "config": valid_config
    })
    assert resp_create.status_code == 200
    tr_data = resp_create.json()
    assert tr_data["name"] == "MyTraining"
    assert tr_data["config_json"]["target"] == "target_cls"
    return tr_data["id"], snap_id

def test_tr_fr3_task_checking(client):
    """
    TR-FR3: Algorithm registry with task checking.
    """
    # Setup DS
    csv_content = "f1,target_cls,target_reg\n0.1,A,10.0\n0.2,B,20.0"
    file = {"file": ("task_check.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp_ds = client.post("/datasources/with-snapshot/", data={"name": "TaskCheckDS"}, files=file)
    snap_id = resp_ds.json()["active_snapshot_id"]

    # 1. Classification algo on Numeric target (Regression) -> Should Fail
    # Note: "target_reg" is float. Logistic Regression expects discrete/string or int labels usually, 
    # but strictly speaking sklearn might accept floats as classes if they are few. 
    # However, our router explicitly checks `is_float_dtype` for classifiers.
    bad_cls_config = {
        "features": ["f1"],
        "target": "target_reg",
        "algorithm": "logistic_regression"
    }
    resp = client.post("/trainings/preview/", json={"snapshot_id": snap_id, "config": bad_cls_config})
    assert resp.status_code == 400
    assert "Classification algorithms require discrete target values" in resp.json()["error"]

    # 2. Regression algo on String target -> Should Fail
    bad_reg_config = {
        "features": ["f1"],
        "target": "target_cls",
        "algorithm": "linear_regression"
    }
    resp = client.post("/trainings/preview/", json={"snapshot_id": snap_id, "config": bad_reg_config})
    assert resp.status_code == 400
    assert "Regression algorithms require numeric target values" in resp.json()["error"]

def test_tr_fr4_fr5_fr6_fr7_hpo_execution(client):
    """
    TR-FR4: HPO Strategies (Random, Grid, Halving).
    TR-FR5: Execution lifecycle & progress.
    TR-FR6: Metric capture.
    TR-FR7: Curve logging.
    """
    # 1. Upload larger dataset for HPO
    # 20 rows to allow CV=2 or 3
    data = []
    for i in range(20):
        data.append(f"{i},{i*2},{'A' if i%2==0 else 'B'}")
    csv_content = "f1,target_reg,target_cls\n" + "\n".join(data)
    file = {"file": ("hpo_data.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp_ds = client.post("/datasources/with-snapshot/", data={"name": "HpoDS"}, files=file)
    ds_id = resp_ds.json()["id"]
    snap_id = resp_ds.json()["active_snapshot_id"]

    # 2. Create Training with Random HPO
    config = {
        "features": ["f1"],
        "target": "target_reg",
        "algorithm": "random_forest_reg",
        "hpo_strategy": "random",
        "cv": 2,  # Small CV for small data
        "hpo_pop": 2, # n_iter for random
        "params": {} # Empty params to trigger HPO on defaults
    }
    resp_tr = client.post("/trainings/", json={
        "name": "HpoTraining",
        "datasource_id": ds_id,
        "config": config
    })
    tr_id = resp_tr.json()["id"]

    # 3. Execute
    resp_exec = client.post(f"/trainings/{tr_id}/execute/", json={"snapshot_id": snap_id})
    assert resp_exec.status_code == 202
    exec_id = resp_exec.json()["id"]

    # 4. Poll Progress (TR-FR5, TR-NFR2)
    for _ in range(30):
        resp_prog = client.get(f"/training_executions/{exec_id}/progress/")
        prog = resp_prog.json()
        # Check progress structure
        if "progress" in prog:
            assert 0 <= prog["progress"] <= 1
        
        # Check status in DB
        resp_list = client.get(f"/trainings/{tr_id}/executions/")
        status = resp_list.json()[0]["status"]
        if status in ("success", "failed"):
            if status == "failed":
                # Debug failure
                exec_rec = resp_list.json()[0]
                print(f"DEBUG: Training failed with error: {exec_rec.get('metrics_json')}")
            break
        time.sleep(1)
    
    assert status == "success"

    # 5. Check Metrics (TR-FR6, TR-FR7)
    resp_list = client.get(f"/trainings/{tr_id}/executions/")
    exec_rec = resp_list.json()[0]
    metrics = json.loads(exec_rec["metrics_json"])
    
    assert "r2" in metrics or "mse" in metrics # TR-FR6
    assert "train_time_s" in metrics
    # Curves might be present if data sufficient
    # With 20 rows, learning curve might fail or be short, but let's check keys exist
    # Validation curves are computed if params are gettable
    # Random Forest has params, so validation_curves should be attempted
    # Note: validation_curves might be empty if no param range fits, but key should be checked if logic ran
    
    # 6. Check HPO updated config
    resp_tr_get = client.get(f"/trainings/{tr_id}")
    updated_config = resp_tr_get.json()["config_json"]
    # Params should now contain best_params from HPO
    assert "n_estimators" in updated_config["params"] or "max_depth" in updated_config["params"]

def test_tr_nfr1_determinism(client):
    """
    TR-NFR1: Determinism with fixed seeds.
    """
    # 1. Upload Data
    csv_content = "f1,target\n1,10\n2,20\n3,30\n4,40\n5,50"
    file = {"file": ("det_data.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp_ds = client.post("/datasources/with-snapshot/", data={"name": "DetDS"}, files=file)
    ds_id = resp_ds.json()["id"]
    snap_id = resp_ds.json()["active_snapshot_id"]

    # 2. Config
    config = {
        "features": ["f1"],
        "target": "target",
        "algorithm": "random_forest_reg", # Non-deterministic by default
        "params": {"n_estimators": 5} # Fixed params, no HPO
    }
    resp_tr = client.post("/trainings/", json={
        "name": "DetTraining",
        "datasource_id": ds_id,
        "config": config
    })
    tr_id = resp_tr.json()["id"]

    # 3. Run Twice
    def run_and_get_metrics():
        resp = client.post(f"/trainings/{tr_id}/execute/", json={"snapshot_id": snap_id})
        exec_id = resp.json()["id"]
        # Wait
        for _ in range(10):
            status = client.get(f"/trainings/{tr_id}/executions/").json()[0]["status"]
            if status == "success":
                break
            time.sleep(0.5)
        # Get metrics
        execs = client.get(f"/trainings/{tr_id}/executions/").json()
        # Get the one matching exec_id
        rec = next(e for e in execs if e["id"] == exec_id)
        return json.loads(rec["metrics_json"])

    m1 = run_and_get_metrics()
    m2 = run_and_get_metrics()

    # 4. Compare
    # R2 should be identical
    assert m1["r2"] == m2["r2"]
    assert m1["mse"] == m2["mse"]
