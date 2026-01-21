
import pytest
import io
import hashlib
import pandas as pd
from fastapi.testclient import TestClient

def test_eda_imputation_encoding_reproducibility(client: TestClient):
    # --- Step 1: Upload Data (EDA) ---
    # Data has missing values and categorical columns
    csv_content = "id,category,value\n1,A,10\n2,B,\n3,A,30\n4,,40"
    file = {"file": ("eda_data.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    
    resp_ds = client.post("/datasources/with-snapshot/", data={"name": "EdaDS"}, files=file)
    assert resp_ds.status_code == 200
    ds_id = resp_ds.json()["id"]
    snap_id = resp_ds.json()["active_snapshot_id"]
    
    # --- Step 2: EDA (Summary) ---
    # Verify we can get summary stats
    resp_summary = client.get(f"/datasources/{ds_id}/snapshots/{snap_id}/summary/")
    assert resp_summary.status_code == 200
    summary = resp_summary.json()
    # Check if summary detects missing values
    # Structure depends on implementation, usually dict of columns
    # Assuming summary returns something like { "columns": { "value": { "missing_count": 1, ... } } }
    # Let's just assert status for now and maybe inspect structure if needed
    
    # --- Step 3: Configure Preprocess (Imputation + Encoding) ---
    pp_config = {
        "steps": [
            {
                "op": "impute_missing",
                "params": {"column": "value", "strategy": "mean"}
            },
            {
                "op": "one_hot_encode",
                "params": {"column": "category", "categories": ["A", "B"], "drop_original": True}
            }
        ]
    }
    
    resp_pp = client.post("/preprocesses/", json={
        "name": "CleanEdaDS",
        "parent_ids": [ds_id],
        "config": pp_config
    })
    assert resp_pp.status_code == 200
    pp_id = resp_pp.json()["id"]
    
    # --- Step 4: Execute ---
    resp_exec1 = client.post(f"/preprocesses/{pp_id}/execute/", json={"snapshot_id": snap_id})
    assert resp_exec1.status_code == 200
    exec1_data = resp_exec1.json()
    sn_out_id = exec1_data["output_snapshot"]
    child_ds_id = resp_pp.json()["child_id"]
    
    # --- Step 5: Verify Output ---
    resp_out = client.get(f"/datasources/{child_ds_id}/snapshots/{sn_out_id}/download")
    assert resp_out.status_code == 200
    df_out = pd.read_csv(io.BytesIO(resp_out.content))
    
    # Verify Imputation: 'value' should have no NaNs
    assert not df_out["value"].isna().any()
    # Mean of 10, 30, 40 is 26.66...
    # Row 2 was missing, should be filled
    assert abs(df_out.loc[1, "value"] - 26.66) < 0.1
    
    # Verify One-Hot: 'category_A', 'category_B' should exist, 'category' gone
    assert "category_A" in df_out.columns
    assert "category_B" in df_out.columns
    assert "category" not in df_out.columns
    
    hash_1 = hashlib.sha256(resp_out.content).hexdigest()
    
    # --- Step 6: Reproducibility ---
    resp_exec2 = client.post(f"/preprocesses/{pp_id}/execute/", json={"snapshot_id": snap_id})
    assert resp_exec2.status_code == 200
    sn_out_2_id = resp_exec2.json()["output_snapshot"]
    
    resp_out_2 = client.get(f"/datasources/{child_ds_id}/snapshots/{sn_out_2_id}/download")
    hash_2 = hashlib.sha256(resp_out_2.content).hexdigest()
    
    assert hash_1 == hash_2


def test_join_two_snapshots(client: TestClient):
    # --- Step 1: Upload Two Datasets ---
    # Users: id, name
    users_csv = "user_id,name\n1,Alice\n2,Bob\n3,Charlie"
    file_users = {"file": ("users.csv", io.BytesIO(users_csv.encode()), "text/csv")}
    resp_users = client.post("/datasources/with-snapshot/", data={"name": "Users"}, files=file_users)
    users_ds_id = resp_users.json()["id"]
    users_snap_id = resp_users.json()["active_snapshot_id"]
    
    # Orders: id, user_id, amount
    orders_csv = "order_id,u_id,amount\n101,1,50\n102,1,20\n103,2,100"
    file_orders = {"file": ("orders.csv", io.BytesIO(orders_csv.encode()), "text/csv")}
    resp_orders = client.post("/datasources/with-snapshot/", data={"name": "Orders"}, files=file_orders)
    orders_ds_id = resp_orders.json()["id"]
    orders_snap_id = resp_orders.json()["active_snapshot_id"]
    
    # --- Step 2: Create Preprocess with 2 Parents ---
    # Join on user_id == u_id
    pp_config = {
        "steps": [
            {
                "op": "join",
                "params": {
                    "left_keys": ["user_id"],
                    "right_keys": ["u_id"],
                    "how": "inner"
                }
            }
        ]
    }
    
    resp_pp = client.post("/preprocesses/", json={
        "name": "UserOrders",
        "parent_ids": [users_ds_id, orders_ds_id],
        "config": pp_config
    })
    assert resp_pp.status_code == 200
    pp_id = resp_pp.json()["id"]
    
    # --- Step 3: Execute Join ---
    # For join, we pass a map of datasource_id -> snapshot_id in 'snapshots' field of execution request
    # Or rely on active snapshots if not provided (but let's be explicit)
    exec_payload = {
        "snapshots": {
            users_ds_id: users_snap_id,
            orders_ds_id: orders_snap_id
        }
    }
    
    resp_exec = client.post(f"/preprocesses/{pp_id}/execute/", json=exec_payload)
    if resp_exec.status_code != 200:
        with open("debug_error.txt", "w") as f:
            f.write(f"Status: {resp_exec.status_code}\nBody: {resp_exec.text}")
    assert resp_exec.status_code == 200
    
    out_snap_id = resp_exec.json()["output_snapshot"]
    child_ds_id = resp_pp.json()["child_id"]
    
    # --- Step 4: Verify Output ---
    resp_out = client.get(f"/datasources/{child_ds_id}/snapshots/{out_snap_id}/download")
    assert resp_out.status_code == 200
    df_out = pd.read_csv(io.BytesIO(resp_out.content))
    
    # Should have merged columns
    assert "name" in df_out.columns
    assert "amount" in df_out.columns
    # Inner join: User 3 (Charlie) has no orders, should be dropped
    assert 3 not in df_out["user_id"].values
    # User 1 has 2 orders, User 2 has 1 order -> Total 3 rows
    assert len(df_out) == 3
    
    # --- Step 5: Verify EP Entry ---
    # Check history
    resp_history = client.get(f"/preprocesses/{pp_id}/executions/")
    assert resp_history.status_code == 200
    history = resp_history.json()
    assert len(history) == 1
    assert history[0]["output_snapshot"] == out_snap_id
    assert len(history[0]["input_snapshots"]) == 2

def test_join_invalid_keys(client: TestClient):
    # Setup similar to above
    users_csv = "user_id,name\n1,Alice"
    file_users = {"file": ("users.csv", io.BytesIO(users_csv.encode()), "text/csv")}
    resp_users = client.post("/datasources/with-snapshot/", data={"name": "Users"}, files=file_users)
    users_ds_id = resp_users.json()["id"]
    
    orders_csv = "order_id,u_id,amount\n101,1,50"
    file_orders = {"file": ("orders.csv", io.BytesIO(orders_csv.encode()), "text/csv")}
    resp_orders = client.post("/datasources/with-snapshot/", data={"name": "Orders"}, files=file_orders)
    orders_ds_id = resp_orders.json()["id"]
    
    # Config with WRONG keys
    pp_config = {
        "steps": [
            {
                "op": "join",
                "params": {
                    "left_keys": ["WRONG_KEY"],
                    "right_keys": ["u_id"],
                    "how": "inner"
                }
            }
        ]
    }
    
    resp_pp = client.post("/preprocesses/", json={
        "name": "BadJoin",
        "parent_ids": [users_ds_id, orders_ds_id],
        "config": pp_config
    })
    pp_id = resp_pp.json()["id"]
    
    # Execute should fail
    resp_exec = client.post(f"/preprocesses/{pp_id}/execute/", json={
        "snapshots": {
            users_ds_id: resp_users.json()["active_snapshot_id"],
            orders_ds_id: resp_orders.json()["active_snapshot_id"]
        }
    })
    
    # Expecting failure (400 or 500)
    assert resp_exec.status_code in [400, 500]
