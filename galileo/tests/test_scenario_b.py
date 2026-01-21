import pytest
import hashlib
import os
import io

def calculate_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def test_scenario_b_snapshot_versioning_and_replay(client):
    # --- Step 1: Initial Snapshot (SN1) ---
    csv_content_v1 = "col1,col2\n1,10\n2,20"
    file_v1 = {"file": ("data_v1.csv", io.BytesIO(csv_content_v1.encode()), "text/csv")}
    
    response = client.post("/datasources/with-snapshot/", data={"name": "TestDS"}, files=file_v1)
    assert response.status_code == 200
    data = response.json()
    ds_id = data["id"]
    sn1_id = data["active_snapshot_id"]
    assert sn1_id is not None
    
    # Verify SN1 file exists
    # We need to find the path. The response returns the full object.
    # we can verify the hash by downloading it.
    
    resp_sn1 = client.get(f"/datasources/{ds_id}/snapshots/{sn1_id}/download")
    assert resp_sn1.status_code == 200
    hash_sn1 = hashlib.sha256(resp_sn1.content).hexdigest()
    
    # --- Step 2: First Execution (SN1 -> SN2) ---
    # First, create a Preprocess definition
    pp_config = {
        "steps": [
            {"op": "rename_column", "params": {"from": "col2", "to": "col2_renamed"}}
        ]
    }
    resp_pp = client.post("/preprocesses/", json={
        "name": "TestPP",
        "parent_ids": [ds_id],
        "config": pp_config
    })
    assert resp_pp.status_code == 200
    pp_id = resp_pp.json()["id"]
    
    # Execute
    resp_exec1 = client.post(f"/preprocesses/{pp_id}/execute/", json={"snapshot_id": sn1_id})
    assert resp_exec1.status_code == 200
    exec1_data = resp_exec1.json()
    ep1_id = exec1_data["id"]
    sn2_id = exec1_data["output_snapshot"]
    
    # Verify SN2 content/hash
    # We can get child_id from PP definition
    child_ds_id = resp_pp.json()["child_id"]
    resp_sn2 = client.get(f"/datasources/{child_ds_id}/snapshots/{sn2_id}/download")
    assert resp_sn2.status_code == 200
    hash_sn2 = hashlib.sha256(resp_sn2.content).hexdigest()
    
    # --- Step 3: Replay (SN1 -> SN2_Replay) ---
    resp_exec2 = client.post(f"/preprocesses/{pp_id}/execute/", json={"snapshot_id": sn1_id})
    assert resp_exec2.status_code == 200
    exec2_data = resp_exec2.json()
    ep2_id = exec2_data["id"]
    sn2_replay_id = exec2_data["output_snapshot"]
    
    # Verify it's a new execution record
    assert ep1_id != ep2_id
    # Verify it's a new snapshot ID (since we always create new files for executions)
    assert sn2_id != sn2_replay_id
    
    # Verify Content Hash is Identical (Reproducibility)
    resp_sn2_replay = client.get(f"/datasources/{child_ds_id}/snapshots/{sn2_replay_id}/download")
    hash_sn2_replay = hashlib.sha256(resp_sn2_replay.content).hexdigest()
    
    if hash_sn2 != hash_sn2_replay:
        print(f"SN2 Content:\n{resp_sn2.content.decode()}")
        print(f"SN2 Replay Content:\n{resp_sn2_replay.content.decode()}")
        
    assert hash_sn2 == hash_sn2_replay
    
    # --- Step 4: New Version Ingestion (SN3) ---
    csv_content_v2 = "col1,col2\n1,100\n2,200\n3,300" # Changed data AND added row to ensure different output
    file_v2 = {"file": ("data_v2.csv", io.BytesIO(csv_content_v2.encode()), "text/csv")}
    
    resp_sn3 = client.post(f"/datasources/{ds_id}/snapshots/", files=file_v2)
    assert resp_sn3.status_code == 200
    sn3_id = resp_sn3.json()["id"]
    assert sn3_id != sn1_id
    
    # --- Step 5: Execution on New Version (SN3 -> SN4) ---
    resp_exec3 = client.post(f"/preprocesses/{pp_id}/execute/", json={"snapshot_id": sn3_id})
    assert resp_exec3.status_code == 200
    exec3_data = resp_exec3.json()
    sn4_id = exec3_data["output_snapshot"]
    
    # Verify SN4 content hash is DIFFERENT from SN2 (Different input -> Different output)
    resp_sn4 = client.get(f"/datasources/{child_ds_id}/snapshots/{sn4_id}/download")
    hash_sn4 = hashlib.sha256(resp_sn4.content).hexdigest()
    assert hash_sn4 != hash_sn2
    
    # --- Step 6: Immutability Check ---
    # Verify SN1 is still retrievable and unchanged
    resp_sn1_check = client.get(f"/datasources/{ds_id}/snapshots/{sn1_id}/download")
    assert resp_sn1_check.status_code == 200
    assert hashlib.sha256(resp_sn1_check.content).hexdigest() == hash_sn1
    
    # Verify SN2 is still retrievable and unchanged
    resp_sn2_check = client.get(f"/datasources/{child_ds_id}/snapshots/{sn2_id}/download")
    assert resp_sn2_check.status_code == 200
    assert hashlib.sha256(resp_sn2_check.content).hexdigest() == hash_sn2

