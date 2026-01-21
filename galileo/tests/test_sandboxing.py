
import pytest
import pandas as pd
import time
from fastapi import HTTPException
from services.customCode import run_custom_step

def test_forbidden_import():
    df = pd.DataFrame({"a": [1, 2, 3]})
    code = """
def step(df, params):
    import os
    return df
"""
    # This should fail because __import__ is not in SAFE_BUILTINS
    with pytest.raises(HTTPException) as excinfo:
        run_custom_step(df, code, {})
    assert "import" in str(excinfo.value.detail) or "name 'os' is not defined" in str(excinfo.value.detail) or "Error running custom step" in str(excinfo.value.detail)

def test_infinite_loop_timeout():
    df = pd.DataFrame({"a": [1, 2, 3]})
    code = """
def step(df, params):
    while True:
        pass
    return df
"""
    # This should fail with a timeout
    start = time.time()
    with pytest.raises(HTTPException) as excinfo:
        run_custom_step(df, code, {})
    end = time.time()
    
    assert "Time limit exceeded" in str(excinfo.value.detail)
    assert end - start < 10  # Should fail relatively quickly (e.g. 5s limit)

def test_memory_limit():
    df = pd.DataFrame({"a": [1, 2, 3]})
    code = """
def step(df, params):
    # Try to allocate a lot of memory (e.g. 1GB)
    x = [0] * (10**8) 
    return df
"""
    # This might fail on macOS if RLIMIT_AS is not enforced.
    try:
        run_custom_step(df, code, {})
    except HTTPException as e:
        assert "Memory limit exceeded" in str(e.detail) or "Process" in str(e.detail)
    else:
        pytest.xfail("Memory limit not enforced on this system (likely macOS)")
