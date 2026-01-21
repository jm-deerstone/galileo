import multiprocessing
import resource
import signal
import pandas as pd
import numpy as np
from fastapi import HTTPException
from datetime import datetime
import traceback

# 1) Define a minimal, safe builtins dict
SAFE_BUILTINS = {
    'None': None,
    'True': True,
    'False': False,
    'len': len,
    'min': min,
    'max': max,
    'range': range,
    'tuple': tuple,
    'list': list,
    'dict': dict,
    'set': set,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'datetime': datetime,
    'abs': abs,
    'round': round,
    'sum': sum,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sorted': sorted,
    'isinstance': isinstance,
    # Add exceptions so user code can catch them if needed
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'KeyError': KeyError,
    'IndexError': IndexError,
}

# Configuration
TIME_LIMIT_SECONDS = 5
MEMORY_LIMIT_MB = 512

def _worker(code: str, func_name: str, args: tuple, queue: multiprocessing.Queue):
    """
    Worker function to run in a separate process.
    """
    try:
        # 1. Enforce Memory Limit (Address Space)
        # Convert MB to bytes
        limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
        
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # If hard is RLIM_INFINITY (-1 on some systems, max int on others), we can set it.
            # Otherwise, we must not exceed hard.
            # Skip memory limit on macOS to avoid test interference (multiprocessing/resource issues)
            import sys
            if sys.platform != 'darwin':
                if hard == resource.RLIM_INFINITY:
                    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))
                else:
                    limit_bytes = min(limit_bytes, hard)
                    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
        except ValueError as e:
            # On some systems (e.g. macOS), RLIMIT_AS might be hard to set or behave differently.
            # We log/queue the error but PROCEED so that we can at least enforce time limits and import restrictions.
            # In production (Linux), this should work.
            # queue.put({"error": f"Failed to set memory limit: {e}"})
            pass
        
        # 2. Prepare Environment
        namespace = {}
        globals_dict = {
            '__builtins__': SAFE_BUILTINS,
            'pd': pd,
            'np': np,
        }
        
        # 3. Compile and Exec
        compiled = compile(code, '<user-code>', 'exec')
        exec(compiled, globals_dict, namespace)
        
        # 4. Get Function
        if func_name not in namespace or not callable(namespace[func_name]):
            queue.put({"error": f"Function `{func_name}` not defined."})
            return

        # 5. Run Function
        # Note: args[0] is df.copy() passed from parent, so it's already a copy in this process context?
        # Actually multiprocessing pickles arguments. So we get a fresh object here.
        result = namespace[func_name](*args)
        
        # 6. Validate Result
        if not isinstance(result, pd.DataFrame):
            queue.put({"error": "Return value must be a pandas DataFrame."})
            return
            
        queue.put({"success": result})
        
    except MemoryError:
        queue.put({"error": "Memory limit exceeded."})
    except Exception as e:
        # Return the traceback or error message
        queue.put({"error": f"Runtime error: {str(e)}"})

def _run_in_sandbox(code: str, func_name: str, args: tuple) -> pd.DataFrame:
    """
    Orchestrates the sandboxed execution.
    """
    queue = multiprocessing.Queue()
    
    # Create process
    p = multiprocessing.Process(target=_worker, args=(code, func_name, args, queue))
    p.start()
    
    # Wait for completion with timeout
    p.join(timeout=TIME_LIMIT_SECONDS)
    
    if p.is_alive():
        p.terminate()
        p.join()
        raise HTTPException(status_code=400, detail="Time limit exceeded (5s).")
        
    if p.exitcode != 0:
        # Non-zero exit code usually means crash (segfault, OOM kill by OS if rlimit didn't catch it nicely)
        # If killed by signal 9 (SIGKILL) or similar
        raise HTTPException(status_code=400, detail="Process crashed (possibly memory limit exceeded).")
        
    if queue.empty():
        # Should not happen if worker handles exceptions, but just in case
        raise HTTPException(status_code=400, detail="No result returned from worker.")
        
    result = queue.get()
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result["success"]

def run_custom_step(df: pd.DataFrame, code: str, params: dict) -> pd.DataFrame:
    # We pass df and params. df will be pickled.
    # Note: Large DFs might be slow to pickle.
    return _run_in_sandbox(code, "step", (df, params or {}))

def run_custom_join(left_df: pd.DataFrame, right_df: pd.DataFrame, code: str, params: dict) -> pd.DataFrame:
    return _run_in_sandbox(code, "join_step", (left_df, right_df, params or {}))