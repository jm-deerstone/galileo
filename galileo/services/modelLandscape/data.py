import pandas as pd

def _build_sliding_window(df: pd.DataFrame, spec: dict):
    fw = [{"name":f["name"], "start_idx":int(f["start_idx"]), "end_idx":int(f["end_idx"])}
          for f in spec["features"]]
    tw = {"name": spec["target"]["name"], "start_idx": int(spec["target"]["start_idx"]), "end_idx": int(spec["target"]["end_idx"])}
    max_end = max(p["end_idx"] for p in fw + [tw])
    X_rows, y_rows = [], []
    for i in range(max_end, len(df)):
        feats = []
        for p in fw:
            window = df[p["name"]].iloc[i-p["end_idx"]:i-p["start_idx"]+1]
            feats.extend(window.tolist())
        X_rows.append(feats)
        tvals = df[tw["name"]].iloc[i-tw["end_idx"]:i-tw["start_idx"]+1]
        y_rows.append(tvals.iloc[0] if len(tvals)==1 else tvals.tolist())
    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows) if all(not isinstance(v,(list,tuple)) for v in y_rows) else pd.DataFrame(y_rows)
    return X, y