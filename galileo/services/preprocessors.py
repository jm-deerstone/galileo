import json
from fastapi import HTTPException

import pandas as pd
import numpy as np
import re
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def rename_column(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """params: { from: str, to: str }"""
    return df.rename(columns={params['from']: params['to']})


def drop_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """params: { columns: List[str] }"""
    return df.drop(columns=params['columns'], errors='ignore')


def filter_rows(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: {
      column: str,
      operator: one of ['>','<','>=','<=','==','!='],
      value: numeric or string
    }
    """
    op = params['operator']
    col = df[params['column']]
    val = params['value']
    ops = {
        '>': col > val,
        '<': col < val,
        '>=': col >= val,
        '<=': col <= val,
        '==': col == val,
        '!=': col != val,
    }
    mask = ops.get(op)
    return df[mask] if mask is not None else df


def filter_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: { column: str, method: 'zscore'|'iqr', threshold: float }
    """
    col = params['column']
    method = params.get('method', 'zscore')
    thresh = params.get('threshold', 3)
    series = df[col].dropna().astype(float)
    if method == 'zscore':
        z = np.abs(zscore(series))
        good = z < thresh
        return df.iloc[good.index[good]]
    else:  # IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))
        return df.iloc[mask.index[mask]]


def impute_missing(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: {
      column: str|list[str]|None,   # Optional: if None or "__ALL__", apply to all columns
      strategy: 'mean'|'median'|'mode'|'constant'|'ffill'|'bfill',
      fill_value: optional (for 'constant')
    }
    """
    columns = params.get('column')
    strat = params.get('strategy', 'mean')

    if columns is None or columns == "__ALL__":
        columns = df.columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]

    result_df = df.copy()
    for col in columns:
        if strat in ('mean', 'median', 'mode'):
            if strat == 'mode':
                val = result_df[col].mode().iloc[0]
            else:
                val = getattr(result_df[col], strat)()
            result_df[col] = result_df[col].fillna(val)
        elif strat == 'constant':
            result_df[col] = result_df[col].fillna(params.get('fill_value'))
        elif strat in ('ffill', 'bfill'):
            result_df[col] = result_df[col].fillna(method=strat)
        # else: do nothing (could add error/warning here)
    return result_df


def one_hot_encode(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params:
      column: str,
      categories: list[str] or JSON-stringified list,
      drop_original: bool or 'true'/'false'
    """
    col = params.get("column")
    if col not in df.columns:
        raise HTTPException(400, f"One-hot failed: unknown column '{col}'")

    # 1) parse categories list
    cats = params.get("categories")
    if isinstance(cats, str):
        try:
            cats = json.loads(cats)
        except json.JSONDecodeError:
            raise HTTPException(400, f"One-hot failed: invalid JSON for categories of '{col}'")
    if not isinstance(cats, (list, tuple)) or not all(isinstance(c, str) for c in cats):
        raise HTTPException(400, f"One-hot failed: categories must be a list of strings for '{col}'")

    # 2) get dummies for this column
    dummies = pd.get_dummies(df[col], prefix=col)
    # 3) ensure only the requested categories (and in requested order)
    expected_cols = [f"{col}_{cat}" for cat in cats]
    for ec in expected_cols:
        if ec not in dummies.columns:
            # fill missing category with zeros
            dummies[ec] = 0
    # drop any extra categories that were in the data but not in the list
    dummies = dummies[expected_cols]

    # 4) concat back
    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    # 5) drop original if asked
    drop_orig = params.get("drop_original", False)
    if isinstance(drop_orig, str):
        drop_orig = drop_orig.lower() == "true"
    if drop_orig:
        out = out.drop(columns=[col])

    return out


def label_encode(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params:
      column: str,
      categories: optional list[str] or JSON-stringified list
    If categories list is provided, unseen values will raise.
    Otherwise uses pandas Categorical.
    """
    col = params.get("column")
    if col not in df.columns:
        raise HTTPException(400, f"Label-encode failed: unknown column '{col}'")

    cats = params.get("categories")
    if cats is not None:
        if isinstance(cats, str):
            try:
                cats = json.loads(cats)
            except json.JSONDecodeError:
                raise HTTPException(400, f"Label-encode failed: invalid JSON for categories of '{col}'")
        if not isinstance(cats, (list, tuple)) or not all(isinstance(c, str) for c in cats):
            raise HTTPException(400, f"Label-encode failed: categories must be a list of strings for '{col}'")
        # force pandas Categorical with specified categories
        cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
        try:
            codes = pd.Series(df[col], dtype=cat_type).cat.codes
        except Exception as e:
            raise HTTPException(400, f"Label-encode failed: {e}")
        df[col] = codes
    else:
        # default: infer categories, unseen won't happen in preview
        df[col] = df[col].astype("category").cat.codes

    return df


def scale_numeric(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: {
      columns: List[str],
      method: 'minmax'|'standard'
    }
    """
    cols = params['columns']
    method = params.get('method', 'standard')
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def log_transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: { columns: List[str], offset: float (to shift before log) }
    """
    cols = params['columns']
    offset = params.get('offset', 1e-6)
    for c in cols:
        df[c] = np.log(df[c] + offset)
    return df


def extract_datetime_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: { column: str, features: List['year','month','day','hour','weekday'] }
    """
    col = params['column']
    dt = pd.to_datetime(df[col], errors='coerce')
    for feat in params.get('features', ['year', 'month', 'day']):
        df[f"{col}_{feat}"] = getattr(dt.dt, feat)
    return df


def remove_duplicates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: { subset: optional List[str], keep: 'first'|'last'|False }
    """
    return df.drop_duplicates(subset=params.get('subset'), keep=params.get('keep', 'first'))


def bin_numeric(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    raw_bins = params.get("bins")

    # 1) If the user typed a JSON array like "[0, 10, 20]", parse it
    bins: int | list[float]
    if isinstance(raw_bins, str):
        try:
            # Try JSON‐decoding first (e.g. "[0, 10, 20]")
            bins = json.loads(raw_bins)
        except json.JSONDecodeError:
            # If that fails, assume it’s an integer string like "5"
            bins = int(raw_bins)
    else:
        # Already an int or a list
        bins = raw_bins

    labels = params.get("labels", None)
    # If labels is also a JSON string, you might do something similar:
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError:
            labels = None  # or raise an error/warning

    return df.assign(**{
        params["column"] + "_binned": pd.cut(
            df[params["column"]],
            bins=bins,
            labels=labels,
        )
    })


def normalize_text(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: { column: str, lowercase: bool, strip: bool }
    """
    col = params['column']
    s = df[col].astype(str)
    if params.get('lowercase', True):
        s = s.str.lower()
    if params.get('strip', True):
        s = s.str.strip()
    return df.assign(**{col: s})


def cap_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    params: { column: str, method: 'clip'|'winsorize', lower_pct: float, upper_pct: float }
    """
    c = params['column']
    lower, upper = df[c].quantile(params.get('lower_pct', 0.01)), df[c].quantile(params.get('upper_pct', 0.99))
    if params.get('method', 'clip') == 'clip':
        df[c] = df[c].clip(lower, upper)
    else:
        # winsorize in-place
        df[c] = np.where(df[c] < lower, lower, np.where(df[c] > upper, upper, df[c]))
    return df


def join_step(leftDf: pd.DataFrame, rightDf: pd.DataFrame, params: dict) -> pd.DataFrame:
    # 1. Perform pandas merge
    how = params.get('how', 'inner')
    left_on = params['left_keys']
    right_on = params['right_keys']
    suffixes = tuple(params.get('suffixes', '_x,_y').split(','))
    
    # print(f"DEBUG: Left cols: {leftDf.columns.tolist()}")
    # print(f"DEBUG: Right cols: {rightDf.columns.tolist()}")
    # print(f"DEBUG: Join keys: left={left_on}, right={right_on}")

    try:
        result = leftDf.merge(
            rightDf,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes
        )
    except KeyError as e:
        raise HTTPException(400, f"Join failed: Missing key {e}")
    except Exception as e:
        raise HTTPException(400, f"Join failed: {str(e)}")
    return result
