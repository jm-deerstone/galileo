import os
import io
import json
import re
import csv
import base64
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ydata_profiling

from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException

from config import SNAPSHOT_BASE
from models import DataSource, Snapshot, Preprocess
from shemas.datasource import RowsInsertRequest, ColumnSummary, SnapshotSummary

logger = logging.getLogger(__name__)

class DatasourceService:
    def __init__(self, db: Session):
        self.db = db

    async def create_datasource_with_snapshot(self, name: str, file: UploadFile) -> DataSource:
        # 1) Create DataSource row
        ds = DataSource(name=name)
        self.db.add(ds)
        self.db.flush()  # assign ds.id

        # 2) Read CSV bytes and introspect schema with pandas
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        schema = {"columns": []}

        for column in df.columns:
            series = df[column]
            str_vals = series.fillna("").astype(str)

            # Check for full timestamp format
            is_datetime = bool(
                str_vals.map(lambda v: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', v)))
                .all()
            )
            # Check for date-only format
            is_date = bool(
                str_vals.map(lambda v: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', v)))
                .all()
            )

            # Decide dtype
            if is_datetime:
                inferred_type = 'datetime64'
            elif is_date:
                inferred_type = 'date'
            else:
                inferred_type = pd.api.types.infer_dtype(series, skipna=True)

            # Append to schema
            schema["columns"].append({
                "name": column,
                "dtype": inferred_type,
                "null_count": int(series.isna().sum())
            })

        ds.schema_json = json.dumps(schema)

        # 3) Persist file under datasources/{id}/
        dest_dir = os.path.join(SNAPSHOT_BASE, ds.id)
        os.makedirs(dest_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        filename = f"{timestamp}_{file.filename}"
        full_path = os.path.join(dest_dir, filename)
        with open(full_path, "wb") as out:
            out.write(content)

        # 4) Create initial Snapshot row
        snap = Snapshot(datasource_id=ds.id, path=full_path)
        self.db.add(snap)
        self.db.flush()
        
        ds.active_snapshot_id = snap.id

        # 5) Commit transaction
        self.db.commit()
        self.db.refresh(ds)
        return ds

    async def upload_snapshot(self, datasource_id: str, file: UploadFile) -> Snapshot:
        # 1) Lookup the existing DataSource
        ds: DataSource = self.db.query(DataSource).filter_by(id=datasource_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Datasource not found")

        # 2) Parse stored schema
        try:
            stored = json.loads(ds.schema_json or '{"columns": []}')["columns"]
            stored_cols = [(c["name"], c["dtype"]) for c in stored]
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid stored schema")

        # 3) Read the uploaded CSV and infer its schema
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        inferred_cols: list[tuple[str,str]] = []
        for col in df.columns:
            series = df[col]
            str_vals = series.fillna("").astype(str)
            is_datetime = bool(
                str_vals.map(lambda v: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', v))).all()
            )
            is_date = bool(
                str_vals.map(lambda v: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', v))).all()
            )
            if is_datetime:
                dtype = "datetime64"
            elif is_date:
                dtype = "date"
            else:
                dtype = pd.api.types.infer_dtype(series, skipna=True)
            inferred_cols.append((col, dtype))

        # 4) Validate schema match
        if stored_cols != inferred_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Schema mismatch: expected {stored_cols}, got {inferred_cols}"
            )

        # 5) Persist file in filesystem
        dest_dir = os.path.join(SNAPSHOT_BASE, datasource_id)
        os.makedirs(dest_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        filename = f"{ts}_{file.filename}"
        full_path = os.path.join(dest_dir, filename)
        with open(full_path, "wb") as out:
            out.write(content)

        # 6) Create Snapshot record
        snap = Snapshot(datasource_id=datasource_id, path=full_path)
        self.db.add(snap)
        self.db.commit()
        self.db.refresh(snap)

        return snap

    def append_rows(self, ds_id: str, req: RowsInsertRequest) -> int:
        # 1) load datasource
        ds = self.db.get(DataSource, ds_id)
        if not ds:
            raise HTTPException(404, "Datasource not found")

        # 2) only allow root datasources
        has_parent = (
                self.db.query(Preprocess)
                .filter(Preprocess.datasource_child_id == ds_id)
                .first()
                is not None
        )
        if has_parent:
            raise HTTPException(400, "Can only append to a root datasource")

        # 3) must have an active snapshot
        snap_id = ds.active_snapshot_id
        if not snap_id:
            raise HTTPException(400, "No active snapshot set for this datasource")
        snap = self.db.get(Snapshot, snap_id)
        if not snap:
            raise HTTPException(500, "Active snapshot record not found")

        # 4) make sure file exists
        csv_path = snap.path
        if not os.path.exists(csv_path):
            raise HTTPException(500, "Snapshot file not found on disk")

        # 5) load schema
        if not ds.schema_json:
            raise HTTPException(500, "Datasource has no schema on record")
        schema = json.loads(ds.schema_json)
        columns = [c["name"] for c in schema.get("columns", [])]
        if not columns:
            raise HTTPException(500, "Empty schema, cannot append")

        # 6) ensure newline
        try:
            with open(csv_path, "rb+") as f_check:
                f_check.seek(0, os.SEEK_END)
                if f_check.tell() > 0:
                    f_check.seek(-1, os.SEEK_END)
                    if f_check.read(1) != b"\n":
                        f_check.write(b"\n")
        except Exception as e:
            raise HTTPException(500, f"Could not ensure newline before append: {e}")

        # 7) append rows
        try:
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                for row in req.rows:
                    missing = set(columns) - row.keys()
                    extra   = set(row.keys()) - set(columns)
                    if missing or extra:
                        raise HTTPException(
                            400,
                            f"Row keys must exactly match columns; missing={missing}, extra={extra}"
                        )
                    writer.writerow(row)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to append rows: {e}")

        return len(req.rows)

    def get_snapshot_summary(self, ds_id: str, snap_id: str) -> SnapshotSummary:
        snap = self.db.get(Snapshot, snap_id)
        if not snap or snap.datasource_id != ds_id:
            raise HTTPException(404, "Snapshot not found")

        try:
            df = pd.read_csv(snap.path)
        except Exception as e:
            raise HTTPException(500, f"Could not read snapshot: {e}")

        n = len(df)
        out: list[ColumnSummary] = []

        for col in df.columns:
            series = df[col]
            missing = int(series.isna().sum() + (series == "").sum())
            missing_pct = (missing / n * 100) if n > 0 else 0.0
            unique = int(series.dropna().nunique())

            if pd.api.types.is_numeric_dtype(series):
                col_type = "numeric"
                vals = series.dropna().astype(float)
                if not vals.empty:
                    mean = vals.mean()
                    std = vals.std()
                    mn, mx = vals.min(), vals.max()
                    stats = f"{mean:.2f}±{std:.2f} (range {mn:.2f}–{mx:.2f})"
                else:
                    stats = ""
            elif pd.api.types.is_datetime64_any_dtype(series) or pd.to_datetime(series, errors="coerce").notna().all():
                col_type = "date"
                dates = pd.to_datetime(series, errors="coerce").dropna()
                if not dates.empty:
                    mn = dates.min().strftime("%Y-%m-%d")
                    mx = dates.max().strftime("%Y-%m-%d")
                    stats = f"{mn}–{mx}"
                else:
                    stats = ""
            else:
                col_type = "categorical"
                freq = series.dropna().value_counts()
                if not freq.empty:
                    top, cnt = freq.index[0], int(freq.iloc[0])
                    stats = f"{top} ({cnt/n*100:.1f}%)"
                else:
                    stats = ""

            out.append(
                ColumnSummary(
                    column=col,
                    type=col_type,
                    missing=missing,
                    missing_pct=round(missing_pct, 1),
                    unique=unique,
                    stats=stats,
                )
            )

        return SnapshotSummary(summary=out)

    def generate_histogram(self, snap_id: str, col: str, bins: int) -> io.BytesIO:
        df = self._load_snapshot_df(snap_id)
        if col not in df.columns:
            raise HTTPException(400, f"Unknown column: {col}")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")
        GRID_KW = dict(color="#444444", linestyle="--", linewidth=0.5)

        vals = df[col].dropna()
        ax.hist(vals, bins=bins, color="#00C49F", edgecolor="#333333")
        ax.set_title(f"Histogram of {col}", color="white", fontsize=14)
        ax.set_xlabel(col, color="white")
        ax.set_ylabel("Count", color="white")
        ax.grid(**GRID_KW)
        ax.tick_params(colors="white")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_scatter(self, snap_id: str, x: str, y: str) -> io.BytesIO:
        df = self._load_snapshot_df(snap_id)
        for col in (x, y):
            if col not in df.columns:
                raise HTTPException(400, f"Unknown column: {col}")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")
        GRID_KW = dict(color="#444444", linestyle="--", linewidth=0.5)

        ax.scatter(df[x], df[y], s=10, alpha=0.6, color="#8884d8")
        ax.set_xlabel(x, color="white")
        ax.set_ylabel(y, color="white")
        ax.set_title(f"{y} vs {x}", color="white")
        ax.grid(**GRID_KW)
        ax.tick_params(colors="white")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_pie(self, snap_id: str, col: str) -> io.BytesIO:
        df = self._load_snapshot_df(snap_id)
        if col not in df.columns:
            raise HTTPException(400, f"Unknown column: {col}")

        counts = df[col].fillna("NULL").value_counts()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()
        colors = plt.cm.tab20.colors

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#2b2b2b")
        ax.pie(
            sizes,
            labels=labels,
            colors=colors[: len(labels)],
            autopct="%1.1f%%",
            textprops={"color": "white"},
            wedgeprops={"edgecolor": "#2b2b2b"},
        )
        ax.set_title(f"Distribution of {col}", color="white")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_line(self, snap_id: str, date_col: str, value_col: str, granularity: str) -> io.BytesIO:
        df = self._load_snapshot_df(snap_id)
        for col in (date_col, value_col):
            if col not in df.columns:
                raise HTTPException(400, f"Unknown column: {col}")
        df = df[[date_col, value_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        if granularity == "week":
            df["group"] = df[date_col].dt.to_period("W").astype(str)
        elif granularity == "month":
            df["group"] = df[date_col].dt.to_period("M").astype(str)
        else:
            df["group"] = df[date_col].dt.date.astype(str)

        grouped = df.groupby("group")[value_col].sum().reset_index()
        x = grouped["group"].tolist()
        y = grouped[value_col].tolist()

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2b2b2b")
        ax.plot(x, y, marker="o", color="#f03e3e")
        ax.set_xlabel("Time", color="white")
        ax.set_ylabel(value_col, color="white")
        ax.set_title(f"{value_col} over time ({granularity})", color="white")
        ax.grid(color="#444444", linestyle="--", linewidth=0.5)
        ax.tick_params(colors="white", rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_profile_report(self, snap_id: str) -> str:
        df = self._load_snapshot_df(snap_id)
        profile = ydata_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        html = profile.to_html()
        
        # Add dark CSS (simplified for brevity, can be expanded)
        dark_css = """
        <style>
        html, body { background: #181A1B !important; }
        .profile-report, .container, .container-fluid { background: #181A1B !important; border: none !important; }
        .bg-body-tertiary { background-color: #23272C !important; }
        .table, .dataframe, table { background: #181A1B !important; color: #F2F3F7 !important; border-color: #444 !important; }
        </style>
        """
        return html + dark_css

    def _load_snapshot_df(self, snapshot_id: str) -> pd.DataFrame:
        snap = self.db.get(Snapshot, snapshot_id)
        if not snap:
            raise HTTPException(404, "Snapshot not found")
        try:
            return pd.read_csv(snap.path)
        except Exception as e:
            raise HTTPException(500, f"Failed to read CSV: {e}")
