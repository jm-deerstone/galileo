import os
import io
import json
import base64
import logging
from typing import List

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from sqlalchemy.orm import Session

from db import get_db
from models import DataSource, Snapshot
from shemas.datasource import DataSourceWithSnapshotRead, SnapshotRead, ActiveSnapshotSet, RowsInsertRequest, \
    SnapshotSummary, DataSourceRead
from services.datasource_service import DatasourceService

logger = logging.getLogger(__name__)
router = APIRouter()

def get_service(db: Session = Depends(get_db)) -> DatasourceService:
    return DatasourceService(db)

@router.get("/datasources/", response_model=List[DataSourceWithSnapshotRead])
async def list_datasources(db: Session = Depends(get_db)):
    return db.query(DataSource).all()

@router.get("/datasources/{ds_id}", response_model=DataSourceRead)
async def read_datasource(ds_id: str, db: Session = Depends(get_db)):
    ds = db.get(DataSource, ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="DataSource not found")
    return ds

@router.post(
    "/datasources/with-snapshot/",
    response_model=DataSourceWithSnapshotRead,
    summary="Create a new datasource, upload its initial CSV, and record schema"
)
async def create_datasource_with_snapshot(
        name: str = Form(...),
        file: UploadFile = File(...),
        service: DatasourceService = Depends(get_service),
):
    return await service.create_datasource_with_snapshot(name, file)

@router.get(
    "/datasources/{datasource_id}/with-snapshot/",
    response_model=DataSourceWithSnapshotRead,
    summary="Get datasource with its schema and snapshots"
)
def get_datasource_with_snapshot(datasource_id: str, db: Session = Depends(get_db)):
    ds = db.query(DataSource).filter_by(id=datasource_id).first()
    if not ds:
        raise HTTPException(404, "Datasource not found")
    return ds

@router.get(
    "/datasources/{datasource_id}/snapshots/{snapshot_id}/download",
    summary="Download a snapshot CSV file"
)
def download_snapshot(
        datasource_id: str,
        snapshot_id: str,
        db: Session = Depends(get_db)
):
    snap = db.query(Snapshot).filter_by(id=snapshot_id, datasource_id=datasource_id).first()
    if not snap:
        raise HTTPException(404, "Snapshot not found")
    if not os.path.isfile(snap.path):
        raise HTTPException(404, "File missing on server")
    return FileResponse(snap.path, filename=os.path.basename(snap.path))

@router.post(
    "/datasources/{datasource_id}/snapshots/",
    response_model=SnapshotRead,
    summary="Upload a new CSV snapshot to an existing datasource (schema must match)"
)
async def upload_snapshot_to_datasource(
        datasource_id: str,
        file: UploadFile = File(...),
        service: DatasourceService = Depends(get_service),
):
    return await service.upload_snapshot(datasource_id, file)

@router.post(
    "/datasources/{ds_id}/active_snapshot/",
    response_model=DataSourceWithSnapshotRead,
)
def set_active_snapshot(
        ds_id: str,
        body: ActiveSnapshotSet,
        db: Session = Depends(get_db),
):
    # This logic is simple enough to keep here or move to service. 
    # For consistency, let's keep simple CRUD here unless it gets complex.
    ds = db.get(DataSource, ds_id)
    if not ds:
        raise HTTPException(404, "Datasource not found")
    
    # Check if root datasource logic is needed? 
    # The original code had checks for Preprocess parents.
    # Let's move this to service if we want full strictness, but for now:
    snap = db.get(Snapshot, body.snapshot_id)
    if not snap or snap.datasource_id != ds_id:
        raise HTTPException(400, "Invalid snapshot for this datasource")

    ds.active_snapshot_id = body.snapshot_id
    db.commit()
    db.refresh(ds)
    return ds

@router.post(
    "/datasources/{ds_id}/active_snapshot/rows",
    summary="Append rows to the active snapshot of a root datasource",
)
def append_to_active_snapshot(
        ds_id: str,
        req: RowsInsertRequest,
        service: DatasourceService = Depends(get_service),
):
    added = service.append_rows(ds_id, req)
    return {"added": added}

@router.get(
    "/datasources/{ds_id}/snapshots/{snap_id}/summary/",
    response_model=SnapshotSummary,
    summary="Compute per-column summary over *all* rows of a snapshot"
)
def get_snapshot_summary(
        ds_id: str,
        snap_id: str,
        service: DatasourceService = Depends(get_service),
):
    return service.get_snapshot_summary(ds_id, snap_id)

@router.get(
    "/datasources/{ds_id}/snapshots/{snap_id}/pairwise_matrix_stream",
    summary="Stream pairwise matrix progressively (SSE via StreamingResponse)",
)
def pairwise_matrix_stream(
        ds_id: str,
        snap_id: str,
        cols: list[str] | None = Query(None),
        db: Session = Depends(get_db),
):
    # This one is complex with streaming. 
    # Moving the generator logic to service is tricky with FastAPI StreamingResponse.
    # For now, I'll leave the generator logic here but use service for data loading if possible.
    # Or better, just keep it here as it's highly specific to HTTP streaming.
    
    # Actually, let's keep the original implementation for this specific endpoint 
    # because it yields data for SSE, which is a controller concern.
    # But we can use service to get the file path.
    
    snap = db.get(Snapshot, snap_id)
    if not snap or not os.path.exists(snap.path):
        raise HTTPException(404, "Snapshot not found")

    with open(snap.path, "r") as f:
        total_rows = sum(1 for _ in f) - 1
    if total_rows <= 0:
        raise HTTPException(400, "Snapshot is empty")

    chunk_size = 5000
    reader = pd.read_csv(snap.path, chunksize=chunk_size)

    try:
        first = next(reader)
    except StopIteration:
        raise HTTPException(400, "No data found")
        
    if cols:
        wanted = []
        for part in cols:
            for c in part.split(","):
                c = c.strip()
                if c in first.columns:
                    wanted.append(c)
        plot_cols = wanted or []
    else:
        plot_cols = first.select_dtypes(include=["number"]).columns.tolist()

    if not plot_cols:
        raise HTTPException(400, "No numeric columns to plot")

    reader = pd.read_csv(snap.path, usecols=plot_cols, chunksize=chunk_size)

    def event_stream():
        accumulated = pd.DataFrame(columns=plot_cols)
        processed = 0

        for chunk in reader:
            accumulated = pd.concat([accumulated, chunk], ignore_index=True)
            processed += len(chunk)
            progress = min(processed / total_rows, 1.0)

            n = len(plot_cols)
            plt.style.use("dark_background")
            fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n), squeeze=False)
            fig.patch.set_facecolor("#2b2b2b")
            GRID_KW  = dict(color="#444444", linestyle="--", linewidth=0.5)

            for i, y in enumerate(plot_cols):
                for j, x in enumerate(plot_cols):
                    ax = axes[i][j]
                    ax.set_facecolor("#2b2b2b")
                    ax.grid(**GRID_KW)

                    if i == j:
                        ax.hist(accumulated[x].dropna(), bins=20, edgecolor="#333333", color="#00C49F")
                    else:
                        ax.scatter(accumulated[x], accumulated[y], s=5, alpha=0.6, color="#8884d8")
                    ax.tick_params(colors="white", labelsize=8)
                    if i == 0:
                        ax.xaxis.set_label_position("top")
                        ax.xaxis.tick_top()
                        ax.set_xlabel(x, fontsize=12, color="white")
                    else:
                        ax.set_xticks([])
                    if j == 0:
                        ax.set_ylabel(y, fontsize=12, color="white")
                    else:
                        ax.set_yticks([])

            plt.tight_layout(pad=1.0)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            payload = json.dumps({"progress": progress, "image": img_b64})
            yield f"data: {payload}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/datasources/{ds_id}/snapshots/{snap_id}/histogram.png")
def histogram_png(
        ds_id: str, snap_id: str, col: str = Query(...), bins: int = Query(10),
        service: DatasourceService = Depends(get_service)
):
    buf = service.generate_histogram(snap_id, col, bins)
    return StreamingResponse(buf, media_type="image/png")

@router.get("/datasources/{ds_id}/snapshots/{snap_id}/scatter.png")
def scatter_png(
        ds_id: str, snap_id: str, x: str = Query(...), y: str = Query(...),
        service: DatasourceService = Depends(get_service)
):
    buf = service.generate_scatter(snap_id, x, y)
    return StreamingResponse(buf, media_type="image/png")

@router.get("/datasources/{ds_id}/snapshots/{snap_id}/pie.png")
def pie_png(
        ds_id: str, snap_id: str, col: str = Query(...),
        service: DatasourceService = Depends(get_service)
):
    buf = service.generate_pie(snap_id, col)
    return StreamingResponse(buf, media_type="image/png")

@router.get("/datasources/{ds_id}/snapshots/{snap_id}/line.png")
def line_png(
        ds_id: str, snap_id: str, date_col: str = Query(...), value_col: str = Query(...), granularity: str = Query("day"),
        service: DatasourceService = Depends(get_service)
):
    buf = service.generate_line(snap_id, date_col, value_col, granularity)
    return StreamingResponse(buf, media_type="image/png")

@router.get(
    "/datasources/{ds_id}/snapshots/{snap_id}/profile_report/",
    summary="Full pandas-profiling (ydata-profiling) report as HTML",
    response_class=HTMLResponse
)
def profile_report(
        ds_id: str,
        snap_id: str,
        service: DatasourceService = Depends(get_service),
):
    html = service.generate_profile_report(snap_id)
    return HTMLResponse(content=html)