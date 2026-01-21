# backend/main.py
import matplotlib
matplotlib.use('Agg')
from routers import datasource, preprocess, training, deployment


from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from fastapi import FastAPI

from db import Base, engine


import logging

# Clear existing handlers, force new config (Python 3.8+ supports force=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

# --- Application Setup ---
app = FastAPI(
    title="Versioning Graph API",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




app.include_router(datasource.router, prefix="", tags=["datasources"])
app.include_router(preprocess.router, prefix="", tags=["preprocess"])
app.include_router(training.router, prefix="", tags=["training"])
app.include_router(deployment.router, prefix="", tags=["deployment"])


