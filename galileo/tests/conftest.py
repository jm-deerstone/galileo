import pytest
import os
import shutil
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Set Environment Variables BEFORE importing app
# This ensures the app loads with test paths
TEST_DB_URL = "sqlite:///./test_galileo.db"
TEST_SNAPSHOT_BASE = "./test_data_snapshots"

os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["SNAPSHOT_BASE"] = TEST_SNAPSHOT_BASE

from db import Base, engine
import models  # Register models with Base
# Create tables BEFORE importing main (which starts scheduler)
Base.metadata.create_all(bind=engine)

from main import app
from db import get_db

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    # Setup: Create test directories
    if os.path.exists(TEST_SNAPSHOT_BASE):
        shutil.rmtree(TEST_SNAPSHOT_BASE)
    os.makedirs(TEST_SNAPSHOT_BASE, exist_ok=True)
    
    yield # Run all tests
    
    # Teardown: Delete test database and test files
    if os.path.exists("./test_galileo.db"):
        os.remove("./test_galileo.db")
    if os.path.exists(TEST_SNAPSHOT_BASE):
        shutil.rmtree(TEST_SNAPSHOT_BASE)

@pytest.fixture(scope="function")
def client():
    # Create a fresh database for every test function
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()

    # Override the dependency in the FastAPI app
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as c:
        yield c
        
    # Drop tables after test to ensure clean slate
    # Drop tables after test to ensure clean slate
    from sqlalchemy import text
    with engine.connect() as con:
        con.execute(text("PRAGMA foreign_keys=OFF"))
        con.commit()
    Base.metadata.drop_all(bind=engine)
    # Also clean up files created during the test
    # (Optional: might want to keep them for debugging if test fails, but for now clean up)
    # shutil.rmtree(TEST_SNAPSHOT_BASE) 
    # os.makedirs(TEST_SNAPSHOT_BASE, exist_ok=True)
