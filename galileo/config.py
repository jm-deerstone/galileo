import os

# 1) Define filesystem base dir for snapshots
SNAPSHOT_BASE = os.getenv("SNAPSHOT_BASE", os.path.join(os.getcwd(), "datasources"))

# 2) Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./versioning.db")