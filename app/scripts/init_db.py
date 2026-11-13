"""Database initialization CLI script."""
from __future__ import annotations

import sys
from pathlib import Path

# Add app to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database.db import init_db
from backend.services.job_service import JobService


def main():
    print("Initializing SQLite database tables...")
    init_db()
    recovered = JobService.recover_stale_jobs()
    print(f"Database initialized successfully. (Recovered jobs: {recovered})")


if __name__ == "__main__":
    main()
