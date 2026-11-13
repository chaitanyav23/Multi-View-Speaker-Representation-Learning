"""Pytest Configuration and Fixtures for App Test Suite with Strict Test Isolation."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

# Add app directory to sys.path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database.db import (
    DEFAULT_DB_PATH,
    get_db_path,
    init_db,
    set_db_path,
)
from backend.main import create_app

PROD_DB_PATH = DEFAULT_DB_PATH.resolve()
PROD_STORAGE_DIR = (APP_DIR / "storage").resolve()


@pytest.fixture(scope="session", autouse=True)
def isolate_test_environment(tmp_path_factory) -> Generator[dict[str, Path], None, None]:
    """Isolate database and filesystem storage to a temporary directory for all tests.

    Ensures zero mutations occur against production app/storage/app.db or production storage/.
    """
    test_root = tmp_path_factory.mktemp("cs698b_isolated_test")
    test_storage = test_root / "storage"
    test_storage.mkdir(parents=True, exist_ok=True)
    test_db = test_storage / "test_app.db"

    # 1. Point SQLite database to temporary test database
    set_db_path(test_db)
    os.environ["APP_DB_PATH"] = str(test_db)

    # 2. Point storage directories to temporary test storage
    original_storage_dir = settings.storage_dir
    settings.storage_dir = test_storage
    os.environ["STORAGE_DIR"] = str(test_storage)

    # 3. Initialize schema in isolated test database
    init_db(test_db)

    yield {
        "test_root": test_root,
        "test_storage": test_storage,
        "test_db": test_db,
    }

    # Teardown: Restore production paths
    set_db_path(None)
    os.environ.pop("APP_DB_PATH", None)
    os.environ.pop("STORAGE_DIR", None)
    settings.storage_dir = original_storage_dir


@pytest.fixture(autouse=True)
def fail_safe_isolation_guard():
    """Fail immediately if any test attempts to access production database or storage."""
    active_db = get_db_path().resolve()
    if active_db == PROD_DB_PATH:
        raise RuntimeError(
            f"CRITICAL ISOLATION FAIL-SAFE: Test attempted to connect to production database: {active_db}"
        )

    active_storage = settings.storage_dir.resolve()
    if active_storage == PROD_STORAGE_DIR:
        raise RuntimeError(
            f"CRITICAL ISOLATION FAIL-SAFE: Test attempted to write to production storage: {active_storage}"
        )


@pytest.fixture
def client(isolate_test_environment) -> TestClient:
    """Return a FastAPI test client backed by the isolated test environment."""
    app = create_app()
    return TestClient(app)
