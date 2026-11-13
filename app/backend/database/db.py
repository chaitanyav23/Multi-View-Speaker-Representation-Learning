"""SQLite Database Connection and Schema Management."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "storage" / "app.db"
_custom_db_path: Optional[Path] = None


def get_db_path() -> Path:
    """Return the currently active SQLite database path."""
    if _custom_db_path is not None:
        return _custom_db_path
    env_path = os.getenv("APP_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def set_db_path(path: Optional[Path]) -> None:
    """Override or reset the SQLite database path (used for test isolation)."""
    global _custom_db_path
    _custom_db_path = Path(path) if path is not None else None


# Backward-compatibility alias
DB_PATH = DEFAULT_DB_PATH


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a thread-safe connection to the SQLite database."""
    target_path = Path(db_path) if db_path is not None else get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite database tables and indexes."""
    conn = get_db_connection(db_path=db_path)
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                duration REAL DEFAULT 0.0,
                size_bytes INTEGER DEFAULT 0,
                sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                stage TEXT DEFAULT 'idle',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                recording_id TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                turn_id TEXT,
                word_index INTEGER,
                original_text TEXT,
                corrected_text TEXT,
                original_speaker TEXT,
                corrected_speaker TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            CREATE TABLE IF NOT EXISTS plugin_results (
                result_id TEXT PRIMARY KEY,
                plugin_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                plugin_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                execution_time_seconds REAL DEFAULT 0.0,
                error TEXT,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            CREATE INDEX IF NOT EXISTS idx_recordings_sha256 ON recordings(sha256);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_plugin_rec ON plugin_results(plugin_id, recording_id);
            CREATE INDEX IF NOT EXISTS idx_plugin_created ON plugin_results(created_at);
        """)
    conn.close()
