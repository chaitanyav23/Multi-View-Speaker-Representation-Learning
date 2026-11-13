"""Database Data Models and Query Helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .db import get_db_connection


@dataclass
class RecordingRecord:
    recording_id: str
    filename: str
    filepath: str
    duration: float
    size_bytes: int
    sha256: str
    created_at: str


@dataclass
class JobRecord:
    job_id: str
    recording_id: str
    status: str
    progress: float
    stage: str
    error: Optional[str]
    created_at: str
    updated_at: str


class RecordingRepository:
    """Persistence operations for recordings and media files."""

    @staticmethod
    def create(rec: RecordingRecord) -> None:
        conn = get_db_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recordings (recording_id, filename, filepath, duration, size_bytes, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rec.recording_id, rec.filename, rec.filepath, rec.duration, rec.size_bytes, rec.sha256, rec.created_at),
            )
        conn.close()

    @staticmethod
    def get(recording_id: str) -> Optional[RecordingRecord]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM recordings WHERE recording_id = ?", (recording_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return RecordingRecord(
            recording_id=row["recording_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            duration=row["duration"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            created_at=row["created_at"],
        )

    @staticmethod
    def get_by_hash(sha256: str) -> Optional[RecordingRecord]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM recordings WHERE sha256 = ?", (sha256,)).fetchone()
        conn.close()
        if not row:
            return None
        return RecordingRecord(
            recording_id=row["recording_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            duration=row["duration"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            created_at=row["created_at"],
        )

    @staticmethod
    def list_all() -> list[RecordingRecord]:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM recordings ORDER BY created_at DESC").fetchall()
        conn.close()
        return [
            RecordingRecord(
                recording_id=r["recording_id"],
                filename=r["filename"],
                filepath=r["filepath"],
                duration=r["duration"],
                size_bytes=r["size_bytes"],
                sha256=r["sha256"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


class JobRepository:
    """Persistence operations for asynchronous speech jobs."""

    @staticmethod
    def create(job: JobRecord) -> None:
        conn = get_db_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (job_id, recording_id, status, progress, stage, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job.job_id, job.recording_id, job.status, job.progress, job.stage, job.error, job.created_at, job.updated_at),
            )
        conn.close()

    @staticmethod
    def update_status(job_id: str, status: str, progress: float, stage: str, updated_at: str, error: Optional[str] = None) -> None:
        conn = get_db_connection()
        with conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, progress = ?, stage = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, progress, stage, error, updated_at, job_id),
            )
        conn.close()

    @staticmethod
    def get(job_id: str) -> Optional[JobRecord]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return JobRecord(
            job_id=row["job_id"],
            recording_id=row["recording_id"],
            status=row["status"],
            progress=row["progress"],
            stage=row["stage"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def get_by_recording(recording_id: str) -> Optional[JobRecord]:
        conn = get_db_connection()
        row = conn.execute(
            "SELECT * FROM jobs WHERE recording_id = ? ORDER BY created_at DESC LIMIT 1",
            (recording_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return JobRecord(
            job_id=row["job_id"],
            recording_id=row["recording_id"],
            status=row["status"],
            progress=row["progress"],
            stage=row["stage"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def recover_interrupted_jobs(timestamp: str) -> int:
        """Mark stale queued/processing jobs as failed on application startup."""
        conn = get_db_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = 'Job interrupted by application restart. Please re-run.',
                    updated_at = ?
                WHERE status IN ('queued', 'processing')
                """,
                (timestamp,),
            )
            count = cursor.rowcount
        conn.close()
        return count


class TranscriptRepository:
    """Persistence operations for serialized canonical transcripts."""

    @staticmethod
    def save(recording_id: str, transcript_data: dict[str, Any], created_at: str) -> None:
        conn = get_db_connection()
        json_str = json.dumps(transcript_data)
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO transcripts (recording_id, result_json, created_at)
                VALUES (?, ?, ?)
                """,
                (recording_id, json_str, created_at),
            )
        conn.close()

    @staticmethod
    def get(recording_id: str) -> Optional[dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute("SELECT result_json FROM transcripts WHERE recording_id = ?", (recording_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return json.loads(row["result_json"])


@dataclass
class PluginAnalysisRecord:
    """Domain model representing a stored plugin execution result."""
    result_id: str
    plugin_id: str
    recording_id: str
    status: str
    result_json: str
    plugin_version: str
    created_at: str
    execution_time_seconds: float = 0.0
    error: Optional[str] = None


class PluginRepository:
    """Persistence operations for third-party plugin analysis results."""

    @staticmethod
    def create(record: PluginAnalysisRecord) -> None:
        conn = get_db_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO plugin_results (
                    result_id, plugin_id, recording_id, status, result_json,
                    plugin_version, created_at, execution_time_seconds, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.result_id,
                    record.plugin_id,
                    record.recording_id,
                    record.status,
                    record.result_json,
                    record.plugin_version,
                    record.created_at,
                    record.execution_time_seconds,
                    record.error,
                ),
            )
        conn.close()

    @staticmethod
    def get(result_id: str) -> Optional[PluginAnalysisRecord]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM plugin_results WHERE result_id = ?", (result_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return PluginAnalysisRecord(
            result_id=row["result_id"],
            plugin_id=row["plugin_id"],
            recording_id=row["recording_id"],
            status=row["status"],
            result_json=row["result_json"],
            plugin_version=row["plugin_version"],
            created_at=row["created_at"],
            execution_time_seconds=float(row["execution_time_seconds"]),
            error=row["error"],
        )

    @staticmethod
    def get_latest(plugin_id: str, recording_id: str) -> Optional[PluginAnalysisRecord]:
        conn = get_db_connection()
        row = conn.execute(
            """
            SELECT * FROM plugin_results
            WHERE plugin_id = ? AND recording_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (plugin_id, recording_id),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return PluginAnalysisRecord(
            result_id=row["result_id"],
            plugin_id=row["plugin_id"],
            recording_id=row["recording_id"],
            status=row["status"],
            result_json=row["result_json"],
            plugin_version=row["plugin_version"],
            created_at=row["created_at"],
            execution_time_seconds=float(row["execution_time_seconds"]),
            error=row["error"],
        )

    @staticmethod
    def list_by_recording(recording_id: str, plugin_id: Optional[str] = None) -> list[PluginAnalysisRecord]:
        conn = get_db_connection()
        if plugin_id:
            rows = conn.execute(
                """
                SELECT * FROM plugin_results
                WHERE recording_id = ? AND plugin_id = ?
                ORDER BY created_at DESC
                """,
                (recording_id, plugin_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM plugin_results
                WHERE recording_id = ?
                ORDER BY created_at DESC
                """,
                (recording_id,),
            ).fetchall()
        conn.close()
        return [
            PluginAnalysisRecord(
                result_id=r["result_id"],
                plugin_id=r["plugin_id"],
                recording_id=r["recording_id"],
                status=r["status"],
                result_json=r["result_json"],
                plugin_version=r["plugin_version"],
                created_at=r["created_at"],
                execution_time_seconds=float(r["execution_time_seconds"]),
                error=r["error"],
            )
            for r in rows
        ]
