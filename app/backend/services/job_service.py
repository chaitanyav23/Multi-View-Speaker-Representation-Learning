"""Asynchronous Processing Job Service with Concurrency Control and Startup Recovery."""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from backend.database.models import JobRecord, JobRepository

logger = logging.getLogger("job_service")


class JobService:
    """Manages background job dispatch, execution state machine, and concurrency limits."""

    # Concurrency lock ensuring single GPU/heavy inference task executes at any given time
    _inference_semaphore = threading.Semaphore(1)

    @classmethod
    def recover_stale_jobs(cls) -> int:
        """Find and recover jobs interrupted by server restart."""
        now_iso = datetime.now(timezone.utc).isoformat()
        count = JobRepository.recover_interrupted_jobs(now_iso)
        if count > 0:
            logger.warning("Recovered %d interrupted processing jobs on startup.", count)
        return count

    @classmethod
    def create_job(cls, recording_id: str) -> JobRecord:
        """Create a new job in queued state."""
        # Check if an active job already exists for this recording
        active_job = JobRepository.get_by_recording(recording_id)
        if active_job and active_job.status in ("queued", "processing"):
            return active_job

        job_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        job = JobRecord(
            job_id=job_id,
            recording_id=recording_id,
            status="queued",
            progress=0.0,
            stage="Queued for processing",
            error=None,
            created_at=now_iso,
            updated_at=now_iso,
        )
        JobRepository.create(job)
        return job

    @classmethod
    def submit_task(cls, job_id: str, task_func: Callable[[str, Callable[[float, str], None]], Any]) -> None:
        """Run task_func in a background thread governed by the concurrency semaphore."""
        def _worker():
            # Acquire concurrency token
            with cls._inference_semaphore:
                now_iso = datetime.now(timezone.utc).isoformat()
                JobRepository.update_status(job_id, "processing", 0.05, "Initializing engine", now_iso)

                def progress_cb(prog: float, stage_name: str) -> None:
                    upd_iso = datetime.now(timezone.utc).isoformat()
                    JobRepository.update_status(job_id, "processing", prog, stage_name, upd_iso)

                try:
                    task_func(job_id, progress_cb)
                    fin_iso = datetime.now(timezone.utc).isoformat()
                    JobRepository.update_status(job_id, "completed", 1.0, "Processing complete", fin_iso)
                except Exception as exc:
                    err_iso = datetime.now(timezone.utc).isoformat()
                    logger.error("Job %s failed with exception: %s", job_id, exc, exc_info=True)
                    JobRepository.update_status(job_id, "failed", 0.0, "Error during processing", err_iso, error=str(exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    @classmethod
    def get_job(cls, job_id: str) -> Optional[JobRecord]:
        return JobRepository.get(job_id)

    @classmethod
    def get_job_by_recording(cls, recording_id: str) -> Optional[JobRecord]:
        return JobRepository.get_by_recording(recording_id)
