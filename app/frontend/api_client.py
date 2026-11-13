"""HTTP Client for communicating with the FastAPI backend."""
from __future__ import annotations

from typing import Any, Optional
import requests


class APIClient:
    """Encapsulates REST communication with the FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def upload_media(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(f"{self.base_url}/api/v1/media/upload", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_recordings(self) -> list[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/media", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def trigger_transcription(
        self, recording_id: str, use_mock: bool = False, initial_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "use_mock": use_mock,
            "initial_prompt": initial_prompt,
        }
        resp = requests.post(f"{self.base_url}/api/v1/transcriptions", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_job_status(self, job_or_rec_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/transcriptions/{job_or_rec_id}/status", timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_transcript(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/transcriptions/{recording_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code in (202, 404):
            return None  # In progress (202) or not yet transcribed / not found (404)
        resp.raise_for_status()
        return None

    def get_analytics(self, recording_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/analytics/{recording_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def submit_feedback(
        self,
        recording_id: str,
        turn_id: Optional[str],
        original_text: str,
        corrected_text: str,
        original_speaker: Optional[str] = None,
        corrected_speaker: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "turn_id": turn_id,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "original_speaker": original_speaker,
            "corrected_speaker": corrected_speaker,
            "notes": notes,
        }
        resp = requests.post(f"{self.base_url}/api/v1/feedback", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_vocabulary(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/v1/feedback/vocabulary", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_export_url(self, recording_id: str, format_str: str) -> str:
        return f"{self.base_url}/api/v1/transcriptions/{recording_id}/export?format={format_str}"

    def get_stream_url(self, recording_id: str) -> str:
        return f"{self.base_url}/api/v1/media/{recording_id}/stream"

    # Stage 1.6.1: Developer API & Plugin Methods

    def list_plugins(self) -> list[dict[str, Any]]:
        """List registered plugin manifests from the backend."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/plugins", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_recording_context(self, recording_id: str) -> Optional[dict[str, Any]]:
        """Retrieve consolidated plugin context for a recording."""
        resp = requests.get(f"{self.base_url}/api/v1/recordings/{recording_id}/context", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        resp.raise_for_status()
        return None

    def run_plugin(
        self, plugin_id: str, recording_id: str, parameters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Trigger an execution of a registered plugin."""
        payload = {"recording_id": recording_id, "parameters": parameters or {}}
        resp = requests.post(f"{self.base_url}/api/v1/plugins/{plugin_id}/runs", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_plugin_result(self, plugin_id: str, recording_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the latest computed result for a plugin and recording."""
        resp = requests.get(
            f"{self.base_url}/api/v1/plugins/{plugin_id}/results/{recording_id}", timeout=5
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        resp.raise_for_status()
        return None

    def get_job_by_id(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve job status explicitly by job_id."""
        resp = requests.get(f"{self.base_url}/api/v1/jobs/{job_id}", timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    # ConvoFuse Unified Framework Methods (Part 6.2 & 6.3)

    def list_convofuse_benchmarks(self) -> list[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/benchmarks", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("benchmarks", [])
        return []

    def trigger_convofuse_job(
        self,
        recording_id: str,
        execution_mode: str = "FROZEN_INPUT_REPLAY",
        condition_preference: str = "auto",
        enable_refinement: bool = True,
        use_locality_gating: bool = True,
        sync: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "execution_mode": execution_mode,
            "condition_preference": condition_preference,
            "enable_refinement": enable_refinement,
            "use_locality_gating": use_locality_gating,
            "sync": sync,
        }
        resp = requests.post(f"{self.base_url}/api/v1/convofuse/jobs", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_job_status(self, job_or_rec_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/jobs/{job_or_rec_id}/status", timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_metadata(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/{recording_id}/metadata", timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_transcript(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/{recording_id}/transcript", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_words(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/{recording_id}/words", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_turns(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/{recording_id}/turns", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_speakers(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/{recording_id}/speakers", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_evaluation(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/convofuse/{recording_id}/evaluation", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_convofuse_export_url(self, recording_id: str, format_str: str) -> str:
        return f"{self.base_url}/api/v1/convofuse/{recording_id}/export?format={format_str}"
