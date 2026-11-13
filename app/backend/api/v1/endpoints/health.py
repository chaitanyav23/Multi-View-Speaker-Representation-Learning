"""Health Check and Engine Status Endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from backend.config import settings

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check() -> dict[str, str | bool]:
    """Return application server status and engine configuration."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "mock_inference": settings.mock_inference,
        "device": settings.device,
    }
