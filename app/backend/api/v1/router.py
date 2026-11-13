"""Consolidated API v1 Router."""
from __future__ import annotations

from fastapi import APIRouter

from .endpoints import analytics, convofuse, developer, feedback, health, media, transcriptions

api_v1_router = APIRouter()

api_v1_router.include_router(health.router)
api_v1_router.include_router(media.router)
api_v1_router.include_router(transcriptions.router)
api_v1_router.include_router(analytics.router)
api_v1_router.include_router(feedback.router)
api_v1_router.include_router(developer.router)
api_v1_router.include_router(convofuse.router, prefix="/convofuse", tags=["convofuse"])
