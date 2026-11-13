"""Application Configuration and Environment Settings."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parents[1]

# Known non-secret placeholder values commonly found in template files
_PLACEHOLDER_TOKENS = {
    "your_huggingface_token_here",
    "your_hf_token_here",
    "replace_with_your_token",
    "placeholder",
    "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "none",
    "null",
}


def load_env_file(env_path: Optional[Path] = None) -> bool:
    """Attempt to load a .env file if present and python-dotenv is available.

    Parameters
    ----------
    env_path : Optional[Path]
        Explicit path to .env file. If None, checks app/.env then root .env.

    Returns
    -------
    bool
        True if an existing .env file was successfully loaded; False otherwise.

    Preserves existing environment variables (override=False).
    Emits an actionable diagnostic if .env exists but python-dotenv is not installed.
    Never exposes secrets or file contents in logs.
    """
    if env_path is None:
        _app_env = APP_DIR / ".env"
        _root_env = APP_DIR.parent / ".env"
        target_path = _app_env if _app_env.is_file() else (_root_env if _root_env.is_file() else None)
    else:
        target_path = env_path if env_path.is_file() else None

    if target_path is None:
        return False

    try:
        import dotenv
        # override=False preserves existing environment variable precedence
        return bool(dotenv.load_dotenv(target_path, override=False))
    except ImportError:
        logger.warning(
            "A .env configuration file was detected at %s, but 'python-dotenv' is not installed. "
            "Variables from this file will not be loaded into the environment.",
            target_path,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Failed to load .env configuration file from %s (%s). File content is omitted to protect potential secrets.",
            target_path,
            type(exc).__name__,
        )
        return False


# Execute optional .env loading on module initialization
load_env_file()


def _is_valid_token(value: Optional[str]) -> bool:
    """Return True if value is a non-empty, non-placeholder token string."""
    if not value or not isinstance(value, str):
        return False
    val = value.strip()
    if not val or val.lower() in _PLACEHOLDER_TOKENS:
        return False
    return True


def resolve_hf_token() -> Optional[str]:
    """
    Resolve Hugging Face authentication token following strict priority:
    1. HF_TOKEN environment variable (ignoring known placeholder templates)
    2. HUGGINGFACE_HUB_TOKEN environment variable
    3. huggingface_hub.get_token() local cache (~/.cache/huggingface/token)
    4. None if no token is found

    Security & Privacy Rules:
    - Never prints or logs token contents.
    - Never persists tokens to database or disk.
    - Strips whitespace safely; ignores empty strings and placeholder values.
    """
    # 1. Primary explicit environment variable
    token = os.getenv("HF_TOKEN")
    if _is_valid_token(token):
        return token.strip()

    # 2. Official Hugging Face Hub secondary environment variable
    hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if _is_valid_token(hub_token):
        return hub_token.strip()

    # 3. Local Hugging Face CLI / SDK cached token
    try:
        import huggingface_hub
        cached = huggingface_hub.get_token()
        if _is_valid_token(cached):
            return str(cached).strip()
    except Exception:
        pass

    # 4. Fallback: No authentication token available
    return None


class Settings(BaseModel):
    """Central configuration parameters for the application."""
    app_name: str = "CS698B Video Transcription with Confidence Analytics"
    app_version: str = "1.0.0"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))

    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_url: str = Field(default_factory=lambda: os.getenv("API_URL", "http://127.0.0.1:8000"))

    storage_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("STORAGE_DIR")) if os.getenv("STORAGE_DIR") else APP_DIR / "storage"
    )
    max_upload_size_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "250")))
    max_duration_seconds: int = Field(default_factory=lambda: int(os.getenv("MAX_DURATION_SECONDS", "1800")))

    # ML Inference Settings (Phase 20 Real Pretrained Models)
    whisper_model: str = Field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")))
    calibration_method: str = Field(default_factory=lambda: os.getenv("CALIBRATION_METHOD", "isotonic_regression"))
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    compute_type: str = Field(default_factory=lambda: os.getenv("COMPUTE_TYPE", "int8"))
    hf_token: Optional[str] = Field(default_factory=resolve_hf_token)
    mock_inference: bool = Field(default=False)

    @property
    def uploads_dir(self) -> Path:
        p = self.storage_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_dir(self) -> Path:
        p = self.storage_dir / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = self.storage_dir / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
