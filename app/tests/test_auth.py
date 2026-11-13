"""Unit tests for Hugging Face authentication resolution and error handling.

Tests:
1. Canonical 4-tier resolution priority:
   - HF_TOKEN environment variable takes priority
   - HUGGINGFACE_HUB_TOKEN is used when HF_TOKEN is absent
   - Local cached token (~/.cache/huggingface/token) is used when both env vars are absent
   - None returned when no token is available
2. Credential privacy:
   - Token values never appear in get_model_info(), logs, or exception messages
3. Inference requirements:
   - NeMo local-checkpoint diarization operates without requiring HF token
4. Actionable error handling for gated Pyannote models:
   - Missing token yields clear authentication blocker with setup instructions
   - Invalid token (HTTP 401) is clearly distinguished
   - Gated model permission denial (HTTP 403) provides exact acceptance URL
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.config import APP_DIR, load_env_file, resolve_hf_token
from engine.phase20_engine import (
    Phase20SpeechEngine,
    ensure_hf_hub_download_compat,
    sanitize_credential_text,
)


@pytest.fixture
def dummy_audio_16k(tmp_path: Path) -> Path:
    """Generate a 2.0-second 16 kHz mono synthetic sine tone for pipeline tests."""
    import numpy as np
    import soundfile as sf

    audio_file = tmp_path / "test_tone_16k.wav"
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    waveform = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(audio_file), waveform, sr)
    return audio_file


def test_resolve_hf_token_explicit_hf_token_priority():
    """Verify that HF_TOKEN takes highest precedence over other sources."""
    with patch.dict(
        os.environ,
        {
            "HF_TOKEN": "mock_token_primary",
            "HUGGINGFACE_HUB_TOKEN": "mock_token_secondary",
        },
        clear=True,
    ), patch("huggingface_hub.get_token", return_value="mock_token_cached"):
        token = resolve_hf_token()
        assert token == "mock_token_primary"


def test_resolve_hf_token_hub_token_fallback():
    """Verify fallback to HUGGINGFACE_HUB_TOKEN when HF_TOKEN is absent or empty."""
    with patch.dict(
        os.environ,
        {
            "HF_TOKEN": "",
            "HUGGINGFACE_HUB_TOKEN": "mock_token_secondary",
        },
        clear=True,
    ), patch("huggingface_hub.get_token", return_value="mock_token_cached"):
        token = resolve_hf_token()
        assert token == "mock_token_secondary"


def test_resolve_hf_token_cache_fallback():
    """Verify fallback to local Hugging Face CLI cache when environment variables are absent."""
    with patch.dict(os.environ, {}, clear=True), patch(
        "huggingface_hub.get_token", return_value="mock_token_cached"
    ):
        token = resolve_hf_token()
        assert token == "mock_token_cached"


def test_resolve_hf_token_none_when_empty():
    """Verify that resolve_hf_token() returns None when no token exists anywhere."""
    with patch.dict(os.environ, {}, clear=True), patch(
        "huggingface_hub.get_token", return_value=None
    ):
        token = resolve_hf_token()
        assert token is None


def test_resolve_hf_token_placeholder_ignored():
    """Verify that placeholder strings from templates are ignored in favor of cached token."""
    with patch.dict(
        os.environ,
        {"HF_TOKEN": "your_huggingface_token_here"},
        clear=True,
    ), patch("huggingface_hub.get_token", return_value="mock_token_cached"):
        token = resolve_hf_token()
        # The placeholder should be safely discarded, falling back to cached token
        assert token == "mock_token_cached"

    # Also verify case-insensitivity and other common dummy placeholders
    with patch.dict(
        os.environ,
        {
            "HF_TOKEN": "YOUR_HUGGINGFACE_TOKEN_HERE",
            "HUGGINGFACE_HUB_TOKEN": "placeholder",
        },
        clear=True,
    ), patch("huggingface_hub.get_token", return_value="mock_token_cached"):
        token = resolve_hf_token()
        assert token == "mock_token_cached"


def test_env_example_template_does_not_override_cached_token(tmp_path: Path):
    """Regression test: copying .env.example without configuring HF_TOKEN does not override cached token."""
    env_example_path = APP_DIR / ".env.example"
    assert env_example_path.is_file(), ".env.example must exist"

    content = env_example_path.read_text(encoding="utf-8")
    # Verify HF_TOKEN is commented out in template
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    active_hf_token_lines = [l for l in lines if l.startswith("HF_TOKEN=")]
    assert len(active_hf_token_lines) == 0, "HF_TOKEN must be commented out in .env.example"

    # Simulate copying .env.example to .env
    test_env = tmp_path / ".env"
    test_env.write_text(content, encoding="utf-8")

    with patch.dict(os.environ, {}, clear=True), patch(
        "huggingface_hub.get_token", return_value="mock_token_cached"
    ):
        loaded = load_env_file(test_env)
        # HF_TOKEN should not have been set into environment
        assert os.getenv("HF_TOKEN") is None
        # Resolution falls through cleanly to cached token
        assert resolve_hf_token() == "mock_token_cached"


def test_dotenv_optional_loading_success(tmp_path: Path):
    """Verify load_env_file successfully loads variables from a .env file when python-dotenv is available."""
    test_env = tmp_path / ".env"
    test_env.write_text("TEST_AUTH_CUSTOM_KEY=auth_custom_value_xyz\n", encoding="utf-8")

    with patch.dict(os.environ, {}, clear=True):
        loaded = load_env_file(test_env)
        assert loaded is True
        assert os.getenv("TEST_AUTH_CUSTOM_KEY") == "auth_custom_value_xyz"


def test_dotenv_preserves_existing_environment_precedence(tmp_path: Path):
    """Verify existing environment variables take precedence over .env file entries."""
    test_env = tmp_path / ".env"
    test_env.write_text("PRE_EXISTING_AUTH_KEY=overridden_from_dotenv\n", encoding="utf-8")

    with patch.dict(os.environ, {"PRE_EXISTING_AUTH_KEY": "original_system_value"}, clear=True):
        loaded = load_env_file(test_env)
        assert loaded is True
        # Original value in os.environ must NOT be replaced
        assert os.getenv("PRE_EXISTING_AUTH_KEY") == "original_system_value"


def test_dotenv_missing_package_warning_diagnostic(tmp_path: Path, caplog):
    """Verify that if .env exists but python-dotenv is missing, a clear warning is logged without raising."""
    import sys
    test_env = tmp_path / ".env"
    secret_value = "super_secret_dont_log_me"
    test_env.write_text(f"SECRET_KEY={secret_value}\n", encoding="utf-8")

    # Simulate missing python-dotenv by patching sys.modules or __import__
    import builtins
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("No module named 'dotenv'")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        loaded = load_env_file(test_env)
        assert loaded is False

    # Verify diagnostic message was emitted
    assert any("python-dotenv' is not installed" in r.message for r in caplog.records)
    # Verify no secret from the file was leaked into the log
    for r in caplog.records:
        assert secret_value not in r.message


def test_dotenv_parsing_exception_sanitizes_secrets(tmp_path: Path, caplog):
    """Verify that exceptions raised by python-dotenv do not leak secret token text into logs."""
    import logging
    test_env = tmp_path / ".env"
    synthetic_token = "hf_synthetic_env_parse_error_token_98765"
    test_env.write_text(f"HF_TOKEN={synthetic_token}\n", encoding="utf-8")

    # Simulate a parser exception containing the synthetic token line
    with caplog.at_level(logging.WARNING), patch(
        "dotenv.load_dotenv",
        side_effect=ValueError(f"Malformed line in .env: HF_TOKEN={synthetic_token}"),
    ):
        loaded = load_env_file(test_env)
        assert loaded is False

    # 1. Verify warning is logged with exception class name
    assert any("Failed to load .env configuration file" in r.message for r in caplog.records)
    assert any("(ValueError)" in r.message for r in caplog.records)
    assert any("File content is omitted to protect potential secrets" in r.message for r in caplog.records)

    # 2. Verify synthetic token is strictly NOT exposed in logs
    log_text = caplog.text
    assert synthetic_token not in log_text, "Synthetic token leaked into dotenv loading logger!"


def test_actual_local_cached_token_lookup_inspection():
    """Inspect actual local cached token on host machine without printing or logging secret content.

    Security protection:
    - Uses hmac.compare_digest() for all comparisons.
    - Zero plain equality assertions on actual credentials.
    - Fixed generic failure messages prevent pytest assertion introspection from leaking tokens.
    """
    import huggingface_hub
    with patch.dict(os.environ, {}, clear=True):
        cached = huggingface_hub.get_token()
        if cached is not None:
            token_str = str(cached).strip()
            # Verify prefix and length safely without exposing secret in assertion introspection
            valid_format = token_str.startswith("hf_") and len(token_str) > 10
            if not valid_format:
                pytest.fail("Cached token does not exhibit standard Hugging Face token formatting.")

            resolved = resolve_hf_token()
            if resolved is None:
                pytest.fail("Cached token resolution returned None when local cache was populated.")

            is_match = hmac.compare_digest(resolved, token_str)
            if not is_match:
                pytest.fail("Cached token resolution did not return the configured token.")
        else:
            # If running in environment without cached token, ensure None is returned
            if resolve_hf_token() is not None:
                pytest.fail("resolve_hf_token() did not return None when cache was empty.")


def test_private_app_env_token_precedence_over_cache():
    """Verify that the token supplied through app/.env takes precedence over secondary env vars and local cache.

    Demonstrates:
    - Precedence over HUGGINGFACE_HUB_TOKEN
    - Precedence over ~/.cache/huggingface/token
    - huggingface_hub.get_token() is NOT consulted when HF_TOKEN is already present
    - Safe hmac.compare_digest comparisons with zero credential exposure in failure messages or introspection.
    """
    app_env_path = APP_DIR / ".env"
    if not app_env_path.is_file():
        pytest.skip("Private app/.env not present")

    with patch.dict(os.environ, {}, clear=True), patch("huggingface_hub.get_token") as mock_cache_get_token:
        # Configure mock_cache_get_token so that if called, it returns a distinct dummy token
        mock_cache_get_token.return_value = "hf_dummy_cache_token_lower_precedence_999"

        loaded = load_env_file(app_env_path)
        if not loaded:
            pytest.fail("Failed to load private app/.env file.")

        env_token = os.environ.get("HF_TOKEN")
        if not env_token:
            pytest.fail("HF_TOKEN was not defined in app/.env.")

        valid_format = env_token.startswith("hf_") and len(env_token) > 10
        if not valid_format:
            pytest.fail("app/.env HF_TOKEN does not exhibit standard Hugging Face token formatting.")

        # Set conflicting secondary environment variable
        secondary_hub_token = "hf_secondary_hub_token_xyz"
        os.environ["HUGGINGFACE_HUB_TOKEN"] = secondary_hub_token

        # Resolve token
        resolved = resolve_hf_token()
        if resolved is None:
            pytest.fail("resolve_hf_token() returned None when app/.env HF_TOKEN was set.")

        # Verify that huggingface_hub.get_token() was NOT consulted because HF_TOKEN was resolved at Tier 1
        if mock_cache_get_token.called:
            pytest.fail("huggingface_hub.get_token() was unexpectedly consulted when HF_TOKEN was already available.")

        # Safe comparison: must match app/.env token
        is_env_match = hmac.compare_digest(resolved, env_token)
        if not is_env_match:
            pytest.fail("Resolved token did not match expected app/.env token.")

        # Safe comparison: must NOT match secondary token
        is_secondary_match = hmac.compare_digest(resolved, secondary_hub_token)
        if is_secondary_match:
            pytest.fail("Resolved token unexpectedly matched secondary hub token instead of app/.env token.")


def test_token_never_logged_or_exposed(dummy_audio_16k: Path):
    """Verify that sensitive token strings are never exposed in engine info or logs."""
    secret_token = "hf_secret_value_do_not_leak_12345"
    engine = Phase20SpeechEngine(
        model_size="tiny",
        hf_token=secret_token,
    )
    info = engine.get_model_info()

    # Model info should only report boolean presence
    assert "has_hf_token" in info
    assert info["has_hf_token"] is True
    assert "hf_token" not in info
    assert secret_token not in str(info)


def test_nemo_does_not_require_hf_token(dummy_audio_16k: Path):
    """Verify that local NeMo MSDD inference proceeds successfully without an HF token."""
    engine = Phase20SpeechEngine(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        hf_token=None,
        enable_diarization=True,
        diarization_backend="nemo",
    )

    mock_whisper = MagicMock()
    mock_seg = MagicMock(start=0.0, end=1.0, text="TEST", avg_logprob=-0.1)
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [{"word": "TEST", "start": 0.1, "end": 0.9, "confidence": 0.9}]
    engine._aligner = mock_aligner

    mock_nemo = MagicMock()
    mock_nemo.is_available.return_value = (True, None)
    mock_nemo.diarize.return_value = (
        [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}],
        "DIARIZATION_AVAILABLE",
        None,
        {"speaker_count": 1, "speakers": ["speaker_0"]},
    )
    engine._nemo_diarizer = mock_nemo

    with patch("faster_whisper.vad.get_speech_timestamps", return_value=[{"start": 0, "end": 16000}]):
        result = engine.transcribe(dummy_audio_16k, recording_id="nemo_no_token_test")

    assert result.summary["diarization_status"] == "DIARIZATION_AVAILABLE"
    assert result.summary["speakers_detected"] == 1
    assert result.words[0].speaker == "speaker_0"


def test_pyannote_missing_token_actionable_error(dummy_audio_16k: Path):
    """Verify actionable authentication error when gated Pyannote is requested without a token."""
    with patch("backend.config.resolve_hf_token", return_value=None):
        engine = Phase20SpeechEngine(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            hf_token=None,
            enable_diarization=True,
            diarization_backend="pyannote",
        )

        mock_whisper = MagicMock()
        mock_seg = MagicMock(start=0.0, end=1.0, text="TEST", avg_logprob=-0.1)
        mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
        engine._whisper_model = mock_whisper

        mock_aligner = MagicMock()
        mock_aligner.align_segment.return_value = [{"word": "TEST", "start": 0.1, "end": 0.9, "confidence": 0.9}]
        engine._aligner = mock_aligner

        result = engine.transcribe(dummy_audio_16k, recording_id="pyannote_no_token")

    assert result.summary["diarization_status"] == "DIARIZATION_FAILED"
    blocker = result.summary["diarization_blocker"]
    assert "Hugging Face authentication required" in blocker
    assert "huggingface-cli login" in blocker
    assert result.words[0].speaker == "SPEAKER_UNASSIGNED"


def test_pyannote_returns_none_repo_not_found_handling(dummy_audio_16k: Path):
    """Verify error handling when Pipeline.from_pretrained returns None (e.g. unaccepted gated terms)."""
    engine = Phase20SpeechEngine(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        hf_token="mock_valid_token_123",
        enable_diarization=True,
        diarization_backend="pyannote",
    )

    mock_whisper = MagicMock()
    mock_seg = MagicMock(start=0.0, end=1.0, text="TEST", avg_logprob=-0.1)
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [{"word": "TEST", "start": 0.1, "end": 0.9, "confidence": 0.9}]
    engine._aligner = mock_aligner

    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=None):
        result = engine.transcribe(dummy_audio_16k, recording_id="pyannote_none")

    assert result.summary["diarization_status"] == "DIARIZATION_FAILED"
    blocker = result.summary["diarization_blocker"]
    assert "Could not load 'pyannote/speaker-diarization-3.1'" in blocker
    assert "https://hf.co/pyannote/speaker-diarization-3.1" in blocker


def test_pyannote_invalid_token_error_distinction(dummy_audio_16k: Path):
    """Verify distinct error messaging for invalid or unauthorized Hugging Face token (HTTP 401)."""
    fake_token = "hf_invalid_token_example_123"
    engine = Phase20SpeechEngine(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        hf_token=fake_token,
        enable_diarization=True,
        diarization_backend="pyannote",
    )

    mock_whisper = MagicMock()
    mock_seg = MagicMock(start=0.0, end=1.0, text="TEST", avg_logprob=-0.1)
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [{"word": "TEST", "start": 0.1, "end": 0.9, "confidence": 0.9}]
    engine._aligner = mock_aligner

    with patch("pyannote.audio.Pipeline.from_pretrained", side_effect=RuntimeError("HTTP 401 Client Error: Unauthorized for url")):
        result = engine.transcribe(dummy_audio_16k, recording_id="pyannote_401")

    assert result.summary["diarization_status"] == "DIARIZATION_FAILED"
    blocker = result.summary["diarization_blocker"]
    assert "HTTP 401 Unauthorized" in blocker
    assert "invalid, expired, or revoked" in blocker
    # Ensure token secret was not reflected into blocker
    assert fake_token not in blocker


def test_pyannote_gated_model_permission_error_distinction(dummy_audio_16k: Path):
    """Verify distinct error messaging for missing gated repo acceptance terms (HTTP 403)."""
    fake_token = "hf_valid_but_unaccepted_terms_123"
    engine = Phase20SpeechEngine(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        hf_token=fake_token,
        enable_diarization=True,
        diarization_backend="pyannote",
    )

    mock_whisper = MagicMock()
    mock_seg = MagicMock(start=0.0, end=1.0, text="TEST", avg_logprob=-0.1)
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [{"word": "TEST", "start": 0.1, "end": 0.9, "confidence": 0.9}]
    engine._aligner = mock_aligner

    with patch("pyannote.audio.Pipeline.from_pretrained", side_effect=RuntimeError("HTTP 403 Client Error: Forbidden. GatedRepoError")):
        result = engine.transcribe(dummy_audio_16k, recording_id="pyannote_403")

    assert result.summary["diarization_status"] == "DIARIZATION_FAILED"
    blocker = result.summary["diarization_blocker"]
    assert "HTTP 403 Forbidden" in blocker
    assert "https://hf.co/pyannote/speaker-diarization-3.1" in blocker
    assert fake_token not in blocker


def test_hf_hub_download_compat_idempotent():
    """Verify that ensure_hf_hub_download_compat installs once and prevents nested wrapper accumulation."""
    import huggingface_hub

    # Create a mock hf_hub_download that only accepts `token` (simulating modern huggingface_hub >= 0.24.0)
    mock_calls = []

    def mock_download(repo_id, filename, *, token=None, **kwargs):
        mock_calls.append({"repo_id": repo_id, "token": token, "kwargs": kwargs})
        return f"/mock/path/{filename}"

    with patch.object(huggingface_hub, "hf_hub_download", mock_download):
        # 1. First installation
        res1 = ensure_hf_hub_download_compat()
        assert res1 is True
        installed_func = huggingface_hub.hf_hub_download
        assert getattr(installed_func, "_is_pyannote_compat_wrapper", False) is True
        assert getattr(installed_func, "_original_func", None) is mock_download

        # 2. Repeated installation calls must be idempotent
        res2 = ensure_hf_hub_download_compat()
        assert res2 is True
        res3 = ensure_hf_hub_download_compat()
        assert res3 is True

        # Verify no nested accumulation of wrappers
        assert huggingface_hub.hf_hub_download is installed_func
        assert huggingface_hub.hf_hub_download._original_func is mock_download


def test_hf_hub_download_compat_translation_and_preservation():
    """Verify use_auth_token is translated to token, and explicit token arguments are preserved."""
    import huggingface_hub

    captured_kwargs = {}

    def mock_download(repo_id, filename, *, token=None, **kwargs):
        captured_kwargs["token"] = token
        captured_kwargs.update(kwargs)
        return f"/path/{filename}"

    with patch.object(huggingface_hub, "hf_hub_download", mock_download):
        ensure_hf_hub_download_compat()

        # 3. Translation of use_auth_token to token
        huggingface_hub.hf_hub_download("test/repo", "config.yaml", use_auth_token="token_from_auth")
        assert captured_kwargs.get("token") == "token_from_auth"
        assert "use_auth_token" not in captured_kwargs

        # 4. Explicit token preservation (token not overwritten by use_auth_token)
        captured_kwargs.clear()
        huggingface_hub.hf_hub_download(
            "test/repo",
            "config.yaml",
            token="explicit_token_val",
            use_auth_token="ignored_fallback_token",
        )
        assert captured_kwargs.get("token") == "explicit_token_val"


def test_hf_hub_download_compat_unimported_module_graceful():
    """Verify that if pyannote modules are not yet imported, ensure_hf_hub_download_compat succeeds gracefully."""
    import sys
    import huggingface_hub

    def mock_download(repo_id, filename, *, token=None, **kwargs):
        return f"/mock/{filename}"

    with patch.object(huggingface_hub, "hf_hub_download", mock_download):
        # Temporarily ensure pyannote targets are not in sys.modules
        saved_modules = {}
        for target in ["pyannote.audio.core.pipeline", "pyannote.audio.pipelines.speaker_diarization"]:
            if target in sys.modules:
                saved_modules[target] = sys.modules.pop(target)

        try:
            res = ensure_hf_hub_download_compat()
            assert res is True
        finally:
            sys.modules.update(saved_modules)


def test_hf_hub_download_compat_unexpected_failure_raises():
    """Verify that unexpected exceptions during module patching are raised rather than silently suppressed."""
    import sys
    import huggingface_hub

    def mock_download(repo_id, filename, *, token=None, **kwargs):
        return f"/mock/{filename}"

    # Create a mock module that raises an exception when hf_hub_download is set
    class FaultyModule:
        @property
        def hf_hub_download(self):
            return None

        @hf_hub_download.setter
        def hf_hub_download(self, value):
            raise TypeError("Read-only or corrupted module attribute")

    with patch.object(huggingface_hub, "hf_hub_download", mock_download):
        with patch.dict(sys.modules, {"pyannote.audio.core.pipeline": FaultyModule()}):
            with pytest.raises(RuntimeError) as exc_info:
                ensure_hf_hub_download_compat()
            assert "Failed to install hf_hub_download compatibility patch" in str(exc_info.value)
            assert "pyannote.audio.core.pipeline" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, TypeError)


def test_hf_hub_download_compat_synthetic_token_in_pyannote_patch_exception_sanitized(caplog):
    """Verify that token strings in pyannote module patch exceptions are sanitized in logs and RuntimeError."""
    import logging
    import sys
    import huggingface_hub

    synthetic_token = "hf_synthetic_test_token_never_expose_123456789"

    def mock_download(repo_id, filename, *, token=None, **kwargs):
        return f"/mock/{filename}"

    class FaultyModuleWithSecret:
        @property
        def hf_hub_download(self):
            return None

        @hf_hub_download.setter
        def hf_hub_download(self, value):
            raise TypeError(f"Module assignment failed for credential {synthetic_token}")

    with caplog.at_level(logging.ERROR):
        with patch.object(huggingface_hub, "hf_hub_download", mock_download):
            with patch.dict(sys.modules, {"pyannote.audio.core.pipeline": FaultyModuleWithSecret()}):
                with pytest.raises(RuntimeError) as exc_info:
                    ensure_hf_hub_download_compat(token=synthetic_token)

                # 1. Propagated RuntimeError text must be sanitized
                err_text = str(exc_info.value)
                assert synthetic_token not in err_text, "Synthetic token leaked into propagated RuntimeError!"
                assert "[REDACTED_HF_TOKEN]" in err_text

                # 2. Exception chaining must be preserved
                assert exc_info.value.__cause__ is not None
                assert isinstance(exc_info.value.__cause__, TypeError)

                # 3. Logger output must be sanitized
                log_text = caplog.text
                assert synthetic_token not in log_text, "Synthetic token leaked into logger output!"
                assert "[REDACTED_HF_TOKEN]" in log_text


def test_hf_hub_download_compat_synthetic_token_in_hub_patch_exception_sanitized(caplog):
    """Verify that token strings in huggingface_hub patch exceptions are sanitized in logs and RuntimeError."""
    import logging
    import huggingface_hub

    synthetic_token = "hf_synthetic_hub_patch_secret_9876543210"

    class FaultyHub:
        @property
        def hf_hub_download(self):
            def mock_dl(repo_id, filename, *, token=None, **kwargs):
                return "/mock"
            return mock_dl

        @hf_hub_download.setter
        def hf_hub_download(self, value):
            raise AttributeError(f"Root hub patch rejected with secret {synthetic_token}")

    faulty_hub = FaultyHub()

    with caplog.at_level(logging.ERROR):
        import sys
        with patch.dict(sys.modules, {"huggingface_hub": faulty_hub}):
            with pytest.raises(RuntimeError) as exc_info:
                ensure_hf_hub_download_compat(token=synthetic_token)

            # 1. Propagated RuntimeError text must be sanitized
            err_text = str(exc_info.value)
            assert synthetic_token not in err_text, "Synthetic token leaked into propagated RuntimeError!"
            assert "[REDACTED_HF_TOKEN]" in err_text

            # 2. Exception chaining must be preserved
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, AttributeError)

            # 3. Logger output must be sanitized
            log_text = caplog.text
            assert synthetic_token not in log_text, "Synthetic token leaked into logger output!"
            assert "[REDACTED_HF_TOKEN]" in log_text


def test_hf_hub_download_compat_native_support_returns_true():
    """Verify that when hf_hub_download already accepts use_auth_token natively, no wrapper is installed."""
    import huggingface_hub

    def mock_native_download(repo_id, filename, *, use_auth_token=None, token=None, **kwargs):
        return f"/mock/{filename}"

    with patch.object(huggingface_hub, "hf_hub_download", mock_native_download):
        res = ensure_hf_hub_download_compat()
        assert res is True
        # Original function should remain un-wrapped
        assert huggingface_hub.hf_hub_download is mock_native_download
        assert getattr(huggingface_hub.hf_hub_download, "_is_pyannote_compat_wrapper", False) is False


def test_pyannote_generic_exception_redacts_token(dummy_audio_16k: Path, caplog):
    """Verify that deliberately token-containing generic exceptions are redacted from logs and blockers."""
    import logging
    leak_token = "hf_sensitive_secret_leak_candidate_1234567890"

    engine = Phase20SpeechEngine(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        hf_token=leak_token,
        enable_diarization=True,
        diarization_backend="pyannote",
    )

    mock_whisper = MagicMock()
    mock_seg = MagicMock(start=0.0, end=1.0, text="TEST", avg_logprob=-0.1)
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [{"word": "TEST", "start": 0.1, "end": 0.9, "confidence": 0.9}]
    engine._aligner = mock_aligner

    # Deliberately include the secret token in a generic error message
    simulated_error_msg = f"HTTP connection failed for url https://huggingface.co/api?token={leak_token} (Connection reset)"

    with caplog.at_level(logging.ERROR), patch(
        "pyannote.audio.Pipeline.from_pretrained",
        side_effect=ConnectionError(simulated_error_msg),
    ):
        result = engine.transcribe(dummy_audio_16k, recording_id="pyannote_leak_test")

    # 1. Blocker check: secret token must NOT appear in diarization blocker
    blocker = result.summary.get("diarization_blocker", "")
    assert leak_token not in blocker, "Secret token leaked into diarization blocker!"
    assert "[REDACTED_HF_TOKEN]" in blocker
    assert "ConnectionError" in blocker
    assert "Connection reset" in blocker

    # 2. Log check: secret token must NOT appear in captured log records
    for record in caplog.records:
        assert leak_token not in record.message, "Secret token leaked into logger message!"

    # 3. Model metadata check: get_model_info() must not reveal the token
    info = engine.get_model_info()
    assert info.get("has_hf_token") is True
    assert leak_token not in str(info)


def test_env_hf_token_source_precedence_over_hub_token_and_cache(tmp_path: Path):
    """Verify that HF_TOKEN loaded from a .env file strictly precedes secondary env vars and local cache."""
    # Write a test .env file with a test token
    test_env = tmp_path / ".env"
    mock_env_token = "hf_env_token_from_explicit_file_abc123"
    test_env.write_text(f"HF_TOKEN={mock_env_token}\n", encoding="utf-8")

    with patch.dict(os.environ, {}, clear=True), patch(
        "huggingface_hub.get_token", return_value="hf_cached_token_lower_precedence_xyz"
    ):
        # Load the test .env file
        load_success = load_env_file(test_env)
        assert load_success is True

        # Set secondary token in environment to verify Tier 1 precedes Tier 2
        os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_hub_token_tier2_precedence"

        # Resolve token
        resolved = resolve_hf_token()

        # Must resolve to Tier 1 (.env HF_TOKEN)
        assert resolved == mock_env_token
        assert resolved != "hf_hub_token_tier2_precedence"
        assert resolved != "hf_cached_token_lower_precedence_xyz"
