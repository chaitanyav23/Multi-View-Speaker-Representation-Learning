"""Unit tests for the NeMo MSDD Diarization Adapter and Worker Protocol.

Deterministic unit verification without requiring GPU, model loading, or speech inference.
Tests:
- Environment and checkpoint detection
- Oracle VAD RTTM formatting (speech-only dummy 'speaker' label)
- NIST RTTM parsing
- Empty VAD short-circuiting (pre-worker and in-worker NO_SPEECH protocol)
- Workspace isolation and unique child directory creation
- Deterministic RTTM output location and workspace provenance checks
- Strict JSON protocol validation (missing, malformed, unknown status, failed status)
- Path traversal and security boundary rejection
- Subprocess error capture and timeout enforcement
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.nemo_diarizer import NemoMsddDiarizer


def test_nemo_diarizer_availability_checks(tmp_path: Path):
    """Test environment and checkpoint detection."""
    real_diarizer = NemoMsddDiarizer()
    avail, reason = real_diarizer.is_available()
    assert avail is True, f"Expected NeMo diarizer to be available, got reason: {reason}"
    assert reason is None

    # Test missing python
    missing_py = NemoMsddDiarizer(nemo_python=tmp_path / "nonexistent_py")
    avail, reason = missing_py.is_available()
    assert avail is False
    assert "NeMo Python executable not found" in str(reason)

    # Test missing checkpoint
    fake_py = tmp_path / "fake_py"
    fake_py.touch(mode=0o755)
    missing_ckpt = NemoMsddDiarizer(
        nemo_python=fake_py,
        checkpoint_path=tmp_path / "nonexistent.nemo",
    )
    avail, reason = missing_ckpt.is_available()
    assert avail is False
    assert "NeMo MSDD checkpoint not found" in str(reason)


def test_generate_oracle_vad_rttm(tmp_path: Path):
    """Verify standard NIST RTTM generation for oracle VAD with dummy 'speaker' label."""
    intervals = [
        {"start": 0.500, "end": 2.750},
        {"start": 3.100, "end": 5.800},
        {"start": 6.000, "end": 6.000},  # Zero duration - should be excluded
        {"start": 7.000, "end": 6.500},  # Negative duration - should be excluded
    ]
    out_rttm = tmp_path / "subdir" / "test_vad.rttm"
    res = NemoMsddDiarizer.generate_oracle_vad_rttm(intervals, out_rttm, "test_audio")
    assert res == out_rttm
    assert out_rttm.exists()

    lines = [line.strip() for line in out_rttm.read_text().splitlines() if line.strip()]
    assert len(lines) == 2

    # Verify standard NIST RTTM structure:
    # SPEAKER <file_id> 1 <start> <duration> <NA> <NA> speaker <NA> <NA>
    line1_parts = lines[0].split()
    assert line1_parts[0] == "SPEAKER"
    assert line1_parts[1] == "test_audio"
    assert line1_parts[2] == "1"
    assert float(line1_parts[3]) == pytest.approx(0.500)
    assert float(line1_parts[4]) == pytest.approx(2.250)
    assert line1_parts[7] == "speaker"

    line2_parts = lines[1].split()
    assert float(line2_parts[3]) == pytest.approx(3.100)
    assert float(line2_parts[4]) == pytest.approx(2.700)
    assert line2_parts[7] == "speaker"


def test_parse_rttm(tmp_path: Path):
    """Verify parsing NIST RTTM files into segment dictionaries."""
    sample_rttm = tmp_path / "test.rttm"
    content = (
        "SPEAKER sample_audio 1 0.50 1.25 <NA> <NA> speaker_0 <NA> <NA>\n"
        "# Comment line should be ignored\n"
        "SPEAKER sample_audio 1 2.00 0.80 <NA> <NA> speaker_1 <NA> <NA>\n"
        "MALFORMED LINE\n"
        "SPEAKER sample_audio 1 3.50 0.00 <NA> <NA> speaker_0 <NA> <NA>\n"  # 0 duration skipped
        "SPEAKER sample_audio 1 4.10 1.50 <NA> <NA> speaker_0 <NA> <NA>\n"
    )
    sample_rttm.write_text(content, encoding="utf-8")

    segments = NemoMsddDiarizer.parse_rttm(sample_rttm)
    assert len(segments) == 3
    assert segments[0] == {"start": 0.5, "end": 1.75, "speaker": "speaker_0"}
    assert segments[1] == {"start": 2.0, "end": 2.8, "speaker": "speaker_1"}
    assert segments[2] == {"start": 4.1, "end": 5.6, "speaker": "speaker_0"}

    # Test nonexistent path
    assert NemoMsddDiarizer.parse_rttm(tmp_path / "missing.rttm") == []


def test_diarize_missing_audio_or_vad(tmp_path: Path):
    """Test error handling when input audio or VAD information is missing."""
    diarizer = NemoMsddDiarizer()

    # Missing audio file
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=tmp_path / "missing.wav",
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "Audio file not found" in str(blocker)
    assert segs == []

    # Missing VAD
    real_audio = tmp_path / "test.wav"
    real_audio.touch()
    segs, status, blocker, stats = diarizer.diarize(audio_path=real_audio)
    assert status == "DIARIZATION_FAILED"
    assert "Neither vad_intervals nor vad_rttm provided" in str(blocker)


@patch("subprocess.run")
def test_diarize_empty_vad_intervals_short_circuits(mock_run, tmp_path: Path):
    """Verify that an empty list of VAD intervals short-circuits without worker invocation."""
    fake_audio = tmp_path / "silence.wav"
    fake_audio.touch()

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[],
    )
    mock_run.assert_not_called()
    assert status == "NO_SPEECH"
    assert segs == []
    assert stats["speaker_count"] == 0
    assert "No speech intervals" in str(blocker)


@patch("subprocess.run")
def test_diarize_empty_vad_rttm_short_circuits(mock_run, tmp_path: Path):
    """Verify that a VAD RTTM with zero turns short-circuits without worker invocation."""
    fake_audio = tmp_path / "silence.wav"
    fake_audio.touch()

    empty_rttm = tmp_path / "empty_vad.rttm"
    empty_rttm.write_text("# Only comments\n\n", encoding="utf-8")

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_rttm=empty_rttm,
    )
    mock_run.assert_not_called()
    assert status == "NO_SPEECH"
    assert segs == []
    assert stats["speaker_count"] == 0
    assert "0 speech intervals" in str(blocker)


@patch("subprocess.run")
def test_diarize_subprocess_mock_success(mock_run, tmp_path: Path):
    """Test successful diarization orchestration with isolated workspace and valid protocol."""
    fake_audio = tmp_path / "meeting.wav"
    fake_audio.touch()

    parent_out_dir = tmp_path / "diar_parent"
    parent_out_dir.mkdir()

    def fake_subprocess_exec(cmd, **kwargs):
        # Determine unique child directory passed to worker
        out_dir_idx = cmd.index("--out-dir") + 1
        worker_work_dir = Path(cmd[out_dir_idx])
        assert worker_work_dir.is_dir()
        assert worker_work_dir.parent == parent_out_dir

        # Simulate worker creating predicted.rttm
        predicted_rttm = worker_work_dir / "predicted.rttm"
        predicted_rttm.write_text(
            "SPEAKER meeting 1 0.00 2.00 <NA> <NA> speaker_0 <NA> <NA>\n"
            "SPEAKER meeting 1 2.50 1.50 <NA> <NA> speaker_1 <NA> <NA>\n"
        )

        worker_payload = {
            "status": "SUCCESS",
            "speaker_count": 2,
            "speakers": ["speaker_0", "speaker_1"],
            "turn_count": 2,
            "rttm_path": str(predicted_rttm.resolve()),
            "inference_time_seconds": 1.45,
        }
        return MagicMock(
            returncode=0,
            stdout=f"INFO: Log before\n{json.dumps(worker_payload)}\nINFO: Log after",
            stderr="",
        )

    mock_run.side_effect = fake_subprocess_exec

    diarizer = NemoMsddDiarizer()
    intervals = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 4.0}]
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=intervals,
        out_dir=parent_out_dir,
    )

    assert status == "DIARIZATION_AVAILABLE"
    assert blocker is None
    assert len(segs) == 2
    assert segs[0] == {"start": 0.0, "end": 2.0, "speaker": "speaker_0"}
    assert segs[1] == {"start": 2.5, "end": 4.0, "speaker": "speaker_1"}
    assert stats["speaker_count"] == 2
    assert stats["inference_time_seconds"] == 1.45

    # Check subprocess command arguments
    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    assert "--audio" in called_cmd
    assert str(fake_audio.resolve()) in called_cmd
    assert "--vad-rttm" in called_cmd
    assert "--out-dir" in called_cmd


@patch("subprocess.run")
def test_diarize_missing_json_payload(mock_run, tmp_path: Path):
    """Test rejection when worker exits 0 but produces no structured JSON output."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="INFO: finished with no JSON string at all\nDone.",
        stderr="",
    )

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "emitted no structured JSON" in str(blocker)


@patch("subprocess.run")
def test_diarize_malformed_json_payload(mock_run, tmp_path: Path):
    """Test rejection when worker emits malformed JSON."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="{status: invalid_json_here}\n",
        stderr="",
    )

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "malformed JSON" in str(blocker)


@patch("subprocess.run")
def test_diarize_unrecognized_worker_status(mock_run, tmp_path: Path):
    """Test rejection of non-protocol status strings."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"status": "MAYBE_DONE", "turns": 2}),
        stderr="",
    )

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "Unrecognized worker status: 'MAYBE_DONE'" in str(blocker)


@patch("subprocess.run")
def test_diarize_worker_reported_failure_status(mock_run, tmp_path: Path):
    """Test handling of explicit worker FAILED status despite 0 returncode."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"status": "FAILED", "error": "Internal clustering convergence failure"}),
        stderr="",
    )

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "Internal clustering convergence failure" in str(blocker)


@patch("subprocess.run")
def test_diarize_worker_reported_no_speech_status(mock_run, tmp_path: Path):
    """Test handling of explicit worker NO_SPEECH status."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({
            "status": "NO_SPEECH",
            "reason": "VAD RTTM contains 0 speech intervals.",
            "speaker_count": 0,
            "speakers": [],
            "turn_count": 0,
            "total_speech_seconds": 0.0,
        }),
        stderr="",
    )

    diarizer = NemoMsddDiarizer()
    # Pass dummy vad_rttm with a turn so adapter invokes worker
    dummy_vad = tmp_path / "dummy_vad.rttm"
    dummy_vad.write_text("SPEAKER audio 1 0.0 1.0 <NA> <NA> speaker <NA> <NA>\n")

    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_rttm=dummy_vad,
    )
    assert status == "NO_SPEECH"
    assert segs == []
    assert stats["speaker_count"] == 0
    assert "0 speech intervals" in str(blocker)


@patch("subprocess.run")
def test_diarize_worker_provenance_violation(mock_run, tmp_path: Path):
    """Test security boundary: worker returning RTTM outside the unique workspace is rejected."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    external_rttm = tmp_path / "outside_workspace.rttm"
    external_rttm.write_text("SPEAKER audio 1 0.0 1.0 <NA> <NA> speaker_0 <NA> <NA>\n")

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"status": "SUCCESS", "rttm_path": str(external_rttm.resolve())}),
        stderr="",
    )

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "outside invocation workspace" in str(blocker)


@patch("subprocess.run")
def test_diarize_worker_missing_output_file(mock_run, tmp_path: Path):
    """Test rejection when worker reports SUCCESS but referenced RTTM does not exist on disk."""
    fake_audio = tmp_path / "audio.wav"
    fake_audio.touch()

    def fake_subprocess_exec(cmd, **kwargs):
        out_dir_idx = cmd.index("--out-dir") + 1
        worker_work_dir = Path(cmd[out_dir_idx])
        # Return path inside workspace, but do NOT create the file
        nonexistent = worker_work_dir / "predicted.rttm"
        return MagicMock(
            returncode=0,
            stdout=json.dumps({"status": "SUCCESS", "rttm_path": str(nonexistent.resolve())}),
            stderr="",
        )

    mock_run.side_effect = fake_subprocess_exec

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "does not exist as a regular file" in str(blocker)


@patch("subprocess.run")
def test_diarize_subprocess_timeout(mock_run, tmp_path: Path):
    """Test timeout handling when worker exceeds timeout threshold."""
    fake_audio = tmp_path / "meeting.wav"
    fake_audio.touch()

    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python"], timeout=10)

    diarizer = NemoMsddDiarizer(timeout_seconds=10)
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "timed out after 10s" in str(blocker)
    assert segs == []


@patch("subprocess.run")
def test_diarize_subprocess_worker_crash(mock_run, tmp_path: Path):
    """Test error handling when worker exits with non-zero status."""
    fake_audio = tmp_path / "meeting.wav"
    fake_audio.touch()

    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="Traceback (most recent call last):\nRuntimeError: CUDA out of memory",
    )

    diarizer = NemoMsddDiarizer()
    segs, status, blocker, stats = diarizer.diarize(
        audio_path=fake_audio,
        vad_intervals=[{"start": 0.0, "end": 1.0}],
    )
    assert status == "DIARIZATION_FAILED"
    assert "Worker exited with code 1" in str(blocker)
    assert "CUDA out of memory" in str(blocker)
    assert segs == []
