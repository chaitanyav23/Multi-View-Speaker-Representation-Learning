# Phase 20 Dual Path Improved VAD

Self-contained dual-path speech processing experiment for `raw_audio/bwc.wav`.
Phase 20 is derived from Phase 19 and keeps preprocessing, DeepFilterNet, loudness normalization, WhisperX, MSDD, fusion, and evaluation unchanged.
The only architectural change is the improved probability-guided Pyannote VAD implementation.

Run the complete pipeline:

```bash
python experiments/phase20_dual_path_improved_vad/run_phase20.py
```

Each script in `scripts/` is independently executable and exposes `--help`.
