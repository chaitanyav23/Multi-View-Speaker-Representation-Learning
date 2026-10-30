# Recreated Environment

Recreated source snapshot: `/home/ubuntu/marston/phase20_44e9bb7_repro` exported from `44e9bb7:phase20_standalone` via `git archive`.
Virtual environment: `/home/ubuntu/marston/.venv_phase20_44e9bb7`.

## Installation Method

Conda was unavailable, so the original `install.sh` fallback path was used conceptually: `python3 -m venv`. Exact `pip install -r requirements.txt` failed under dependency resolution because the archived pins are internally inconsistent:

```text
DeepFilterNet==0.5.6 requires numpy<2.0,>=1.22, but requirements.txt pins numpy==2.2.6.
```

To reproduce the archived pin set, the environment was installed with `pip install --no-deps -r requirements.txt` after the resolver failure. This preserved the exact package pins but leaves known metadata conflicts.

## Pip Check

```text
deepfilternet 0.5.6 has requirement numpy<2.0,>=1.22, but you have numpy 2.2.6.
wheel 0.47.0 has requirement packaging>=24.0, but you have packaging 23.2.
WARNING: The directory '/ephemeral/cache/pip' or its parent directory is not owned or is not writable by the current user. The cache has been disabled. Check the permissions and owner of that directory. If executing pip with sudo, you should use sudo's -H flag.
```

## Recreated Key Versions

| Package | Version | Matches 44e9bb7 Pin? |
|---|---:|---:|
| python | not installed | NO/NA |
| torch | 2.8.0 | YES |
| torchaudio | 2.8.0 | YES |
| whisperx | 3.8.5 | YES |
| faster-whisper | 1.2.1 | YES |
| ctranslate2 | 4.7.2 | YES |
| transformers | 4.57.6 | YES |
| numpy | 2.2.6 | YES |
| scipy | 1.15.3 | YES |
| soundfile | 0.13.1 | YES |
| pyannote-audio | 4.0.4 | YES |
| pyannote-core | 6.0.1 | YES |
| pyannote-metrics | 4.1 | YES |
| nemo-toolkit | 2.7.3 | YES |
| deepfilternet | 0.5.6 | YES |
| deepfilterlib | 0.5.6 | YES |
| onnxruntime | 1.23.2 | YES |
| onnxruntime-gpu | 1.23.2 | YES |
| av | 17.0.1 | YES |
| ffmpeg | 1.4 | YES |
| ffmpeg-python | 0.2.0 | YES |
| cuda-toolkit | 13.0.2 | YES |

## Runtime Verification

The recreated environment imported `torch`, `torchaudio`, `whisperx`, `faster_whisper`, `ctranslate2`, `transformers`, `pyannote.audio`, `nemo_toolkit`, `DeepFilterNet`, `soundfile`, `numpy`, and `scipy`. Outside the sandbox it reported CUDA available on Tesla T4.
