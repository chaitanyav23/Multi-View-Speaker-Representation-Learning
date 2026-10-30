# System Dependencies

Observed source machine: Ubuntu 22.04.

| Package / tool | Observed version | Ubuntu install command | Why it is needed |
|---|---|---|---|
| `ffmpeg` | `4.4.2-0ubuntu0.22.04.1` | `sudo apt-get install -y ffmpeg` | Audio preprocessing and loudness normalization |
| `ffprobe` | `4.4.2-0ubuntu0.22.04.1` | `sudo apt-get install -y ffmpeg` | Audio duration probing in evaluation helpers |
| `libsndfile` | system package | `sudo apt-get install -y libsndfile1` | Runtime backend for `soundfile` |
| `git` | `2.34.1` | `sudo apt-get install -y git` | Python package/model tooling commonly shells out to Git |
| `gcc` | `11.4.0` | `sudo apt-get install -y build-essential` | Building Python packages with native extensions |
| `g++` | `11.4.0` | `sudo apt-get install -y build-essential` | Building Python packages with native C++ extensions |
| `cmake` | `3.22.1` | `sudo apt-get install -y cmake` | Native package build dependency |
| NVIDIA driver / CUDA | driver not active on captured machine; CUDA packages pinned in `requirements.txt` | Install a driver compatible with the selected PyTorch/CUDA wheels | Required for GPU execution; CPU fallback exists for several stages but the original experiment targeted CUDA where available |

`deepFilter` is provided by the Python package `DeepFilterNet==0.5.6`; it was not present on the base shell PATH during extraction because the original virtual environment was not active.
