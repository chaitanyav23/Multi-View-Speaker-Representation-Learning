# WhisperX Audio Intelligence Platform

A modular, deployable audio intelligence system built using WhisperX, Faster-Whisper, and Pyannote Audio.

This project provides:

* Automatic Speech Recognition (ASR)
* Word-level forced alignment
* Optional speaker diarization
* CPU-optimized inference
* Gradio-based interactive interface
* Hugging Face Spaces deployment support

---

# Features

## Automatic Speech Recognition (ASR)

* Powered by WhisperX and Faster-Whisper
* Fast and accurate transcription pipeline
* Automatic language detection
* Multi-language support

## Word-Level Alignment

* Precise timestamp alignment for words and segments
* Improved subtitle and transcript quality
* Alignment using phoneme-based models

## Speaker Diarization

* Optional speaker diarization using `pyannote-audio`
* Assigns speaker labels to transcript segments
* Lazy-loaded diarization model for improved performance

## CPU-Optimized Deployment

* Uses CPU-only PyTorch wheels
* `int8` compute optimization
* Designed for Hugging Face CPU Spaces
* Audio length validation to avoid long-running inference

## Interactive Gradio Interface

* Upload audio directly in browser
* Toggle diarization on/off
* View formatted transcripts with timestamps
* Hugging Face Spaces ready

---

# Demo Architecture

```text
Audio Input
    ↓
WhisperX ASR
    ↓
Word-Level Alignment
    ↓
Optional Speaker Diarization
    ↓
Structured Transcript Output
    ↓
Gradio Interface
```

---

# Project Structure

```text
project/
│
├── app.py              # Gradio frontend entrypoint
├── pipeline.py         # Main inference pipeline
├── utils.py            # Utilities and model loaders
├── transcribe.py       # ASR transcription module
├── align.py            # Alignment module
├── diarize.py          # Speaker diarization utilities
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies for HF Spaces
├── README.md
└── .gitignore
```

---

# Technologies Used

* WhisperX
* Faster-Whisper
* PyTorch
* Pyannote Audio
* Gradio
* Transformers
* Librosa
* FFmpeg

---

# Setup

## 1. Clone Repository

```bash
git clone https://github.com/chaitanyav23/WhisperX-Audio-Intelligence-Platform.git

cd WhisperX-Audio-Intelligence-Platform/project
```

---

# 2. Create Environment

Recommended Python version:

```text
Python 3.10
```

Create conda environment:

```bash
conda create -n whisperx python=3.10

conda activate whisperx
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install FFmpeg:

## Ubuntu/Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

---

# 4. Hugging Face Token

Speaker diarization requires a Hugging Face token.

Create a `.env` file (this is the only source used for the token):

```env
HF_TOKEN=your_huggingface_token
```

You must also:

* Accept access to `pyannote/speaker-diarization-community-1`
* Accept Hugging Face model terms

---

# Running the Application

Launch the Gradio application:

```bash
python app.py
```

The application will start locally at:

```text
http://127.0.0.1:7860
```

---

# CLI Usage

Run the pipeline directly (diarization is **off** by default):

```bash
python pipeline.py path/to/audio.wav
```

Enable diarization explicitly:

```bash
python pipeline.py path/to/audio.wav --diarization
```

---

# Usage

1. Upload an audio file
2. Optionally enable speaker diarization (disabled by default)
3. Run the pipeline
4. View timestamped transcript output

Recommended audio length:

```text
< 2 minutes
```

for CPU inference.

---

# Example Output

```text
[1.23s - 4.88s] SPEAKER_00: Well, from my point of view, what Paul is proposing sounds fine.

[5.66s - 16.65s] SPEAKER_00: I am a bit concerned about working with a system of core hours and then flexible hours.
```

---

# Hugging Face Spaces Deployment

This project is configured for deployment on Hugging Face Spaces.

## Required Files

```text
app.py
requirements.txt
packages.txt
```

## packages.txt

```text
ffmpeg
```

## Add Secret

Inside Hugging Face Space settings:

```text
Settings → Variables and Secrets
```

Add:

```text
HF_TOKEN
```

---

# Optimization Features

## Lazy Diarization Loading

The diarization model is loaded only when required.

Benefits:

* Faster startup
* Reduced memory usage
* Better CPU deployment performance

## CPU-Safe Torch Installation

Uses:

```text
torch==2.8.0+cpu
```

instead of CUDA wheels for lightweight deployment.

## int8 Inference

Configured for CPU-efficient inference:

```python
COMPUTE_TYPE = "int8"
```

---

# Future Improvements

Planned additions:

* SRT/VTT subtitle export
* JSON transcript export
* Speaker-aware summarization
* Waveform visualization
* Speaker timeline visualization
* FastAPI backend deployment
* Streaming ASR
* Model selection support
* Multilingual analytics

---

# Known Limitations

* CPU inference is slower than GPU inference
* Diarization significantly increases runtime
* Very long audio files are not recommended on free CPU deployments
* First diarization load may take time due to model downloads

---

# Credits

## Core Technologies

* WhisperX
  [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)

* OpenAI Whisper
  [https://github.com/openai/whisper](https://github.com/openai/whisper)

* Pyannote Audio
  [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)

* Faster-Whisper
  [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

# Repository

GitHub Repository:

[https://github.com/chaitanyav23/WhisperX-Audio-Intelligence-Platform](https://github.com/chaitanyav23/WhisperX-Audio-Intelligence-Platform)
