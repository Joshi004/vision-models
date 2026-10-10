# Parakeet TDT ASR Service

Audio transcription service using NVIDIA Parakeet TDT (0.6B-v3) model via NeMo toolkit with native segment timestamps.

## On demand via model-api (recommended)

This project no longer needs its own always-on GPU allocation for normal use. `model-api`
(`/home/naresh/Vision/model-api`) submits one short Slurm job per transcription request --
`POST /v1/parakeet/transcribe` -- holding a GPU only for the length of that one job, instead of a
dedicated `salloc` running for hours between requests.

- **Entry point for that job:** `run_transcribe.sh` / `run_transcribe.py` in this directory --
  decodes the input (any format ffmpeg can read -- audio or video) to 16kHz mono WAV, loads
  `models/parakeet-tdt-0.6b-v3.nemo` locally (no network access needed), transcribes (chunking
  automatically past 10 minutes of audio), and writes a JSON file in the same response shape the
  legacy endpoint below always returned (`transcription`, `word_timestamps`, `segment_timestamps`,
  `metadata`).
- **API side:** see `model-api/services/parakeet/GUIDE.md`, or just call `GET /v1/guide` on a
  running model-api instance -- that single document covers the full upload -> submit -> poll ->
  download workflow (with curl/Python examples) that every model-api backend shares.
- **One-time setup this mode needed** (already done on this deployment): `imageio-ffmpeg` installed
  into `/home/naresh/venvs/parakeet-service` (no system ffmpeg exists on this cluster), and
  `models/parakeet-tdt-0.6b-v3.nemo` downloaded locally so Slurm jobs never need network access to
  fetch it.

Everything below this point (`app.py`, `start_service.sh`, the `/transcribe` URL-based endpoint)
describes the **legacy always-on mode**: a single long-lived `uvicorn` process holding one GPU for
as long as it runs, started by hand via a dedicated `salloc`. It still works and its own code is
unchanged, but for day-to-day use the on-demand path above is what avoids tying up a GPU between
requests. Keep reading if you specifically need a persistent, low-latency server instead (e.g. many
requests in a short burst, where repeated model cold-starts would dominate).

## Overview

This service provides a REST API endpoint for transcribing audio files from URLs. It automatically generates:
- Full transcription with punctuation and capitalization
- Word-level timestamps for every word
- Segment-level timestamps with populated text (native from model)
- Metadata about the transcription

## Key Features

- **High Accuracy**: Uses Parakeet TDT 0.6B-v3 model with ~6.05% WER (better than RNNT 1.1B)
- **Native Punctuation**: Automatic punctuation and capitalization built into the model
- **Native Segmentation**: Segment timestamps with text provided directly by the model
- **Word Timestamps**: Precise timing for every word
- **Segment Timestamps**: Sentences with start/end times, text, and word counts
- **Multilingual Support**: Supports 25 European languages with automatic language detection

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv /home/naresh/venvs/parakeet-service
source /home/naresh/venvs/parakeet-service/bin/activate
```

### 2. Install Dependencies

```bash
cd /home/naresh/Vision/ModelService_Parakeet
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Service

Edit `config.env` to adjust settings if needed:
- `PORT`: Service port (default: 8006)
- `MODEL_NAME`: Model identifier (default: nvidia/parakeet-tdt-0.6b-v3)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 100)

## Usage

### Start the Service

```bash
cd /home/naresh/Vision/ModelService_Parakeet
./start_service.sh
```

The service will:
- Load the Parakeet TDT model (downloads automatically on first run)
- Start the FastAPI service on port 8006
- Log output to `logs/service.log`

### API Endpoint

#### Transcribe Audio from URL

```bash
curl -X POST "http://localhost:8006/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "http://localhost:8080/audio/sample.wav"}'
```

**Request Body:**
```json
{
  "audio_url": "http://example.com/audio.wav"
}
```

**Response:**
```json
{
  "transcription": "Hello, how are you today? I am doing great. Thank you for asking!",
  "processing_time": 3.45,
  "word_timestamps": [
    {"word": "hello", "start": 0.0, "end": 0.5},
    {"word": "how", "start": 0.6, "end": 0.8},
    ...
  ],
  "segment_timestamps": [
    {
      "text": "Hello, how are you today?",
      "start": 0.0,
      "end": 2.5,
      "word_count": 5
    },
    {
      "text": "I am doing great.",
      "start": 2.5,
      "end": 4.2,
      "word_count": 4
    },
    ...
  ],
  "metadata": {
    "total_segments": 3,
    "total_words": 12,
    "duration": 6.5
  }
}
```

#### Health Check

```bash
curl http://localhost:8006/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "service": "Parakeet TDT ASR Service",
  "version": "3.0.0"
}
```

### API Documentation

Once the service is running, access interactive API docs at:
- Swagger UI: http://localhost:8006/docs
- ReDoc: http://localhost:8006/redoc

## Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

## Segmentation

The TDT model provides segment timestamps natively with populated text. Segments are automatically generated by the model based on natural sentence boundaries and punctuation. No post-processing is required - the model handles segmentation internally.

## Configuration

All configuration is managed via `config.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PORT` | 8006 | Service port |
| `HOST` | 0.0.0.0 | Bind address |
| `MODEL_NAME` | nvidia/parakeet-tdt-0.6b-v3 | NeMo model identifier |
| `MAX_FILE_SIZE_MB` | 100 | Maximum upload size |
| `TEMP_DIR` | /tmp/parakeet_uploads | Temporary file storage |
| `LOG_DIR` | logs | Log file directory |

## Architecture

```
parakeet-service/
├── app.py              # Legacy always-on mode: FastAPI app, routes, model loading
├── config.py           # Legacy always-on mode: configuration management
├── config.env          # Legacy always-on mode: environment configuration
├── run_transcribe.sh   # On-demand mode: Slurm job entrypoint (CUDA env fix, execs run_transcribe.py)
├── run_transcribe.py   # On-demand mode: one-shot transcription (see model-api/services/parakeet/)
├── segment_utils.py    # Shared by both modes: metadata calculation utilities
├── models/             # On-demand mode: local parakeet-tdt-0.6b-v3.nemo checkpoint (gitignored)
├── requirements.txt    # Python dependencies (shared venv, both modes)
└── logs/               # Legacy always-on mode: service logs
```

## Logs

Service logs are written to `logs/service.log` with INFO level logging.

## Performance

- **Model**: Parakeet TDT 0.6B-v3 (~600M parameters)
- **Speed**: Real-time factor of 3386x (processes 60 minutes in ~1 second)
- **Accuracy**: WER ~6.05% (better than RNNT 1.1B's ~7.0%)
- **Memory**: ~2GB GPU memory (more efficient than RNNT 1.1B)
- **Example**: 146-second audio processes in ~2-3 seconds

## Notes

- The model is loaded once at startup and reused for all requests
- Temporary files are automatically cleaned up after processing
- Word and segment timestamps are always generated natively by the model
- Segment text is populated automatically by the TDT model
- First run downloads model weights (~600MB) and caches them
- Supports up to 24 minutes of audio with full attention (or 3 hours with local attention)

## Troubleshooting

### Model not loading
- Ensure CUDA is available (check with `nvidia-smi`)
- Verify NeMo toolkit is installed correctly
- Check logs in `logs/service.log`

### Slow transcription
- Check GPU availability and memory
- Verify audio is being preprocessed correctly
- Monitor CPU/GPU usage during transcription

### Empty segments
- TDT model provides segments with populated text natively
- If segments are empty, check `logs/service.log` for warnings
- Verify model loaded correctly and timestamps are enabled

## Changelog

### Version 3.0.0
- Migrated to Parakeet TDT 0.6B-v3 model
- Removed manual segment creation logic (model provides segments natively)
- Improved accuracy (6.05% WER vs 7.0% for RNNT)
- Faster processing and lower memory usage
- Native punctuation and capitalization
- Segment timestamps include populated text from model

### Version 2.0.0
- Simplified API to single `/transcribe` endpoint
- Removed file upload endpoint (URL-based only)
- Automatic generation of word and segment timestamps
- Intelligent segmentation with punctuation + duration fallback
- Added metadata in response (word count, segment count, duration)
- Modular code structure (separated segment logic)

### Version 1.0.0
- Initial release with Parakeet RNNT 1.1B
- Support for word and segment timestamps
- File upload and URL-based transcription
