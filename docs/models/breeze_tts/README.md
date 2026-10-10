# ModelService_BreezeTTS2

Text-to-speech with [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) (BreezeBlue). Served on demand by `model-api` at `/v1/breeze-tts/synthesize` (see `model-api/services/breeze_tts/GUIDE.md`). Each request is one Slurm GPU job; there is no always-on service.

## Modes

| Mode | Inputs |
|------|--------|
| Voice Design | `text` + `instruction` (natural-language voice description) |
| Voice Clone | `text` + reference audio + exact reference transcript |
| Voice Direction | Voice Clone inputs + `instruction` (tone, pace, emotion) |

English and Chinese. Vocal events inline in text: `(laugh)`, `(sigh)` in English; `[笑]`, `[叹气]` in Chinese. Output: 24 kHz mono 16-bit WAV.

## Layout

```
breeze-tts/          upstream inference code (breezeblue-ai/breeze-tts, Apache 2.0), pinned in UPSTREAM_SHA.txt
models/breeze-tts-2/ weights (gitignored, ~7.2 GB)
.venv/               Python 3.12, torch 2.9.1+cu128 (gitignored)
run_synthesize.py    entrypoint model-api's Slurm job runs
normalize_audio.py   ffmpeg -> WAV helper for reference audio (libsndfile here can't read M4A/AAC)
```

## Setup (already done on this node)

```bash
git clone --depth 1 https://github.com/breezeblue-ai/breeze-tts.git breeze-tts   # then record the SHA, drop breeze-tts/.git
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r breeze-tts/requirements.txt imageio-ffmpeg huggingface_hub \
    --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('BreezeBlue/Breeze-TTS-2', local_dir='models/breeze-tts-2')"
```

Weights are loaded from disk, so jobs run fine with `HF_HUB_OFFLINE=1`. Eager inference needs about 7.7 GiB of GPU memory.

## Manual run (on a GPU node)

```bash
.venv/bin/python run_synthesize.py --model-dir models/breeze-tts-2 --work-dir /tmp/bt --output-path /tmp/bt/out.wav \
    --text="Hello there." --instruction="A calm, warm young woman." --cfg-scale=4
```

## License

Upstream code: Apache 2.0. **Model weights and self-hosted outputs: BreezeBlue Research and Non-Commercial License - research / non-commercial use only.** Only clone voices you have the rights and consent to use.
