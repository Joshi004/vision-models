# Stable Audio 3 (Medium)

Instrumental music, stems, solo instruments and sound effects with [Stable Audio 3 Medium](https://github.com/Stability-AI/stable-audio-3) (`stabilityai/stable-audio-3-medium`, post-trained, 8 steps), full precision on one GPU. Served by the gateway at `/v1/stable-audio3/audio/generate`, `/audio/transform` and `/audio/inpaint` (caller guide: `gateway/services/stable_audio3/GUIDE.md`; prompting and comparison: `docs/research/MUSIC_MODEL_COMPARISON.md`).

**Licence:** Stability AI Community Licence (free below USD 1M yearly revenue). The weights are **gated** on Hugging Face. The model does not produce intelligible vocals.

## What it does

| Endpoint | Inputs | Output |
|---|---|---|
| `audio/generate` | `prompt`, `duration` (default 30 s), `num_inference_steps` (default 8), `seed` | 44.1 kHz stereo FLAC |
| `audio/transform` | an uploaded recording plus `prompt`, `init_noise_level` (0-1, default 0.5) | re-imagined recording |
| `audio/inpaint` | an uploaded recording plus `mask_start`, `mask_end` (may pass the end to extend), `prompt` | recording with that window regenerated |

There is no `cfg_scale` or `negative_prompt`: they have no effect on this checkpoint. The maximum duration is `STABLE_AUDIO3_MAX_DURATION` (default **380 s**; upstream documents both about 285 s and 380 s, and a 380 s clip was generated here). The runner's own hard bound is also 380 s. `inpaint` with `mask_start` equal to the recording's length extends it.

## Layout

```
models/stable_audio3/
  download_weights.sh         fetches the gated repo into a private HF cache
  env/requirements.lock.txt   frozen environment (python 3.12, torch 2.7.1+cu128, flash-attn prebuilt wheel, transformers 5.20.2)
  runner/run_generate.py      one request from the command line (also the load/generate code the worker uses)
  runner/worker.py            the long-lived worker the gateway starts
third_party/stable-audio-3    pinned upstream checkout, installed editable
$VISION_DATA_DIR/envs/stable_audio3/
$VISION_DATA_DIR/weights/stable_audio3/hf          private HF cache, ~10.4 GB; also holds the T5Gemma text encoder
```

## Setup

```bash
git submodule update --init third_party/stable-audio-3
scripts/build_env.sh stable_audio3
# accept the terms at https://huggingface.co/stabilityai/stable-audio-3-medium, then log in:
$VISION_DATA_DIR/envs/stable_audio3/bin/hf auth login
models/stable_audio3/download_weights.sh
```

## Manual run (on a GPU node)

```bash
$VISION_DATA_DIR/envs/stable_audio3/bin/python models/stable_audio3/runner/run_generate.py \
    --model-dir $VISION_DATA_DIR/weights/stable_audio3 --mode generate \
    --prompt="Warm lo-fi hip hop, dusty drums, mellow Rhodes piano, 80 BPM" --duration 30 \
    --output-path /tmp/out.flac
```

## Warm workers

Up to its "max jobs" limit (GPU capacity page, default 4) one-GPU Slurm jobs, each loading the model once; idle for `STABLE_AUDIO3_IDLE_MINUTES` (default 10) stops a worker. `STABLE_AUDIO3_JOB_TIME` (20 min) is the longest one request may run. See `docs/models/qwen_image21/README.md` for the shared pool behaviour.

## Measured on this cluster (H100 80GB, 2026-10-23)

Through an isolated test gateway (cap 1 worker). The gated weights were downloaded first (9.8 GB private cache).

| Case | Result |
|---|---|
| worker load | 39 s; a cold request, from submit to a finished file, took 65-140 s |
| `generate`, 30 s | 6.5 s for the first request of a fresh process, then about 0.5-0.6 s |
| `generate`, **380 s** (the maximum) | 1.4 s of generation, valid 380.0 s stereo FLAC, peak memory 5.3 GiB |
| `transform` and `inpaint`, 30 s source | 0.6 s and 0.5 s |
| `inpaint` extending a 30 s clip to 45 s (`mask_start` 30, `mask_end` 45) | 0.7 s; the first 29 s match the source (waveform correlation 0.98) |
| `inpaint` of 10-20 s in a 30 s clip | the parts outside the window match the source (correlation 0.96-0.97) |
| `transform`, `init_noise_level` 0.5 / 0.1 | correlation with the source 0.20 / 0.87 (lower = further from the source) |
| peak GPU memory | 5.1-5.3 GiB |
| idle stop and restart | the worker idled out after 10 minutes and the next request started a new one (65 s) |

Outputs were valid 44.1 kHz stereo FLAC, no NaN, almost no clipped samples (at most 0.01 %). Offline loading from the private cache works (the workers run with `HF_HUB_OFFLINE=1`).

## Not verified yet

- **Nobody has listened to the results.** Musical quality, and whether a 380 s clip stays coherent for its whole length, are unjudged.
- `inpaint` with a very short mask (a few seconds) was not tried.
