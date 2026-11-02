# Stable Audio Open 1.0

Short sound effects, foley, drum loops and ambience with [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0), run through `stable-audio-tools` in float32 on one GPU. Served by the gateway at `/v1/stable-audio-open/audio/generate` and `/audio/transform` (caller guide: `gateway/services/stable_audio_open/GUIDE.md`; prompting and comparison: `docs/research/MUSIC_MODEL_COMPARISON.md`).

**Licence:** Stability AI Community Licence (free below USD 1M yearly revenue). The weights are **gated** on Hugging Face. Clips are at most 47 s; for longer music or better musical structure use Stable Audio 3.

## What it does

| Endpoint | Inputs | Output |
|---|---|---|
| `audio/generate` | `prompt`, `negative_prompt`, `duration` (default 30, max 47 s), `num_inference_steps` (default 100), `cfg_scale` (default 7), `seed` | 44.1 kHz stereo FLAC |
| `audio/transform` | an uploaded recording (first 47 s used) plus `prompt`, `init_noise_level` (0.05-20, default 0.5) | re-imagined recording |

Sampler: `dpmpp-3m-sde`. `init_noise_level` is in the sampler's noise units, not a 0-1 fraction (unlike Stable Audio 3): about 0.1 stays very close to the source, around 1 changes the sound but keeps the character, 5 or more ignores the source almost entirely.

## Layout

```
models/stable_audio_open/
  download_weights.sh         fetches the gated repo plus the ungated T5 text encoder (t5-base)
  env/requirements.lock.txt   frozen environment (python 3.10, torch 2.7.1+cu128, transformers 5.20.2)
  runner/run_generate.py      one request from the command line (also the load/generate code the worker uses)
  runner/worker.py            the long-lived worker the gateway starts
third_party/stable-audio-tools   pinned upstream checkout, installed editable
$VISION_DATA_DIR/envs/stable_audio_open/
$VISION_DATA_DIR/weights/stable_audio_open/{model,hf}   ~5 GB model + the T5 cache
```

The upstream pins torch 2.7 with the cu126 index; the lock installs the cu128 build instead, to match the cluster.

## Setup

```bash
git submodule update --init third_party/stable-audio-tools
scripts/build_env.sh stable_audio_open
# accept the terms at https://huggingface.co/stabilityai/stable-audio-open-1.0, then log in:
$VISION_DATA_DIR/envs/stable_audio_open/bin/hf auth login
models/stable_audio_open/download_weights.sh
```

## Manual run (on a GPU node)

```bash
$VISION_DATA_DIR/envs/stable_audio_open/bin/python models/stable_audio_open/runner/run_generate.py \
    --model-dir $VISION_DATA_DIR/weights/stable_audio_open --mode generate \
    --prompt="128 BPM tech house drum loop" --duration 20 --output-path /tmp/out.flac
```

## Warm workers

Up to its "max jobs" limit (GPU capacity page, default 4) one-GPU Slurm jobs, each loading the model once; idle for `STABLE_AUDIO_OPEN_IDLE_MINUTES` (default 10) stops a worker. `STABLE_AUDIO_OPEN_JOB_TIME` (20 min) is the longest one request may run. See `docs/models/qwen_image21/README.md` for the shared pool behaviour.

## Measured on this cluster (H100 80GB, 2026-10-23)

Through an isolated test gateway (cap 1 worker), after downloading the gated weights (4.6 GB model, 0.85 GB T5 cache).

| Case | Result |
|---|---|
| worker load | 29-34 s; a cold request took about 2 minutes from submit to a finished file |
| `generate`, 20 s, 100 steps, cfg 7 | 16.5 s, peak memory 9.6 GiB |
| `transform`, 30 s source, 100 steps | 13.4 s, peak memory 10.1 GiB |
| `transform`, waveform correlation with the 30 s source at `init_noise_level` 0.1 / 1 / 5 | 0.94 / 0.78 / 0.06 |

Outputs were valid 44.1 kHz stereo FLAC, no NaN, almost no clipped samples.

### Fixes the first real run needed

Both were environment gaps that the CPU import check missed; `requirements.lock.txt` is updated, and an environment built from the old lock needs `uv pip install pytorch-lightning==2.5.5 soundfile pywavelets==1.8.0`:
- `pytorch-lightning` is imported by upstream's model code but declared only in its `train` extra.
- The pinned `pywavelets==1.4.1` is binary-incompatible with numpy 2.2 (now 1.8.0).
- `soundfile` was missing (the runner writes FLAC with it).

Also, `download_weights.sh` now downloads `t5-base` with xet disabled: the legacy `t5-base` name has no xet token endpoint (404).

## Not verified yet

- **Nobody has listened to the results**, so sound quality is unjudged (upstream says it is weaker than Stable Audio 3 at musical structure).
- The `negative_prompt` effect and step counts other than 100 were not compared.
