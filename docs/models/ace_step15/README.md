# ACE-Step 1.5

Song generation with vocals (or instrumentals), covers and local repaint with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) in its **XL** configuration: DiT `acestep-v15-xl-sft` (4B, 50 steps) plus the 4B language-model planner `acestep-5Hz-lm-4B`, both in bf16 on one GPU, no quantization and no offload. Served by the gateway at `/v1/ace-step15/music/generate`, `/music/cover` and `/music/repaint` (caller guide: `gateway/services/ace_step15/GUIDE.md`; how to prompt, and when to choose it over the other music models: `docs/research/MUSIC_MODEL_COMPARISON.md`).

## What it does

| Endpoint | Inputs | Output |
|---|---|---|
| `music/generate` | `caption`, `lyrics`, `instrumental`, `vocal_language`, `duration` (10-600 s), `bpm`, `keyscale`, `time_signature`, `thinking` | 48 kHz stereo FLAC |
| `music/cover` | an uploaded recording (`source_audio_asset_id`) plus `caption`, `lyrics`, `cover_strength` | as long as the source |
| `music/repaint` | an uploaded recording plus `repaint_start`, `repaint_end` and `caption` | as long as the source |

Common to all: `num_inference_steps` (default 50), `guidance_scale` (default 7), `seed`. Source recordings may be up to 600 s (checked against the real length at submit time). Cover and repaint skip the language model.

## Layout

```
models/ace_step15/
  download_weights.sh         fetches the four checkpoint folders (none is gated)
  env/requirements.lock.txt   frozen environment (python 3.11, torch 2.10.0+cu128, flash-attn prebuilt wheel, nano-vllm from the submodule)
  runner/run_generate.py      one request from the command line (also the load/generate code the worker uses)
  runner/worker.py            the long-lived worker the gateway starts
third_party/ACE-Step-1.5      pinned upstream checkout, installed editable into the env
$VISION_DATA_DIR/envs/ace_step15/
$VISION_DATA_DIR/weights/ace_step15/checkpoints/   ~28 GB (folder names must stay exactly as downloaded)
```

## Setup

```bash
git submodule update --init third_party/ACE-Step-1.5
scripts/build_env.sh ace_step15
models/ace_step15/download_weights.sh
```

## Manual run (on a GPU node)

```bash
$VISION_DATA_DIR/envs/ace_step15/bin/python models/ace_step15/runner/run_generate.py \
    --model-dir $VISION_DATA_DIR/weights/ace_step15 --mode generate \
    --caption="melancholic indie folk, fingerpicked acoustic guitar, soft female vocal" \
    --lyrics=$'[Verse 1]\nWalking home through the quiet rain' --output-path /tmp/out.flac
```

## Warm workers

Up to its "max jobs" limit (GPU capacity page, default 4) one-GPU Slurm jobs, each loading the DiT and the planner once and serving queued requests one at a time. A worker idle for `ACE_STEP15_IDLE_MINUTES` (default 10) stops. `ACE_STEP15_JOB_TIME` (30 min) is the longest one request may run; a request that fails or is cancelled while running ends its worker. See `docs/models/qwen_image21/README.md` for the shared pool behaviour.

## Measured on this cluster (H100 80GB, 2026-10-23)

Through an isolated test gateway (pool cap 1, idle stop 3 minutes), four requests: generate, cover, repaint and a second generate.

| Case | Result |
|---|---|
| worker load (DiT + planner, weights not yet in page cache) | 96 s; a whole cold request, from submit to a finished file, took 165 s |
| `generate`, 30 s song, 50 steps, planner on | 8.7 s (first request of a fresh process), 32 s from hand-over to finished file as the gateway sees it |
| `repaint`, 10-20 s of a 30 s song | 4.6 s of generation |
| `cover`, 30 s source, strength 0.7 | finished in about 43 s from hand-over (it waited behind the repaint) |
| peak GPU memory (PyTorch's own count, DiT side) | 22-23 GiB; the planner's vLLM cache is allocated separately (about 20 GiB target) |
| warm start | requests 2 and 3 went to the loaded worker (`warm start` in the gateway log) |
| idle stop | the worker was stopped by the gateway after 3 idle minutes (the configured test value; production default is 10) |

Outputs were valid 48 kHz stereo FLAC of the requested length (30.0 s), not silent (peak 0.89). Validation answered correctly: a repaint start past the end of the source returns 400 with the real length; an unknown asset returns 400; a bad `time_signature` returns 422.

## Not verified yet

- **Nobody has listened to the results.** Whether the songs are good, whether the lyrics are sung clearly, and whether cover and repaint do what the prompts say are unjudged. The cover output is uncorrelated with the source waveform (as a restyle should be); repaint outside the window is close to the source but not sample-identical.
- Longer songs (several minutes): time and memory are unmeasured.
- A node whose GPU is already occupied fails the load with an out-of-memory error (seen once on `vlm-0`, where the first attempt of both ACE-Step and YuE2 died during start-up; the gateway started a replacement worker and the job succeeded). That is the node, not the model.
- Upstream says generated music may resemble existing works; check originality before publishing.
