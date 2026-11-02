# Vision

A model-serving gateway for a GPU cluster (Slurm). One FastAPI app accepts jobs, submits each as a Slurm job, and serves the results. Each model runs in its own isolated Python environment.

Models: LTX-2.3 and LTX-2.5 (video), MiniMax H3 (video with sound, 4 or 8 GPUs per job), Wan-Animate v1 (character replace), RVC via Applio (voice conversion), Parakeet (transcription), Breeze TTS 2 (text to speech), Qwen-Image-2.1 (image generation and editing), Qwen3.8-27B-FP8 (text and JSON, one shared warm server), TBDub (lip sync, also available as `lip_sync` on LTX audio-to-video), and four music models, each its own backend with warm workers: ACE-Step 1.5 (songs, cover, repaint), YuE2 (full songs, cover from a score), Stable Audio 3 (instrumentals, stems, SFX, transform, inpaint) and Stable Audio Open 1.0 (short SFX and loops). Which to use for what: `docs/research/MUSIC_MODEL_COMPARISON.md`.

## Layout

```
vision/
├── gateway/          the API: server.py, common/ (job engine), services/<model>/ (adapters), ui/ (React app)
├── models/<id>/      our own code for each model: runner/ (runs on the GPU node), env/ (frozen env lock)
├── third_party/      upstream projects as pinned git submodules. Never edited here.
├── scripts/          build_env.sh
└── docs/             status, research notes, per-model notes

../vision-data/       NOT in git (override with VISION_DATA_DIR)
├── envs/<id>/        Python environments, built by scripts/build_env.sh
├── weights/<id>/     model weights
├── jobs/  jobs.db    job workspaces and the job database
└── wheels/           local wheels (flash-attn for Wan-Animate)
```

Rules of thumb: code and pins go in this repo; weights, environments, jobs and outputs go in the data directory. Weights are never committed.

## Setup

```bash
git clone --recurse-submodules <repo-url> vision        # or: git submodule update --init
cd vision

# Python environments (one per model, built from the frozen lock files)
for e in gateway ltx23 ltx25 wan_animate rvc parakeet breeze_tts qwen_image21 tbdub h3 qwen38 ace_step15 stable_audio_open stable_audio3 yue2; do scripts/build_env.sh $e; done

# Web UI
(cd gateway/ui && npm ci && npm run build)
```

Weights go under `$VISION_DATA_DIR/weights/<id>/`:

| Model | Folder | How to get it |
|---|---|---|
| ltx25 | `weights/ltx25/ltx-2.5` | `models/ltx25/download_weights.sh` (gated, needs `HF_TOKEN`) |
| ltx23 | `weights/ltx23/ltx-2.3`, `weights/ltx23/gemma-3-12b` | no download script; the expected file names are in `gateway/services/ltx/config.py` |
| wan_animate | `weights/wan_animate/Wan2.2-Animate-14B` | `docs/models/wan_animate/README.md` (Hugging Face `Wan-AI/Wan2.2-Animate-14B`) |
| rvc | `weights/rvc/voices/<name>/{model.pth,model.index}` | your own voice models; Applio's pretrained files stay inside `third_party/Applio` (see below) |
| parakeet | `weights/parakeet/parakeet-tdt-0.6b-v3.nemo` | no download script; see `docs/models/parakeet/README.md` |
| breeze_tts | `weights/breeze_tts/breeze-tts-2` | `docs/models/breeze_tts/README.md` (Hugging Face `BreezeBlue/Breeze-TTS-2`) |
| qwen_image21 | `weights/qwen_image21/{base,turbo}` | `models/qwen_image21/download_weights.sh` (not gated, ~31 GB each) |
| tbdub | `weights/tbdub` | `models/tbdub/download_weights.sh` (not gated, ~15 GB; build the `tbdub` env first, the script uses its `hf`) |
| h3 | `weights/h3/MiniMax-H3` | `models/h3/download_weights.sh` (not gated but under the MiniMax H3 Community License, ~210 GB; build the `h3` env first, the script uses its `hf`) |
| qwen38 | `weights/qwen38/Qwen3.8-27B-FP8` | `models/qwen38/download_weights.sh` (not gated, ~29 GB; build the `qwen38` env first, the script uses its `hf`) |
| ace_step15 | `weights/ace_step15/checkpoints` | `models/ace_step15/download_weights.sh` (not gated, ~30 GB; build the `ace_step15` env first, the script uses its `hf`) |
| yue2 | `weights/yue2/{YuE2-3B,YuE2-Vae}` | `models/yue2/download_weights.sh` (not gated, ~7.8 GB; CC BY-NC 4.0; build the `yue2` env first) |
| stable_audio3 | `weights/stable_audio3/hf` (a Hugging Face cache) | `models/stable_audio3/download_weights.sh` (gated: accept the terms for `stabilityai/stable-audio-3-medium` and log in with `hf auth login`; ~10 GB; build the `stable_audio3` env first) |
| stable_audio_open | `weights/stable_audio_open/{model,hf}` | `models/stable_audio_open/download_weights.sh` (gated: accept the terms for `stabilityai/stable-audio-open-1.0` and log in with `hf auth login`; ~5.3 GB; build the `stable_audio_open` env first) |

Applio looks up its pretrained files relative to its own folder, so those (`rvc/models/**`, `assets/config.json`) live inside `third_party/Applio` as git-ignored files. The submodule is configured with `ignore = untracked` so they never show up in `git status`.

## Run

```bash
cp vision.env.example vision.env     # optional, edit
gateway/run.sh                       # (re)starts the API in a tmux session, always on port 8012
```

`gateway/run.sh` loads `vision.env` itself (its values replace same-named variables already set in your shell), so settings only take effect the next time you run it. `VISION_ENV_FILE=/path/to/file` loads another file instead.

`gateway/run.sh` always restarts: it stops the running gateway (on this node, anything else of yours on port 8012, and -- over ssh if it can -- one started on another login node), starts a fresh one and waits for `/v1/health`. It prints which node it is on: forward port 8012 from that node (`ssh -N -L 8012:localhost:8012 <you>@<node>`). The workspace's `.vscode/settings.json` pins Cursor's forward of 8012 to local port 8012.

By default the gateway keeps everything about a job (logs, uploaded inputs, outputs, the request) for `MODEL_API_RETENTION_DAYS` or until the job is purged. A request sent with `"incognito": true` (the web UI's Incognito switch) is the exception: it keeps no logs, its uploaded inputs are deleted when its job ends, and its result can be downloaded once, after which it is deleted. `./purge_all_data.sh` deletes every job's logs, inputs and outputs on demand (`--dry-run` to preview; it refuses while jobs are running).

The API docs are at `/docs`, the integration guide at `/v1/guide`, the web UI at `/ui/`.

## Upstream pins

| Submodule | Upstream | Commit |
|---|---|---|
| `third_party/ltx-2.3` | Lightricks/LTX-2 | `4f89057` |
| `third_party/ltx-2.5` | Lightricks/LTX-2 | `9ec55f9` |
| `third_party/Wan2.2` | Wan-Video/Wan2.2 | `1ea34ff` |
| `third_party/Applio` | IAHispano/Applio | `c7665ac` |
| `third_party/breeze-tts` | breezeblue-ai/breeze-tts | `58ec70c` |
| `third_party/TBDub` | TaoLiveAIGC/TBDub | `19e18e4` |
| `third_party/ACE-Step-1.5` | ACE-Step/ACE-Step-1.5 | `ca1e85f` |
| `third_party/stable-audio-tools` | Stability-AI/stable-audio-tools | `3241adb` |
| `third_party/stable-audio-3` | Stability-AI/stable-audio-3 | `77e0398` |
| `third_party/YuE` | multimodal-art-projection/YuE | `1647252` |
| diffusers (no submodule; pinned in `models/qwen_image21/env/requirements.lock.txt`) | huggingface/diffusers | `1d5d056` |

To move a pin: `git -C third_party/<name> fetch && git -C third_party/<name> checkout <sha>`, rebuild that model's env if its dependencies changed, then commit the updated submodule pointer. LTX-2.3 and 2.5 are the same upstream repository at two different commits, on purpose.

## Notes

- `models/parakeet/legacy/` is the old always-on Parakeet service from before the gateway. Nothing imports it. It is kept as-is, so it still contains old absolute paths.
- Some docstrings and notes under `docs/` still mention the previous folder names (`model-api`, `ModelService_*`, `ltx-2.3/`). They describe the same code; the table above maps old to new.
- Environments are rebuilt from `requirements.lock.txt` files frozen from the original working environments, so versions match what was in use before the restructure.
