# YuE2

Full-song generation with sung vocals from a style line and lyrics, with [YuE2](https://huggingface.co/m-a-p/YuE2-3B) (YuE2-3B language model plus YuE2-Vae, full precision on one GPU). Served by the gateway at `/v1/yue2/songs/generate` and `/songs/cover` (caller guide: `gateway/services/yue2/GUIDE.md`; prompting and comparison: `docs/research/MUSIC_MODEL_COMPARISON.md`).

**Licence:** the weights are **CC BY-NC 4.0** and the authors ask for permission; companies need a commercial licence from them. The API does not enforce this.

## What it does

| Endpoint | Inputs | Output |
|---|---|---|
| `songs/generate` | `style`, `lyrics`, `cot` (`full`, `melody`, `off`), `cfg_scale`, `seed` | 48 kHz stereo FLAC, a whole song |
| `songs/cover` | the same plus `abc`, a melody score (`cot` is `melody` or `full`) | a song following that melody in the new style |

- `style` is one line, language first: genre, instruments, vocal character and tempo. There is no BPM field.
- One request makes **one** candidate. The authors' best quality comes from generating several and choosing; send several seeds.
- Turning a recording into an ABC score (upstream: SheetSage2) is **not hosted**; the caller supplies the score.

## Layout

```
models/yue2/
  download_weights.sh         fetches YuE2-3B and YuE2-Vae (not gated)
  env/requirements.lock.txt   frozen environment (python 3.12, torch 2.10.0+cu128)
  runner/run_generate.py      one request from the command line (also the load/generate code the worker uses)
  runner/worker.py            the long-lived worker the gateway starts
third_party/YuE               pinned upstream checkout (provides YuE2Pipeline), installed editable
$VISION_DATA_DIR/envs/yue2/
$VISION_DATA_DIR/weights/yue2/{YuE2-3B,YuE2-Vae}   ~7.3 GB
```

## Setup

```bash
git submodule update --init third_party/YuE
scripts/build_env.sh yue2
models/yue2/download_weights.sh
```

## Manual run (on a GPU node)

```bash
$VISION_DATA_DIR/envs/yue2/bin/python models/yue2/runner/run_generate.py \
    --model-dir $VISION_DATA_DIR/weights/yue2 --mode generate \
    --style="English, warm piano pop, expressive female voice, acoustic piano, light drums, 88 BPM" \
    --lyrics=$'[Verse]\nWalking home through the quiet rain\n\n[Chorus]\nHold on to the little light' \
    --output-path /tmp/out.flac
```

## Warm workers

Up to its "max jobs" limit (GPU capacity page, default 4) one-GPU Slurm jobs, each loading the model once and serving queued requests one at a time; a worker idle for `YUE2_IDLE_MINUTES` (default 10) stops. `YUE2_JOB_TIME` (45 min) is the longest one request may run. The pipeline is created with a 72 GiB memory budget, so it stays fully on the GPU (an 80 GB card is assumed). See `docs/models/qwen_image21/README.md` for the shared pool behaviour.

## Measured on this cluster (H100 80GB, 2026-10-23)

Through an isolated test gateway (pool cap 1, idle stop 3 minutes), one `songs/generate` request with 8 lines of lyrics.

| Case | Result |
|---|---|
| worker load | 32 s; a whole cold request, from submit to a finished file, took 150 s (including one worker that died on a node with an occupied GPU, and its replacement) |
| generation, 36.9 s of audio | 17.1 s; `cot` full, no truncation reported (`abc` and `semantic` both `False`) |
| peak GPU memory | 7.9 GiB |
| idle stop | the worker was stopped by the gateway after 3 idle minutes (the configured test value; production default is 10) |

The output was a valid 48 kHz stereo FLAC, not silent (peak 0.74). The length came out at 37 s for 8 short lyric lines; whole songs of 3-5 minutes need longer lyrics and were not tried.

`songs/cover` was run with upstream's included example (`third_party/YuE/examples/melody.abc`, `cot: melody`, a new folk style, 8 lyric lines): 65.6 s of audio in 20.4 s, 7.9 GiB peak, no truncation, valid FLAC with almost no clipping.

## Not verified yet

- **Nobody has listened to the results**, so singing quality, lyric clarity, and whether the cover follows the supplied melody, are unjudged.
- Full-length songs (minutes): time, memory and the `truncated` behaviour are unmeasured.
- A node whose GPU is already occupied fails the load with an out-of-memory error (seen once on `vlm-0`); the gateway replaced the worker and the job succeeded.
