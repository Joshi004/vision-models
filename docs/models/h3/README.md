# MiniMax H3

MiniMax H3 ([MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), released 2026-07-31, weights 2026-08) is an
open-weight model that generates a video and its stereo soundtrack **together, in one denoising loop**: a 33B
single-stream transformer over one packed sequence of text, conditioning media, audio latents and video latents, with
Qwen3-VL-32B as the text and vision encoder. This project runs the **open release only (H3-Base, bf16)**; MiniMax's hosted
service adds two models that are not in the open weights and therefore not here: *Context-IR* (rewrites a plain request into
the structured prompt) and *Regenerate-2K* (the 2K upscaler).

> **Status (2026-10-17): the 4- and 8-GPU layout below is built and checked everywhere except on a GPU. It has not run yet.**
> Jobs have run on this cluster on the earlier two-GPU layouts (the notes below recorded only `split` as completing; an
> `ulysses` attempt with 328 frames, and two requests with large reference images, ran out of memory -- see "Why four or more GPUs"). All the numbers
> that decide how a request is sized (limits, speed-ups, load times) are **initial guesses**, to be refitted from the usage log.
> The "Measured on this cluster" section at the end is the checklist for the first runs of the new layout.

It is used through two endpoints, one per checkpoint partition:

- `POST /v1/h3/videos/generate` -- text, or a first and/or last frame, to video + sound (diffusers workflows `t2va` and
  `fl2va`, the `transformer/` partition, MiniMax's "FL2VA" checkpoint).
- `POST /v1/h3/videos/audio-to-video` -- image and/or video references (up to 9 and 3), with optional audio references (up to 3,
  12 references in all), to video + sound (workflow `ref2va`, the `transformer_ref/` partition, the "Ref2VA" checkpoint). Since
  0.14.0 the audio is `audio_asset_ids` (a list; it was a single `audio_asset_id`), and `video_audio` switches a reference
  video's own soundtrack off per video.

License: **MiniMax H3 Community License** (`LICENSE` in the weights folder; summary in `gateway/services/h3/GUIDE.md` §7).
Read it before relying on this. In particular the weights and their outputs are **not licensed for use or hosting in the
United States, the European Union, the United Kingdom or South Korea** without MiniMax's separate written authorization, and
commercial products above 20 million USD yearly revenue need authorization too. The Hugging Face repository itself is not
gated (checked 2026-10-13); nothing here enforces the license.

## How it works (read from diffusers 0.41's source)

Integration is diffusers' **Modular Diffusers** blocks (`diffusers.modular_pipelines.minimax_h3`, class
`MiniMaxH3ModularPipeline`; there is no `DiffusionPipeline` half). One repository holds both partitions in diffusers layout;
selecting a workflow loads only its own transformer. Facts that shaped the integration:

- **Workflows.** `t2va` (text only), `fl2va` (first and/or last keyframe), `ref2va` (omni-reference: up to 9 images, 3 videos,
  3 audio, 12 in total, an audio reference never alone). Blocks run in the order `before_encode` (media preparation),
  `text_encoder`, `vae_encoder`, `denoise`, `decode`.
- **Guidance-distilled.** Guidance is baked into both partitions: no guider, no negative prompt, no guidance scale, one forward
  pass per step. Two schedulers step the video (`shift` 12) and audio (`shift` 3) latents inside one transformer call.
  `num_inference_steps` defaults to 50.
- **Fixed geometry.** 24 fps; frame counts are `17*n+5` (what the video VAE decodes); 5 to 15 s, which on that grid is 124
  (5.17 s) to 345 (14.4 s) frames (346 would round up to 362 = 15.08 s and be rejected); a 768 px short edge by default
  (1344x768), sizes multiples of 32. Reference images do not bind the output shape (default 16:9); the first keyframe does.
- **Prompt.** The encoder prepends a label per reference, numbered per kind (`<Picture i>: `, `<Audio j>: `, `<Video k>: `),
  and reads the hidden state after the 50th decoder layer. Nothing rewrites the prompt text. MiniMax's two prompt guides
  (in the weights folder under `docs/`) describe the structured format the model was trained on.
- **Reference order and `<Audio j>` numbering.** The references are packed in the order they are passed, and the order sets the
  labels and each reference's place on the shared rotary clock, so a different order is a different request. The runner always
  passes images, then videos, then audio files, each in the order given. `<Picture i>` and `<Video k>` count their own kind;
  `<Audio j>` counts every audio: a video that carries a soundtrack takes the next `<Audio j>` **before** its `<Video k>`, then the
  audio files follow (`encoders.py`, the presentation tokenizer). The limits (9 images, 3 videos, 3 audio, 12 in all) count
  references, not soundtracks, so three videos with sound plus three audio files is allowed. A video clip and an audio clip are
  documented as 2 to 15 s each and 15 s in total per kind; diffusers does not check that, it cuts each reference to the output's
  length (`_normalize_video_condition`, `_normalize_audio_condition`), and neither does the gateway.
- **The audio reference.** It never reaches the Qwen3-VL conditioner: it is encoded by the audio VAE alone, into clean
  conditioning rows of the packed sequence. A reference video's own soundtrack is read as well, unless the runner is told to
  drop it (`--mute-reference-video N`, which the gateway sets from `video_audio: false`: the reference's `audio` is cleared, so
  it gets no `<Audio j>` and is conditioned on for its picture alone). What the model does with the
  audio is stated in the prompt (`fully_copy`, `partially_copy`, `reference`, `weak_reference`) -- so it can be reused as
  the soundtrack or only followed for voice timbre. Either way the result is the model's own rendition, not a muxed copy.
  *How close a `fully_copy` is has not been measured here.* (An earlier research note in `LIP_SYNC_RESEARCH.md` described
  the audio only as a timbre reference; section 11 there corrects that.)
- **Components and sizes** (the diffusers layout, as downloaded):

| Folder | Size | Used by |
|---|---|---|
| `transformer/` | 66.3 GB (61.7 GiB), bf16, 14 shards | `generate` |
| `transformer_ref/` | 66.3 GB (61.7 GiB), bf16, 14 shards | `audio-to-video` |
| `text_encoder/` | 66.7 GB (62.1 GiB), Qwen3-VL-32B in bf16, 14 shards | both |
| `vae/` | 10.4 GB, the video VAE (released in float32) | both |
| `audio_vae/` | 0.6 GB | both |

That is **210 GB (196 GiB) on disk**; each job reads one transformer, the encoder and the two VAEs, about 144 GB. These are the
complete released weights: nothing is pruned or quantized. (The repository also holds the original, non-diffusers copy of the
same weights in `FL2VA/` and `Ref2VA/`, 290 GB more; the download script skips it.) The VAEs are loaded under diffusers'
`dtype=bfloat16` recipe, but diffusers pins the cast-sensitive modules of the video VAE to float32 itself.

## How a job runs: 4 or 8 GPUs, always ulysses

The text encoder and the transformer are each about 62 GiB, so one 80 GB card cannot hold both. Each job reserves **N H100s
on one node** (`--nodes=1 --gres=gpu:N`, N = 4 or 8, chosen per request by the gateway, see "Sizing") and
`models/h3/runner/run_h3.py` (`--gpus N --parallelism ulysses`, always passed) lays the work out like this, with the job's GPUs
in `CUDA_VISIBLE_DEVICES` order `g0, g1, g2, ...`:

- **g0 (rank 0):** loads the transformer and the VAEs, VAE-encodes the references, broadcasts the transformer to the others,
  denoises its share, and decodes the result.
- **g1 and g2:** hold the **text/vision encoder** while the prompt and references are read: Qwen3-VL's 64 decoder layers are
  split between them (layers 0-31 on g1, 32-63 on g2), so the encoder's per-layer outputs for a long sequence of reference-image
  tokens have two cards to live on. When the encoder has finished and exited they become ranks 1 and 2.
- **g3 and up (8-GPU jobs have g3 .. g7):** idle until rank 0 is ready, then ranks 3 and up. They only receive the weights and
  denoise.
- **Denoising uses every GPU** (`ulysses_degree=N`): each holds the whole transformer (61.7 GiB) and runs 1/N of the packed
  token sequence through every block. 56 attention heads divide evenly by 4 and by 8. A 2-GPU layout (`--gpus 2`, the encoder on one
  GPU) and the `split` mode (transformer blocks divided between two GPUs) are still in the runner as operator fallbacks, but the
  gateway never asks for either.

```
Slurm job (N GPUs)  ->  run_h3.py (rank 0, g0: generation)
                          |-- spawns child run_h3.py --stage condition (g1 + g2: encoder split over two GPUs)
                          |      before_encode -> text_encoder           loads ~62 GiB encoder, layers 0-31 | 32-63
                          |      writes work/h3_conditioning.pt, exits
                          |-- spawns N-1 run_h3.py --stage denoise-worker --rank r (r = 1 .. N-1, will take g<r>)
                          |      each imports, then waits for rank 0 to open a localhost rendezvous store
                          |-- loads transformer(_ref) + VAEs on g0      loads ~77 GB, at the same time
                          |-- waits for the child (it exits, g1 + g2 are free), reads the handoff
                          |-- vae_encoder (references/keyframes) on g0, VAEs to host RAM,
                          |     encoded latents written to work/h3_encoded.pt
                          |-- opens the store + NCCL group; the workers join and take their GPUs
                          |-- copies the whole transformer g0 -> every other GPU over NCCL (nothing is read from disk again)
                          |-- all N ranks: denoise (steps), each on 1/N of the token sequence, same seed
                          |-- workers exit; rank 0 moves the transformer to host RAM, decodes on g0 -> MP4 (PyAV)
```

- The conditioning half's outputs are exactly the inputs the rest reads by name (checked against diffusers' block graph on
  CPU): the text embedding and its row tags, plus the resolved `height`/`width`, `num_frames`, `keyframes` or
  `normalized_references`. They cross as one pickle file in the job's own folder, which the runner deletes when it ends.
- **Separate processes, not threads**, so the weights are read from disk in parallel: accelerate's weight-loading context patches
  torch globally, which is not safe across threads. The child sees only the encoder's GPUs (as its `cuda:0` and `cuda:1`); the
  parent sees all N: g0 as `cuda:0`, and it only touches the others after the child has exited.
- **The encoder is placed by hand** (`split_text_encoder`): the vision tower, token embeddings, rotary embedding and layers 0-31
  go to `cuda:0`, layers 32-63 and the final norm to `cuda:1`, and a forward pre-hook on layer 32 moves the hidden states and
  position embeddings across. The first parameter of the whole encoder stays on `cuda:0`, which is the device the offload manager
  (`ComponentsManager.enable_auto_cpu_offload`) looks at, so it does not move the layers back.
- Loading is lazy: `load_components` memory-maps the weights in seconds, and they are only read from disk when a model first
  moves onto its GPU. The encoder moves when it first runs; the generation half moves the transformer and VAEs onto g0 straight
  away, while the encoder works, so both reads really overlap.
- The VAEs leave g0 right after the references are encoded. The offload manager only makes room for a model's weights, never for
  activations, so with the transformer (61.7 GiB) and both VAEs (10.3 GiB) resident, the first audio-to-video jobs ran out of
  memory on the first denoising step even at 124 frames. Their offload hooks bring them back for the decode.
- **`ulysses`: how the ranks work together.**
  - *Context parallelism.* diffusers' `MiniMaxH3Transformer3DModel` already declares how to split its packed sequence (`_cp_plan`):
    the hidden states, the rotary table and the per-row indices are split N ways at the first block, and the two output heads
    gather the sequence back, so every rank sees the full prediction each step and runs the (cheap) scheduler step itself.
    `enable_parallelism(ContextParallelConfig(ulysses_degree=N, ulysses_anything=True))` is called on every rank;
    `ulysses_anything` because the packed sequence has whatever length the request makes it. The attention backend
    (`_flash_3_hub` by default) must support it, and does. The two-layer text refiner runs before the split, on text every rank
    holds in full, so the runner switches the exchange off for it only (`enable_parallel_denoiser`).
  - *Inputs.* Every rank starts from identical inputs: the conditioning handoff, plus the encoded reference/keyframe latents rank
    0 writes to `work/h3_encoded.pt`. Each rank draws its noise from its own generator seeded with `--seed`, so the noise is
    identical.
  - *Weights.* A worker builds the transformer without weights (meta device) and receives every parameter and buffer from rank 0
    over NCCL (one broadcast to all ranks), shapes and dtypes included (the checkpoint is mixed precision). Expected to take
    seconds to tens of seconds depending on the GPU link (**not measured yet**: the runner logs
    `transformer copied to the other N GPU(s) in ...s`), where reading 61.7 GiB from shared storage again took 5-11 min.
  - *Order of events.* The workers are started with the encoder so their imports are done early, but they touch no GPU until rank
    0 opens the rendezvous store, which happens only after the encoder has exited and the references are encoded; until then g1 and
    g2 are the encoder's. `NCCL_NVLS_ENABLE=0` is set for the job to save NCCL buffer memory.
  - *Failure handling.* A thread in rank 0 watches every worker: if one dies, rank 0 prints that worker's error as the last line
    (`ERROR: a denoising process failed: ...`) and stops at once instead of waiting for a GPU operation to time out (10 min). A
    failure on rank 0 kills the workers.
  - *Decode.* Rank 0 moves its transformer to host RAM (about 62 GiB, logged), so the VAEs get g0 back, and decodes alone. The
    workers have exited by then.
- **Why four or more GPUs.** Memory per GPU while denoising is the 61.7 GiB of weights plus activations for 1/N of the sequence:
  about 15 GiB are left and each row of the sequence costs about 0.18 MiB, so more GPUs means a longer sequence fits, and each
  step also runs faster. Two earlier failures led here: (1) requests with **two large reference images** (13,673 and 16,809
  encoder tokens) ran the encoder out of memory on its single GPU before denoising even started, because the encoder keeps every
  layer's output (about 0.63 MiB per token) next to its 62 GiB of weights; (2) a 328-frame video in the two-GPU `ulysses` mode ran
  out of memory on the first GPU. `split` was the only mode recorded as completing, but it runs two GPUs one after the other and cannot help the
  encoder.
- **`split` (operator fallback, `--gpus 2 --parallelism split` only).** With all 61.7 GiB on one card, a 345-frame 1344x768 clip
  (about 103k video tokens) ran out of memory in the first denoising step. `split_transformer` in `run_h3.py` moves blocks 25-49
  of the 50 onto the conditioning GPU once the encoder has exited, so each GPU holds about 31 GiB of weights and the blocks run
  one after another. Hooks carry the inputs of block 25 to the second GPU and the output of block 49 back.
- **Warm workers.** A request does not get a Slurm job of its own. The gateway keeps up to `H3_MAX_WORKERS` (default 8)
  long-lived Slurm jobs of 4 or 8 GPUs (`models/h3/runner/worker.py`, started and fed by `gateway/common/worker_pool.py`, the
  pool Qwen-Image-2.1 uses too). A worker holds its allocation and, for each request, runs the same `run_h3.py` command a
  cold job would run, so the runner is unchanged and **nothing stays loaded in a process**. What a second request saves is the
  reading: the checkpoints the first one read are in the node's page cache, so loading takes seconds. Measured with a tiny
  `t2va` request (22 frames, 256x448, 4 steps, 4 GPUs) run twice in one allocation: transformer load 263.8 s -> 12.5 s, encoder
  load 308.2 s -> 11.6 s, whole run 440.7 s -> 76.5 s. Through the gateway (two cold workers started together on one node,
  then a third request): 590 s and 621 s cold, 92 s warm. About 33 s of the remaining time is `transformer_to_host` and about
  10 s the NCCL group set-up, both of which a persistent process could remove; that was not built, because the page cache
  alone gets most of the saving and leaves `run_h3.py` untouched.
  - Workers are separate per transformer checkpoint (`transformer` for `t2va` and `fl2va`, `transformer_ref` for `ref2va`) and
    per GPU count. A worker serving a checkpoint is kept in the pool for `H3_IDLE_MINUTES` (default 10) after its last
    request; at the cap, the longest-idle worker is stopped to make room for one of another kind. A worker is also replaced
    before its own Slurm time (`H3_WORKER_TIME`, 12 h) cannot fit another request.
  - A request that fails ends its worker (the next one gets a fresh one). Cancelling a running request stops its whole worker,
    since the run is several processes on every GPU of the allocation. A request that runs past its time limit (the sizer's
    `--time`) is stopped the same way and fails with a message saying so.
  - The page-cache saving depends on the node keeping the files cached; it is a property of the node's memory, not something the
    worker enforces, so a busy node may evict them and a "warm" request be slower than the figures above.
  - `estimated_run_seconds` still includes the 330 s load, so a warm request finishes well inside it.
  - The usage log (below) records a pooled request with `worker_id` and `warm`; its `slurm.elapsed_s` is the request's own run
    time and `slurm.max_rss_gib` is null (sacct's figures would be the worker's).

## Sizing: how many GPUs, and the estimate

Before a job is queued, `gateway/services/h3/sizing.py` sizes the request from the request body and the **headers** of the uploaded
files (Pillow reads an image's size, PyAV a video's or audio's length, picture size and whether it has a soundtrack; nothing is
decoded). It mirrors what diffusers' pipeline does with the same inputs:

| Quantity | Rule |
|---|---|
| Canvas | the request's `orientation` / `height` + `width`; else 16:9 (768x1344); else, for a keyframe request, the first keyframe's aspect ratio (short edge 768, area at most 768x1344, multiples of 32) |
| Target video rows | `(5n + 2)` latent frames (a `17n + 5` frame clip) x `(H/32) x (W/32)` |
| Target audio rows | 80 per second of output (40 latents per second, two channels) |
| Prompt tokens | `ceil(characters / 3.5)` plus 16 per reference for its label |
| Image reference | scaled to a 2048 px short edge (upscaling included, no area cap, sides rounded to 32); `tokens = rows = (w/32) x (h/32)`; an aspect ratio beyond 1:4 / 4:1 is refused. Checked exactly against the two failed jobs (13,673 and 16,809 encoder tokens) |
| Keyframe | a canvas worth of tokens and of rows each |
| Video reference | cut to the output's length (and the VAE snaps its frames down to `17n + 5`); put on the canvas of its **own** aspect ratio; the encoder reads it at 2 frames per second, two frames per block, and a block is one canvas worth of tokens plus 8 (about one block per second of video; an upper bound beyond ~12 s, where Qwen3-VL starts to shrink the frames); its picture rows are `(5n + 2)` x canvas rows; plus 80 rows per second for its soundtrack unless `video_audio` turns it off |
| Audio reference | cut to the output's length, 80 rows per second; no encoder tokens beyond the label |
| `match_audio_duration` | the first audio's length is read from its header here, so the frame count is known at submission |

`encoder_tokens` is the prompt plus every reference's tokens; `denoise_rows` is those tokens plus all picture rows, audio rows and the
target video's rows. Then:

- **GPU count:** the smallest value in `H3_GPU_OPTIONS` (default `4,8`) with `denoise_rows / N <= H3_ROWS_PER_GPU` (initial guess 40,000
  per GPU; about 80,000 would just fit, from 15 GiB / 0.18 MiB) whose estimated time is within `H3_API_TIME`.
- **Refused (HTTP 422, nothing queued)** when `encoder_tokens > H3_MAX_ENCODER_TOKENS` (initial guess 60,000), when no option fits,
  or when even the largest option's estimate is over the job time limit. The `detail` is one sentence with the numbers.
- **Estimate (`estimated_run_seconds`)**, in seconds, before the queue: `330` loading + `60` weight copy + reference encode
  (0.0004 s per picture row: **a placeholder, never measured**) + `120` decode + `862 x (rows / 73000)^1.5 x steps / 50 / speedup[N]`, with
  speed-ups `{4: 3.0, 8: 5.0}`, all multiplied by 2 for the denoising when `H3_ATTENTION_BACKEND` is not a FlashAttention-3 backend. The
  exponent 1.5 sits between linear (feed-forward) and quadratic (attention). Slurm's `--time` is twice the estimate plus 10 minutes,
  kept between 30 minutes and `H3_API_TIME`.
- **The Slurm allocation:** `--gres=gpu:N --cpus-per-task=8N --mem=${H3_MEM:-256G}` (both halves load into host RAM at once, about
  144 GB).

The result goes back to the client as `gpus` and `estimated_run_seconds` on the submit response, `GET /v1/jobs/{id}` and
`GET /v1/jobs`; the web UI shows them on every job row and in the form after submitting. They are also stored with the job
(`jobs.db` columns `gpus`, `estimated_run_seconds`) and in the job folder as `sizing.json` (numbers only).

### The usage log (numbers only, for calibration)

Whenever an H3 job finishes, the gateway's poller appends one JSON line to **`$VISION_DATA_DIR/metrics/h3_jobs.jsonl`**
(`gateway/services/h3/usage_log.py`). A line holds: the job's outcome, GPUs, estimate, queue wait and run time; `sizing.json`; the runner's
`metrics.<role>.json` files (`run`, `condition`, `worker<r>`: stage timings, peak GPU memory per GPU, the encoder's actual token
count, the packed sequence's actual length, and for a failure the stage and the out-of-memory sizes parsed from the error); and one
`sacct` call (state, node, elapsed seconds, peak host RAM; for a request run by a warm worker the request's own run time and node,
without the RAM figure, plus `worker_id` and `warm`). **Never the prompt, file names, asset ids or error messages** (only an error's type and the out-of-memory sizes in it). A job sent with
`incognito: true` is recorded the same way with its job id left out. The log lives outside `jobs/`, so retention does not delete it, and
`purge_all_data.sh` leaves `metrics/` alone.

```bash
# Add the jobs that ran before the log existed (from their run.log, args.json and sacct; marked "source": "backfill").
$VISION_DATA_DIR/envs/gateway/bin/python scripts/h3_usage_log.py backfill --dry-run     # then without --dry-run
# One row per job, one column per number, for a spreadsheet.
$VISION_DATA_DIR/envs/gateway/bin/python scripts/h3_usage_log.py csv --output h3_jobs.csv
```

The thresholds and speed-ups above are meant to be refitted from this data once 4- and 8-GPU jobs have run: compare `sizing.denoise_rows`
with `metrics.run.sequence_rows` and `sizing.encoder_tokens` with `metrics.condition.encoder_tokens` (the formulas), the peak memory per
GPU with the limits, and `sizing.estimated_run_seconds` with `run_s` (the time model).

## Setup

```bash
scripts/build_env.sh h3               # Python 3.12, torch 2.9.1 (CUDA 12.8), diffusers 0.41.0, transformers 5.19.0
models/h3/download_weights.sh         # ~210 GB into $VISION_DATA_DIR/weights/h3/MiniMax-H3, pinned revision, not gated
gateway/run.sh                        # restart the gateway to pick up the new backend and the rebuilt UI
```

The download script pins revision `42ed227e` (MiniMax's self-hosting guide asks for that), checks that every shard each index
names exists, and is safe to re-run. The environment is its own because it needs diffusers 0.41 and transformers 5.19 together
with `torchaudio` (used to resample a reference soundtrack that is not already at 32 kHz) and `av` (decodes the reference media
and writes the MP4). The upstream code is not vendored: it is the installed diffusers package, pinned in
`models/h3/env/requirements.lock.txt`. The gateway side is `gateway/services/h3/`; the Slurm-side wrapper is
`models/h3/runner/run_h3.py`.

Operator settings (environment variables of the gateway process, see `gateway/run.sh` and `vision.env.example`; all are read when the
gateway starts): `H3_MODEL_DIR`; `H3_GPU_OPTIONS` (`4,8`; `4` pins every job to four GPUs); `H3_ROWS_PER_GPU` (40000) and
`H3_MAX_ENCODER_TOKENS` (60000), the limits a request is sized against; `H3_MEM` (256G for the whole job), `H3_CPUS_PER_GPU` (8) and
`H3_API_TIME` (01:30:00, also the longest estimate a request may have); `H3_MAX_WORKERS` (8), `H3_IDLE_MINUTES` (10) and
`H3_WORKER_TIME` (12:00:00, must exceed `H3_API_TIME`), the warm workers described above (`gateway/run.sh` passes these
three on to the gateway); and `H3_ATTENTION_BACKEND` (`_flash_3_hub` when started with
`gateway/run.sh`). There is no `H3_PARALLELISM` any more: jobs always use `ulysses`. The gateway environment now also needs Pillow and
PyAV (`gateway/requirements.lock.txt`, the same pins as the `h3` environment) to read the uploaded files' headers.

## Things the wrapper does because diffusers' recipe does not

| Situation | What `run_h3.py` does |
|---|---|
| Loading both halves one after the other would add the two load times | Runs the conditioning half as a child process so both are read at once |
| A duration or frame count is not on the `17*n+5` grid, or beyond 345 frames | Rounds up to the grid and clamps to 124-345 (the gateway applies the same rule so the stored request is explicit) |
| `ref2va` needs `num_frames` before anything runs | Always resolves it; `match_audio_duration` computes it from the decoded audio (`round(samples / rate * 24)`) |
| Weights missing or half downloaded | Fails first with "weights are incomplete ... run models/h3/download_weights.sh" instead of an obscure loader error |
| The handoff holds processed copies of the request's reference media | Deletes it (and the scratch folder) when the run ends, success or not |
| A failed conditioning half | The parent reads its error file and prints `ERROR: conditioning step failed: ...` as the last line, which is what the job status shows |
| A denoising worker dies while rank 0 waits on it inside a GPU operation | A watcher thread in the parent prints `ERROR: a denoising process failed: ...` (the worker's own error) as the last line and stops at once, instead of waiting out the 10 min GPU-operation timeout |
| A bad `H3_GPU_OPTIONS`, `H3_ROWS_PER_GPU` or `H3_MAX_ENCODER_TOKENS` value | `gateway/run.sh` and `gateway/services/h3/config.py` refuse to start the gateway with it, so it never becomes a failure in every request |
| A request that cannot fit on 8 GPUs, or whose encoder input is too long | The gateway refuses it with 422 and the numbers before any job is queued (see "Sizing") |
| A failed or finished job | Every process writes `metrics.<role>.json` into the job folder (numbers and stage names only) for the usage log |

## Limits

- **768p only.** The 2K pass is not in the open weights.
- **5.17 to 14.4 s, 24 fps.**
- **No prompt rewriting, no negative prompt, no guidance scale.** Write the structured prompt yourself.
- **The audio reference is not a muxed copy** (see above). For an exact recording with the mouth matched to it, use LTX
  audio-to-video with `lip_sync`, or TBDub on its own.
- **Not tried:** several audio references together, a reference video's soundtrack with `video_audio` on and off, the 12-reference
  maximum (the gateway and runner accept it; memory is unmeasured), several speakers, languages other than English, the
  community turbo LoRAs (4 and 8 steps), quantized or pruned variants, ComfyUI.

## Measured on this cluster

**Nothing yet for the 4- and 8-GPU layout.** Figures that circulate elsewhere (an independent benchmark: ~94 s for a 5.17 s 960x544 clip,
20 steps, on one H100 with CPU offload; a 4x H100 cloud session averaging ~4.7 min per clip at 50 steps) were taken with different settings
and are not predictions for this deployment. From the first runs on, **the usage log records these numbers for every job** (see "The usage
log"), so there is nothing to copy by hand; the first runs are for checking them against what the gateway predicted:

| | Where to read it |
|---|---|
| Load time, per half | `metrics.run.timings_s.load_gpu`, `metrics.condition.timings_s.load_encoder` (and `loaded in ...s` in `run.log`) |
| Peak GPU memory, each GPU | `metrics.*.peak_gib`; `run.log` prints `peak GPU memory during denoising` per rank, and the conditioning half prints its own peak |
| Peak host RAM | `slurm.max_rss_gib` (expected ~134 GiB, about 144 GB, while both halves load; rank 0 adds ~62 GiB when it moves its transformer off the GPU for the decode, after the encoder is long gone) |
| Rows the gateway predicted vs rows that ran | `sizing.denoise_rows` vs `metrics.run.sequence_rows`; `sizing.encoder_tokens` vs `metrics.condition.encoder_tokens` |
| Estimated vs actual time | `sizing.estimated_run_seconds` and `sizing.estimate_parts_seconds` vs `run_s` and `metrics.run.timings_s` (reference encode, weight copy, denoise, decode) |
| Weight copy and GPU link | `metrics.run.timings_s.weights_copy`; `nvidia-smi topo -m` inside a job shows whether the GPUs share NVLink |
| Attention backend | `H3_ATTENTION_BACKEND=_flash_3_hub` (diffusers: ~3x faster on Hopper) is the default under `gateway/run.sh`; the estimate doubles its denoising time without it |
| `fully_copy` fidelity | listen: is the reference audio reproduced closely, or only its voice? |

The first runs to make, each submitted by hand and checked against the table above: the text-only 243-frame baseline (the estimate says
4 GPUs and about 13 minutes with FlashAttention 3), the request with two large reference images that ran the encoder out of memory, five
reference images, and one reference video.

**Likely first problems**

- **Out of GPU memory on the generation GPU** for a very long clip at full size. Measured on the older layout, with the whole
  transformer on one card (peak GPU memory, 1344x768 unless noted): 124 frames 70.4 GiB, 243 frames at 768x992 71.4 GiB, and 345 frames
  ran out of memory in the first denoising step. The 4- and 8-GPU layout gives each GPU only 1/N of the sequence, and the gateway refuses
  requests above `H3_ROWS_PER_GPU` rows per GPU, but that limit is a guess: if a job still runs out of memory, read `metrics.run.peak_gib`
  and `metrics.run.oom`, lower `H3_ROWS_PER_GPU`, and compare the failed job's `sequence_rows` with the estimate.
- **The encoder:** it keeps about 0.63 MiB per token on top of its half of the weights; `H3_MAX_ENCODER_TOKENS` (60,000) is a guess at
  where two GPUs run out. The conditioning stage's `metrics.condition.peak_gib` and `failed_stage` say how close a job came.
- **The VAE encode step** of the references runs on g0 next to the whole transformer (about 72 GiB), and reference videos make it
  heavier; the estimate's time per picture row for it (`REFERENCE_ENCODE_SECONDS_PER_ROW` in `config.py`) is a placeholder.
- **Host RAM:** both halves hold their weights in RAM while loading. If a job is killed for memory, raise `H3_MEM` (256G by default for
  the whole job).
- **`_flash_3_hub` offline:** the kernel is fetched from the Hub by the `kernels` package, which is not in the environment, and
  jobs run with `HF_HUB_OFFLINE=1`. Untested sketch: `uv pip install kernels` into `envs/h3`, then on a node with internet run
  `python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3')"` so the cache is filled, then set
  `H3_ATTENTION_BACKEND=_flash_3_hub` for the gateway. Do this only after a first job has worked with the default backend.

## What was checked without a GPU (2026-10-14)

- The block graph of all three workflows, and that the conditioning half's outputs equal the rest's inputs by name.
- The media-preparation blocks run on CPU with synthetic inputs: an image resized to a 2048 px short edge, a 30 fps video
  resampled to 24 fps and fitted to 1344x768, a 44.1 kHz mono waveform resampled to 32 kHz stereo with `torchaudio`, two
  keyframes put onto the canvas; and the pickle round trip of the handoff.
- The runner's argument parser accepts exactly the argv the gateway builds, rejects the invalid combinations, and the weights
  check names what is missing.
- The gateway: request validation (422), unknown assets (400), the OpenAPI schema, the `/v1/guide` section, the rendered web
  forms against the real schema.
- `scripts/build_env.sh h3` from the lock file, and `hf download --dry-run` of the exact include patterns against the pinned
  revision (210.3 GB).

Not checked: the text encoder step, the transformer, the decode, the process timing, memory, and `sbatch --gres=gpu:4` / `gpu:8`.

## What was checked without a GPU for the 4- and 8-GPU layout (2026-10-17)

- The runner compiles and its argument parser accepts `--gpus`, `--parallelism` and `--metrics-dir`.
- The sizer on synthetic images, videos and audio (made with PyAV and Pillow): the token counts of the two failed jobs are reproduced
  exactly, and the GPU choice, the 422 messages (too many encoder tokens, too many rows, too long) and `sizing.json` come out as intended.
- `dispatch.py` with a stand-in `sbatch`: the `--gres`, `--cpus-per-task`, `--mem` and `--time` it passes, and the runner's argv.
- The usage log: a record built from a fake finished job (the prompt does not appear in it; an incognito job's id is omitted), and the
  `backfill --dry-run` against the existing database (26 earlier H3 jobs).
- The web UI builds.

Not checked: any job on a GPU, `ulysses` with 4 or 8 ranks, the encoder split, the metrics files a real run writes, or the
estimate against real times.
