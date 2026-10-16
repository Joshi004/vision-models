# MiniMax H3

MiniMax H3 ([MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), released 2026-07-31, weights 2026-08) is an
open-weight model that generates a video and its stereo soundtrack **together, in one denoising loop**: a 33B
single-stream transformer over one packed sequence of text, conditioning media, audio latents and video latents, with
Qwen3-VL-32B as the text and vision encoder. This project runs the **open release only (H3-Base, bf16)**; MiniMax's hosted
service adds two models that are not in the open weights and therefore not here: *Context-IR* (rewrites a plain request into
the structured prompt) and *Regenerate-2K* (the 2K upscaler).

> **Status (2026-10-14): built and checked everywhere except on a GPU. No H3 job has run yet** -- the weights had not been
> downloaded when this was written. Everything about speed, memory and output quality below is marked as unmeasured. The
> "Measured on this cluster" section at the end is the checklist for the first real runs.

It is used through two endpoints, one per checkpoint partition:

- `POST /v1/h3/videos/generate` -- text, or a first and/or last frame, to video + sound (diffusers workflows `t2va` and
  `fl2va`, the `transformer/` partition, MiniMax's "FL2VA" checkpoint).
- `POST /v1/h3/videos/audio-to-video` -- an audio reference plus image and/or video references to video + sound (workflow
  `ref2va`, the `transformer_ref/` partition, the "Ref2VA" checkpoint).

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
- **The audio reference.** It never reaches the Qwen3-VL conditioner: it is encoded by the audio VAE alone, into clean
  conditioning rows of the packed sequence. A reference video's own soundtrack is read as well. What the model does with the
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

## How a job runs: two GPUs

The text encoder and the transformer are each about 62 GiB, so one 80 GB card cannot hold both. Each job reserves **two
H100s on one node** (`--nodes=1 --gres=gpu:2`) and `models/h3/runner/run_h3.py` splits the workflow the way diffusers' own
documentation does for "two cards, nothing offloaded", then uses the second card a second time for denoising. How is the
operator setting **`H3_PARALLELISM`** (`gateway/run.sh`, read when the gateway starts; the gateway passes it to every job as
`--parallelism`):

- **`auto` (default): the mode is chosen per request from the video's length.** Up to 10 s (243 frames, which is what a 10 s
  request rounds up to) uses `ulysses`, anything longer uses `split`. A `--match-audio-duration` request is sized from the
  audio's length, read from the file's header. If that length cannot be read, `split` is used. The choice is logged as
  `parallelism auto: ... -> ulysses|split`.
- **`ulysses`: both GPUs compute every denoising step.** Each holds the whole transformer and runs half of the token
  sequence through every block; around each attention layer the two exchange their halves of the queries, keys and values.
- **`split`: the blocks are divided between the GPUs and run one after another.** Slower, but each GPU holds only half of the
  weights, so far more memory is left for activations. The fallback if `ulysses` runs a job out of memory.

```
Slurm job (2 GPUs)  ->  run_h3.py (rank 0, GPU a: generation)
                          |-- spawns child run_h3.py --stage condition (GPU b: conditioning)
                          |      before_encode -> text_encoder           loads ~62 GiB encoder
                          |      writes work/h3_conditioning.pt, exits
                          |-- ulysses only: spawns run_h3.py --stage denoise-worker (rank 1, will take GPU b)
                          |      imports, then waits for rank 0 to open a localhost rendezvous store
                          |-- loads transformer(_ref) + VAEs             loads ~77 GB, at the same time
                          |-- waits for the child (it exits, GPU b is free), reads the handoff
                          |
                          |  --parallelism split
                          |-- moves transformer blocks 25-49 (~31 GiB) onto GPU b
                          '-- vae_encoder -> denoise (N steps; blocks 0-24 on GPU a, 25-49 on GPU b)
                                          -> decode -> MP4 (PyAV)
                          |
                          |  --parallelism ulysses
                          |-- vae_encoder (references/keyframes) on GPU a, VAEs to host RAM,
                          |     encoded latents written to work/h3_encoded.pt
                          |-- opens the store + NCCL group; the worker joins and takes GPU b
                          |-- copies the whole transformer GPU a -> GPU b over NCCL (nothing is read from disk again)
                          |-- both ranks: denoise (N steps), each on half of the token sequence, same seed
                          |-- worker exits; rank 0 moves the transformer to host RAM, decodes on GPU a -> MP4 (PyAV)
```

- The conditioning half's outputs are exactly the inputs the rest reads by name (checked against diffusers' block graph on
  CPU): the text embedding and its row tags, plus the resolved `height`/`width`, `num_frames`, `keyframes` or
  `normalized_references`. They cross as one pickle file in the job's own folder, which the runner deletes when it ends.
- **Two processes, not threads**, so the two sets of weights are read from disk in parallel: accelerate's weight-loading
  context patches torch globally, which is not safe across threads. The child sees only the conditioning GPU (as its
  `cuda:0`). The parent sees both: the generation GPU as `cuda:0` and the conditioning GPU as `cuda:1`, and it only touches
  `cuda:1` after the child has exited.
- Loading is lazy: `load_components` memory-maps the weights in seconds, and they are only read from disk when a model
  first moves onto its GPU. The encoder moves when it first runs; the generation half moves the transformer and VAEs onto
  its GPU straight away, while the encoder works, so both reads really overlap (the first runs showed the transformer
  would otherwise be read only after the encoder finished). Diffusers' `ComponentsManager` (`enable_auto_cpu_offload`)
  still evicts a model to host RAM if a later step needs the room.
- The VAEs leave the generation GPU when denoising starts. The offload manager only makes room for a model's weights,
  never for activations, so with the transformer (61.7 GiB) and both VAEs (10.3 GiB) resident, the first audio-to-video
  jobs ran out of memory on the first denoising step even at 124 frames (generate jobs fit, peaking at 75 GiB). The
  runner moves both VAEs to host RAM at the first transformer step (`split`) or right after the references are encoded
  (`ulysses`); their offload hooks bring them back for the decode.
- **`ulysses`: how the two ranks work together.**
  - *Context parallelism.* diffusers' `MiniMaxH3Transformer3DModel` already declares how to split its packed sequence
    (`_cp_plan`): the hidden states, the rotary table and the per-row indices are split in two at the first block, and the
    two output heads gather the sequence back, so both ranks see the full prediction every step and run the (cheap)
    scheduler step themselves. `enable_parallelism(ContextParallelConfig(ulysses_degree=2, ulysses_anything=True))` is
    called on both ranks; `ulysses_anything` because the packed sequence has whatever length the request makes it. The
    attention backend (`_flash_3_hub` by default) must support it, and does. The two-layer text refiner runs before the
    split, on text both ranks hold in full, so the runner switches the exchange off for it only
    (`enable_parallel_denoiser`).
  - *Inputs.* Both ranks start from identical inputs: the conditioning handoff, plus the encoded reference/keyframe latents
    rank 0 writes to `work/h3_encoded.pt` (large reference videos would be slow and memory-hungry to push through a GPU
    collective, and a file is what the conditioning handoff already uses). Each rank draws its noise from its own generator
    seeded with `--seed`, so the noise is identical.
  - *Weights.* The worker builds the transformer without weights (meta device) and receives every parameter and buffer from
    rank 0 over NCCL, shapes and dtypes included (the checkpoint is mixed precision). Expected to take seconds to tens of
    seconds depending on the GPU link (**not measured yet**: the runner logs `transformer copied to the second GPU in ...s`),
    where reading 61.7 GiB from shared storage again took 5-11 min.
  - *Order of events.* The worker is started with the encoder so its imports are done early, but it touches no GPU until
    rank 0 opens the rendezvous store, which happens only after the encoder has exited and the references are encoded;
    until then GPU b is the encoder's. `NCCL_NVLS_ENABLE=0` is set for the job to save NCCL buffer memory.
  - *Failure handling.* A thread in rank 0 watches the worker: if it dies, rank 0 prints the worker's error as the last
    line (`ERROR: the second denoising process failed: ...`) and stops at once instead of waiting for a GPU operation to
    time out (10 min). A failure on rank 0 kills the worker.
  - *Decode.* Rank 0 moves its transformer to host RAM (about 62 GiB, logged), so the VAEs get the GPU back, and decodes
    alone. The worker has exited by then.
- **`split`: the transformer is split across both GPUs for denoising.** With all 61.7 GiB on one card, a 345-frame 1344x768 clip
  (about 103k video tokens) ran out of memory in the first denoising step: the feed-forward layer alone briefly needs
  12-13 GiB for that sequence, and the 79 GiB card had about 17 GiB left. The conditioning GPU is idle once the encoder's
  process has exited, so `split_transformer` in `run_h3.py` moves blocks 25-49 of the 50 onto it (`SPLIT_AT`). Each GPU then
  holds about 31 GiB of weights. Everything outside the block stack (input projections, text refiner, output heads) stays on
  the generation GPU, which is also the device the offload manager looks at, so it does not move the blocks back. Forward
  hooks carry the inputs of block 25 to the second GPU and the output of block 49 back; the blocks still run one after
  another, so the output is the same. Per denoising step, one hidden-state tensor crosses each way, plus a copy of the
  rotary table (about 0.1 GiB) for each moved block.
- A job is a cold start every time, like every other backend here: nothing stays loaded.

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

Operator settings (environment variables of the gateway process, see `gateway/run.sh`): `H3_MODEL_DIR`, `H3_MEM_PER_GPU`
(112G, so 224G for the job), `H3_CPUS` (16 in total), `H3_API_TIME` (01:30:00), `H3_ATTENTION_BACKEND` (`_flash_3_hub` when
started with `gateway/run.sh`), and `H3_PARALLELISM` (`auto`, the default; `ulysses` or `split` force that mode for every request).
`H3_PARALLELISM` is read when the gateway starts: every request sent after a restart follows it, and a
restart with `H3_PARALLELISM=split gateway/run.sh` (or `=ulysses`) starts the gateway in that mode.

## Things the wrapper does because diffusers' recipe does not

| Situation | What `run_h3.py` does |
|---|---|
| Loading both halves one after the other would add the two load times | Runs the conditioning half as a child process so both are read at once |
| A duration or frame count is not on the `17*n+5` grid, or beyond 345 frames | Rounds up to the grid and clamps to 124-345 (the gateway applies the same rule so the stored request is explicit) |
| `ref2va` needs `num_frames` before anything runs | Always resolves it; `match_audio_duration` computes it from the decoded audio (`round(samples / rate * 24)`) |
| Weights missing or half downloaded | Fails first with "weights are incomplete ... run models/h3/download_weights.sh" instead of an obscure loader error |
| The handoff holds processed copies of the request's reference media | Deletes it (and the scratch folder) when the run ends, success or not |
| A failed conditioning half | The parent reads its error file and prints `ERROR: conditioning step failed: ...` as the last line, which is what the job status shows |
| `ulysses`: the second denoising process dies while rank 0 waits on it inside a GPU operation | A watcher thread in the parent prints `ERROR: the second denoising process failed: ...` (the worker's own error) as the last line and stops at once, instead of waiting out the 10 min GPU-operation timeout |
| A bad `H3_PARALLELISM` value | `gateway/run.sh` and `gateway/services/h3/config.py` refuse to start the gateway with it, so it never becomes a failure in every job |

## Limits

- **768p only.** The 2K pass is not in the open weights.
- **5.17 to 14.4 s, 24 fps.**
- **No prompt rewriting, no negative prompt, no guidance scale.** Write the structured prompt yourself.
- **The audio reference is not a muxed copy** (see above). For an exact recording with the mouth matched to it, use LTX
  audio-to-video with `lip_sync`, or TBDub on its own.
- **Not tried:** more than one audio reference (the model allows 3; the API takes one), several speakers, languages other than
  English, the community turbo LoRAs (4 and 8 steps), quantized or pruned variants, ComfyUI.

## Measured on this cluster

**Nothing yet.** Figures that circulate elsewhere (an independent benchmark: ~94 s for a 5.17 s 960x544 clip, 20 steps, on one
H100 with CPU offload; a 4x H100 cloud session averaging ~4.7 min per clip at 50 steps) were taken with different settings and
are not predictions for this deployment. To fill in on the first runs, per job (both GPUs sampled every few seconds, as for
TBDub):

| | To record |
|---|---|
| Load time, conditioning half / generation half | printed in the job's `run.log` (`loaded in ...s`) |
| Peak GPU memory, each GPU | the conditioning half prints its own peak. `split`: the generation half prints `peak GPU memory since the split: generation GPU ..., second GPU ...` (counted from after the transformer is split, so the earlier full load does not count). `ulysses`: `peak GPU memory during denoising: generation GPU ..., second GPU ...` (counted from the start of denoising; the worker prints its own too, tagged `denoise-worker`) |
| Peak host RAM | from `sacct`/`seff` (expected ~134 GiB, about 144 GB, while both halves load; `ulysses` adds ~62 GiB when rank 0 moves its transformer off the GPU for the decode, after the encoder is long gone) |
| Denoising time, 1344x768, 124 frames and 345 frames | `split`: `generated N frames ... in ...s`. `ulysses`: `denoised in ...s` (then `decoded N frames ... in ...s`). Compare the two modes on the same request and seed |
| `ulysses`: weight copy and GPU link | `transformer copied to the second GPU in ...s`; `nvidia-smi topo -m` inside a job shows whether the GPUs share NVLink |
| `ulysses`: decode handover | `transformer moved off the GPU for the decode in ...s` |
| Whole job, cold | Slurm run time; this replaces `TYPICAL_RUN_SECONDS` (20 min, a guess) and `H3_API_TIME` |
| `fully_copy` fidelity | listen: is the reference audio reproduced closely, or only its voice? |
| Attention backend | try `H3_ATTENTION_BACKEND=_flash_3_hub` (diffusers: ~3x faster on Hopper) -- see below |

**Likely first problems**

- **Out of GPU memory on the generation GPU** for a long clip at full size. Measured with the whole transformer on one
  card (peak GPU memory, 1344x768 unless noted): 124 frames 70.4 GiB, 243 frames at 768x992 71.4 GiB, and 345 frames ran out
  of memory in the first denoising step. The transformer split across both GPUs (see above) is meant to fix this, but **it has
  not yet been run end to end**: record the two `peak GPU memory since the split` figures and the `generated N frames ... in ...s`
  time for 345 frames at 1344x768, and for 124 frames to compare with the earlier ~710 s. If a job still runs out of memory,
  retry with a smaller canvas (960x544 is the documented speed lever) or a shorter clip; the next step would be group
  offloading, which is not wired in.
- **`ulysses` has NOT been run to completion yet; `auto` uses it for videos up to 10 s.** A 328-frame job (13 s, 768x1344) ran
  out of memory on the first GPU in ulysses mode, which is why longer videos go to `split` under `auto`. Whether 243 frames
  fits with room to spare is expected, not measured. It trades memory for speed: each GPU holds all 61.7 GiB, which leaves
  about 17 GiB for activations of half the tokens (plus NCCL buffers). Expected, not measured: it fits a plain 345-frame
  1344x768 clip with a few GiB to spare, and it is the **long clips, large canvases and many references** that can run out of
  memory (the reference tokens are part of the same sequence). Read both `peak GPU memory during denoising` figures, and if a
  job runs out of memory restart the gateway with `H3_PARALLELISM=split`. The VAE-encode step still runs next to the whole
  transformer (about 72 GiB), as it did before the block split; reference videos make it heavier. Also unmeasured: the speedup
  itself (expected 1.6-1.9x on denoising, from published two-GPU results for video transformers, not from this cluster).
- **Host RAM:** both halves hold their weights in RAM while loading. If a job is killed for memory, raise `H3_MEM_PER_GPU`.
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

Not checked: the text encoder step, the transformer, the decode, the two-process timing, memory, and `sbatch --gres=gpu:2`.
