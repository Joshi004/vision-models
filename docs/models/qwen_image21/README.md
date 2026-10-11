# Qwen-Image-2.1

Image generation and editing with [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Alibaba Qwen, released 2026-09-20; 7B visual generation component, Qwen3-VL 8B text encoder). Served on demand by the gateway at `/v1/qwen-image21/images/generate` and `/v1/qwen-image21/images/edit` (see `gateway/services/qwen_image21/GUIDE.md`). Each request is one Slurm GPU job; there is no always-on service.

## What it does

| Endpoint | Inputs |
|---|---|
| `images/generate` | `prompt`, `mode`, `aspect_ratio` (default `1:1`) |
| `images/edit` | the same plus 1-10 reference images; edits one image or composes several |

- `mode: fast` loads the **Turbo** checkpoint (`Qwen/Qwen-Image-2.1-Turbo`, saved 8-step schedule). `mode: quality` loads the **base** checkpoint (`Qwen/Qwen-Image-2.1`, 40 steps by default). Different checkpoints: the same seed gives different images.
- Native 2K output. Aspect ratios: 1:1 = 2048x2048, 4:3 = 2400x1792, 3:4 = 1792x2400, 3:2 = 2528x1696, 2:3 = 1696x2528, 16:9 = 2752x1536, 9:16 = 1536x2752.
- **Transparent (RGBA) output** is chosen by the prompt, not a parameter: `This is an RGBA image with transparency. <description>. The image has alpha channel and the background is transparent.` The pipeline always returns RGBA; the runner saves an ordinary result (alpha minimum 240 or above) as RGB and keeps real transparency as RGBA.
- Not wired in: the official prompt rewriter (extra 9B model), separate mask / circle inputs for local edits, vLLM-Omni / SGLang serving, FP8.

## Layout

```
models/qwen_image21/
  download_weights.sh    fetches both checkpoints (neither is gated)
  env/requirements.lock.txt   frozen environment (python 3.12, torch 2.9.1+cu128, transformers 5.19.0, diffusers from git)
  runner/run_generate.py      entrypoint the gateway's Slurm job runs
$VISION_DATA_DIR/envs/qwen_image21/         built with scripts/build_env.sh qwen_image21
$VISION_DATA_DIR/weights/qwen_image21/{base,turbo}   ~31 GB each
```

There is no `third_party/` checkout: the model runs through diffusers' `QwenImage21Pipeline`, which was not in a diffusers release when this was set up, so the lock installs diffusers from git at commit `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458`. The pipeline also needs `torchvision` (Qwen3-VL's video processor imports it).

## Setup

```bash
scripts/build_env.sh qwen_image21
models/qwen_image21/download_weights.sh
```

## Manual run (on a GPU node)

```bash
$VISION_DATA_DIR/envs/qwen_image21/bin/python models/qwen_image21/runner/run_generate.py \
    --model-dir $VISION_DATA_DIR/weights/qwen_image21/turbo --mode fast \
    --prompt="A capybara reading a book by candlelight" --output-path /tmp/out.png
```

## Measured on this cluster (H100 80GB, 2026-10-10)

| Case | Peak GPU memory | Generation time | Notes |
|---|---|---|---|
| text-to-image, 2048x2048, fast | 56.5 GiB | 10 s | whole job ~135 s, mostly loading ~31 GB of weights (65 s on a cold node) |
| text-to-image, 2048x2048, quality (40 steps) | 56.5 GiB | 39 s | whole job ~105 s |
| edit, 1 reference, 2048x2048, fast | 64.6 GiB | 89 s | |
| edit, 2 references @2048, 2528x1696 | 73.2 GiB | 170 s | works, but too close to the 80 GB limit; the same job without `expandable_segments` ran out of memory in the VAE decode |
| edit, 2 references @1536, 2528x1696 | 66.1 GiB | 97 s | what the runner uses for 2 images |
| edit, 4 references @1024, 2528x1696 | 65.1 GiB | 87 s | what the runner uses for 3-4 images |
| edit, 10 references @768, 2528x1696 | 68.4 GiB | 118 s | what the runner uses for 5-10 images |
| edit, 10 references @2048 (the pipeline default) | not measured | over 11 min, cancelled | impractical |

Everything runs with no CPU offload on one 80 GB GPU. The runner sets `PYTORCH_ALLOC_CONF=expandable_segments:True` and shrinks the reference images as their number grows (`default_condition_resolution` in `run_generate.py`), which is the one real quality trade-off: with several references, fine detail from them is preserved less well.

## Data retention

Results follow the gateway's retention rules (see the main README): by default an uploaded reference image and the generated PNG are kept until the job is purged or expires. A request sent with `incognito: true` is the exception: its reference images are deleted when its job ends and the generated PNG is deleted right after it has been downloaded once.

## License

**Qwen Research License Agreement: research and non-commercial use only.** Commercial use needs a separate agreement with Hangzhou Tongyi Laboratory. (The earlier Qwen-Image was Apache 2.0; 2.1 is not.) The code in this repository does not enforce this; whoever runs the deployment is responsible for how it is used.
