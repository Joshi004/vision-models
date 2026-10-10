#!/usr/bin/env python3
"""Runs Qwen-Image-2.1 for one image request -- the single unit of work the
gateway's Slurm job submits for one `/images/generate` or `/images/edit`
request. See `gateway/services/qwen_image21/dispatch.py`, which builds the
argv below, and `gateway/common/run_pipeline_job.py`, which invokes this on
the compute node (that wrapper owns the partial-path-rename-on-success /
`.failed`-marker-on-failure convention; this script only needs to write
`--output-path` and exit non-zero on any real failure).

Must be run with this model's own environment
(`$VISION_DATA_DIR/envs/qwen_image21/bin/python`): diffusers (from git),
transformers 5.x and torch only exist there.

Two modes, matching the two checkpoints the caller points --model-dir at:
  * quality : base Qwen-Image-2.1, `--steps` denoising steps (default 40)
  * fast    : Qwen-Image-2.1-Turbo; its checkpoint carries its own saved
              8-step schedule, so `--steps` is ignored

Text-to-image when no --image is given, editing otherwise (up to 10
reference images; the pipeline treats the two the same way).

Usage:
    run_generate.py --model-dir DIR --mode fast|quality --output-path FILE \\
        --prompt=TEXT [--image FILE ...] [--width N --height N] \\
        [--steps N] [--seed N]

--prompt is passed as `--prompt=value` (not `--prompt value`) so a value
starting with "-" is never mistaken for an option. Nothing here goes through
a shell, so user text needs no escaping.

Size: --width and --height go together. Without them, text-to-image uses the
model's native 2048x2048 and an edit follows the last reference image's
aspect ratio, at about the same total area as the reference images are
resized to (see default_condition_resolution: smaller with more images, to
fit GPU memory).

Output is always PNG: RGB for an ordinary image, RGBA when the model produced
real transparency (see the RGBA prompt wording in the gateway's guide).

Exit code 0 only if --output-path was actually written.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

MAX_REFERENCE_IMAGES = 10
NATIVE_RESOLUTION = 2048  # the pipeline's own default is 1024; upstream documents 2048
OPAQUE_ALPHA_MIN = 240  # lowest alpha value still treated as "fully opaque"


def default_condition_resolution(image_count: int) -> int:
    """Side length whose area each reference image is resized to, by how many
    there are. Every reference adds tokens the transformer attends to, so GPU
    memory grows with the count; at the native 2048 it only fits one 80 GB GPU
    for one or two images. Measured on an H100 80GB, fast mode, ~4.3 MP output
    (peak GPU memory): 1 image @2048 = 64.6 GiB, 2 @2048 = 73.2 GiB (too close
    to the limit), 2 @1536 = 66.1 GiB, 4 @1024 = 65.1 GiB, 10 @768 = 68.4 GiB.
    """
    if image_count <= 1:
        return NATIVE_RESOLUTION
    if image_count == 2:
        return 1536
    if image_count <= 4:
        return 1024
    return 768


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path, help="Local diffusers checkpoint directory.")
    p.add_argument("--mode", required=True, choices=["fast", "quality"])
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", action="append", default=[], type=Path, help="Reference image (repeatable, max 10).")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int, default=40, help="Quality mode only.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--condition-resolution", type=int, help="Edit only: side length whose area each reference image is resized to (default: chosen from the image count).")
    args = p.parse_args()

    if (args.width is None) != (args.height is None):
        p.error("--width and --height must be given together.")
    if len(args.image) > MAX_REFERENCE_IMAGES:
        p.error(f"At most {MAX_REFERENCE_IMAGES} reference images are supported, got {len(args.image)}.")
    for path in args.image:
        if not path.is_file():
            p.error(f"Reference image not found: {path}")
    if not args.model_dir.is_dir():
        p.error(f"Model directory not found: {args.model_dir}")
    return args


def main() -> int:
    args = parse_args()

    # Less fragmentation at 2K: an out-of-memory failure here reported ~11 GiB reserved but unusable.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Imported late so --help and argument errors stay fast and torch-free.
    import torch
    from diffusers import QwenImage21Pipeline
    from PIL import Image

    started = time.time()
    print(f"Loading {args.model_dir} ({args.mode})", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to("cuda")
    print(f"Loaded in {time.time() - started:.0f}s", flush=True)

    call_kwargs: dict = {
        "prompt": args.prompt,
        "use_kv_cache": True,
        "generator": torch.Generator("cpu").manual_seed(args.seed),
    }
    if args.mode == "quality":
        call_kwargs["num_inference_steps"] = args.steps
    # fast: no num_inference_steps -- Turbo's saved 8-step schedule applies, and
    # upstream notes that setting num_inference_steps alone does not override it.

    if args.image:
        # RGBA, as in upstream's editing examples.
        call_kwargs["image"] = [Image.open(path).convert("RGBA") for path in args.image]
        call_kwargs["output_resolution"] = args.condition_resolution or default_condition_resolution(len(args.image))

    if args.width is not None:
        call_kwargs["width"] = args.width
        call_kwargs["height"] = args.height
    elif not args.image:
        call_kwargs["width"] = NATIVE_RESOLUTION
        call_kwargs["height"] = NATIVE_RESOLUTION

    generate_started = time.time()
    image = pipe(**call_kwargs).images[0]
    print(f"Generated {image.size[0]}x{image.size[1]} {image.mode} in {time.time() - generate_started:.0f}s", flush=True)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # The pipeline always returns RGBA. For an ordinary prompt the alpha channel
    # is opaque (measured minimum 250-253 of 255, i.e. rounding noise), so save
    # those as plain RGB; a real transparent result (minimum alpha 0) keeps it.
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] >= OPAQUE_ALPHA_MIN:
        image = image.convert("RGB")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output_path, format="PNG")
    if not args.output_path.is_file() or args.output_path.stat().st_size == 0:
        print(f"ERROR: {args.output_path} was not written.", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_path} (total {time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
