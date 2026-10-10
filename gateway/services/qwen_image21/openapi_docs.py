"""Long-form OpenAPI/Swagger documentation text for Qwen-Image-2.1's
endpoints, kept out of router.py so the routes themselves stay readable.
Docs shared by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.qwen_image21 import config

_ACCEPTED_FORMATS_LIST = ", ".join(sorted(ext.lstrip(".") for ext in config.REFERENCE_IMAGE_EXTENSIONS))
_SIZES_TABLE = "\n".join(f"| `{name}` | {w}x{h} |" for name, (w, h) in config.ASPECT_RATIOS.items())

_RGBA_WORDING = (
    "`This is an RGBA image with transparency. <your description>. "
    "The image has alpha channel and the background is transparent.`"
)

_COMMON_NOTES = f"""
**Two modes** (`mode`): `fast` (default) runs the Turbo checkpoint in 8
denoising steps; `quality` runs the base checkpoint in
{config.DEFAULT_QUALITY_STEPS} steps (change with `num_inference_steps`, 1-{config.MAX_QUALITY_STEPS}, quality
mode only). They are different checkpoints, so **the same seed gives a
different image in each mode**.

**Transparent images:** the model decides between a normal and a transparent
(RGBA) image from the prompt wording. For a transparent result, write the
prompt as {_RGBA_WORDING} The PNG then has a real alpha channel.

**Result:** `output.png`, downloaded with `GET /v1/jobs/{{job_id}}/result`. The
same inputs and `seed` give the same image on the same hardware.

**Timing:** each request is a one-shot GPU job that loads about 31 GB of
weights before generating, so expect a few minutes end to end including
queueing. No live progress: `progress` is always null.

**License:** Qwen-Image-2.1 is released under the **Qwen Research License,
non-commercial use only**; commercial use needs a separate agreement with the
model's authors. Whoever operates this deployment is responsible for how it
is used; this API does not enforce that restriction itself.
"""

GENERATE = f"""
Generates an image from a text prompt with **Qwen-Image-2.1** (Alibaba Qwen;
7B parameters, strong at typography, portraits and fine detail).

`aspect_ratio` picks one of the model's native 2K sizes (default `1:1`):

| `aspect_ratio` | Output size |
|---|---|
{_SIZES_TABLE}
{_COMMON_NOTES}
"""

EDIT = f"""
Edits an image, or composes a new one from several, with **Qwen-Image-2.1**.
Send 1 to {config.MAX_REFERENCE_IMAGES} reference images and a `prompt` that says what to do with them
-- change one image, or combine subjects ("these three characters sitting
around a campfire"). With several images, refer to them in order in the
prompt. Identity of people and products is preserved well.

Upload each reference first with `POST /v1/uploads` and pass the returned
`asset_id`s in `reference_image_asset_ids`. Accepted formats:
{_ACCEPTED_FORMATS_LIST}.

`aspect_ratio` is optional here. When omitted, the result follows the **last**
reference image's aspect ratio, at about 4 megapixels (the size of a 2048x2048
image) for one image and smaller for several. When given, it is one of:

| `aspect_ratio` | Output size |
|---|---|
{_SIZES_TABLE}

**Several references cost detail.** Every reference image adds work and GPU
memory, so with more than one image the references are resized smaller before
the model sees them (1 image: full size; 2: about 1536x1536 area; 3-4: about
1024x1024; 5-10: about 768x768). The output size is unaffected when you pass
`aspect_ratio`. Fine detail from the references (small text, textures) is
better preserved with fewer images.

Local, region-specific changes are described in words in the prompt (there is
no separate mask field).
{_COMMON_NOTES}
"""

GENERATE_EXAMPLES = {
    "fast": {
        "summary": "Fast (Turbo, 8 steps)",
        "value": {
            "prompt": "A capybara reading a book by candlelight, photorealistic.",
            "mode": "fast",
            "aspect_ratio": "16:9",
        },
    },
    "quality": {
        "summary": "Quality (base checkpoint, 40 steps)",
        "value": {
            "prompt": "A ceramic teapot on a wooden table, soft window light, photorealistic.",
            "mode": "quality",
            "aspect_ratio": "1:1",
            "num_inference_steps": 40,
        },
    },
    "transparent": {
        "summary": "Transparent (RGBA) background",
        "value": {
            "prompt": (
                "This is an RGBA image with transparency. A cute cartoon dragon sticker. "
                "The image has alpha channel and the background is transparent."
            ),
            "mode": "fast",
        },
    },
}

EDIT_EXAMPLES = {
    "single": {
        "summary": "Edit one image",
        "value": {
            "prompt": "Change the candlelight to bright morning sunlight coming through a window.",
            "reference_image_asset_ids": ["<asset_id from POST /v1/uploads>"],
            "mode": "fast",
        },
    },
    "multi": {
        "summary": "Compose from several references",
        "value": {
            "prompt": "These three characters are sitting around a campfire in a forest.",
            "reference_image_asset_ids": [
                "<asset_id 1 from POST /v1/uploads>",
                "<asset_id 2>",
                "<asset_id 3>",
            ],
            "mode": "quality",
            "aspect_ratio": "16:9",
        },
    },
}

GENERATE_RESPONSES = {**common_docs.SUBMIT_RESPONSES}

EDIT_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": "A reference image's asset_id doesn't exist, or its file isn't an accepted format.",
        "content": {
            "application/json": {
                "examples": {
                    "unknown_asset": {
                        "summary": "Unknown or missing asset_id",
                        "value": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"},
                    },
                    "unsupported_format": {
                        "summary": "Unsupported reference file type",
                        "value": {
                            "detail": (
                                f"Unsupported file extension '.xyz' (accepted: "
                                f"{', '.join(sorted(config.REFERENCE_IMAGE_EXTENSIONS))}): reference.xyz"
                            )
                        },
                    },
                }
            }
        },
    },
}
