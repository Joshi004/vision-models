## Qwen-Image-2.1 image generation

**What this is for:** generating and editing still images with Alibaba Qwen's Qwen-Image-2.1 (7B). One model
covers text-to-image, image editing, multi-reference composition (up to 10 images) and transparent (RGBA)
output. It is especially good at legible text in images, portraits and product images.

**Two endpoints:**
- `POST /v1/qwen-image21/images/generate` -- text-to-image. `prompt`, `aspect_ratio` (default `1:1`).
- `POST /v1/qwen-image21/images/edit` -- edit one image, or compose from several. Same fields plus
  `reference_image_asset_ids` (1 to 10 `asset_id`s from `POST /v1/uploads`). `aspect_ratio` is optional; when
  omitted the result follows the last reference image's shape.

**Two modes** (`mode`, both endpoints):
- `fast` (default) -- the Turbo checkpoint, 8 denoising steps.
- `quality` -- the base checkpoint, 40 steps by default (`num_inference_steps` can change it; it is rejected
  with `fast`).
They are different checkpoints, so the same `seed` gives a different image in each. Don't compare them
seed-for-seed.

**Sizes:** native 2K. `aspect_ratio` is one of `1:1` (2048x2048), `4:3` (2400x1792), `3:4` (1792x2400),
`3:2` (2528x1696), `2:3` (1696x2528), `16:9` (2752x1536), `9:16` (1536x2752).

**Transparent images:** the prompt decides. Write it as
`This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent.`
and the PNG comes back with a real alpha channel. Without that wording you get an ordinary opaque image.

**Prompts:** long, specific prompts work best (the model's own examples run to thousands of characters). Put
any text that should appear in the image in quotes. For edits, say what to change and what to keep; with
several reference images, refer to them in order. With several reference images they are resized smaller before the model sees them (2 images ~1536x1536
area, 3-4 ~1024x1024, 5-10 ~768x768), so fine detail from the references survives better with fewer
images. There is no `negative_prompt` and no mask field -- describe
local edits in words.

**Output:** a PNG (`output.png`, RGB or RGBA) via `GET /v1/jobs/{job_id}/result`. Deterministic for the same
inputs and `seed` on the same hardware.

**Timing:** requests are run by long-lived GPU jobs, one per GPU, that load about 31 GB of weights once and then
run requests one after another. A request that has to start one takes minutes (about 5 minutes to load, plus a slower
first image), even in `fast` mode; a request that finds one already loaded and free starts at once and takes about
the generation time (tens of seconds to a couple of minutes, more for edits with many references). Up to 8 run at
once (operator setting `QWEN_IMAGE21_MAX_WORKERS`), so with many requests at once some wait `queued` for a free one;
a GPU job with nothing to do for 10 minutes is stopped (`QWEN_IMAGE21_IDLE_MINUTES`). `fast` and `quality` use
different checkpoints and so different GPU jobs. Cancelling a request that is already running stops its GPU job, so
the next request starts cold. No live progress (`progress` is always null).

**License:** Qwen-Image-2.1 is under the **Qwen Research License -- non-commercial use only**. Commercial use
needs a separate agreement with the model's authors. Whoever operates this deployment is responsible for how it
is used; this API does not enforce that restriction itself.
