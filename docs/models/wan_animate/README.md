# ModelService_Wan-Animate-2 — Starting Document

**Status: v1 (replace mode) is set up, verified end to end, and wired into
`model-api`.** Lives at [`v1/`](v1/) (code, venv, ~68 GB of weights) — see
§2.1 for exactly what was done, the two real gotchas hit along the way, and
real measured timing/memory numbers. Callable today via
`POST /v1/wan-animate/videos/replace` — see
[`model-api/services/wan_animate/`](../model-api/services/wan_animate/) and
`model-api/STATUS.md`. **v2 is still just this document** — nothing for it
has been downloaded, installed, or run; §3 below is a plan, not a record.
This document exists so whoever picks this up next (possibly future-you)
doesn't have to redo the research that led here.

## Why this folder exists

Came out of a specific need: **swap a different character into an existing video**
(e.g., a dance clip) while keeping the original motion, timing, and background —
not generating a new video from a text prompt. That's a different task from
`ltx-2.3` and the abandoned MAGI-2 plan (see `model-api/VIDEO_MODEL_COMPARISON.md`
and `model-api/CHARACTER_SWAP_RESEARCH.md` for the full comparison that led here).

Wan-Animate is the model family that came out on top of that research: open-weight,
actually released (not a paper-only preview), purpose-built for exactly this,
Apache 2.0 licensed, and actively maintained by a serious team (Alibaba's Tongyi
Lab / Wan-Video).

## There are two versions — and they are NOT the same tool for our use case

**Resolved (this was an open question in the first draft of this document, now
confirmed from multiple independent sources including the official ComfyUI docs
and a direct third-party tutorial that specifically covers this confusion):**

**Wan-Animate-2 ("v2") dropped the "replace into the original background" mode
entirely.** It only does what the community calls "Move" mode: reference character
+ driving video → a new clip of that character performing the motion, with a
**freshly generated background from your text prompt** — not the original video's
background. Per ComfyUI's own docs: *"generate a fresh background from your text
prompt instead of copying the driving video's background."*

**Only Wan2.2-Animate-14B ("v1") still has the mode we actually need** — called
"replace" (v1's own naming) or "Mix" (community shorthand): reference character +
*existing* video → the same video, same background, same camera, different person.

| | **Wan2.2-Animate-14B ("v1")** | **Wan-Animate-2 ("v2")** |
|---|---|---|
| Released | Sep 19, 2025 | Aug 7, 2026 |
| Repo | Inside [`Wan-Video/Wan2.2`](https://github.com/Wan-Video/Wan2.2) | Its own repo: [`Wan-Video/Wan-Animate-2`](https://github.com/Wan-Video/Wan-Animate-2) |
| Weights | [`Wan-AI/Wan2.2-Animate-14B`](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) (+ `-Diffusers`) | [`Wan-AI/Wan2.2-Animate-2-14B`](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B) (+ `-Diffusers` / `-Distilled-Diffusers`) |
| **Preserves the original video's background?** | **✅ Yes — "replace" mode, this is the whole point of that mode** | **❌ No — always generates a fresh background from your prompt** |
| Motion input | Needs a separate preprocessing script (pose/face/mask extraction) | Skips that — consumes the driving video directly in the model |
| Confirmed single-GPU support | **Yes**, for both modes | Not shown — official docs say "tuned for 8× A800... tested on 2× A800," no 1-GPU example |
| Prompt format | Plain text | Structured **Chinese-language** caption via an LLM step (see §3) |
| Known hard limits | Character-mask extraction is **single-person videos only** (official warning) | N/A for our use case (see above) |
| Ecosystem | Diffusers (`WanAnimatePipeline`), ComfyUI | Diffusers (`WanAnimate2Pipeline`), ComfyUI, DiffSynth-Studio, and a real-time **Lite** variant (confirmed released, not just promised — see Comfy-Org's repackaged files) |
| License | Apache 2.0 | Apache 2.0 |

**What this means in practice: v2 is not "v1 but better" for our task — it's a
different tool for a different job.** v2 is aimed at avatars/mascots/streaming
hosts performing on a *new* backdrop; v1 is aimed at editing *existing* footage.
Newer doesn't mean strictly better here — it narrowed scope in a direction that
happens not to be ours.

**Recommendation: use v1 (`Wan2.2-Animate-14B`), replacement mode. Not "pilot it
first" — it's the only one of the two that does what was asked for.** v2's
real-time Lite variant might become relevant later for a genuinely different
use case (e.g., a live avatar), which is the only reason this folder still tracks
it at all.

**Why not just use v2 anyway, since it's newer/better?** Because its improvements
(direct video-conditioning instead of pose-skeleton extraction, viewpoint control,
real-time Lite mode) are all upgrades to the *animate-on-a-new-background* job —
that's the exact use case named in its own paper's abstract (digital avatars,
live-streaming hosts). Its published pipeline signature has no background/mask
input at all (`image, driving_video, prompt` — compare v1's
`image, pose_video, face_video, background_video, mask_video, mode="replace"`),
and the architecture paper describes the driving video as the *only* reference
input structurally, not a missing flag. Faking it by compositing v2's output onto
the original footage yourself doesn't really work either: v2's camera control is
12×4 discrete angles chosen by text description, not a track of the source
video's actual pans/zooms — so the generated clip's camera wouldn't match the
original's, and you'd end up rebuilding v1's job on a worse-fitting tool. The
honest tradeoff: v1 uses an older, somewhat more error-prone motion-transfer
mechanism (explicit pose skeletons — the same body-proportion-mismatch risk noted
above), but it's the one that structurally has the feature we need. v2 is more
polished at a job that isn't this one.

---

## 1. What it actually does

v1's two modes (v2 only has the equivalent of the first one — see above):

- **Animate mode** (both versions): character reference image + a driving video →
  a new video of *that character* performing the driving video's motion, on a
  generated background (not the original one).
- **Replace mode** (v1 only): character reference image + an *existing* video →
  the same existing video, with the original person swapped out for the reference
  character — background, camera motion, and lighting preserved. **This is the
  dance-video use case.** v1 also applies a dedicated "Relighting LoRA" here so the
  new character's lighting/color tone matches the original scene instead of
  looking pasted in.

Inputs needed either way: one clear reference image of the target character, plus
the driving/source video.

## 2. Setup — v1 (Wan2.2-Animate-14B), the recommended starting point

Straight from the official repo/model card (verified directly, not secondhand):

```bash
# 1. Get the code (this is inside the main Wan2.2 repo, not a separate one)
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# 2. Environment (torch >= 2.4.0 required)
pip install -r requirements.txt
# If flash_attn fails to install, install everything else first, flash_attn last

# 3. Download the Animate-14B weights specifically
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B
```

**Before generation, inputs must be preprocessed** (pose/face extraction) — this is
a real, required step, not optional:

```bash
# For replace mode specifically:
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path <your_dance_video.mp4> \
    --refer_path <your_character_reference.jpg> \
    --save_path ./process_results \
    --resolution_area 1280 720 \
    --iterations 3 --k 7 --w_len 1 --h_len 1 \
    --replace_flag
```

**Then generate** (single GPU, confirmed working per official docs):

```bash
python generate.py --task animate-14B \
    --ckpt_dir ./Wan2.2-Animate-14B/ \
    --src_root_path ./process_results/ \
    --refert_num 1 \
    --replace_flag --use_relighting_lora
```

A multi-GPU version exists too (`--dit_fsdp --t5_fsdp --ulysses_size 8`, tested up to
8 GPUs) — purely a speed optimization per the docs, not required to get results.

**One documented compatibility note:** the official docs specifically advise
*against* using LoRAs trained on plain Wan2.2 with Wan-Animate — "weight changes
during training may lead to unexpected behavior."

**A cleaner alternative to the raw scripts above: v1 also has a `diffusers`
pipeline**, confirmed working from a real usage example on the model's own HF
discussion page:

```python
import torch
from diffusers import WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = WanAnimatePipeline.from_pretrained(
    "Wan-AI/Wan2.2-Animate-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Inputs still come from the same preprocess_data.py step above.
replace_video = pipe(
    image=load_image("src_ref.png"),
    pose_video=load_video("src_pose.mp4"),
    face_video=load_video("src_face.mp4"),
    background_video=load_video("src_bg.mp4"),
    mask_video=load_video("src_mask.mp4"),
    prompt="People in the video are doing actions.",   # plain English, no template needed
    mode="replace",
    guidance_scale=1.0,
    num_inference_steps=20,
).frames[0]
export_to_video(replace_video, "output.mp4", fps=24)
```

Note this still needs the same preprocessing step (pose/face/background/mask
videos) — the `diffusers` path just replaces the `generate.py` CLI call, not the
preprocessing.

## 2.1 What we actually did on this cluster (verified, not just planned)

Went with the raw-scripts path above (not the `diffusers` pipeline), since
preprocessing needs the official repo either way. Everything lives under
[`v1/`](v1/): `Wan2.2/` (the cloned repo, pinned to commit `1ea34ff4`),
`.venv/` (Python 3.11, torch 2.8.0+cu128), `models/Wan2.2-Animate-14B/`
(~68 GB — the official `hf download` command above, unmodified), and
`run_replace.py`, a wrapper this API actually calls (see §6) that runs
preprocessing → generation → an audio re-mux step as one job, since the
model itself never touches audio at all (see §2.1.3).

### 2.1.1 Two real gotchas the official docs don't mention

**Neither of these is a Wan-Animate problem specifically** — both are bugs/
quirks in dependencies it happens to pull in — but both were real blockers
until diagnosed:

1. **SAM2's own `pip install` is missing its config files.** Its
   `sam2/__init__.py` registers a Hydra config search path pointing at a
   `sam2_configs` package, but that package's `setup.py` never declares its
   YAML files as `package_data` — so a plain `pip install` from the repo
   installs an *empty* `sam2_configs` package (just `__init__.py`, no
   YAMLs), and any preprocessing call that touches SAM2 (i.e. every
   replace-mode call) fails immediately with
   `hydra.errors.MissingConfigException: Cannot find primary config
   'sam2_hiera_l.yaml'`. **Fix:** manually copy the 4 YAML files from
   [the repo's own `sam2_configs/`
   directory](https://github.com/facebookresearch/sam2/tree/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2_configs)
   into the installed package's directory after installing. Not a code
   change — just restoring files a correct install should have shipped.
2. **Plain CPU `onnxruntime` (what `requirements_animate.txt` actually
   specifies) makes pose extraction impractically slow.** Preprocessing
   runs a YOLO detector + a "Huge" ViTPose backbone per frame; on CPU, a
   real test on this cluster was still stuck on frame-level pose
   extraction after 13.5+ minutes with zero output files, confirmed
   genuinely busy (not hung) via `ps`/`/proc` — just far too slow for
   interactive use. **Fix:** installed `onnxruntime-gpu==1.22.0` instead
   (pinned specifically — the latest 1.30.x targets CUDA 13, which doesn't
   match this cluster's torch/CUDA 12.8 stack). That single swap cut
   preprocessing to under 2 minutes.

A smaller third thing, not really a "gotcha" so much as an upstream
oddity worth knowing: `wan/__init__.py` unconditionally imports the
speech-to-video (S2V) submodule even though this project only ever calls
`WanAnimate` — so importing the package at all requires `librosa` (used by
S2V's audio encoder), even though `librosa` isn't declared anywhere in the
main `requirements.txt` and S2V's own separate, officially-"optional"
`requirements_s2v.txt` is not otherwise needed. Installed just `librosa`
rather than that entire file — everything else S2V's import chain needs
(`transformers`, `einops`, `diffusers`) was already present.

### 2.1.2 Real measured numbers (answers README open question 3 below)

From a full end-to-end run on this cluster — one H100, the official repo's
own replace-mode example inputs (a 6.8s, 1280×720 driving video) — measured
twice: once as a standalone `sbatch` smoke test, once as a real job
submitted through `model-api`'s HTTP API. Both matched closely:

| Phase | Time |
|---|---|
| Preprocessing (pose/face/background/mask extraction, GPU-accelerated ONNX) | ~90–115s |
| Model loading (T5-XXL, CLIP, VAE, the 14B DiT's 4 shards, relighting LoRA) | ~140s |
| Denoising | ~19.4s/step × 20 steps × 3 clips (77-frame segments, `refert_num=1` overlap) ≈ 19.5 min |
| Audio re-mux | <1s |
| **Total wall clock** | **~25 minutes** |

Peak GPU memory: **~74.6 GB of 80 GB** — tight but not OOMing (single-GPU
runs get `offload_model=True` automatically). Peak host RAM: **~40.6 GB**.
**Cost scales with the driving video's length** (more 77-frame segments),
so a longer clip will cost proportionally more of both time and (mildly)
memory. A 25-minute job is long enough to make this cluster's default
pre-emptible `background` partition a real, non-trivial risk for this
specific endpoint — see the API's own docs for the `partition` override.

The visual result of that one run looked correct on manual inspection: the
reference character was convincingly swapped into the original market
scene, same pose/camera/background, relit to match the scene's cool
color grading rather than looking pasted in. One eyeballed example, not a
rigorous quality evaluation — see open question 1 below.

### 2.1.3 The audio gap

Confirmed directly from `generate.py`'s own code: `WanAnimate.generate()`
never touches audio at all — `save_video()` writes a silent MP4. The
official repo doesn't re-mux the source audio back on for animate/replace
tasks (unlike its S2V task, which does). `run_replace.py` does this itself
as a final step with `ffmpeg` (via the `imageio-ffmpeg`-bundled binary,
since no system `ffmpeg` exists on this cluster): copy the generated
video stream, take the source video's original audio track if it has one
(an *optional* `-map` so a source with no audio still produces a valid
silent output), re-encode that audio to AAC, trim to the shorter of the
two. This matches the "keep the original audio, only redo the picture"
approach `CHARACTER_SWAP_RESEARCH.md` already recommended for this use
case.

## 3. Setup — v2 (Wan-Animate-2), if/when you move to it

```bash
git clone --recursive https://github.com/Wan-Video/Wan-Animate-2.git
cd Wan-Animate-2

conda create -n wan_animate_2 python==3.11 -y
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .

huggingface-cli download Wan-AI/Wan2.2-Animate-2-14B --local-dir ./ckpts/
```

**The prompt-format gotcha, verbatim from the official README** — before running
generation, you're expected to run an LLM over your reference image with this exact
instruction to produce the actual prompt Wan-Animate-2 wants (Chinese, structured,
appearance + background only, no action description):

> 用中文客观描述图片中的内容，包括以下要点：人物外观描述，不描述动作行为。 背景描述，忽略主观评价和情绪推测。
> 下面给出描述范例，必须遵循这个范式，不要输出额外的符号： 人物外观描述：穿着一件浅蓝色的校服衬衫，领口和袖口有白色边饰。
> 胸前有一个圆形徽章。 背景描述：背景为明亮、整洁的教室或办公室，氛围安静有序。

(Translation of the instruction: "Objectively describe the image's content in
Chinese: character appearance — not actions/behavior — and background, ignoring
subjective judgments or emotional inference. Follow the given example format
exactly, no extra symbols.") They suggest using Qwen3.7-Plus to generate this
caption. **This is not optional-looking from the docs** — worth confirming with a
real test whether an English caption in the same structure works at all before
assuming you must go through a Chinese-language LLM step.

Then, the simplest path is actually the `diffusers` library, not the raw scripts:

```python
import torch
from diffusers import WanAnimate2Pipeline
from diffusers.utils import export_to_video, load_image

pipe = WanAnimate2Pipeline.from_pretrained(
    "Wan-AI/Wan2.2-Animate-2-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

output = pipe(
    image=load_image("reference.png"),
    driving_video="template.mp4",
    prompt="<the structured caption from the step above>",
    height=800, width=640,
    num_inference_steps=40,
)
export_to_video(output.frames[0], "output.mp4", fps=24)
```

A distilled checkpoint (`Wan2.2-Animate-2-14B-Distilled-Diffusers`) runs in 10 steps
with no classifier-free guidance (`guidance_scale=1.0`) — much faster, presumably
some quality tradeoff (not measured by us yet).

There's also an official hosted demo to sanity-check quality **before spending any
of our own GPU time**: https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate

## 4. Open questions, updated

~~1. Does v2 still have an explicit "replace into the original background" mode?~~
**Resolved: no.** Confirmed via ComfyUI's own docs and independent tutorials
specifically written to clear up this exact confusion between the two versions.
v2 = fresh background always. Use v1 for background-preserving swaps.

~~3. Exact single-GPU memory/time cost for v1's replace mode on our actual H100s?~~
**Resolved, with real numbers:** ~25 minutes wall clock, ~74.6 GB peak VRAM (of
80 GB), ~40.6 GB peak host RAM, for a 6.8s/1280×720 driving video on one H100 —
see §2.1.2 for the full breakdown. Confirmed via two independent runs (a
standalone smoke test and a real job through `model-api`) that matched closely.
Cost scales with driving-video length, not tested yet at durations much longer
than ~7s.

Remaining open questions, still worth a real test rather than more reading:

1. How good is v1's replace-mode quality on a real dance video with fast, complex
   multi-limb motion, rather than the calmer demo footage shown officially? The
   one real run done so far (§2.1.2) used the official repo's own demo clip and
   looked correct on manual inspection — encouraging, but not yet tested against
   our own actual footage or fast/complex motion.
2. Our dance video may have **more than one dancer in frame** — v1's official
   preprocessing docs explicitly warn that mask extraction "is designed for
   single-person videos ONLY and may produce incorrect results or fail in
   multi-person videos." If that's the case here, we'd need our own solution for
   isolating the target person first (a separate segmentation step) before v1's
   pipeline can even start. Worth checking the actual source video for this before
   anything else. (The API layer does not check or enforce this today — see
   `model-api/services/wan_animate/openapi_docs.py`.)
3. Does v2's Lite/real-time variant have any future use for us beyond this
   specific task (e.g., a live avatar use case)? Not needed now, just tracked so
   this folder doesn't need to be rebuilt from scratch if that comes up later.

## 5. Why not VACE (another name that comes up in this space)

`VACE` (`Wan2.1-VACE-1.3B` / `-14B`, Apache 2.0) is a broader, general-purpose
video editing model from the same team — inpainting, outpainting, background
replacement, pose/depth/canny-guided control. It can technically do motion
transfer too. **Deliberately not recommended for this task**, on the Wan team's
own direct comparison (their Wan-Animate paper benchmarks against it): *"VACE has
issues with identity consistency. Furthermore, its general-purpose nature makes it
highly dependent on parameter tuning, resulting in a higher barrier to entry. In
contrast, Wan-Animate is much more user-friendly and performs better in character
replacement."* Worth remembering as a *different* tool for a *different* future
need (e.g., removing an object from a video, or extending a frame outward) — not
a substitute for Wan-Animate here.

## 6. How this fits into the bigger picture

**v1 is wired into `model-api/`** (the shared job-submission API), the same
shape LTX-2.3 uses: a `services/wan_animate/` module (`config.py`,
`schemas.py`, `dispatch.py`, `router.py`, `openapi_docs.py`) that builds a
`run_replace.py` invocation and submits it as a one-GPU Slurm job, mounted
at `POST /v1/wan-animate/videos/replace`. No changes were needed to
`common/` or to LTX's own code — see `model-api/STATUS.md` for the
service-level summary and `model-api/services/wan_animate/openapi_docs.py`
for the endpoint's full documented behavior (including the real timing
numbers from §2.1.2 and the pre-emption-risk note that follows from them).
Job submission/tracking/upload/download all reuse the exact same shared
endpoints LTX-2.3 already uses (`POST /v1/uploads`, `GET /v1/jobs/{id}`,
etc.) — nothing new there.

**A naming note, now resolved:** this folder is called
`ModelService_Wan-Animate-2`, but the model actually in use (per the
research above) is **v1** — which is why it lives at [`v1/`](v1/), a
subfolder, rather than at this folder's own root. `v2/` is reserved for
Wan-Animate-2 itself (§3), if/when its Lite variant earns a place here for
a different use case (e.g. a live avatar). Each gets its own venv/weights/
`services/<name>/` module — they're different pipelines, not two modes of
one setup.

## 7. Primary sources (all fetched and verified directly, not secondhand)

- v1 code: https://github.com/Wan-Video/Wan2.2
- v1 weights: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- v1 diffusers port: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers
- v1 preprocessing guide: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md
- v1 paper (incl. head-to-head vs. VACE/Animate Anyone/Runway Act-two/DreamActor-M1): https://arxiv.org/abs/2509.14055
- v2 code: https://github.com/Wan-Video/Wan-Animate-2
- v2 weights: https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B
- v2 paper: https://arxiv.org/abs/2608.06009
- Project page: https://humanaigc.github.io/wan-animate-2/
- VACE weights: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- Move-vs-Mix explainer (independent, specifically resolves the v1/v2 confusion): https://www.seedance.tv/blog/wan-animate-2-tutorial
