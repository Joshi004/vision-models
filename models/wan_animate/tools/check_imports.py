"""One-off GPU-node import/sanity check for the Wan-Animate v1 environment.
Not part of the shipped pipeline -- run once manually via srun to confirm
the venv built on the (GPU-less) login node actually works on a real H100
node. See the setup plan's gpu-import-check step.
"""

from __future__ import annotations

import sys

print("python:", sys.version)

import torch  # noqa: E402

print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available on this node!"
print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))

import flash_attn  # noqa: E402
from flash_attn import flash_attn_func  # noqa: E402

print("flash_attn:", flash_attn.__version__)

# Exercise flash_attn_func for real on this GPU (not just import it).
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, k, v)
print("flash_attn_func smoke call OK, output shape:", tuple(out.shape))

import sam2._C as _C  # noqa: E402, F401
from sam2.build_sam import _load_checkpoint  # noqa: E402, F401

print("sam2 + compiled CUDA extension import OK")

import decord  # noqa: E402

print("decord:", decord.__version__)

import onnxruntime as ort  # noqa: E402

print("onnxruntime:", ort.__version__, "providers:", ort.get_available_providers())

import peft  # noqa: E402
from diffusers import FluxKontextPipeline  # noqa: E402, F401

print("peft:", peft.__version__, "diffusers FluxKontextPipeline import OK")

# The real test: the `wan` package itself. wan/modules/t5.py evaluates
# torch.cuda.current_device() as a class-body default argument at *import*
# time, so this package cannot be imported at all without a live CUDA
# device -- confirmed failing on the login node earlier. This is why this
# check has to run on a real GPU node.
sys.path.insert(0, "Wan2.2")
import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402

print("wan package import OK; tasks:", list(WAN_CONFIGS.keys()))
print("WanAnimate class:", wan.WanAnimate)

print("\nALL CHECKS PASSED")
