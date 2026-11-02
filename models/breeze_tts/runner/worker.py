#!/usr/bin/env python3
"""A warm Breeze TTS 2 worker: loads the model onto the GPU once, then serves
text-to-speech requests from the gateway one after another (see
`gateway/services/breeze_tts/dispatch.py` and `gateway/common/worker_pool.py`, which
start and feed it, and `models/common/pool_worker.py`, the request loop).

Usage (started by the gateway, with this model's own environment):
    worker.py --worker-dir DIR --model-dir DIR --idle-exit-seconds N

Each request is a job folder whose args.json holds the same argv a one-shot job runs
([python, run_synthesize.py, flags...]). `run_synthesize.py`'s own `_parse_args` and
`prepare` check it and decode the reference; the generation itself is the same sequence
of calls as upstream's `infer.py` (eager mode, the same seeding), run here in-process
on the model loaded once, so the same request and seed give the same audio as a one-shot
job. A request that fails ends the worker: the next request gets a new one, so a failure
such as running out of GPU memory never leaves a damaged process serving.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(RUNNER_DIR.parents[1] / "common"))  # models/common/pool_worker.py

import pool_worker  # noqa: E402
import run_synthesize  # noqa: E402

# Upstream's packages (breeze_infer, models) and infer.py, imported from its own folder as
# infer.py itself would see them.
sys.path.insert(0, str(run_synthesize.UPSTREAM_DIR))

# The id infer.py gives its one request; kept the same so nothing differs from a one-shot job.
_REQUEST_ID = "single-request"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker-dir", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--idle-exit-seconds", required=True, type=float)
    worker_args = p.parse_args()

    state = pool_worker.WorkerState(worker_args.worker_dir)
    state.write("loading")

    import infer  # upstream's infer.py: its generation limits
    import soundfile as sf
    import torch
    from breeze_infer.runtime import load_runtime, resolve_device, set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    tokenizer, model, audio_tokenizer = load_runtime(
        worker_args.model_dir, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=infer.MAX_NEW_TOKENS,
            max_seq_len=infer.MAX_SEQ_LEN,
            fast_all=None,
            fast_text_encoder=False,
            fast_backbone_prefill=False,
            fast_backbone_decode=False,
            fast_depth_decoder=False,
            fast_codec=False,
            repetition_penalty=infer.REPETITION_PENALTY,
        ),
        tokenizer=tokenizer,
    )

    def handle(argv: list[str]) -> None:
        # argv is [python, run_synthesize.py, flags...], as for a one-shot job.
        args = run_synthesize._parse_args(argv[2:])
        if Path(args.model_dir) != worker_args.model_dir:
            raise RuntimeError("The request points at a different model directory than this worker loaded.")
        try:
            reference_wav = run_synthesize.prepare(args)
            request = {"id": _REQUEST_ID, "text": args.text, "speaker": "S0"}
            instruction = args.instruction.strip() if args.instruction else None
            if instruction:
                request["instruction"] = instruction
            if reference_wav is not None:
                request["ref_audio_path"] = str(reference_wav)
                request["ref_text"] = args.ref_text.strip()

            set_all_seeds(args.seed)
            inputs = prepare_inputs(
                tokenizer,
                audio_tokenizer,
                model,
                [request],
                get_template(select_template_name(request)),
                guidance_scale=args.cfg_scale,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(
                output_path, mode="w", samplerate=runtime.sample_rate, channels=1, subtype="PCM_16"
            ) as output_file:
                for chunk in runtime.iter_audio_chunks(inputs, request_id=_REQUEST_ID, seed=args.seed):
                    output_file.write(chunk.audio)
            print(f"Synthesis completed: {output_path}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return pool_worker.serve(worker_args.worker_dir, handle, worker_args.idle_exit_seconds, state)


if __name__ == "__main__":
    sys.exit(main())
