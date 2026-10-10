#!/usr/bin/env python3
"""Runs one Parakeet TDT transcription on a single audio/video file -- the
single unit of work model-api's Slurm job submits for one
`POST /v1/parakeet/transcribe` request -- see
`model-api/services/parakeet/dispatch.py`, which builds the argv below
(via `run_transcribe.sh`, which only adds the CUDA-library-path env fix),
and `model-api/common/run_pipeline_job.py`, which actually invokes this on
the compute node (that wrapper owns the partial-path-rename-on-success /
`.failed`-marker-on-failure convention; this script only needs to write
--output-path and exit non-zero on any real failure).

Deliberately does not import app.py or config.py (the legacy always-on
service's own modules): both have import-time side effects (app.py
constructs a FastAPI app and registers startup/shutdown events; config.py
creates /tmp directories) that make no sense for a one-shot Slurm job. The
chunking/merge/timestamp-extraction logic below is ported from app.py, not
imported from it -- segment_utils.py (pure functions, no side effects) is
the one module actually shared with it.

One deliberate behavior change from app.py: chunking is triggered by the
*decoded audio's duration* (> CHUNK_DURATION_SECONDS), not by the original
file's compressed size (> 50MB) -- a highly-compressed file long enough to
exceed the model's own ~24-minute full-attention limit could pass app.py's
size check without ever being chunked. Same chunk duration/overlap (600s /
10s) either way.

Usage:
    run_transcribe.py --input-path FILE --work-dir DIR \\
        --model-path FILE --output-path FILE

--input-path   The original recording (any format ffmpeg can decode -- see
               model-api/services/parakeet/config.py's INPUT_EXTENSIONS for
               what's actually offered to callers). May be audio or video;
               only its first audio stream is used.
--work-dir     Directory to write the decoded WAV (and, for long audio, its
               chunks) into -- already exists by the time this runs (it's
               the job's own job_dir) and is not cleaned up afterwards,
               same convention ModelService_RVC-Applio/run_convert.py uses
               for its own --work-dir.
--model-path   Path to the local parakeet-tdt-0.6b-v3.nemo checkpoint.
               Loaded via ASRModel.restore_from(), not from_pretrained(),
               so this never needs network access (every other backend's
               Slurm job already runs with HF_HUB_OFFLINE=1 -- see
               dispatch.py's _submit_slurm_job -- restore_from simply never
               needs that network path to begin with, since the .nemo file
               is a fully self-contained local archive).
--output-path  Where to write the transcript JSON. Intercepted here (not
               forwarded anywhere) purely so this script can explicitly
               verify it was written before declaring success -- defense in
               depth, not strictly required, since
               model-api/common/run_pipeline_job.py already checks this
               same path's existence on its own.

Output JSON shape (unchanged from the legacy app.py's TranscribeResponse,
so any downstream code already parsing that response keeps working):
    {
      "transcription": "...",
      "processing_time": 12.34,
      "word_timestamps": [{"word": "...", "start": 0.0, "end": 0.5}, ...],
      "segment_timestamps": [{"text": "...", "start": 0.0, "end": 2.5, "word_count": 5}, ...],
      "metadata": {"total_segments": 3, "total_words": 12, "duration": 6.5}
    }

Exit code 0 only if --output-path was actually written; non-zero on any
failure (an undecodable input, no audio track, CUDA unavailable, or a
transcription error).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import soundfile as sf

# segment_utils.py lives next to this script -- resolved by absolute path,
# not cwd, since a Slurm job's starting cwd isn't guaranteed to be this
# directory (see common/run_pipeline_job.py, which runs this script's own
# argv[0] wherever Slurm happened to start it).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_utils import calculate_metadata, merge_segment_boundaries  # noqa: E402

# Same chunking shape as the legacy app.py's config.py defaults
# (CHUNK_DURATION / CHUNK_OVERLAP) -- not exposed as a request field, since
# there's no evidence either value needs to vary per request.
CHUNK_DURATION_SECONDS = 600.0  # 10 minutes
CHUNK_OVERLAP_SECONDS = 10.0


class TranscribeError(RuntimeError):
    """Something about this specific request failed in an expected way
    (undecodable input, no audio track, model/CUDA problem) -- always
    caught in main() and reported as a clean stderr message plus non-zero
    exit, never an uncaught traceback."""


# ---------------------------------------------------------------------------
# Audio decoding -- same ffmpeg-via-imageio_ffmpeg approach as
# ModelService_RVC-Applio/normalize_audio.py (no system ffmpeg exists on
# this cluster), but resampling straight to the model's own required format
# (mono, 16kHz, PCM16) in one ffmpeg pass instead of a separate librosa
# resample step afterward, since ffmpeg's swresample already does this
# resampling well and avoids an extra full read/write of the decoded audio.
# ---------------------------------------------------------------------------
def decode_to_wav(src: Path, dst: Path) -> None:
    """Decodes `src`'s first audio stream to a mono 16kHz PCM16 WAV at
    `dst`. Raises TranscribeError (with ffmpeg's own stderr) if ffmpeg
    exits non-zero or doesn't actually produce `dst` -- covers both a
    corrupt/truncated input and a video with no audio stream at all."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    argv = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        # Absolute paths: a real upload on this deployment's own RVC
        # backend has already been named with a colon in it (e.g. "Soi
        # Sukhumvit 48:3 3.m4a") -- an absolute path always starts with
        # "/", which can never be misread as an ffmpeg "scheme:" prefix.
        "-i", str(src.resolve()),
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst.resolve()),
    ]
    print(f"Decoding '{src}' -> '{dst}' (mono, 16kHz, PCM16)...", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file():
        raise TranscribeError(
            f"Could not decode input file (ffmpeg exit {proc.returncode}): "
            f"{proc.stderr.strip() or 'no stderr output'}"
        )


# ---------------------------------------------------------------------------
# Chunking for long audio -- ported from the legacy app.py's
# split_audio_into_chunks(), operating on the already-decoded 16kHz mono WAV
# via soundfile directly (no librosa needed here -- that was only used by
# app.py for the mono/resample step, which ffmpeg above now does in one
# pass).
# ---------------------------------------------------------------------------
def split_into_chunks(wav_path: Path, chunks_dir: Path) -> list[dict[str, Any]]:
    """Returns a list of {"path": Path, "offset": float} -- a single entry
    pointing at `wav_path` itself (offset 0.0, no file copy) if it's
    already within CHUNK_DURATION_SECONDS, otherwise several overlapping
    chunk files written under `chunks_dir`. Boundary math matches app.py's
    own split_audio_into_chunks() exactly."""
    info = sf.info(str(wav_path))
    total_duration = info.frames / info.samplerate
    if total_duration <= CHUNK_DURATION_SECONDS:
        return [{"path": wav_path, "offset": 0.0}]

    effective_chunk_duration = CHUNK_DURATION_SECONDS - CHUNK_OVERLAP_SECONDS
    num_chunks = int((total_duration - CHUNK_OVERLAP_SECONDS) / effective_chunk_duration) + 1
    print(
        f"Audio is {total_duration:.1f}s (> {CHUNK_DURATION_SECONDS:.0f}s) -- "
        f"splitting into {num_chunks} chunk(s) of ~{CHUNK_DURATION_SECONDS:.0f}s "
        f"with {CHUNK_OVERLAP_SECONDS:.0f}s overlap.",
        flush=True,
    )

    audio_data, sample_rate = sf.read(str(wav_path), dtype="int16")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for i in range(num_chunks):
        if i == 0:
            start_time, end_time = 0.0, CHUNK_DURATION_SECONDS
        elif i == num_chunks - 1:
            start_time, end_time = i * effective_chunk_duration, total_duration
        else:
            start_time = i * effective_chunk_duration
            end_time = start_time + CHUNK_DURATION_SECONDS

        start_sample = int(start_time * sample_rate)
        end_sample = min(int(end_time * sample_rate), len(audio_data))
        chunk_path = chunks_dir / f"chunk_{i:03d}.wav"
        sf.write(str(chunk_path), audio_data[start_sample:end_sample], sample_rate, format="WAV", subtype="PCM_16")
        offset = i * effective_chunk_duration
        chunks.append({"path": chunk_path, "offset": offset})
        print(f"  chunk {i + 1}/{num_chunks}: {start_time:.1f}s-{end_time:.1f}s (offset {offset:.1f}s)", flush=True)

    return chunks


# ---------------------------------------------------------------------------
# Per-chunk transcription + timestamp extraction -- ported from app.py's
# process_single_chunk/extract_word_timestamps/extract_segment_timestamps.
# Operates on the Hypothesis object model.transcribe() returns; unrelated
# to any change made above.
# ---------------------------------------------------------------------------
def extract_word_timestamps(hypothesis: Any) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    if hasattr(hypothesis, "timestamp") and "word" in hypothesis.timestamp:
        for item in hypothesis.timestamp["word"]:
            if isinstance(item, dict):
                words.append({"word": item.get("word", ""), "start": float(item.get("start", 0)), "end": float(item.get("end", 0))})
    return words


def extract_segment_timestamps(hypothesis: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if hasattr(hypothesis, "timestamp") and "segment" in hypothesis.timestamp:
        for item in hypothesis.timestamp["segment"]:
            if isinstance(item, dict):
                text = item.get("segment", "")
                segments.append(
                    {
                        "text": text,
                        "start": float(item.get("start", 0)),
                        "end": float(item.get("end", 0)),
                        "word_count": len(text.split()) if text else 0,
                    }
                )
    return segments


def transcribe_chunk(model: Any, chunk_path: Path, offset: float) -> dict[str, Any]:
    result = model.transcribe([str(chunk_path)], timestamps=True)[0]
    text = result.text if hasattr(result, "text") else str(result)
    words = extract_word_timestamps(result)
    segments = extract_segment_timestamps(result)
    for word in words:
        word["start"] += offset
        word["end"] += offset
    for segment in segments:
        segment["start"] += offset
        segment["end"] += offset
    return {"transcription": text, "word_timestamps": words, "segment_timestamps": segments}


# ---------------------------------------------------------------------------
# Cross-chunk merge -- ported from app.py's
# deduplicate_overlap/merge_transcription_chunks. merge_segment_boundaries
# itself comes from segment_utils.py (imported above), same as app.py.
# ---------------------------------------------------------------------------
def deduplicate_overlap(prev_words: list[dict[str, Any]], curr_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not prev_words or not curr_words:
        return curr_words

    prev_overlap_start = prev_words[-1]["end"] - CHUNK_OVERLAP_SECONDS
    prev_overlap_words = [w for w in prev_words if w["end"] >= prev_overlap_start]
    curr_overlap_end = curr_words[0]["start"] + CHUNK_OVERLAP_SECONDS
    curr_overlap_indices = [i for i, w in enumerate(curr_words) if w["start"] <= curr_overlap_end]
    if not prev_overlap_words or not curr_overlap_indices:
        return curr_words

    to_remove: set[int] = set()
    for i in curr_overlap_indices:
        curr_word = curr_words[i]
        for prev_word in prev_overlap_words:
            text_match = curr_word["word"].lower() == prev_word["word"].lower()
            time_match = abs(curr_word["start"] - prev_word["start"]) < 0.5
            if text_match and time_match:
                to_remove.add(i)
                break
    return [w for i, w in enumerate(curr_words) if i not in to_remove]


def merge_chunk_results(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(chunk_results) == 1:
        return chunk_results[0]

    merged_text_parts = [chunk_results[0]["transcription"]]
    merged_words = list(chunk_results[0]["word_timestamps"])
    merged_segments = list(chunk_results[0]["segment_timestamps"])

    for curr in chunk_results[1:]:
        deduped_words = deduplicate_overlap(merged_words, curr["word_timestamps"])
        curr_segments = merge_segment_boundaries(merged_segments, curr["segment_timestamps"])
        merged_words.extend(deduped_words)
        merged_segments.extend(curr_segments)
        merged_text_parts.append(curr["transcription"])

    return {
        "transcription": " ".join(merged_text_parts),
        "word_timestamps": merged_words,
        "segment_timestamps": merged_segments,
    }


# ---------------------------------------------------------------------------
# Model loading -- ASRModel.restore_from() against the local .nemo file
# (never touches the network), then the same CUDA-graph-disabling decoding
# strategy override app.py applies at startup, for the same driver-version
# compatibility reason (see CUDA_FIX_SUMMARY.md). Unlike app.py, this does
# NOT set CUDA_VISIBLE_DEVICES itself -- Slurm's own --gres=gpu:1 allocation
# already controls which single GPU this process can see, and overriding it
# again here would silently ignore whichever physical GPU Slurm actually
# assigned.
# ---------------------------------------------------------------------------
def load_model(model_path: Path) -> Any:
    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise TranscribeError("CUDA is not available in this job -- it requires a GPU (--gres=gpu:1).")

    print(f"Loading model from {model_path}...", flush=True)
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(model_path), map_location="cpu")
    model = model.cuda()
    model.eval()

    try:
        decoding_cfg = OmegaConf.create(
            {
                "strategy": "greedy_batch",
                "model_type": "tdt",
                "durations": [0, 1, 2, 3, 4],
                "greedy": {"max_symbols": 10, "loop_labels": True, "use_cuda_graph_decoder": False},
            }
        )
        model.change_decoding_strategy(decoding_cfg)
        print("CUDA graphs disabled for decoding (driver compatibility mode).", flush=True)
    except Exception as error:  # noqa: BLE001 -- same tolerant handling as app.py: non-fatal either way
        print(f"WARNING: could not override decoding strategy: {error}", flush=True)

    print("Model loaded.", flush=True)
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", required=True, help="The original audio/video file to transcribe.")
    parser.add_argument("--work-dir", required=True, help="Directory to write the decoded WAV (and any chunks) into.")
    parser.add_argument("--model-path", required=True, help="Path to the local parakeet-tdt-0.6b-v3.nemo checkpoint.")
    parser.add_argument("--output-path", required=True, help="Where to write the transcript JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start_time = time.monotonic()

    input_path = Path(args.input_path)
    work_dir = Path(args.work_dir)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1
    if not model_path.is_file():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    try:
        decoded_path = work_dir / "decoded.wav"
        decode_to_wav(input_path, decoded_path)

        chunks = split_into_chunks(decoded_path, work_dir / "chunks")
        model = load_model(model_path)

        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{len(chunks)}...", flush=True)
            chunk_results.append(transcribe_chunk(model, chunk["path"], chunk["offset"]))

        merged = merge_chunk_results(chunk_results)
    except TranscribeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    metadata = calculate_metadata(merged["transcription"], merged["word_timestamps"], merged["segment_timestamps"])
    processing_time = time.monotonic() - start_time

    response = {
        "transcription": merged["transcription"],
        "processing_time": round(processing_time, 2),
        "word_timestamps": merged["word_timestamps"],
        "segment_timestamps": merged["segment_timestamps"],
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")

    if not output_path.is_file():
        print(f"ERROR: finished successfully, but failed to write output file: {output_path}", file=sys.stderr)
        return 1

    print(
        f"Transcription completed in {processing_time:.2f}s: "
        f"{metadata['total_words']} words, {metadata['total_segments']} segments -> {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
