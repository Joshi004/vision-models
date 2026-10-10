import { useEffect, useState } from 'react';
import { fetchResultBlob } from '../api.js';
import { getCachedResult, putCachedResult } from '../resultCache.js';
import ProgressBar from './ProgressBar.jsx';
import TranscriptView from './TranscriptView.jsx';
import Button from './ui/Button.jsx';

// Pipelines whose result is audio, not video -- rendered with an <audio>
// element below instead of this component's default <video> assumption.
// `ltx:text-to-audio` always produces .wav (see services/ltx/dispatch.py's
// dispatch_text_to_audio); `rvc:convert` also produces audio, but its real
// extension follows whatever `export_format` its own request chose (see
// resultExtension below) -- unlike text-to-audio, it is not always .wav.
// `breeze-tts:synthesize` always produces .wav (see
// services/breeze_tts/dispatch.py).
const AUDIO_PIPELINES = new Set(['ltx:text-to-audio', 'rvc:convert', 'breeze-tts:synthesize']);

function isAudioPipeline(pipeline) {
  return AUDIO_PIPELINES.has(pipeline);
}

// rvc:batch-convert's result is a zip of several converted files (see
// services/rvc/schemas.py's BatchConvertRequest docstring), not a single
// playable file -- no <audio>/<video> element makes sense for it, just a
// download link (see the render logic below).
function isZipPipeline(pipeline) {
  return pipeline === 'rvc:batch-convert';
}

// parakeet:transcribe's result is a JSON transcript, not a playable media
// file -- rendered with TranscriptView below instead of <audio>/<video>.
// See services/parakeet/openapi_docs.py's TRANSCRIBE text for the exact
// JSON shape.
function isTranscriptPipeline(pipeline) {
  return pipeline === 'parakeet:transcribe';
}

// The extension to download this result as. Every other pipeline has one
// fixed extension; rvc:convert is the one exception -- its own
// export_format (read from `request`, the job's original stored request
// body -- see JobRow.jsx, the only caller that supplies it) decides its
// real extension, which is not always .wav.
function resultExtension(pipeline, request) {
  if (isZipPipeline(pipeline)) return 'zip';
  if (isTranscriptPipeline(pipeline)) return 'json';
  if (pipeline === 'rvc:convert') return (request?.export_format || 'wav').toLowerCase();
  if (isAudioPipeline(pipeline)) return 'wav';
  return 'mp4';
}

// Loaded on demand (not automatically) and fetched as a blob rather than
// used as a plain <video src="..."> -- a plain media element can't report
// download progress or feed the local result cache below, so this goes
// through api.js's fetch() wrapper instead and turns the response into an
// object URL. Streamed with progress reporting (see api.js's
// fetchResultBlob), since a result can be tens of MB and a bare spinner
// would say very little about how much longer it'll take.
//
// A result already fetched once is kept in resultCache.js's IndexedDB
// store, keyed by jobId -- mounting this component re-checks that cache
// first (below) and, on a hit, shows the video/audio immediately with no
// network request at all. A fresh fetch (handleLoad) writes its blob back
// into that same cache once it finishes, so the *next* time this mounts
// for the same job (row re-expanded, Job History instead of the Jobs
// panel, or even a page reload) it's instant too. Caching is best-effort
// (see resultCache.js) -- this component works exactly as before if it's
// unavailable for any reason.
//
// For a transcript result (isTranscriptPipeline), the same blob is also
// read as text and JSON-parsed for TranscriptView -- see handleBlob below.
// The object URL still backs the Download link either way, same as every
// other pipeline.
export default function ResultPreview({ jobId, pipeline, request }) {
  const [url, setUrl] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  // Briefly true while the initial cache check (below) is in flight, so
  // the "Load preview" button doesn't flash on screen for an instant
  // before a cache hit replaces it with the video/audio itself.
  const [checkingCache, setCheckingCache] = useState(true);

  const isTranscript = isTranscriptPipeline(pipeline);

  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [url]);

  // Shared by the cache-hit path below and handleLoad's fresh-fetch path:
  // always backs the Download link with an object URL, and additionally
  // parses the blob as the transcript JSON on a parakeet:transcribe job.
  // A result that somehow isn't valid JSON there (should not normally
  // happen) surfaces as this component's own error state instead of
  // crashing -- the Download link still works either way.
  async function handleBlob(blob) {
    setUrl(URL.createObjectURL(blob));
    if (isTranscript) {
      try {
        setTranscript(JSON.parse(await blob.text()));
      } catch {
        setError('Could not parse the transcript JSON.');
      }
    }
  }

  useEffect(() => {
    let active = true;
    getCachedResult(jobId).then((blob) => {
      if (!active) return;
      if (blob) handleBlob(blob);
      setCheckingCache(false);
    });
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  async function handleLoad() {
    setError(null);
    setProgress({ receivedBytes: 0, totalBytes: 0 });
    try {
      const blob = await fetchResultBlob(jobId, (receivedBytes, totalBytes) => setProgress({ receivedBytes, totalBytes }));
      await handleBlob(blob);
      putCachedResult(jobId, blob); // best-effort; see resultCache.js
    } catch (err) {
      setError(err.message);
    } finally {
      setProgress(null);
    }
  }

  const isZip = isZipPipeline(pipeline);
  const isAudio = isAudioPipeline(pipeline);
  const filename = `${jobId}.${resultExtension(pipeline, request)}`;

  return (
    <div className="flex flex-col items-start gap-2">
      {!url && !progress && !checkingCache && (
        <Button type="button" variant="secondary" size="sm" onClick={handleLoad}>
          {isZip ? 'Load result' : 'Load preview'}
        </Button>
      )}
      {progress && <ProgressBar sentBytes={progress.receivedBytes} totalBytes={progress.totalBytes} label="Downloading" />}
      {error && <p className="text-xs text-danger">{error}</p>}
      {url && !isZip && isTranscript && transcript && <TranscriptView transcript={transcript} />}
      {url && !isZip && !isTranscript && isAudio && <audio src={url} controls className="w-full max-w-md" />}
      {url && !isZip && !isTranscript && !isAudio && <video src={url} controls className="w-full max-w-md rounded-md" />}
      {url && (
        <a href={url} download={filename} className="text-xs text-accent-2 hover:underline">
          Download {filename}
        </a>
      )}
    </div>
  );
}
