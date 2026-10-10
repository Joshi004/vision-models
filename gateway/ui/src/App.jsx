import { useEffect, useRef, useState } from 'react';
import Header from './components/Header.jsx';
import Sidebar from './components/Sidebar.jsx';
import JobsPanel from './components/JobsPanel.jsx';
import JobHistory from './components/JobHistory.jsx';
import { ToastProvider } from './components/ui/Toast.jsx';
import { FilmIcon, LayersIcon, WaveIcon, RedoIcon, MusicIcon, SwapUserIcon, MicIcon, StackIcon, TranscriptIcon } from './components/icons.jsx';
import TextToVideoForm from './forms/TextToVideoForm.jsx';
import Ltx25TextToVideoForm from './forms/Ltx25TextToVideoForm.jsx';
import Ltx25InterpolateForm from './forms/Ltx25InterpolateForm.jsx';
import KeyframeInterpolationForm from './forms/KeyframeInterpolationForm.jsx';
import AudioToVideoForm from './forms/AudioToVideoForm.jsx';
import RetakeForm from './forms/RetakeForm.jsx';
import TextToAudioForm from './forms/TextToAudioForm.jsx';
import WanReplaceForm from './forms/WanReplaceForm.jsx';
import RvcConvertForm from './forms/RvcConvertForm.jsx';
import RvcBatchForm from './forms/RvcBatchForm.jsx';
import TranscribeForm from './forms/TranscribeForm.jsx';
import BreezeTtsForm from './forms/BreezeTtsForm.jsx';
import { getHealth, getOpenApi } from './api.js';
import { extractSchemas, SchemaContext } from './schema.js';
import { usePersistedState } from './storage.js';

// One group per backend this API actually has recipes for -- MAGI-2.2 and
// Qwen3-VL have no endpoints yet (see STATUS.md), so there's nothing to
// add here for them until that changes.
const GROUPS = [
  { id: 'ltx', label: 'LTX-2.3' },
  { id: 'ltx25', label: 'LTX-2.5' },
  { id: 'wan', label: 'Wan-Animate' },
  { id: 'rvc', label: 'Voice Conversion' },
  { id: 'parakeet', label: 'Transcription' },
  { id: 'breeze', label: 'Text to Speech' },
];

// One entry per generation recipe this API actually has today. `pipeline`
// is each recipe's own "<backend>:<recipe>" string, exactly as stored on
// every job row (see services/*/dispatch.py's _dispatch/create_job calls)
// -- Retry (handleRetry below) uses it to find which recipe/form a given
// job.pipeline should reopen into.
const RECIPES = [
  {
    id: 'ltx-generate',
    label: 'Text / image to video',
    description: 'Generate a video from a prompt, optionally conditioned on one or more images.',
    group: 'ltx',
    icon: FilmIcon,
    Component: TextToVideoForm,
    pipeline: 'ltx:text-to-video',
  },
  {
    id: 'ltx-keyframe',
    label: 'Keyframe interpolation',
    description: 'Interpolate motion between two or more keyframe images.',
    group: 'ltx',
    icon: LayersIcon,
    Component: KeyframeInterpolationForm,
    pipeline: 'ltx:keyframe-interpolation',
  },
  {
    id: 'ltx-audio-to-video',
    label: 'Audio to video',
    description: "Generate a video synced to an existing audio/video track's speech.",
    group: 'ltx',
    icon: WaveIcon,
    Component: AudioToVideoForm,
    pipeline: 'ltx:audio-to-video',
  },
  {
    id: 'ltx-retake',
    label: 'Retake',
    description: 'Regenerate a short window of an existing video.',
    group: 'ltx',
    icon: RedoIcon,
    Component: RetakeForm,
    pipeline: 'ltx:retake',
  },
  {
    id: 'ltx-text-to-audio',
    label: 'Text to audio',
    description: 'Generate a standalone audio clip from a text prompt.',
    group: 'ltx',
    icon: MusicIcon,
    Component: TextToAudioForm,
    pipeline: 'ltx:text-to-audio',
  },
  {
    id: 'ltx25-generate',
    label: 'Text / image to video',
    description: 'Generate a video on LTX-2.5, with native multi-shot cuts written into the prompt.',
    group: 'ltx25',
    icon: FilmIcon,
    Component: Ltx25TextToVideoForm,
    pipeline: 'ltx25:text-to-video',
  },
  {
    id: 'ltx25-interpolate',
    label: 'First / last frame',
    description: 'Generate the motion between a given first frame and a given last frame on LTX-2.5.',
    group: 'ltx25',
    icon: LayersIcon,
    Component: Ltx25InterpolateForm,
    pipeline: 'ltx25:interpolate',
  },
  {
    id: 'ltx25-retake',
    label: 'Retake',
    description: 'Regenerate a short window of an existing video on LTX-2.5.',
    group: 'ltx25',
    icon: RedoIcon,
    Component: RetakeForm,
    // Same form as LTX-2.3's Retake (identical fields), pointed at LTX-2.5's
    // own endpoint and request schema -- see the props at the top of
    // forms/RetakeForm.jsx.
    formProps: { formId: 'ltx25-retake', model: 'Ltx25RetakeRequest', endpoint: '/v1/ltx25/videos/retake', title: 'Retake (LTX-2.5)' },
    pipeline: 'ltx25:retake',
  },
  {
    id: 'wan-replace',
    label: 'Replace character',
    description: 'Swap the on-screen person in a video for a reference photo.',
    group: 'wan',
    icon: SwapUserIcon,
    Component: WanReplaceForm,
    pipeline: 'wan-animate:replace',
  },
  {
    id: 'rvc-convert',
    label: 'Convert voice',
    description: 'Convert a recording to one of the installed voices, keeping its own delivery.',
    group: 'rvc',
    icon: MicIcon,
    Component: RvcConvertForm,
    pipeline: 'rvc:convert',
  },
  {
    id: 'rvc-batch',
    label: 'Batch convert',
    description: 'Convert several recordings to the same installed voice in one job. Result is a single zip.',
    group: 'rvc',
    icon: StackIcon,
    Component: RvcBatchForm,
    pipeline: 'rvc:batch-convert',
  },
  {
    id: 'parakeet-transcribe',
    label: 'Transcribe',
    description: 'Transcribe a recording to text, with word- and segment-level timestamps.',
    group: 'parakeet',
    icon: TranscriptIcon,
    Component: TranscribeForm,
    pipeline: 'parakeet:transcribe',
  },
  {
    id: 'breeze-synthesize',
    label: 'Synthesize speech',
    description: 'Generate speech from text: design a voice from a description, or clone and direct a reference voice.',
    group: 'breeze',
    icon: WaveIcon,
    Component: BreezeTtsForm,
    pipeline: 'breeze-tts:synthesize',
  },
];

// True if some recipe form above can actually resubmit this job's exact
// pipeline -- the only jobs JobRow.jsx offers a Retry button for (see
// canRetry prop threaded through JobsPanel.jsx/JobHistory.jsx below).
function canRetry(job) {
  return RECIPES.some((recipe) => recipe.pipeline === job.pipeline);
}

// Selects the full job-history view (components/JobHistory.jsx) instead of
// any one recipe form -- kept out of RECIPES itself since it isn't a
// generation recipe and doesn't take the same defaultPartition/onSubmitted
// props every recipe form does.
const HISTORY_ID = 'job-history';

function isValidActiveId(id) {
  return id === HISTORY_ID || RECIPES.some((recipe) => recipe.id === id);
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  // Persisted so reopening the tab lands back on the recipe last used
  // instead of always resetting to the first one -- validated against
  // RECIPES/HISTORY_ID in case a stored id no longer exists (e.g. after a
  // future recipe rename).
  const [activeId, setActiveId] = usePersistedState('activeRecipeId', RECIPES[0].id, {
    validate: isValidActiveId,
  });
  // The `components.schemas` map from GET /openapi.json -- every field's
  // hover description and required-marker comes from this (see schema.js),
  // so neither can silently drift from what the API actually validates.
  // Loaded once, not polled -- it only changes when the API's own code
  // changes, which already requires a server (and so a page) restart.
  const [schemas, setSchemas] = useState(null);
  const [schemaError, setSchemaError] = useState(null);
  // Bumped after every successful submit so JobsPanel can refetch right
  // away instead of waiting for its next timed poll -- see JobsPanel.jsx.
  const [refreshSignal, setRefreshSignal] = useState(0);
  // Set by Retry (a job row's own Retry button, via handleRetry below) to
  // reopen the matching recipe form prefilled with that job's own stored
  // request -- { job, nonce } while active, null for the normal blank-form
  // case. `nonce` (just a fresh value each time, Date.now() below) is only
  // there to force the form to remount even if the same job is retried
  // twice in a row without navigating away in between, since the form
  // reads its prefilled values once, at mount, from useFormState (see
  // storage.js) -- see ActiveForm's `key` below.
  const [prefill, setPrefill] = useState(null);
  // The <main> element (below) scrolls its own content rather than the
  // page itself -- handleRetry scrolls this back to the top so a form
  // prefilled from a job the user found further down the Jobs panel/job
  // history doesn't open off-screen.
  const mainRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const result = await getHealth();
        if (!cancelled) {
          setHealth(result);
          setHealthError(null);
        }
      } catch (err) {
        if (!cancelled) setHealthError(err.message);
      }
    }
    check();
    const interval = setInterval(check, 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getOpenApi()
      .then((doc) => {
        if (!cancelled) setSchemas(extractSchemas(doc));
      })
      .catch((err) => {
        if (!cancelled) setSchemaError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // A file dragged/dropped anywhere on the page still bubbles up to here.
  // AssetUpload's own onDrop already calls preventDefault for its own
  // field, making this a no-op there; this only actually matters for a
  // drop that missed every upload field, where without it the browser's
  // default action -- opening the file in this tab -- would wipe out
  // whatever's currently in the form.
  useEffect(() => {
    function handleWindowDragOver(event) {
      if (event.dataTransfer?.types.includes('Files')) event.preventDefault();
    }
    function handleWindowDrop(event) {
      if (event.dataTransfer?.types.includes('Files')) event.preventDefault();
    }
    window.addEventListener('dragover', handleWindowDragOver);
    window.addEventListener('drop', handleWindowDrop);
    return () => {
      window.removeEventListener('dragover', handleWindowDragOver);
      window.removeEventListener('drop', handleWindowDrop);
    };
  }, []);

  const showingHistory = activeId === HISTORY_ID;
  const activeRecipe = RECIPES.find((recipe) => recipe.id === activeId) ?? RECIPES[0];
  const ActiveForm = activeRecipe.Component;

  function handleSubmitted() {
    setRefreshSignal((value) => value + 1);
  }

  // Passed to JobRow (via JobsPanel.jsx/JobHistory.jsx) as `onRetry` --
  // only ever called for a job canRetry(job) has already confirmed has a
  // matching recipe below, so `recipe` here should never actually be
  // undefined; the check is only a defensive fallback against that
  // invariant somehow not holding.
  function handleRetry(job) {
    const recipe = RECIPES.find((r) => r.pipeline === job.pipeline);
    if (!recipe) return;
    setActiveId(recipe.id);
    setPrefill({ job, nonce: Date.now() });
    mainRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function selectRecipe(recipe) {
    setActiveId(recipe.id);
    // Switching recipe by hand abandons any pending retry -- otherwise a
    // stale prefill could silently reappear if the user later clicked
    // back to the recipe it came from.
    setPrefill(null);
  }

  return (
    <ToastProvider>
      <SchemaContext.Provider value={schemas}>
        <div className="flex h-screen flex-col overflow-hidden bg-bg text-text">
          <Header health={health} healthError={healthError} />
          {schemaError && (
            <p className="border-b border-border bg-surface px-5 py-1.5 text-xs text-warning">
              Field help unavailable ({schemaError}) &mdash; forms still work, just without hover descriptions or
              required markers.
            </p>
          )}
          <div className="flex flex-1 min-h-0">
            <Sidebar
              groups={GROUPS}
              recipes={RECIPES}
              activeId={activeId}
              showingHistory={showingHistory}
              onSelectRecipe={selectRecipe}
              onSelectHistory={() => setActiveId(HISTORY_ID)}
            />
            <main ref={mainRef} className="flex-1 overflow-y-auto">
              {showingHistory ? (
                <div className="px-5 py-6 lg:px-10">
                  <JobHistory jobRetentionDays={health?.job_retention_days} onRetry={handleRetry} canRetry={canRetry} />
                </div>
              ) : (
                <div className="flex flex-col gap-6 px-5 py-6 lg:flex-row lg:gap-10 lg:px-10">
                  <div className="min-w-0 flex-1">
                    <ActiveForm
                      // A changing key forces React to unmount/remount the
                      // form instead of reusing the previous instance, so
                      // its useFormState initializer (which reads
                      // initialRequest/storage, see storage.js) runs
                      // again -- otherwise switching from one retry
                      // straight to another (or back to a blank form)
                      // would leave stale field values in place.
                      key={prefill ? `retry-${prefill.nonce}` : activeRecipe.id}
                      {...activeRecipe.formProps}
                      defaultPartition={health?.default_partition}
                      onSubmitted={handleSubmitted}
                      initialRequest={prefill?.job.request ?? null}
                      prefillJob={prefill?.job ?? null}
                      onClearPrefill={() => setPrefill(null)}
                    />
                  </div>
                  <aside className="w-full shrink-0 lg:w-80">
                    <JobsPanel refreshSignal={refreshSignal} onRetry={handleRetry} canRetry={canRetry} />
                  </aside>
                </div>
              )}
            </main>
          </div>
        </div>
      </SchemaContext.Provider>
    </ToastProvider>
  );
}
