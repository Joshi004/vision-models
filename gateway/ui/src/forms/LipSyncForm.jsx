import { postJSON } from '../api.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Input from '../components/ui/Input.jsx';
import SegmentedControl from '../components/ui/SegmentedControl.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

const MODEL = 'LipSyncRequest';
const FORM_ID = 'tbdub-lip-sync';

const DEFAULTS = {
  inputMode: 'job',
  sourceJobId: '',
  videoAsset: null,
  audioAsset: null,
  audioStartTime: 0,
  seed: 10,
  audioCfgScale: '',
  refCfgScale: '',
  partition: '',
};

// The inputs (job id, files) and the two CFG dials belong to one specific run,
// so only the seed and partition carry over to the next fresh form -- see
// storage.js.
const REMEMBER_KEYS = ['seed', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    inputMode: initialRequest.source_job_id ? 'job' : 'upload',
    sourceJobId: initialRequest.source_job_id ?? '',
    videoAsset: reusedAsset(initialRequest.video_asset_id),
    audioAsset: reusedAsset(initialRequest.audio_asset_id),
    audioStartTime: initialRequest.audio_start_time ?? 0,
    seed: initialRequest.seed ?? 10,
    audioCfgScale: initialRequest.audio_cfg_scale ?? '',
    refCfgScale: initialRequest.ref_cfg_scale ?? '',
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto LipSyncRequest (services/tbdub/schemas.py): either the id of
// an LTX audio-to-video job to re-sync (its video and audio are reused, so
// nothing is uploaded -- this is what "Retry lip sync" on a job row opens), or
// a video and an audio file uploaded here. See TextToVideoForm.jsx for how
// initialRequest/useFormState/FormShell fit together.
export default function LipSyncForm({
  defaultPartition,
  onSubmitted,
  initialRequest = null,
  prefillJob = null,
  onClearPrefill,
}) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { inputMode, sourceJobId, videoAsset, audioAsset, audioStartTime, seed, audioCfgScale, refCfgScale, partition } = values;
  const fromJob = inputMode === 'job';

  const payload = { seed: seed === '' ? undefined : Number(seed) };
  if (fromJob) {
    payload.source_job_id = sourceJobId.trim() || undefined;
  } else {
    payload.video_asset_id = videoAsset?.status === 'ready' ? videoAsset.assetId : undefined;
    payload.audio_asset_id = audioAsset?.status === 'ready' ? audioAsset.assetId : undefined;
    payload.audio_start_time = audioStartTime === '' ? undefined : Number(audioStartTime);
  }
  if (audioCfgScale !== '') payload.audio_cfg_scale = Number(audioCfgScale);
  if (refCfgScale !== '') payload.ref_cfg_scale = Number(refCfgScale);
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = fromJob
    ? sourceJobId.trim()
      ? []
      : ['Enter the job id of the audio-to-video job to re-sync.']
    : [...assetBlocker(videoAsset, 'the video to re-sync'), ...assetBlocker(audioAsset, 'the audio to sync to')];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/tbdub/videos/lip-sync', payload));
  }

  return (
    <FormShell
      title="Lip sync"
      description="Redraws the speaker's mouth in an existing video to match an audio track exactly (TBDub). One speaker with a visible face; the result is 25 fps. This is always its own job, separate from the job it was started from."
      endpoint="POST /v1/tbdub/videos/lip-sync"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Lip sync video"
      onSubmit={handleSubmit}
    >
      <Section title="What to sync">
        <SegmentedControl
          name="lip-sync-input-mode"
          value={inputMode}
          onChange={(value) => setField('inputMode', value)}
          options={[
            { value: 'job', label: 'An audio-to-video job' },
            { value: 'upload', label: 'Upload a video and audio' },
          ]}
        />
        {fromJob ? (
          <FieldLabel model={MODEL} field="source_job_id" text="source_job_id (a succeeded LTX audio-to-video job)" required>
            <Input
              value={sourceJobId}
              onChange={(event) => setField('sourceJobId', event.target.value)}
              placeholder="job id"
              className="font-mono"
            />
          </FieldLabel>
        ) : (
          <>
            <AssetUpload
              model={MODEL}
              field="video_asset_id"
              label="video to re-sync"
              accept="video/*"
              value={videoAsset}
              onChange={(next) => setField('videoAsset', next)}
              required
            />
            <AssetUpload
              model={MODEL}
              field="audio_asset_id"
              label="audio (or video) to sync to"
              accept="audio/*,video/*"
              value={audioAsset}
              onChange={(next) => setField('audioAsset', next)}
              required
            />
            <FieldLabel model={MODEL} field="audio_start_time" text="audio_start_time (seconds)">
              <Input
                type="number"
                min="0"
                step="0.1"
                value={audioStartTime}
                onChange={(event) => setField('audioStartTime', event.target.value)}
                className="max-w-[10rem]"
              />
            </FieldLabel>
          </>
        )}
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={MODEL} field="audio_cfg_scale" text="audio_cfg_scale (follow the voice; default 6)">
            <Input
              type="number"
              min="1"
              max="20"
              step="0.5"
              value={audioCfgScale}
              onChange={(event) => setField('audioCfgScale', event.target.value)}
              placeholder="default"
            />
          </FieldLabel>
          <FieldLabel model={MODEL} field="ref_cfg_scale" text="ref_cfg_scale (stay like the original; default 2)">
            <Input
              type="number"
              min="1"
              max="20"
              step="0.5"
              value={refCfgScale}
              onChange={(event) => setField('refCfgScale', event.target.value)}
              placeholder="default"
            />
          </FieldLabel>
        </div>
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
