import { useEffect, useRef, useState } from 'react';
import { uploadFile } from '../api.js';
import { formatBytes } from '../utils.js';
import { useFieldDoc } from '../schema.js';
import { InfoBubble } from './FieldLabel.jsx';
import ProgressBar from './ProgressBar.jsx';
import Button from './ui/Button.jsx';
import { UploadIcon, XIcon } from './icons.jsx';

// A file picker that immediately uploads to POST /v1/uploads and reports
// back one of three statuses via onChange -- the two-step "upload first,
// reference by asset_id" flow every generation endpoint expects (see
// server.py):
//   null                                                     nothing chosen
//   { status: 'uploading', filename, sentBytes, totalBytes }  sending
//   { status: 'ready', assetId, filename, sizeBytes }         done
// Forms use this shape directly to compute their own submit blockers --
// see utils.js's assetBlocker -- instead of re-deriving "is it still
// uploading?" from scratch. This exact shape is also what gets persisted
// to sessionStorage/localStorage (see storage.js) -- it's already plain,
// JSON-serializable data, which is why a page refresh can restore a
// "ready" file's filename/size but can never resume an in-progress upload
// (there's no File object in here to resume from).
//
// A fourth, read-only variant of the 'ready' status can arrive as a prop
// from the Retry flow instead of from an upload here: utils.js's
// reusedAsset() builds { status: 'ready', assetId, filename: null,
// sizeBytes: null, reused: true } straight from a prior job's own stored
// asset_id, with no filename/size on hand (the stored request never kept
// them). Shown with its own message below; picking/pasting/dropping a new
// file replaces it exactly like any other 'ready' value.
//
// A file can arrive three ways: the native picker (hidden, but still
// present -- see inputRef below), pasting (Ctrl+V/Cmd+V) while this card
// has focus, or dragging a file onto any part of it. All three end up
// calling the same startUpload(file).
//
// `previewUrl` (an object-URL thumbnail for image files) is local-only
// state, never part of `value` -- it points at an in-memory Blob that
// doesn't exist anymore after a refresh, so a restored draft or reused
// asset simply shows a generic file icon instead of a thumbnail.
//
// The file input is deliberately NOT nested inside a <label> here (unlike
// FieldLabel's usual pattern) -- clicking a <label> natively activates the
// first control nested inside it, which would double-fire alongside this
// component's own click-to-open handling below. Its header row is built
// from the same pieces FieldLabel uses (useFieldDoc/InfoBubble) instead.
function fileMatchesAccept(file, accept) {
  if (!accept || !file.type) return true;
  return accept.split(',').some((pattern) => {
    const type = pattern.trim();
    return type.endsWith('/*') ? file.type.startsWith(type.slice(0, -1)) : file.type === type;
  });
}

// ClipboardEvent.clipboardData and DragEvent.dataTransfer both expose this
// same {files, items} shape, so paste and drop share this one lookup.
function firstFile(dataTransfer) {
  if (dataTransfer.files.length > 0) return dataTransfer.files[0];
  const item = Array.from(dataTransfer.items || []).find((entry) => entry.kind === 'file');
  return item ? item.getAsFile() : null;
}

export default function AssetUpload({ model, field, label, accept, value, onChange, required }) {
  const doc = useFieldDoc(model, field);
  const isRequired = required ?? doc.required;
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function setPreviewFor(file) {
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    });
  }

  async function startUpload(file) {
    setError(null);
    setPreviewFor(file);
    onChange({ status: 'uploading', filename: file.name, sentBytes: 0, totalBytes: file.size });
    try {
      const result = await uploadFile(file, (sentBytes, totalBytes) => {
        onChange({ status: 'uploading', filename: file.name, sentBytes, totalBytes });
      });
      onChange({ status: 'ready', assetId: result.asset_id, filename: result.filename, sizeBytes: result.size_bytes });
    } catch (err) {
      setError(err.message);
      onChange(null);
    }
  }

  function handleFileChange(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    startUpload(file);
  }

  // Shared entry point for a pasted or dropped file, after it's already
  // been pulled out of the clipboard/drag event -- checks its type (the
  // native input's own `accept` never got a say for these), mirrors it
  // onto the native <input>, then uploads exactly like the picker does.
  function intake(file) {
    if (!fileMatchesAccept(file, accept)) {
      setError(`"${file.name}" doesn't look like the expected file type (${accept}).`);
      return;
    }
    if (inputRef.current) {
      try {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        inputRef.current.files = transfer.files;
      } catch {
        // Some older browsers can't construct a DataTransfer by hand --
        // the native input just won't mirror the file then; the upload
        // itself still proceeds below either way, and this UI never
        // displays the native input's own filename anyway (see the
        // ready-state message rendered from `value` below).
      }
    }
    startUpload(file);
  }

  function handlePaste(event) {
    if (uploading) return;
    const file = firstFile(event.clipboardData);
    if (!file) {
      setError('No file found on the clipboard.');
      return;
    }
    // Stops the browser's own default paste-into-file-input handling (a
    // focused native <input type="file"> can do this itself in some
    // browsers) from also firing and double-uploading the same file.
    event.preventDefault();
    intake(file);
  }

  function handleDragOver(event) {
    if (!event.dataTransfer.types.includes('Files')) return;
    // Required so `drop` actually fires here instead of the browser
    // rejecting the drag outright -- see handleDrop.
    event.preventDefault();
    if (!uploading) setDragActive(true);
  }

  function handleDragLeave() {
    setDragActive(false);
  }

  function handleDrop(event) {
    setDragActive(false);
    if (!event.dataTransfer.types.includes('Files')) return;
    // As in handleDragOver: also suppresses the browser's own default
    // drop-into-file-input handling, so only our intake() below runs.
    event.preventDefault();
    if (uploading) return;
    const file = firstFile(event.dataTransfer);
    if (!file) {
      setError('No file found in the dropped item.');
      return;
    }
    intake(file);
  }

  function handleRemove(event) {
    event.stopPropagation();
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    setError(null);
    onChange(null);
    if (inputRef.current) inputRef.current.value = '';
  }

  // Guarded against the input's own click bubbling back up to this same
  // handler: calling inputRef.current.click() dispatches a real,
  // bubbling 'click' event whose target is the input itself, which would
  // otherwise reach this handler again and call .click() a second time
  // (an infinite loop). Skipping whenever the click's target is already
  // the input breaks that loop after exactly one bounce.
  function handleZoneClick(event) {
    if (uploading) return;
    if (event.target === inputRef.current) return;
    inputRef.current?.click();
  }

  function handleZoneKeyDown(event) {
    if (uploading) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      inputRef.current?.click();
    }
  }

  const uploading = value?.status === 'uploading';
  const ready = value?.status === 'ready';
  // The browser reports 100% once the request body is fully sent, but the
  // server still has to read it and write it to disk before it replies --
  // there's no further progress to report during that gap, so show an
  // indeterminate bar instead of sitting at a misleading 100%.
  const sendComplete = uploading && value.sentBytes >= value.totalBytes;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted">
        <span>
          {label}
          {isRequired && (
            <span className="text-danger" aria-hidden="true">
              {' '}
              *
            </span>
          )}
        </span>
        <InfoBubble description={doc.description} />
      </span>

      <div
        role="button"
        tabIndex={uploading ? -1 : 0}
        onClick={handleZoneClick}
        onKeyDown={handleZoneKeyDown}
        onPaste={handlePaste}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`flex items-center gap-3 rounded-lg border border-dashed p-3 text-sm transition-colors ${
          dragActive ? 'border-accent bg-accent/5' : 'border-border'
        } ${uploading ? 'cursor-default opacity-80' : 'cursor-pointer hover:border-accent/60 hover:bg-surface-2'}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={handleFileChange}
          disabled={uploading}
          className="sr-only"
          tabIndex={-1}
          aria-hidden="true"
        />

        {previewUrl ? (
          <img src={previewUrl} alt="" className="h-12 w-12 shrink-0 rounded-md object-cover" />
        ) : (
          <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-md bg-surface-2 text-muted">
            <UploadIcon className="h-5 w-5" />
          </span>
        )}

        <div className="min-w-0 flex-1">
          {!value && <p className="text-muted">Click, paste, or drop a file here</p>}

          {uploading && !sendComplete && (
            <ProgressBar sentBytes={value.sentBytes} totalBytes={value.totalBytes} label={`Uploading ${value.filename}`} />
          )}
          {uploading && sendComplete && <ProgressBar indeterminate label="Processing on server..." />}

          {ready && value.reused && (
            <p className="truncate text-muted">
              Reusing the file from the original job (asset {value.assetId.slice(0, 8)}&hellip;). Click to replace it.
            </p>
          )}
          {ready && !value.reused && (
            <p className="truncate text-text">
              {value.filename} <span className="text-muted">({formatBytes(value.sizeBytes)})</span>
            </p>
          )}
        </div>

        {value && !uploading && (
          <Button type="button" variant="ghost" size="sm" onClick={handleRemove} aria-label="Remove file">
            <XIcon className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  );
}
