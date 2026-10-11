// incognito.js
//
// Whether Incognito mode is currently on. App.jsx owns the real (persisted)
// React state behind the Header's switch and mirrors it here, so api.js's
// postJSON can read it synchronously at submit time without every form
// having to be passed the flag. While it is on, every job submission is sent
// with `incognito: true`, and the server keeps no logs or inputs of that job
// and deletes its result after one download (see common/retention.py).

let incognitoMode = false;

export function getIncognito() {
  return incognitoMode;
}

export function setIncognito(value) {
  incognitoMode = Boolean(value);
}
