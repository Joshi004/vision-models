import { createContext, useContext } from 'react';

// Provided once by App.jsx after a single GET /openapi.json -- every
// field's hover description and required-ness comes from here, straight
// from the same Pydantic models the API itself validates against (see
// services/ltx/schemas.py, services/wan_animate/schemas.py). Never
// hand-copied, so neither can silently drift from what the API actually
// does. Value is the `components.schemas` map from /openapi.json (see
// extractSchemas), or null before it's loaded / if it failed to load --
// every consumer here degrades gracefully rather than crashing.
export const SchemaContext = createContext(null);

export function useSchemas() {
  return useContext(SchemaContext);
}

// { description, required } for one field of one component. Returns
// { description: null, required: false } if the schema hasn't loaded (or
// the model/field name doesn't match) -- callers show no tooltip and no
// asterisk rather than crashing.
export function useFieldDoc(model, field) {
  const schemas = useSchemas();
  const component = schemas?.[model];
  const property = component?.properties?.[field];
  const required = (component?.required || []).includes(field);
  return { description: property?.description ?? null, required };
}

// The schema's own minItems for an array field (e.g. keyframes' >=2 rule),
// or 0 if unknown -- read live so a change to that minimum on the API side
// can't silently drift out of sync with the UI's own messaging.
export function useArrayMinItems(model, field) {
  const schemas = useSchemas();
  return schemas?.[model]?.properties?.[field]?.minItems ?? 0;
}

// Pulls just the `components.schemas` map out of a raw /openapi.json
// response -- the only part this UI needs.
export function extractSchemas(openApiDocument) {
  return openApiDocument?.components?.schemas ?? {};
}

// Missing-required-field messages for one model's current payload, in the
// schema's own required-field order.
//
// Array-typed fields (images/keyframes) are always skipped here -- see
// utils.js's imageListBlockers, which has the row-level context (which
// row is empty vs. still uploading) this function doesn't have.
//
// `skip` additionally excludes fields a form already reports on via a
// more specific message -- e.g. a single-file field mid-upload gets
// "Wait for X to finish uploading" from utils.js's assetBlocker instead of
// the generic "field is required" this function would otherwise also add,
// so the same problem isn't reported twice.
export function missingRequired(schemas, model, payload, skip = []) {
  const component = schemas?.[model];
  if (!component) return [];
  const messages = [];
  for (const field of component.required || []) {
    if (skip.includes(field)) continue;
    const property = component.properties?.[field];
    if (property?.type === 'array') continue;
    const value = payload[field];
    if (value === undefined || value === null || value === '') {
      messages.push(`${field} is required.`);
    }
  }
  return messages;
}
