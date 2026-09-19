// Single data-access surface used by every component. Talks to the real
// API when VITE_API_URL is set, otherwise falls back to the bundled
// fixtures (frontend/fixtures/) so the app runs with no backend at all.
import {
  fixtureCaseIds,
  fixtureManifest,
  fixtureGetCase,
  fixtureGetCaseLinks,
  fixturePostFeedback,
} from "./fixturesClient.js";

const API_URL = import.meta.env.VITE_API_URL?.replace(/\/+$/, "");
export const isLiveApi = Boolean(API_URL);

async function apiFetch(path, options) {
  const res = await fetch(`${API_URL}${path}`, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.message || `request failed: ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

const caseCache = new Map();

export async function getCase(id) {
  if (caseCache.has(id)) return caseCache.get(id);
  const promise = isLiveApi
    ? apiFetch(`/cases/${encodeURIComponent(id)}`)
    : fixtureGetCase(id);
  caseCache.set(id, promise);
  try {
    return await promise;
  } catch (err) {
    caseCache.delete(id);
    throw err;
  }
}

export async function getCaseLinks(id, scope = "same", limit = 50) {
  if (isLiveApi) {
    return apiFetch(`/cases/${encodeURIComponent(id)}/links?scope=${scope}&limit=${limit}`);
  }
  return fixtureGetCaseLinks(id, scope);
}

export async function postFeedback(payload) {
  if (isLiveApi) {
    return apiFetch(`/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }
  return fixturePostFeedback(payload);
}

// Case picker + demo mode need a browsable list of cases and the curated
// examples. There is no "list cases" route in api/openapi.yaml, so this
// stays fixtures-only — in live-API mode the picker is seeded from the
// demo/example ids only, not a live directory listing.
export function browsableCaseIds() {
  return fixtureCaseIds();
}

export function demoManifest() {
  return fixtureManifest();
}
