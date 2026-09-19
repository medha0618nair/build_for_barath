// Bundles frontend/fixtures/ at build time so the app runs standalone with
// no API and no network (Amplify static hosting, `npm run dev` offline).
// Mirrors the three routes in api/openapi.yaml exactly, so swapping to
// dataClient's real-API path is a drop-in replacement.

const caseModules = import.meta.glob("../../fixtures/cases/*.json", { eager: true });
const linkModules = import.meta.glob("../../fixtures/links/*.json", { eager: true });
const manifestModule = import.meta.glob("../../fixtures/manifest.json", { eager: true });
const feedbackExampleModule = import.meta.glob("../../fixtures/feedback_example.json", { eager: true });

function unwrap(mod) {
  return mod.default ?? mod;
}

const casesById = {};
for (const [path, mod] of Object.entries(caseModules)) {
  const id = path.match(/([^/]+)\.json$/)[1];
  casesById[id] = unwrap(mod);
}

const linksByKey = {};
for (const [path, mod] of Object.entries(linkModules)) {
  const key = path.match(/([^/]+)\.json$/)[1]; // "{caseId}_{scope}"
  linksByKey[key] = unwrap(mod);
}

const manifest = unwrap(Object.values(manifestModule)[0]);
const feedbackExample = unwrap(Object.values(feedbackExampleModule)[0]);

export function fixtureCaseIds() {
  return Object.keys(casesById).sort();
}

export function fixtureManifest() {
  return manifest;
}

export async function fixtureGetCase(id) {
  const record = casesById[id];
  if (!record) {
    const err = new Error(`no such case ${id}`);
    err.status = 404;
    throw err;
  }
  return record;
}

export async function fixtureGetCaseLinks(id, scope = "same") {
  const record = linksByKey[`${id}_${scope}`];
  if (!record) {
    const err = new Error(`no fixture for case ${id}`);
    err.status = 404;
    throw err;
  }
  return record;
}

export async function fixturePostFeedback(payload) {
  // No backend in fixtures mode: echo back the shape /feedback would
  // return (api/serve_fixtures.py does the same), stamped locally.
  return {
    ...payload,
    pair_id: `${payload.case_id_a}#${payload.case_id_b}#${Date.now()}`,
    submitted_at: new Date().toISOString(),
  };
}

export function fixtureFeedbackExample() {
  return feedbackExample;
}
