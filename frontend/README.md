# Frontend

Analyst-facing UI for the ranked shortlist described in `TECHNICAL_SPEC.md`
and `api/openapi.yaml`. Vite + React + Tailwind + Recharts.

## Run standalone (no API)

```bash
npm install
npm run dev
```

Case and link data is bundled at build time from `frontend/fixtures/` via
`import.meta.glob` (see `src/lib/fixturesClient.js`), so this works fully
offline. `POST /feedback` is simulated locally in this mode and doesn't
persist anywhere — it just echoes back the record shape the real API
would return (matching `api/serve_fixtures.py`).

To exercise the actual three-route contract instead of the bundled fixtures
(still no AWS), run the fixture server from the repo root and point the
app at it:

```bash
python -m api.serve_fixtures --port 8000
VITE_API_URL=http://localhost:8000/v1 npm run dev
```

## Point at the real API

```bash
VITE_API_URL=https://<api-id>.execute-api.ap-south-1.amazonaws.com/v1 npm run dev
```

Everything routes through `src/lib/dataClient.js`; no other file knows
whether it's talking to fixtures or a live API.

## Build

```bash
npm run build   # outputs dist/, deployable to Amplify Hosting as a static build
```

Set `VITE_API_URL` as an Amplify build-time environment variable to point
a deployed build at the real API; leave it unset to ship the
fixtures-only demo build.

## Notes

- **Never renders a probability** (CLAUDE.md hard rule 4). Every link shows
  rank, signed bits, and the driving field-level contributions instead.
- **Cluster timeline** (`src/components/ClusterTimeline.jsx`) is derived
  client-side from the current case's own ranked links above a bit
  threshold — there's no `/clusters` route in `api/openapi.yaml` yet
  (`TECHNICAL_SPEC.md`'s cut order dropped the persisted batch cluster job
  for this phase). It's labelled as derived, not presented as the real
  transitive cluster output.
- **Demo mode** loads `relocated_demo_pair` from `frontend/fixtures/manifest.json`.
  That pair doesn't clear either case's top-50 shortlist in this corpus —
  it's the product's own honest-limitations case (`TECHNICAL_SPEC.md` §7),
  not a scored success story, so the UI shows it as a direct field
  comparison with no bits rather than fabricating a score.
