# Task: synthetic corpus generator

Read `CLAUDE.md` and `TECHNICAL_SPEC.md` first. This brief covers only the
generator — the first thing to build. Nothing else exists yet.

## Fix these before starting

1. `TECHNICAL_SPEC.md` §4.4 — all-property-crime prior is **−16.35**, not
   −16.38. (Verified: the gap from the burglary-only prior is exactly 2.000
   bits, since 4× the cases means log2(4) = 2 bits of prior.)
2. `CLAUDE.md` points to `docs/SPEC.md`; the file is `TECHNICAL_SPEC.md` at
   root. Either move it or fix the pointer.
3. `evaluate.py` appears in commands but not in the layout. Add it.
4. `anthropic.claude-haiku-4-5` is not a full Bedrock model ID. Do not guess
   the versioned form — run
   `aws bedrock list-foundation-models --region ap-south-1` and use exactly
   what returns. Same for the Cohere embedding model. If nothing returns,
   model access isn't enabled, which is the real first build step.
5. `git init` — this folder is not currently its own repo.

## What this builds

A generator producing a synthetic multi-state property-crime corpus with
ground-truth offender IDs, for training and evaluating the linkage scorer.

No public dataset contains "these two crimes share an offender," so synthetic
is not a fallback — it's the only source of the label. Everything downstream
depends on this being right.

## Division of labour

**You write:** the sampling chain, corruption layer, renderer, validators,
manifest. All mechanical.

**Nithin writes:** `loadings.yaml` and the provenance tags in
`marginals.yaml`. The loadings matrix is the behavioural model — every
non-zero entry is a claim he has to defend. Scaffold it with the full field
structure and exactly two filled examples, then stop and hand it back.

Do not fill in the loadings matrix.

## Settled design decisions

Do not relitigate these; they were worked through already.

### Three-level sampling chain

```
p   population marginals          (config)
 ↓   tilt by offender style
q   offender's centre
 ↓   spread by repeat_rate
θ   offender's realised mix
 ↓   sample
x   one crime's true behaviour
```

Tilting: `q_v ∝ p_v · exp(τ · Σ_k s_k · loading_v,k)`, normalised per field.

Why this and not independent Dirichlets per field: one style vector tilts
entry method, tools, search pattern and counter-forensics **simultaneously**,
so correlation emerges without a correlation matrix. The scorer assumes
conditional independence. The generator must violate it, or the evaluation
just measures whether the model can invert its own assumptions.

### Style axes

Three: force↔stealth, opportunistic↔planned, solo↔group. Styles drawn from a
zero-centred normal. An offender at s = 0 reproduces `p` exactly.

### Series shape

- Length: **negative binomial**, not Poisson. Real criminal careers are
  heavily skewed — mostly short series, a thin long tail. Poisson gives a
  tight band around the mean and removes the population the system exists
  to catch.
- Gaps: two-component mixture, roughly 70% short (mean ~20d) and 30% long
  (mean ~120d). Offending is bursty; pure exponential spreads too evenly.
- Relocation: dormancy gap of 90–365 days inserted at the relocation index.
  State changes there.

### Occurrence as an interval

`occurred_from` / `occurred_to`, width driven by occupancy — hours when
occupants were home, days-to-weeks for a locked vacant house.
`registered_at` follows `occurred_to` by a short delay.

When the window is wide, `time_band` is **unknowable**, which is different
from **missing**. Encode them distinctly; the scorer must treat them
differently.

### Corruption (per state, three independent mechanisms)

- **Confusion** — replace with a *confusable neighbour*, never a random
  value. Needs a small confusability graph per field.
  `lock_broken ↔ forced_open` is plausible recording divergence;
  `lock_broken ↔ no_force` is not.
- **Missingness** — per-state, per-field dropout rates, differing across
  states.
- **Structural absence** — one state has no occupancy column at all. Pairs
  involving it score on fewer fields. This is a real effect and must survive
  into the data.

Recorded crime type is a noisy function of the true one, with confusion at
genuine boundaries (burglary ↔ commercial break-in, snatching ↔ robbery).

### Rendering

Feeds carry **state-native** vocabulary, not a shared schema with different
labels. `TALA_TODA` vs `LOCK BROKEN` vs `Forced - lock`. Different delimiter
per state for multi-valued fields. Different date formats. Different column
names and ordering. One state has a combined free-text `mo_description` where
another has structured fields.

If feeds share a vocabulary, the normalisation layer has nothing to do and
the "adding a state is a YAML diff" demo is theatre.

Narratives render from **recorded** values, not true ones, with two
deliberate mismatches: sometimes the narrative mentions something absent from
the structured fields (extraction adds information), sometimes a structured
field is populated but unmentioned in the narrative (extraction must return
null, not invent).

### Output: four artifacts

| File | Contents |
|---|---|
| `truth.parquet` | offender_id, style position, true values, true dates. **Never enters the pipeline.** |
| `feeds/{state}.csv` | recorded, corrupted, state-native. This is what gets ingested. |
| `offenders.parquet` | style, repeat_rate, series length, home/destination state, relocation index |
| `manifest.json` | seeds, all config used, config marginals **and realised marginals**, timestamp |

## Three gotchas — these were caught the hard way

### 1. α direction is not what it looks like

`repeat_rate` → α must be **inverse**. Small α = high repetition. Verified:

| α | P(two crimes by same offender agree) |
|---|---|
| 0.05 | 0.971 |
| 7.05 | 0.467 |
| 20.05 | 0.420 |
| ∞ | 0.391 (chance) |

Small α pushes θ toward a spike on one category, so every crime draws the
same value — a burglar who always enters by the roof. Large α makes θ track
the population and behaviour looks random.

This reads backwards to careful people. **Do not expose a raw `consistency`
parameter.** Define `repeat_rate` as target `m` at a reference frequency and
solve for α internally. `m_v = (α·q_v + 1)/(α + 1)`, monotonically decreasing
in α.

Getting this backwards makes the headline sweep plot show linkage getting
*worse* as offenders become more consistent.

### 2. Aggregate marginals drift, and that is expected

Averaging tilted distributions over offenders does **not** return `p` —
Jensen's inequality, since `exp` is convex and the normaliser varies.
Measured at τ = 1:

| value | config p | realised | drift |
|---|---|---|---|
| door | 0.550 | 0.536 | −2.5% |
| roof | 0.060 | 0.074 | **+23%** |
| wall_breach | 0.040 | 0.057 | **+42%** |

Drift concentrates in rare, heavily-loaded values — exactly the high-bit
fields. Scales steeply with τ: +2% at τ=0.25, +43% at τ=1.5.

Handling:
- Config marginals are the **style-neutral** distribution. Document this.
- Generate background one-offs at **exactly s = 0** so the validator has a
  clean target.
- Write realised corpus marginals into the manifest. Present measured
  numbers, never config numbers.
- **Pick τ by measurement:** sweep τ, compute cross-field MI, take the
  smallest τ giving meaningful correlation. Minimises drift while preserving
  the anti-circularity property. Ship this as a short script so the chosen
  value has a reason attached.

### 3. Measure MI in the recorded table, not just truth

The anti-circularity argument is that the generator correlates fields while
the scorer assumes independence. But corruption sits between them. If
per-state confusion and dropout wash out the correlation, conditional
independence becomes approximately *true* by the time data reaches the
scorer, and the evaluation silently becomes the mirror it was designed to
avoid.

Track MI in truth, MI in recorded, and the gap. Collapsing recorded MI means
corruption rates are too high.

## Validators — run every regeneration

1. Background one-offs (s = 0) reproduce `marginals.yaml`. Validates the
   tilting maths.
2. Cross-field MI in truth is clearly above zero. Near zero means the
   loadings aren't doing anything.
3. Cross-field MI in the recorded table has not collapsed.
4. **Cheat detector** — train on metadata only (row position, case_id, raw
   timestamp, police station). Above chance means leakage. Also check state
   is not predictable from post-normalisation features.
5. Series lengths are heavy-tailed. A tight bell around the mean means
   Poisson snuck in.
6. Per-state marginals and corruption/dropout rates match config.

## Order of work

1. Doc fixes above, `git init`, Bedrock model access check
2. Config scaffolding — `marginals.yaml` (with provenance tags),
   `loadings.yaml` (**stubbed, two examples**), `states.yaml`
3. **Stop.** Hand back for the loadings matrix.
4. Three-level sampler → validator 1 before anything else
5. τ selection script → validator 2
6. Timing, series shape, occurrence windows
7. Corruption layer → validator 3
8. Rendering to state-native feeds
9. Remaining validators, manifest

Checkpoint after step 4. If background one-offs don't reproduce the config
marginals, nothing downstream is trustworthy.

## Scale and reproducibility

10,000 cases for development, 40–50k for the final run. Seconds to generate
if sampling is vectorised and only offenders are looped. Small data — no
pipeline, no parallelism, no streaming.

One `numpy.random.default_rng(seed)` threaded through everything. No
`random.random()` anywhere. Separate seed stream for corruption so truth can
be held fixed while recording noise varies.

Three planned sweeps — repeat_rate 0.2→0.9, marginals, serial fraction — need
the same offenders with the same series structure differing only in the swept
parameter. Same seed, one changed config value.

Runs locally only. Never on AWS. `data/` is gitignored; configs and code are
committed so any corpus is reproducible from a seed plus a config.

## Definition of done

`python -m linkage.generate --seed 7 --out data/dev/` produces the four
artifacts, all six validators pass, and the manifest records both config and
realised marginals.
