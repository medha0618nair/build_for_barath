# Synthetic corpus

**Synthetic.** Every number measured on this corpus measures whether a model
can invert this generator — not real-world performance.

## Regenerate

```bash
python -m linkage.config                                          # must print READY
python -m linkage.generate --seed 7 --n-cases 45000 --out data/final/
python -m linkage.generate --seed 7 --out data/dev/               # 10k dev corpus
python -m linkage.generate.tau --n-cases 45000                    # re-derive τ
```

`data/final/` is committed so the corpus is usable without running anything;
the rest of `data/` is gitignored. Seed + `config/` reproduce any corpus exactly.

## Artifacts

| File | What | Enters the pipeline? |
|---|---|---|
| `feeds/{MH,MP,KA,TG}.csv` | Recorded, corrupted, state-native feeds | **Yes — ingest these** |
| `truth.parquet` | One row per case: `offender_id`, style, true values and dates, and the recorded canonical values (`rec_*`, `ingested`) | Never. Labels and normaliser answer key |
| `offenders.parquet` | Style, repeat_rate, alpha, series length, home/destination state, relocation index | Never |
| `manifest.json` | Seeds, full config, config vs realised marginals, link counts and priors, validator results, limitations | — |

Join feeds to truth on `case_id = sha1(state_code + fir_no + year)[:16]`.

`rec_*` tokens: `__MISSING__` blank cell · `__UNKNOWABLE__` occurrence window
too wide to know the time band · `__ABSENT__` state has no such column.

## Final corpus (seed 7, 45,000 cases)

- 1,809 serial offenders (628 relocate across a state border), 7,204 serial cases
- Feed rows: MH 15,620 · MP 10,779 · KA 9,839 · TG 8,295
- 467 cases recorded as ROBBERY and dropped at ingestion; 44,533 ingested
- 15,810 true same-offender pairs: 5,907 cross-type, 2,768 cross-state
- Prior, all property crime: −15.94 bits. Same-type: −15.08 (residential
  burglary) to −12.16 (ATM). Cross-type: −16.93.
- Top 10% of offenders hold 49.5% of true pairs — report per-offender metrics
  alongside per-pair ones.
- All six validators pass. The 10k dev corpus fails validator 3 (recorded MI
  not significant at that size); use the 45k corpus for evaluation.

## Limitations

- Every marginal is an **assumption** (see provenance tags); the loadings
  were drafted by Claude at the owner's request and need owner review.
- τ = 0.75 was chosen by the rule in `config/tau_selection.json`. Expected
  drift in serial crimes reaches +229% for rare heavily-loaded values
  (ATM explosive); corpus-level realised drift is in the manifest.
- Narratives are templated English. Cross-lingual retrieval is **not**
  demonstrated, and retrieval on templated text will look easier than on
  real FIRs.
- Background one-offs have style exactly 0 (brief decision), so rare,
  heavily-loaded values are over-represented among serial crimes: an artifact
  a scorer can exploit. The ATM same-type prior (−12.16) partly reflects it.
- Crime type is drawn through the style chain (proposal adopted), so style
  selects offenders into crime types and amplifies within-type drift.
- Non-links here are clean. Real unlabelled pairs are not confirmed
  non-links (positive-unlabelled); don't carry accuracy-style metrics over.
- Not checked: state predictability after normalisation (no normaliser yet).
