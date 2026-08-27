# 8/23/2026
- added docs folder to pitching_plus/
- created purpose.md and outlined preliminary thoughts in there
- created dev_log.md to log work
- created stuff.ipynb, no changes made

# 8/24/2026
- Cell(s) under "Data Loading and Packages" created
- Started feature engineering, needs to run quicker

# 8/25/2026
- Stuff+: Attempting to optimize calculation for features
- Stuff+: Proofread outputs and logic for entropy
- Stuff+: Replaced all `.iterrows()`/`.apply(axis=1)` feature calculation with vectorized
  pandas/numpy (merge-based pairwise comparison + sparse wide pivot for the arsenal
  "_vs_" columns). Also localized arsenal comparisons to game_year (was pooling all
  5 seasons per pitcher) and dropped junk/non-standard pitch types (FA/EP/FO/KN/CS/SC/PO/UN)
  before feature engineering.
- Verified vectorized output is bit-identical to the old row-wise code on a real subset
  (13 pitchers, ~35k rows) before rewriting the notebook.
- Full run on all 3.55M rows: ~53s end-to-end, 13.5GB peak RAM (was previously unable
  to finish). df_v2 is 3,526,127 x 3,750 columns, with the wide arsenal-comparison block
  stored as sparse (pandas SparseDtype) since it's ~1-2% dense.

# 8/26/2026
- Stuff+: reframed as a standardized composite index rather than a trained model
  (matches purpose.md -- Stuff+ is pure physics "outlierness," no outcome data needed;
  that's Location+/Pitching+'s job).
- Explored PCA for automatic feature weighting instead of hand-picked weights. Found
  PC1 alone explains ~40-51% of variance by pitch type (not dominant), and that
  including a redundant feature (reaction time is r=0.99 with velocity) inflates
  weights -- deduped down to: release_speed, release_spin_rate, release_extension,
  acceleration_mag, horizontal_acceleration, and a new derived feature
  movement_per_reaction_time (horizontal_acceleration / time_30ft, i.e. purpose.md's
  "Reaction x Movement" concept) -- which ended up the single highest-loading feature
  on PC1 across every pitch type.
- Tested for a genuine velocity x movement synergy (beyond additive credit) by adding
  an explicit z_velocity * z_movement term to the PCA. It loads weakly on PC1 (mostly
  under 0.2) and lands on PC2 instead -- a separate, secondary signal, not part of the
  main "how good is this pitch" axis. Decided to keep the composite purely additive
  (PC1 only) rather than hand-add the synergy term, since additive credit for
  "movement given fixed velocity" (and vice versa) is already confirmed present
  (checked directly: movement still varies within a narrow 95.0-95.2mph FF band).
- Building the 100+ scale conversion: composite scored per pitch, but the "100 =
  average" calibration is done on (pitcher, pitch_type, game_year) aggregates, not
  raw pitches -- matches how real "+" stats (wRC+, ERA-) are reported, and purpose.md's
  "aggregated to pitch type." Using a genuine ratio transform (exp(k*z), renormalized
  to a league-average of exactly 100) rather than the industry-typical "100 + 10*z"
  SD-relabeling, so "101 = 1% better" is literally true, not just a labeling
  convention.
- Also need pitch-level Stuff+ (not just the pitcher-season average) since
  Pitching+/bestPitch+ combine Stuff+ with per-pitch situational Location+. Solved by
  calibrating BOTH levels off the same reference distribution (built only from
  "reliable" pitcher-seasons with >=20 pitches, so noisy small samples can't skew the
  league mean/SD) -- averaging a pitcher's pitch-level scores for a season lands within
  ~0.3 points of their aggregate score (small, expected Jensen's-gap bias from the
  nonlinear exp transform, up to ~2.7 points in rare cases).
- Wrote the full Stuff+ scoring section into stuff.ipynb (config, per-pitch PCA
  composite, aggregate + calibration + dual-level scale) and ran the complete
  pipeline end-to-end (V1 -> V2 -> Stuff+) on all 3.55M rows: ~184s, 18.4GB peak RAM.
  Produces stuff_df (pitch-level, for Pitching+/bestPitch+ later) and
  pitcher_stuff_plus (14,343 pitcher x pitch_type x season rows, the headline figure).
- Noticed the actual Stuff+ scoring cells only ever read from `df` (V1), never
  `df_v2` -- the V2 arsenal pairwise/sparse block wasn't actually feeding the
  score. Before dropping it from the production script, tested whether it should:
  added the cheap arsenal-contrast signal (`usage_weighted_distance_30ft`, no full
  sparse pivot needed) to STUFF_FEATURES and reran scoring on all 3.55M rows to
  compare against V1-only. Result: Spearman rho 0.998 overall (0.97-1.0 per pitch
  type), mean |diff| 0.36 points (max 8.1 in a 62-row group), 12-15/15 top-15
  overlap per pitch type -- no meaningful ranking change. Decided to ship V1-only
  for now; the V2 pairwise block can come back later if something (e.g.
  Pitching+/bestPitch+) actually needs arsenal-relative features directly.
- Moved the cleaning + V1 feature engineering + Stuff+ scoring flow out of
  stuff.ipynb and into pitching_plus/stuff.py, as the first stage of the
  pipeline scripts (stuff.ipynb stays as the exploration/derivation notebook).
  Public API is `add_stuff_plus(raw_df) -> raw_df`: takes a raw Statcast
  dataframe and returns it unchanged in row count/order with `pitch_stuff_plus`
  (pitch-level), `stuff_plus` (pitcher x pitch_type x season aggregate,
  broadcast per pitch), and `stuff_plus_reliable` added -- NaN for out-of-scope
  pitches (junk types, incomplete physics, or pitch types too rare to
  calibrate). Also runnable standalone via CLI (`python stuff.py --input ...
  --output ...`).
- Started pitching_plus/full_pipeline.py (WIP): loads the raw CSV, calls
  `add_stuff_plus`, writes the output, with TODOs for Location+/Pitching+/
  bestPitch+ once those scripts exist.
- Smoke-tested `add_stuff_plus` on a 20k-row sample (import, shape, NaN
  handling all correct) but have NOT yet run stuff.py/full_pipeline.py on the
  full 3.55M-row dataset -- next step.
- Started Location+ in docs/location.ipynb (purpose.md: expected run value of a
  pitch thrown into a location, given the situation -- RE288, pitch type,
  batter/pitcher handedness). Confirmed Statcast's `delta_run_exp` is itself
  computed from an RE288-consistent table (24 base-out states x 12 counts),
  not just RE24 -- checked directly: holding base-out state fixed (empty/0
  outs), a called strike is worth -0.040 runs at 0-0 but -0.344 at 3-2, and a
  ball is worth +0.040 at 0-0 but +0.361 at 3-2. So Location+ uses
  `delta_pitcher_run_exp` (= -delta_run_exp, sign-flipped so positive favors
  the pitcher) directly as its target rather than rebuilding a run-expectancy
  matrix from half-inning outcomes.
- Unlike Stuff+, this is an actual trained model per purpose.md ("given pitch
  type and situation, expected run value of a pitch thrown into a certain
  area"): per pitch type, HistGradientBoostingRegressor predicting
  delta_pitcher_run_exp from location (plate_x, batter-zone-relative plate_z,
  arm-side-adjusted plate_x), count, base-out state, an explicit RE288 state
  id, and handedness (stand, p_throws, platoon matchup flag). Every pitch's
  `location_run_value` comes from 5-fold out-of-fold CV, not the same-data fit
  -- the whole point of Location+ is the model's expectation for that
  situation, not the noisy realized single-pitch outcome (a well-located pitch
  can still get bloop-hit). Final per-type models are refit on all data for
  future scoring.
- Applied the same dual-level 100+ calibration as Stuff+ (reliable pitcher-
  seasons >=20 pitches set the (pitch_type, season) reference; exp(k*z)
  renormalized to a league average of exactly 100; same constants applied to
  pitch-level and pitcher-season-level scores). Outputs: `location_run_value`
  (raw expected runs, positive = good for pitcher) and `location_plus` /
  `pitch_location_plus` (100+ scaled).
- Ran end-to-end on the full 3.55M-row dataset: ~94s. R^2 is low but real and
  consistent across pitch types (0.02-0.04, corr 0.14-0.21) -- expected, since
  most single-pitch run-value variance is swing decision/contact quality the
  model has no access to; the aggregated pitcher-season signal is still
  meaningful (14,357 reliable pitcher-type-seasons scored). location.ipynb is
  exploration only so far -- no location.py yet.
- Per-pitch-type Location+ leaderboard is hard to trust at face value (top of
  the list is small samples like Blake Treinen's 22 sinkers in 2022 --
  22 pitches isn't a real read on command). Added an "Overall Leaderboard"
  section to location.ipynb: average the raw `location_run_value` (directly
  comparable in run units across pitch types, no per-type calibration needed)
  across EVERY in-scope pitch a pitcher threw that season, then calibrate that
  seasonal average against all reliable pitcher-seasons that year (100 =
  league average across the whole arsenal, same season) -- naturally usage-
  weighted since a pitch type thrown more often contributes more rows to the
  average. Used a much higher reliability bar for this (>=100 pitches on the
  season, vs. 20 for the per-type breakdown). Result looks far more credible:
  reliable samples jump to 105-1,027 pitches, and the top of the list is
  known-command guys (Ryu, Cobb, Chris Martin) instead of one-off small
  samples. Still, "best command" landing on some fairly middle-of-the-road
  names is the expected symptom of Location+ not yet being weighted by how
  much a mistake actually costs given the pitch's Stuff+ / the situation --
  that reconciliation is explicitly Pitching+'s job, not something to force
  into Location+ itself.
- Ported location.ipynb into pitching_plus/location.py, mirroring stuff.py's
  shape: public API is `add_location_plus(raw_df, models_dir=..., retrain=
  False) -> raw_df`, adding `location_run_value`, `pitch_location_plus`
  (per pitch_type/season, 100+), `location_plus` and `location_plus_reliable`
  (the season-overall "final" score, 100+) -- NaN for out-of-scope pitches,
  row count/order unchanged. Reuses PITCHER_COL/PITCH_TYPE_COL/SEASON_COL/
  JUNK_PITCH_TYPES/MIN_PITCHES_FOR_SCORE from stuff.py instead of redefining
  them.
- Training the per-pitch-type gradient-boosted models with 5-fold CV is the
  expensive step (94s-340s on the full dataset, depending on system load --
  timed two identical runs and saw a >3x spread from contention alone), so it
  shouldn't happen on every pipeline run. Cached under pitching_plus/models/
  (joblib, gitignored -- regenerable from data/, doesn't belong in git):
  the fitted per-pitch-type models (for scoring genuinely new pitches later)
  AND the out-of-fold historical scores keyed by (game_pk, at_bat_number,
  pitch_number) -- a real unique-pitch key, verified no nulls/duplicates.
  This matters because the cached *model*'s own .predict() on data it was
  trained on would be in-sample (mildly optimistic, exactly the leakage the
  notebook's OOF approach was built to avoid) -- caching the OOF scores
  themselves means re-running the pipeline on the same historical data keeps
  the same rigor without re-paying the CV cost. A pitch not found in the
  cache (new data added later) falls back to direct model.predict(), which is
  the correct/unavoidable thing to do for a pitch the model hasn't seen.
  Smoke-tested both paths on a 300k-row sample: train call ~37s, cached call
  ~2.5s, identical location_run_value between the two.
- Added pitching_plus/location.py's call to full_pipeline.py, after
  add_stuff_plus.
- Found (and fixed) a real memory bug while training the full model cache for
  the first time: `add_stuff_plus`/`add_location_plus` both built their
  in-scope working frame as `raw_df[row_filter].copy()` -- keeping ALL ~119
  raw Statcast columns through every intermediate step, not just the ~15-20
  actually needed. Fine on the trimmed exploration scripts (which only ever
  loaded a handful of columns via usecols), but full_pipeline.py reads the
  ENTIRE raw CSV -- running location.py's training pass against it multiplied
  that 119-column width across several full in-memory copies (in_scope,
  engineered, scored, result...) and peaked at ~14.9GB, dropping system free
  memory to 0.7GB/31.7GB before it was caught and killed. Fixed in both
  scripts by selecting only REQUIRED_COLS (+ BASE_STATE_COLS for location.py)
  immediately when building in_scope, rather than keeping the full row width;
  peak main-process memory on the same full run dropped to ~6.6GB. Retrained
  the full cache afterward with the fix: `python location.py` end-to-end on
  all 3,565,743 raw pitches, 3,540,371 scored, memory stayed healthy
  throughout (system free memory never dropped below ~11GB).
- Reorganized pitching_plus/ into subfolders: scripts/ (stuff.py, location.py,
  full_pipeline.py) and notebooks/ (stuff.ipynb, location.ipynb) -- docs/
  (dev_log.md, purpose.md) and models/ (gitignored cache) stay where they
  were, directly under pitching_plus/. stuff.ipynb was already git-tracked,
  so moved with `git mv` to preserve history; everything else was untracked
  and moved with a plain `mv`. Updated every path reference that depended on
  the old layout: stuff.py/location.py/full_pipeline.py's default data/
  input-output paths (Path(__file__).resolve().parent...) needed one more
  `.parent` now that scripts/ is a level deeper than pitching_plus/ used to
  be; location.py's DEFAULT_MODELS_DIR needed the same adjustment to still
  resolve to pitching_plus/models/ (not pitching_plus/scripts/models/); and
  docstring/comment mentions of "docs/location.ipynb" and bare "stuff.py"
  paths were updated to notebooks/location.ipynb and scripts/stuff.py.
  Verified after the move: DEFAULT_MODELS_DIR resolves to the existing cache
  and full_pipeline's default input/output resolve correctly; add_stuff_plus
  and add_location_plus both ran clean from pitching_plus/scripts/ against a
  sample (location.py correctly loaded the real cached models rather than
  retraining). .gitignore's `pitching_plus/models/` entry needed no change,
  since models/ didn't move.
- Started notebooks/pitching.ipynb: a Pitching+-prep notebook that reconciles
  Stuff+ against Location+ empirically -- how much of realized run value
  (`delta_pitcher_run_exp`) each one actually explains, individually and
  combined -- ahead of building purpose.md's Pitching+ stage. Loads data
  through add_stuff_plus/add_location_plus (no logic reimplemented) and
  reasons about the output with correlation, pitch-level OLS, pitcher x
  pitch-type x season WLS, PCA, a season holdout, and a pitch-sequencing
  significance test.
- Found (and fixed) a real bug in stuff.py's pitch-level score assignment
  while sanity-checking pitching.ipynb's inputs: a pitch's own release_speed
  should correlate strongly positively with its own pitch_stuff_plus (PCA
  composite is sign-anchored on release_speed), and it didn't -- ~0 for every
  pitch type. Root cause: `_score_stuff_plus`'s
  `stuff_df = stuff_df.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")`
  silently resets stuff_df's index to a fresh RangeIndex, and
  `add_stuff_plus` reattaches pitch_stuff_plus back onto raw_df
  *positionally* via `result.loc[stuff_df.index, ...]` -- after the reset,
  that index no longer points at the original rows, so pitch_stuff_plus
  landed on the wrong pitches for most of the dataset. `stuff_plus` (the
  pitcher-season aggregate) was unaffected, since it's reattached via an
  actual key-based merge on pitcher/pitch_type/season rather than
  positionally -- exactly why this stayed invisible in stuff.py's own output
  and only surfaced once pitch-level Stuff+ was checked against something
  external to it. Fixed by saving/restoring stuff_df's index around the
  merge, mirroring the identical fix already present in location.py's
  `_calibrate`. Confirmed via the same sanity check: own-release_speed vs.
  own-pitch_stuff_plus correlation went from ~0 to 0.60-0.86 across pitch
  types (matches location.py's diff to `_calibrate`, which had already
  caught the same class of bug there). No cached artifacts needed
  invalidating -- stuff.py doesn't cache anything, and location.py's cache
  only stores `_train_models`'s output, not `_calibrate`'s.
- That fix changed pitching.ipynb's results substantially -- with correct
  pitch-level scores, Stuff+ has a real, significant relationship with run
  value (aggregate R^2 ~0.02 alone, both R^2 ~0.08 combined with Location+,
  up from ~0.05 for Location+ alone), and the two combine for a better,
  holdout-validated fit than either alone.
- Reviewed FanGraphs' Stuff+/Location+/Pitching+ primer
  (library.fangraphs.com) for methodology to cross-check against. Two of its
  claims motivated new checks in pitching.ipynb: (1) Stuff+ and Location+
  stabilize at very different pitch counts (~80 vs. ~400) -- swept the
  reliability bar from 20 to 600 pitches and found Stuff+'s R^2 climbs with
  more data and overtakes Location+'s past a few hundred pitches, while
  Location+'s is already close to its ceiling at 20-50; (2) Stuff+ is
  reported as far more year-over-year sticky than Location+ -- confirmed
  directly (season Y -> Y+1 correlation ~0.89 for Stuff+ vs. ~0.41 for
  Location+ on pitcher-pitch-type-season aggregates). Also noted the primer
  states real Pitching+ is not a weighted average of Stuff+ and Location+ but
  a separately trained ("third") model on physical + location + count
  features against run value -- folded into pitching.ipynb's closing
  discussion as the more principled direction for the actual Pitching+ build,
  vs. this notebook's regression-based weight as a reasonable starting point.

# 8/27/2026
- New branch feat/fangraphs-style-pitching, off main: builds the FanGraphs-style
  Pitching+ pitching.ipynb's conclusion flagged but didn't build -- one jointly
  trained model per pitch type on physics + location + count features together,
  rather than a weighted blend of the two separately-computed scores. Added
  notebooks/fg_pitching.ipynb. Feature engineering is reused directly from
  stuff.py's `_build_v1_features` and location.py's `_build_features` (not
  reimplemented); the joint model is a `HistGradientBoostingRegressor` per pitch
  type with the exact same hyperparameters and 5-fold OOF-CV scheme
  location.py's `_train_models` already uses, just fed `STUFF_FEATURES +
  LOCATION_FEATURES` instead of `LOCATION_FEATURES` alone. Calibrated to the
  same 100+ ratio scale via location.py's own `_ratio_calibration`/
  `_to_100_scale` helpers, at the same pitch/pitcher-pitch-type-season dual
  level as Stuff+/Location+. Top of the reliable leaderboard: Devin Williams'
  changeup (2025, 148.4) -- a sensible result given the pitch's reputation.
- Compared the joint model against a weighted-average blend (literally
  purpose.md's "reward maximized Location+ w/ weight for Stuff+" framing) on
  a strict 2021-2024-train / 2025-test split, built to be leak-checked in both
  directions: the joint model only ever trains on 2021-2024, and -- since
  location.py's cached production model is cross-validated across all five
  seasons pooled together, which would hand the blend an unfair look-ahead
  edge on a 2025 test -- the blend's Location+ input is a from-scratch
  train-only refit via location.py's own `_train_models`, not the cached
  model. Noted one remaining, much smaller asymmetry left as-is: Stuff+'s PCA
  loadings are fit pooled across all five seasons by stuff.py's existing
  design, not re-derived train-only.
- Result: the weighted-average blend generalizes better (train R^2=0.095,
  2025 holdout R^2=0.052) than the joint model (train R^2=0.132, holdout
  R^2=0.039), despite the joint model's richer feature access. The joint
  model's holdout shrinkage is much steeper (71% relative drop vs. the
  blend's 45%) -- consistent with overfitting from its far larger parameter
  count (a full GBM ensemble per pitch type over 19 features vs. an
  intercept and two slopes) against a target this noisy. Recorded as the
  headline finding for this branch; not merged into main.
- Followed up rather than shelving the joint model: added a regularization
  sweep to fg_pitching.ipynb testing whether tuning HistGradientBoostingRegressor's
  hyperparameters (left unchanged from location.py's originals for the first,
  fair comparison) closes the gap. Three configs, identical per-pitch-type
  5-fold OOF-CV and aggregate/holdout evaluation harness as the original
  comparison: shallower trees + much stronger L2 (max_depth=3,
  l2_regularization=10.0, min_samples_leaf=50), aggressive early stopping
  (max_iter=1000, validation_fraction=0.2, n_iter_no_change=5), and both
  combined.
- Result: none close the gap to the blend's 0.0523 holdout R^2. Aggressive
  early stopping alone is the best variant (holdout R^2 0.0386 -> 0.0417,
  ~8% relative gain, shrinkage 71% -> 67%) -- some support for the original
  fixed max_iter=300 having run a bit past the point of real generalization.
  Shallow+L2 alone barely moves it (0.0395). Combining both regularizers is
  worse than either alone (train R^2 drops to 0.1204, the lowest of any
  joint variant, holdout drops to 0.0380) -- stacked regularization tips
  into underfitting rather than compounding.
- Conclusion: rules out "the joint model just needed more regularization" as
  the explanation for the original result. Points instead to a structural
  reason no amount of hyperparameter tuning on this same model family
  reaches: the blend's inputs (Stuff+, Location+) are each already
  calibrated/regularized per-season summaries, not raw per-pitch features
  fit flat against four prior seasons. Doesn't rule out a joint model in
  general (more data, a different regularization family, or better inputs
  could change this) -- but for this repo's current data/tools, the
  weighted-blend recommendation from the first synopsis stands as the basis
  for building Pitching+ next.

# 8/27/2026 (cont'd): Pitching+ and bestPitch+ shipped
- Ported the validated weighted-blend recipe into pitching_plus/pitching.py:
  `add_pitching_plus(raw_df) -> raw_df`, mirroring stuff.py/location.py's
  shape. One pooled WLS blend (z-scored log(stuff_plus) + mean
  location_run_value against delta_pitcher_run_exp, weighted by n_pitches),
  fit fresh on every call like stuff.py's PCA (cheap -- a few thousand
  aggregate rows, no need for location.py's model-caching machinery). Applies
  the same fitted weights at both the pitch level and the (pitcher,
  pitch_type, season) aggregate level, then calibrates both to the 100+ ratio
  scale via location.py's own `_ratio_calibration`/`_to_100_scale` (reused,
  not reimplemented). Wired into full_pipeline.py after add_location_plus.
  Ran end-to-end on the full 3,565,743-row dataset (chained with the existing
  add_stuff_plus/add_location_plus): reliable-population calibration lands
  exactly on 100 per (pitch_type, season) as designed; top of the leaderboard
  (Treinen 2022 sinker 143.0, Murray 2025 sinker 141.3, Montero 2022 changeup
  135.8) are plausible names/pitches, not small-sample noise.
- Refactored pitching.py's internal blend fit into `_fit_pitching_plus_model`
  (returns blend_params + the 100+ calibration table + pitcher_agg) and
  `_score_pitching_plus` (pure function: applies an already-fitted
  blend/calibration to arbitrary stuff/location inputs) -- needed so
  bestpitch.py can score *hypothetical* pitches on the exact same fitted
  scale as real ones, not a separately re-derived one. Confirmed via the
  full pytest suite that add_pitching_plus's output is unchanged after the
  refactor.
- Built pitching_plus/bestpitch.py for purpose.md's last stage: "given
  pitcher arsenal, calculate hypothetical Pitching+ score (bestPitch)" then
  `bestPitch - Pitching+ = bestPitch+`. Design: for each in-scope pitch, hold
  the real situation fixed (count, outs, base state, handedness) and search
  every (candidate pitch type from that pitcher's own RELIABLE arsenal that
  season x candidate zone) combination -- zones are Statcast's real `zone`
  1-9 in-zone codes (purpose.md's "zone, not pinpoint"), represented by that
  zone's dataset-wide average (plate_x, plate_z_rel), not a synthetic point
  or a per-pitch-type-specific location (a pitch type's own typical spot
  within a zone is a targeting choice the location model itself already
  captures). Each candidate is scored by combining the pitcher's own real
  stuff_plus for that pitch type with location.py's own cached model's
  prediction for that zone (added `location.load_cached_models` to expose
  the cache for this), run through pitching.py's `_score_pitching_plus` --
  the identical fitted blend/calibration as real Pitching+ scores, so
  "best" and "actual" are directly comparable. The max across all valid
  candidates is `best_pitching_plus`.
- Found and fixed a real scope-filtering bug while building this: initially
  folded BASE_STATE_COLS (on_1b/on_2b/on_3b) into the dropna gate alongside
  the rest of REQUIRED_COLS, which wrongly required all three bases occupied
  to keep a row (NaN there is a real "base empty" state, per location.py's
  own established convention) -- on a synthetic test fixture this dropped
  ~98% of rows (19/900 scored) before the fix. Caught by smoke-testing
  against the same synthetic fixture used for stuff.py/location.py's tests
  before writing formal pytest cases; fixed by keeping BASE_STATE_COLS out of
  the dropna subset, exactly mirroring location.py's own in-scope filtering.
- Result on the same synthetic fixture (900 in-scope pitches, 3 pitch types,
  10 pitchers): best_pitching_plus meets or exceeds the realized
  pitch_pitching_plus for ~97.6% of pitches. The ~2.4% exceptions are
  expected, not a bug: `best_pitching_plus` uses a *zone-average* location
  for its candidates, so a real pitch placed unusually well within its own
  zone can occasionally out-score the zone-average hypothetical for that
  same zone.
- bestPitch+ (`bestPitch - Pitching+`) is a maximum-over-candidates quantity,
  so it's >= the realized value almost by construction -- positive or close
  to 0, confirmed at ~97.6% non-negative on the synthetic fixture and ~99.1%
  on the full 2021-2025 dataset. Corrected purpose.md's original draft note
  (it had stated the opposite sign) and the bestpitch.py/bestpitch.ipynb text
  to match.
- Ran add_bestpitch_plus end-to-end on the full 3,565,743-row dataset
  (against already-scored stuff/location/pitching columns, so this run only
  paid the counterfactual search's own cost): 910s, 3,521,353 pitches scored.
  Built notebooks/bestpitch.ipynb in the same style as the others -- scores
  the full pipeline, sanity-checks the zone reference grid against Statcast's
  real 1-9 layout, checks the bestPitch+ sign distribution, and a
  pitcher-season "closest to optimal pitch selection" leaderboard (same
  MIN_PITCHES_FOR_SEASON_SCORE reliability bar as location_plus's own
  whole-arsenal score).
- Added tests/test_pitching_plus.py coverage for both pitching.py (missing
  columns, junk-type NaN, reliable-population calibration lands on exactly
  100, and a check that add_pitching_plus reuses already-present
  stuff/location columns instead of recomputing them) and bestpitch.py
  (missing columns, junk-type NaN, best-meets-or-exceeds-actual on >90% of
  in-scope pitches). Added a `zone` column to the shared synthetic fixture
  (tests/conftest.py) via a rough 3x3 plate_x/plate_z_rel grid, matching
  Statcast's real 1-9 numbering closely enough to exercise the zone-grid
  logic (not intended to test real zone semantics).