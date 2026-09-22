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
  (matches purpose.md: Stuff+ is pure physics "outlierness," no outcome data needed;
  that's Location+/Pitching+'s job).
- Explored PCA for automatic feature weighting instead of hand-picked weights. Found
  PC1 alone explains ~40-51% of variance by pitch type (not dominant), and that
  including a redundant feature (reaction time is r=0.99 with velocity) inflates
  weights. Deduped down to: release_speed, release_spin_rate, release_extension,
  acceleration_mag, horizontal_acceleration, and a new derived feature
  movement_per_reaction_time (horizontal_acceleration / time_30ft, i.e. purpose.md's
  "Reaction x Movement" concept), which ended up the single highest-loading feature
  on PC1 across every pitch type.
- Tested for a genuine velocity x movement synergy (beyond additive credit) by adding
  an explicit z_velocity * z_movement term to the PCA. It loads weakly on PC1 (mostly
  under 0.2) and lands on PC2 instead, a separate, secondary signal, not part of the
  main "how good is this pitch" axis. Decided to keep the composite purely additive
  (PC1 only) rather than hand-add the synergy term, since additive credit for
  "movement given fixed velocity" (and vice versa) is already confirmed present
  (checked directly: movement still varies within a narrow 95.0-95.2mph FF band).
- Building the 100+ scale conversion: composite scored per pitch, but the "100 =
  average" calibration is done on (pitcher, pitch_type, game_year) aggregates, not
  raw pitches. This matches how real "+" stats (wRC+, ERA-) are reported, and
  purpose.md's "aggregated to pitch type." Using a genuine ratio transform (exp(k*z),
  renormalized to a league-average of exactly 100) rather than the industry-typical
  "100 + 10*z" SD-relabeling, so "101 = 1% better" is literally true, not just a
  labeling convention.
- Also need pitch-level Stuff+ (not just the pitcher-season average) since
  Pitching+/bestPitch+ combine Stuff+ with per-pitch situational Location+. Solved by
  calibrating BOTH levels off the same reference distribution (built only from
  "reliable" pitcher-seasons with >=20 pitches, so noisy small samples can't skew the
  league mean/SD). Averaging a pitcher's pitch-level scores for a season lands within
  ~0.3 points of their aggregate score (small, expected Jensen's-gap bias from the
  nonlinear exp transform, up to ~2.7 points in rare cases).
- Wrote the full Stuff+ scoring section into stuff.ipynb (config, per-pitch PCA
  composite, aggregate + calibration + dual-level scale) and ran the complete
  pipeline end-to-end (V1 -> V2 -> Stuff+) on all 3.55M rows: ~184s, 18.4GB peak RAM.
  Produces stuff_df (pitch-level, for Pitching+/bestPitch+ later) and
  pitcher_stuff_plus (14,343 pitcher x pitch_type x season rows, the headline figure).
- Noticed the actual Stuff+ scoring cells only ever read from `df` (V1), never
  `df_v2`. The V2 arsenal pairwise/sparse block wasn't actually feeding the
  score. Before dropping it from the production script, tested whether it should:
  added the cheap arsenal-contrast signal (`usage_weighted_distance_30ft`, no full
  sparse pivot needed) to STUFF_FEATURES and reran scoring on all 3.55M rows to
  compare against V1-only. Result: Spearman rho 0.998 overall (0.97-1.0 per pitch
  type), mean |diff| 0.36 points (max 8.1 in a 62-row group), 12-15/15 top-15
  overlap per pitch type, no meaningful ranking change. Decided to ship V1-only
  for now; the V2 pairwise block can come back later if something (e.g.
  Pitching+/bestPitch+) needs arsenal-relative features directly.
- Moved the cleaning + V1 feature engineering + Stuff+ scoring flow out of
  stuff.ipynb and into pitching_plus/stuff.py, as the first stage of the
  pipeline scripts (stuff.ipynb stays as the exploration/derivation notebook).
  Public API is `add_stuff_plus(raw_df) -> raw_df`: takes a raw Statcast
  dataframe and returns it unchanged in row count/order with `pitch_stuff_plus`
  (pitch-level), `stuff_plus` (pitcher x pitch_type x season aggregate,
  broadcast per pitch), and `stuff_plus_reliable` added, NaN for out-of-scope
  pitches (junk types, incomplete physics, or pitch types too rare to
  calibrate). Also runnable standalone via CLI (`python stuff.py --input ...
  --output ...`).
- Started pitching_plus/full_pipeline.py (WIP): loads the raw CSV, calls
  `add_stuff_plus`, writes the output, with TODOs for Location+/Pitching+/
  bestPitch+ once those scripts exist.
- Smoke-tested `add_stuff_plus` on a 20k-row sample (import, shape, NaN
  handling all correct) but have NOT yet run stuff.py/full_pipeline.py on the
  full 3.55M-row dataset; that's next.
- Started Location+ in docs/location.ipynb (purpose.md: expected run value of a
  pitch thrown into a location, given the situation: RE288, pitch type,
  batter/pitcher handedness). Confirmed Statcast's `delta_run_exp` is itself
  computed from an RE288-consistent table (24 base-out states x 12 counts),
  not just RE24. Checked directly: holding base-out state fixed (empty/0
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
  `location_run_value` comes from 5-fold out-of-fold CV, not the same-data fit.
  That's the whole point of Location+: the model's expectation for that
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
  consistent across pitch types (0.02-0.04, corr 0.14-0.21). Expected, since
  most single-pitch run-value variance is swing decision/contact quality the
  model has no access to; the aggregated pitcher-season signal is still
  meaningful (14,357 reliable pitcher-type-seasons scored). location.ipynb is
  exploration only so far; no location.py yet.
- Per-pitch-type Location+ leaderboard is hard to trust at face value (top of
  the list is small samples like Blake Treinen's 22 sinkers in 2022; 22
  pitches isn't a real read on command). Added an "Overall Leaderboard"
  section to location.ipynb: average the raw `location_run_value` (directly
  comparable in run units across pitch types, no per-type calibration needed)
  across EVERY in-scope pitch a pitcher threw that season, then calibrate that
  seasonal average against all reliable pitcher-seasons that year (100 =
  league average across the whole arsenal, same season), naturally usage-
  weighted since a pitch type thrown more often contributes more rows to the
  average. Used a much higher reliability bar for this (>=100 pitches on the
  season, vs. 20 for the per-type breakdown). Result looks far more credible:
  reliable samples jump to 105-1,027 pitches, and the top of the list is
  known-command guys (Ryu, Cobb, Chris Martin) instead of one-off small
  samples. Still, "best command" landing on some fairly middle-of-the-road
  names is the expected symptom of Location+ not yet being weighted by how
  much a mistake costs given the pitch's Stuff+ / the situation.
  That reconciliation is explicitly Pitching+'s job, not something to force
  into Location+ itself.
- Ported location.ipynb into pitching_plus/location.py, mirroring stuff.py's
  shape: public API is `add_location_plus(raw_df, models_dir=..., retrain=
  False) -> raw_df`, adding `location_run_value`, `pitch_location_plus`
  (per pitch_type/season, 100+), `location_plus` and `location_plus_reliable`
  (the season-overall "final" score, 100+), NaN for out-of-scope pitches,
  row count/order unchanged. Reuses PITCHER_COL/PITCH_TYPE_COL/SEASON_COL/
  JUNK_PITCH_TYPES/MIN_PITCHES_FOR_SCORE from stuff.py instead of redefining
  them.
- Training the per-pitch-type gradient-boosted models with 5-fold CV is the
  expensive step (94s-340s on the full dataset, depending on system load;
  timed two identical runs and saw a >3x spread from contention alone), so it
  shouldn't happen on every pipeline run. Cached under pitching_plus/models/
  (joblib, gitignored, regenerable from data/, doesn't belong in git):
  the fitted per-pitch-type models (for scoring genuinely new pitches later)
  AND the out-of-fold historical scores keyed by (game_pk, at_bat_number,
  pitch_number), a real unique-pitch key, verified no nulls/duplicates.
  This matters because the cached *model*'s own .predict() on data it was
  trained on would be in-sample (mildly optimistic, exactly the leakage the
  notebook's OOF approach was built to avoid). Caching the OOF scores
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
  in-scope working frame as `raw_df[row_filter].copy()`, keeping ALL ~119
  raw Statcast columns through every intermediate step, not just the ~15-20
  actually needed. Fine on the trimmed exploration scripts (which only ever
  loaded a handful of columns via usecols), but full_pipeline.py reads the
  ENTIRE raw CSV. Running location.py's training pass against it multiplied
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
  full_pipeline.py) and notebooks/ (stuff.ipynb, location.ipynb). docs/
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
  Stuff+ against Location+ empirically: how much of realized run value
  (`delta_pitcher_run_exp`) each one explains, individually and
  combined, ahead of building purpose.md's Pitching+ stage. Loads data
  through add_stuff_plus/add_location_plus (no logic reimplemented) and
  reasons about the output with correlation, pitch-level OLS, pitcher x
  pitch-type x season WLS, PCA, a season holdout, and a pitch-sequencing
  significance test.
- Found (and fixed) a real bug in stuff.py's pitch-level score assignment
  while sanity-checking pitching.ipynb's inputs: a pitch's own release_speed
  should correlate strongly positively with its own pitch_stuff_plus (PCA
  composite is sign-anchored on release_speed), and it didn't (~0 for every
  pitch type). Root cause: `_score_stuff_plus`'s
  `stuff_df = stuff_df.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")`
  silently resets stuff_df's index to a fresh RangeIndex, and
  `add_stuff_plus` reattaches pitch_stuff_plus back onto raw_df
  *positionally* via `result.loc[stuff_df.index, ...]`. After the reset,
  that index no longer points at the original rows, so pitch_stuff_plus
  landed on the wrong pitches for most of the dataset. `stuff_plus` (the
  pitcher-season aggregate) was unaffected, since it's reattached via an
  actual key-based merge on pitcher/pitch_type/season rather than
  positionally, which is exactly why this stayed invisible in stuff.py's own output
  and only surfaced once pitch-level Stuff+ was checked against something
  external to it. Fixed by saving/restoring stuff_df's index around the
  merge, mirroring the identical fix already present in location.py's
  `_calibrate`. Confirmed via the same sanity check: own-release_speed vs.
  own-pitch_stuff_plus correlation went from ~0 to 0.60-0.86 across pitch
  types (matches location.py's diff to `_calibrate`, which had already
  caught the same class of bug there). No cached artifacts needed
  invalidating: stuff.py doesn't cache anything, and location.py's cache
  only stores `_train_models`'s output, not `_calibrate`'s.
- That fix changed pitching.ipynb's results substantially. With correct
  pitch-level scores, Stuff+ has a real, significant relationship with run
  value (aggregate R^2 ~0.02 alone, both R^2 ~0.08 combined with Location+,
  up from ~0.05 for Location+ alone), and the two combine for a better,
  holdout-validated fit than either alone.
- Reviewed FanGraphs' Stuff+/Location+/Pitching+ primer
  (library.fangraphs.com) for methodology to cross-check against. Two of its
  claims motivated new checks in pitching.ipynb: (1) Stuff+ and Location+
  stabilize at very different pitch counts (~80 vs. ~400). Swept the
  reliability bar from 20 to 600 pitches and found Stuff+'s R^2 climbs with
  more data and overtakes Location+'s past a few hundred pitches, while
  Location+'s is already close to its ceiling at 20-50; (2) Stuff+ is
  reported as far more year-over-year sticky than Location+. Confirmed
  directly (season Y -> Y+1 correlation ~0.89 for Stuff+ vs. ~0.41 for
  Location+ on pitcher-pitch-type-season aggregates). Also noted the primer
  states real Pitching+ is not a weighted average of Stuff+ and Location+ but
  a separately trained ("third") model on physical + location + count
  features against run value. Folded into pitching.ipynb's closing
  discussion as the more principled direction for the actual Pitching+ build,
  vs. this notebook's regression-based weight as a reasonable starting point.

# 8/27/2026 (cont'd): Pitching+ and bestPitch+ shipped
- Ported the validated weighted-blend recipe into pitching_plus/pitching.py:
  `add_pitching_plus(raw_df) -> raw_df`, mirroring stuff.py/location.py's
  shape. One pooled WLS blend (z-scored log(stuff_plus) + mean
  location_run_value against delta_pitcher_run_exp, weighted by n_pitches),
  fit fresh on every call like stuff.py's PCA (cheap: a few thousand
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
  blend/calibration to arbitrary stuff/location inputs), needed so
  bestpitch.py can score *hypothetical* pitches on the exact same fitted
  scale as real ones, not a separately re-derived one. Confirmed via the
  full pytest suite that add_pitching_plus's output is unchanged after the
  refactor.
- Built pitching_plus/bestpitch.py for purpose.md's last stage: "given
  pitcher arsenal, calculate hypothetical Pitching+ score (bestPitch)" then
  `bestPitch - Pitching+ = bestPitch+`. Design: for each in-scope pitch, hold
  the real situation fixed (count, outs, base state, handedness) and search
  every (candidate pitch type from that pitcher's own RELIABLE arsenal that
  season x candidate zone) combination. Zones are Statcast's real `zone`
  1-9 in-zone codes (purpose.md's "zone, not pinpoint"), represented by that
  zone's dataset-wide average (plate_x, plate_z_rel), not a synthetic point
  or a per-pitch-type-specific location (a pitch type's own typical spot
  within a zone is a targeting choice the location model itself already
  captures). Each candidate is scored by combining the pitcher's own real
  stuff_plus for that pitch type with location.py's own cached model's
  prediction for that zone (added `location.load_cached_models` to expose
  the cache for this), run through pitching.py's `_score_pitching_plus`,
  the identical fitted blend/calibration as real Pitching+ scores, so
  "best" and "actual" are directly comparable. The max across all valid
  candidates is `best_pitching_plus`.
- Found and fixed a real scope-filtering bug while building this: initially
  folded BASE_STATE_COLS (on_1b/on_2b/on_3b) into the dropna gate alongside
  the rest of REQUIRED_COLS, which wrongly required all three bases occupied
  to keep a row (NaN there is a real "base empty" state, per location.py's
  own established convention). On a synthetic test fixture this dropped
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
  so it's >= the realized value almost by construction: positive or close
  to 0, confirmed at ~97.6% non-negative on the synthetic fixture and ~99.1%
  on the full 2021-2025 dataset. Corrected purpose.md's original draft note
  (it had stated the opposite sign) and the bestpitch.py/bestpitch.ipynb text
  to match.
- Ran add_bestpitch_plus end-to-end on the full 3,565,743-row dataset
  (against already-scored stuff/location/pitching columns, so this run only
  paid the counterfactual search's own cost): 910s, 3,521,353 pitches scored.
  Built notebooks/bestpitch.ipynb in the same style as the others: scores
  the full pipeline, sanity-checks the zone reference grid against Statcast's
  real 1-9 layout, checks the bestPitch+ sign distribution, and a
  pitcher-season "closest to optimal pitch selection" leaderboard (same
  MIN_PITCHES_FOR_SEASON_SCORE reliability bar as location_plus's own
  whole-arsenal score).
- That leaderboard surfaced a real interpretive caveat: the closest-to-optimal
  end sits in a plausible 72-84 range, but the most-room-for-improvement end
  runs 780-1420, an order of magnitude larger, and concentrated in the 2024
  season. best_pitching_plus is a max over ~(arsenal size x 9 zones)
  candidates per pitch, and the underlying 100+ scale is unbounded above
  (exp(k*z)); a max over that many candidates systematically favors
  whichever pitcher-season happened to have one candidate land in the long
  right tail, not a stable read on how much better their pitch selection
  could realistically be. Documented in bestpitch.ipynb's synopsis rather
  than treated as a bug; a bounded-candidate or trimmed-mean variant would be
  the fix if this metric needs to support that specific claim later.
- Added tests/test_pitching_plus.py coverage for both pitching.py (missing
  columns, junk-type NaN, reliable-population calibration lands on exactly
  100, and a check that add_pitching_plus reuses already-present
  stuff/location columns instead of recomputing them) and bestpitch.py
  (missing columns, junk-type NaN, best-meets-or-exceeds-actual on >90% of
  in-scope pitches). Added a `zone` column to the shared synthetic fixture
  (tests/conftest.py) via a rough 3x3 plate_x/plate_z_rel grid, matching
  Statcast's real 1-9 numbering closely enough to exercise the zone-grid
  logic (not intended to test real zone semantics).

# 8/30/2026: bestPitch+ was inflated, two real fixes not one
- User question ("is bestPitch's outcome really that much better than
  average, and what defines the zone") led to checking the
  magnitude rather than just the sign. Isolated the cause by re-running the
  candidate search restricted to ONLY the pitch actually thrown (no
  pitch-type switching, just its 9 in-zone locations); still produced a
  mean gap of 90.8 (median 75.2), nearly as large as the full ~33-candidate
  search's 127.4 (median 99.7). That ruled out "too many candidates" as the
  main driver.
- Root cause #1: pitching.py's blend z-scores location_run_value using
  `loc_sigma` fit from the (pitcher, pitch_type, season) AGGREGATE
  population's spread, much narrower than a single pitch's natural spread,
  since an aggregate is a mean over many pitches. Applying that narrow sigma
  to pitch-level values inflates every pitch-level z-score, and taking a max
  over several such over-wide draws compounds it further via order
  statistics (max of 9 std~61 draws being ~60*1.5 above the mean is roughly
  what was observed). Fixed by giving the location term a separate,
  pitch-level-appropriate sigma (`loc_sigma_pitch`, computed from real
  pitch-level location_run_value spread among reliable-arsenal pitches) and
  its own from-scratch 100+ calibration (`pitch_calibration`, fit on the
  pitch-level raw_pitching_value distribution itself, not borrowed from the
  aggregate one). `_fit_pitching_plus_model` now returns both calibrations;
  `_apply_blend`/`_score_pitching_plus` take an explicit `level` ("aggregate"
  or "pitch") so callers can't accidentally cross the two. Stuff+'s
  contribution needed no such split; both levels already use the same
  (pitcher, pitch_type, season) aggregate stuff_plus value, never a separate
  pitch-level composite.
- Root cause #2 (the user's own proposed fix, tested empirically before
  committing to it): a single-point model query at each candidate zone's
  reference location lets bestPitch+'s max-over-candidates search exploit
  any local spike in the fitted regression surface. Averaging the
  prediction over a small "target" area (5-point sample: center plus
  N/S/E/W, within a radius of 3 baseball-diameters, ~0.36ft) cut the
  same-pitch-type-only mean gap from 90.8 to 52.2 (median 75.2 -> 43.7) on a 200k-row
  sample, confirmed empirically before implementing at full scale.
  TARGET_RADIUS_FT/CANDIDATE_ZONE_CODES are in bestpitch.py; the candidate
  zone set was also widened from just the 9 in-strike-zone codes to include
  Statcast's 4 chase/waste corners (11-14), since "up-and-in chase" is a
  real pitch-location idea purpose.md's "zone, not pinpoint" should cover
  and the original implementation excluded it entirely.
- Combining both fixes overshot: bestPitch+ flipped to roughly symmetric
  around 0 (mean -7.72, median -4.85, only 46.4% non-negative on the 200k
  sample) instead of reliably non-negative. Root cause: `best_pitching_plus`
  is now a *smoothed* (target-averaged) quantity, but it was still being
  compared against the real `pitch_pitching_plus`, a *pinpoint* value: an
  apples-to-oranges comparison where a specific real pitch's own single-point
  luck can beat a smoothed area estimate close to half the time.
- Fixed (also the user's proposed design) by scoring the actual pitch's own
  (pitch type, zone) combination through the identical target-averaging
  machinery (`_actual_smoothed_pitching_plus`) before comparing, rather than
  reusing pitch_pitching_plus for this comparison. Since that combination is
  itself always one of the candidates the search considers, best_pitching_plus
  is a true max over a set that includes it again. Result on the same
  200k-row sample: 99.95% non-negative, mean 10.2, median 8.7, std 8.4, max
  86.9, down from the original single-point version's mean 127.4/max
  17,624 by roughly an order of magnitude, and now a believable "typical
  achievable improvement" figure rather than a max-search artifact.
  pitch_pitching_plus itself (the "real," reported metric used elsewhere) is
  untouched by any of this; the smoothed-actual value is computed
  internally, used only for bestPitch+'s own comparison.

# 8/31/2026: new branch off main, continuing the FanGraphs joint-model check
- Provenance note, so this isn't read as a from-scratch redo. Everything
  below builds on real work from feat/fangraphs-style-pitching (tip commit
  b62b8eb, pushed to origin so it survives even if deleted locally later;
  `git log main..origin/feat/fangraphs-style-pitching` has the full
  9-commit history). That branch's technical work holds up and is credited
  by name throughout this entry: the FanGraphs joint-model comparison
  (fg_pitching.ipynb) and its regularization sweep, the weighted-blend
  Pitching+ implementation (pitching.py), and bestPitch+'s counterfactual
  search (bestpitch.py) with two rounds of real, non-obvious bug fixes (the
  aggregate-vs-pitch-level calibration mismatch and the pinpoint-vs-smoothed
  scoring mismatch; see the 8/27 and 8/30 entries above). The problem wasn't
  that the work was bad. That branch's own name and scope was "investigate
  FanGraphs' joint model," and it kept going well past that investigation's
  conclusion into shipping production code, without the basic project infra
  (pinned dependencies, package init files) that should have existed from
  the start. This branch redoes the scoping, not the engineering. Ported
  code below is called out as ported, not rewritten, unless a real change
  is noted.
- New branch feat/pitching-plus-v2, off main. Fixed purpose.md first: a typo,
  a Stuff+ claim contradicting stuff.py's actual (physics-only) design, the
  bestPitch+ sign (was documented backwards), and the Pitching+ workflow
  item's target metric (named speculative xwOBA/xERA/xRE options that were
  superseded once Location+ settled on delta_pitcher_run_exp).
- Added requirements.txt/requirements-dev.txt, pitching_plus/__init__.py and
  scripts/__init__.py, and a pytest suite for stuff.py/location.py (ported
  from the prior branch) before any new modeling work. main had been
  missing this despite already depending on numpy/pandas/scikit-learn/
  statsmodels/joblib/pybaseball.
- Brought over notebooks/fg_pitching.ipynb (the FanGraphs-style joint-model
  comparison) and reran it end-to-end to confirm it reproduces: identical
  numbers to the prior branch's run (weighted blend train R^2=0.0954/holdout
  R^2=0.0523; joint model 0.1318/0.0386; the three regularized joint variants
  match to 4 decimal places).
- Pushed the investigation one step further than the prior branch did: does
  feeding the joint model the blend's own two calibrated inputs (log_stuff,
  location_run_value) instead of 19 raw physics/location features close the
  gap, rather than just regularizing the raw-feature model harder? Same
  per-pitch-type 5-fold OOF-CV scheme, same unregularized hyperparameters,
  same train(2021-2024)/test(2025) split and evaluation harness as every
  other row in the comparison.
- Result: worse, not better. Holdout R^2 0.0343 (75% shrinkage from train's
  0.1380). That's the worst generalization of every joint variant tried,
  including the explicitly-regularized ones, despite the highest train R^2
  of the bunch. An unconstrained HistGradientBoostingRegressor with only two
  features still has plenty of capacity to carve fine-grained threshold
  interactions between them that fit train-season noise and don't carry over
  to a new season. This rules out "wrong inputs" as the explanation
  alongside the prior branch's "needs more regularization." The blend's
  advantage is the linear, additive, two-parameter functional form itself,
  not what it's fed or how it's tuned.
- Conclusion: for this repo's current data and tools, the weighted-average
  blend is the validated choice for Pitching+, not a placeholder pending a
  better joint model. Two independent angles on "make the joint model
  competitive" (regularization, then input choice) both failed to close the
  gap. Ported the prior branch's pitching.py and bestpitch.py (see the
  8/27-8/30 entries above for their construction and bug-fix history;
  unchanged here, since neither's design depended on which Pitching+
  approach won) and wired them into full_pipeline.py, along with their
  pytest coverage and README's bestPitch+ description. Only location.py
  needed a real change: added `load_cached_models` (bestpitch.py's
  dependency), which hadn't been ported in the earlier infra pass since
  nothing needed it yet. Full suite (18 tests, stuff/location/pitching/
  bestpitch) passes.

# 9/2/2026: hypercritical architecture review (stuff/location/pitching/bestpitch)
- User asked for a hypercritical review of the whole pipeline plus an explicit
  verdict on whether Stuff+ should be outcome-independent. Reviewed all four
  scripts end-to-end (including the uncommitted working-tree state of
  bestpitch.py/pitching.py from the separate counterfactual-search-
  optimization session -- see that entry's context below), cross-checked
  every dev-log-claimed fix against the current code, and where a claim was
  checkable, verified it empirically against the real cached models and
  data/MLB_2021-2025_plus.csv rather than trusting the code/comments alone.
- Verdict on the central question: keep Stuff+'s fitting process
  outcome-independent. The factored pipeline design, Stuff+'s much higher
  year-over-year stability than Location+ (~0.89 vs ~0.41, see 8/27 entry),
  and the joint-model experiment (fg_pitching.ipynb: a single model fed more
  information generalizes worse than the two-input blend) all support it.
  But the design has never been validated against anything outcome-shaped,
  even as a diagnostic -- every existing check on Stuff+ is either
  near-tautological (own release_speed vs. own pitch_stuff_plus, and
  release_speed is one of the composite's own six inputs) or a check on the
  downstream Pitching+ blend, never a check on the composite in isolation
  against a bat-missing proxy (whiff rate, CSW%, chase rate, hard-hit rate).
  Recommended next step: add a held-out validation notebook cell correlating
  pitch_stuff_plus/stuff_plus against such a proxy per pitch type -- a
  diagnostic only, not a change to how the composite is fit.
- Found a real, empirically-confirmed bug: location.py's `_calibrate`
  (lines 265-302) has the identical aggregate-vs-pitch-level sigma mismatch
  that was fixed in pitching.py on 8/30, but unfixed here. `_ratio_calibration`
  fits agg_mu/agg_sigma from the spread of per-(pitcher, pitch_type, season)
  MEANS, and that sigma is then applied to raw per-pitch location_run_value
  to produce pitch_location_plus (line 285). Checked directly against FF/2023
  in the real production data: pitch-level std of location_run_value is
  0.0461; the aggregate sigma actually used is 0.0065, ~7x too narrow.
  Result: pitch_location_plus for FF/2023 has mean 121.4, std 64.6, max 806.5
  (n=230,188), instead of the intended ~100-centered, comparably-scaled
  distribution the season-level location_plus correctly has (mean 100.0,
  std 9.7). This doesn't reach pitching.py/bestpitch.py's own scoring math
  (both recompute their own correctly-split sigmas from raw
  location_run_value directly, never consuming pitch_location_plus), but
  pitch_location_plus itself is a documented headline output and is
  currently untrustworthy at the pitch level. Not caught by any existing
  test: test_location.py has no test on pitch_location_plus's calibration at
  all, and the one test that exercises _ratio_calibration/_to_100_scale
  fits and applies to the SAME distribution, so it can't detect a
  cross-level mismatch by construction. stuff.py's _score_stuff_plus has the
  identical code pattern (agg_sigma from spread-of-means, applied to raw
  pitch_composite) but doesn't show the same failure in practice --
  physics has much lower pitch-to-pitch noise relative to between-pitcher
  spread than run value does, so the two sigmas don't diverge nearly as
  much (checked FF/2023: pitch_stuff_plus mean 100.9, std 12.8, sane) -- but
  this hasn't been measured systematically per pitch type and is worth a
  direct spread check before fully trusting pitch_stuff_plus's tails.
- Found a likely real, unfixed bug: location.py's plate_x_armside
  (line 122-123) is computed as `np.where(stand == "R", plate_x, -plate_x)`
  -- keyed on BATTER handedness (stand) -- while its own inline comment
  claims the result is "regardless of batter stand." Arm-side is by
  definition pitcher-relative; the formula should key off p_throws, not
  stand, and as written only gives the claimed sign for same-handed
  matchups (inverted for platoon matchups). Likely low impact on the
  location model's actual accuracy, since LOCATION_FEATURES also carries
  raw plate_x, stand_R, and p_throws_R separately, so the model can learn
  the real interaction regardless of this one derived column's labeling.
  But the identical formula is copy-pasted into bestpitch.py in 4 places
  (_predict_at_point and the 3 batched-array equivalents), and
  test_location.py::test_plate_z_rel_and_armside currently asserts the
  batter-keyed sign as ground truth, so it would resist rather than catch a
  correct fix. Flagging for a domain-expert sign check before touching --
  the categorical bug (code contradicts its own comment) is unambiguous;
  the exact correct sign convention is worth verifying against Statcast's
  real coordinate convention first.
- Found that bestpitch.py's "reliable arsenal" gate can violate its own
  documented invariant. _arsenal_wide (lines 165-181) restricts candidates
  to stuff_plus_reliable == True pitch types per pitcher-season
  (MIN_PITCHES_FOR_SCORE, 20). A pitch type thrown fewer than 20 times that
  season never enters the candidate search at all -- including for scoring
  that type's own pitches -- so for those pitches best_pitching_plus is
  computed only from the pitcher's OTHER reliable pitch types. This directly
  contradicts the module docstring's claim (lines 39-42) that the actual
  (pitch type, zone) combination is "always" a candidate, which is what
  makes best_pitching_plus >= actual a near-structural guarantee. Confirmed
  via a synthetic repro (a pitcher with 3 CU pitches, below the reliability
  bar): cand_stuff_CU is NaN for every row in that pitcher-season. Didn't
  flip any signs in the small repro (other pitch types still had a higher
  ceiling) but the mechanism is real and will bite hardest for a starter's
  rarely-used show-me pitch -- a realistic, non-edge-case population the
  current test fixture doesn't cover (every synthetic pitch type clears
  every threshold).
- Found that both regression tests written for the 8/30 fixes would not
  catch either bug's reintroduction, confirmed empirically, not just by
  inspection:
    - test_pitching.py::test_pitch_level_calibration_uses_its_own_spread_not_the_aggregates:
      monkeypatched _apply_blend to always use loc_sigma_agg regardless of
      `level` (i.e. reintroduced the original bug) and reran the test's own
      logic -- it still passed (group means still land on exactly 100.0,
      since the calibration-fit step and the scoring call route through the
      same, consistently-wrong dispatch). The loc_sigma_pitch > loc_sigma_agg
      assertion only checks how the sigma is computed, never whether it's
      actually selected correctly per level, and no test touches
      add_pitching_plus's own pitch_pitching_plus output magnitude via the
      public entry point at all.
    - test_bestpitch.py::test_add_bestpitch_plus_junk_and_best_meets_or_exceeds_actual:
      reproduced the exact pre-fix formula (best_pitching_plus -
      pitch_pitching_plus, pinpoint instead of smoothed) on the test's own
      fixture -- it still clears the >0.9 non-negative threshold (95.6%),
      because the fixture's pitch-level noise is far gentler than real
      Statcast data, where this bug drove the non-negative fraction down to
      46.4% (see 8/30 entry).
  Both fixes ARE correctly implemented in the current code (every call site
  of _apply_blend/_score_pitching_plus/_fit_pitching_plus_model traced and
  confirmed passing the right level/calibration) -- this finding is
  specifically that the tests guarding them have no teeth.
- Smaller findings, roughly in order of severity: stuff.py's
  horizontal_acceleration (line 146, sqrt(ax^2+ay^2)) is likely mislabeled --
  ay is forward-path drag deceleration, not lateral movement, and typically
  dwarfs ax in magnitude, yet this feeds directly into
  movement_per_reaction_time, the single highest-loading PC1 feature per the
  8/26 entry; worth a direct check since a physics-only composite with no
  outcome feedback has no way to notice this on its own. stuff.py's
  pitcher_agg groupby key includes PLAYER_NAME_COL (line 231) for no
  computational reason, creating a latent row-duplication risk in the final
  merge if any (pitcher, pitch_type, season) ever has more than one distinct
  player_name string (untestable with the current fixture, which assigns
  player_name as a deterministic function of pitcher id). PITCHING_SCALE_K
  (pitching.py:53) is defined and never used anywhere -- _to_100_scale always
  uses location.py's hardcoded LOCATION_SCALE_K instead, so tuning either
  constant doesn't do what it looks like it does. location.py's model cache
  (_load_or_score, lines 199-226) has no staleness protection beyond file
  existence -- editing LOCATION_FEATURES or the HGB hyperparameters and
  calling with retrain=False silently scores against the stale model.
  bestpitch.py's _actual_smoothed_pitching_plus scores real historical
  pitches with the refit-on-all-data cached model rather than location.py's
  OOF scores (keeps the best-vs-actual comparison internally consistent,
  which is what the 8/30 fix needed, but means bestPitch+'s absolute
  magnitude carries a different, slightly more optimistic rigor level than
  the headline location_plus/pitching_plus columns -- currently
  undocumented). stuff.py computes and discards a lot of dead feature
  engineering (spin_axis_sin/cos, velocity_mag, horizontal_velocity, and
  every 10/20/40ft trajectory column except time_30ft) left over from the
  notebook's V2 arsenal-comparison work. MIN_PITCHES_FOR_SCORE=20 is
  directly contradicted by the 8/27 entry's own finding that Stuff+'s R^2
  keeps climbing well past 20 pitches. full_pipeline.py has zero test
  coverage (no test calls run_pipeline end-to-end).
- On the uncommitted bestpitch.py rewrite specifically (vectorized/batched
  counterfactual search -- _build_base_array,
  _batched_target_averaged_location_run_value,
  _all_zones_target_averaged_location_run_value -- from the separate
  counterfactual-search-optimization session): checked the core dedup
  assumption by hand and it holds -- re288_state = (outs*8 + base_state)*12
  + count_state (location.py:115-117) is a genuine bijection over its
  inputs, so center_key correctly identifies rows the location model can't
  tell apart. But it's an invariant enforced only by convention across two
  files with no assertion checking group-homogeneity; a future column added
  to LOCATION_FEATURES without a matching update here would silently
  corrupt whole groups of predictions rather than crash. More concretely:
  _predict_at_point (the pre-rewrite reference implementation) is no longer
  called from the production path and isn't wired into any pytest as an
  equivalence oracle for the new batched path (it's still used by
  notebooks/location_smoothing_check.ipynb for offline validation, but that
  isn't checked-in test coverage). And tests/conftest.py's fixture holds
  sz_top/sz_bot constant across every row, so zone_height never varies --
  the new ZONE_HEIGHT_DEDUP_ROUND_FT bucketing logic (the actual novel part
  of this rewrite) is structurally untested; the fixture can't distinguish
  "dedup by rounded zone height works" from "there's only one zone height
  anyway." The docstrings' equivalence claims ("verified... max abs diff
  ~1e-13") describe an ad hoc check, not a checked-in regression test.
- No code changes made in this entry -- review only. Step-by-step fix list
  handed to the user directly (not duplicated here); highest priority items
  are the location.py sigma fix (mirrors pitching.py's existing pattern) and
  strengthening the two tautological regression tests above.

# 9/2/2026 (cont'd): implemented the review's fix list
- Fixed location.py's pitch-level calibration sigma mismatch (`_calibrate`):
  `type_calibration` now fits agg_mu/agg_sigma on the real per-pitch
  location_run_value spread among reliable-arsenal pitches
  (`pitch_level_reliable`), not the far narrower spread of per-pitcher-season
  means, mirroring pitching.py's own aggregate/pitch split. Added
  test_location.py::test_pitch_location_plus_calibration_uses_pitch_level_spread_not_the_aggregates,
  which checks the actual magnitude the bug broke (mean-100 alone can't
  catch this bug class -- confirmed both here and for pitching.py's
  equivalent test, see below).
- Fixed bestpitch.py's reliable-arsenal invariant gap: `add_bestpitch_plus`
  now takes `np.fmax(best_pitching_plus, actual_smoothed)` (only where a
  candidate search actually ran for that row) before computing
  pitch_bestpitch_plus, restoring the "actual combination is always a
  candidate" guarantee even when the pitch's own type falls outside that
  pitcher-season's reliable arsenal. Added
  test_bestpitch.py::test_add_bestpitch_plus_own_pitch_always_a_candidate_even_if_unreliable.
- Hardened two regression tests that were empirically shown (by the review)
  to have no power against the bug classes they're supposed to guard:
  test_pitching.py::test_apply_blend_level_dispatch_actually_uses_the_right_sigma
  (checks _apply_blend's actual level="aggregate" vs. level="pitch" effect,
  not just that mean lands on 100) and
  test_bestpitch.py::test_actual_smoothed_differs_from_pinpoint_pitch_pitching_plus
  (checks the smoothed-vs-pinpoint mechanism directly, immune to fixture
  noise magnitude, unlike the existing end-to-end sign-rate test).
- Fixed plate_x_armside (location.py `_build_features`): now keyed on
  `p_throws` (pitcher) instead of `stand` (batter), per the derivation in
  the review (RHB stands 3B-side/negative plate_x, LHB 1B-side/positive;
  pitcher faces the opposite direction from the catcher-perspective
  convention plate_x uses, so a RHP's own arm side is negative plate_x).
  Propagated to the 4 mirrored spots in bestpitch.py's batched search
  (`_predict_at_point`, `_build_base_array` now returns `p_throws_is_R`
  instead of `stand_is_R`, and both `_batched_target_averaged_location_run_value`/
  `_all_zones_target_averaged_location_run_value`). Rewrote
  test_location.py::test_plate_z_rel_and_armside, which previously
  certified the batter-keyed sign as correct (four rows varying stand and
  p_throws independently now confirm the sign tracks p_throws only).
- Varied sz_top/sz_bot per row in tests/conftest.py's fixture (previously a
  fixed 3.5/1.5 constant) and added
  test_bestpitch.py::test_batched_target_averaging_matches_unbatched_reference_with_varying_zone_height,
  an equivalence test between the uncommitted counterfactual-search
  rewrite's batched/deduplicated path and the original per-row
  `_predict_at_point` reference, using real per-row zone-height variation so
  ZONE_HEIGHT_DEDUP_ROUND_FT's bucketing is actually exercised (previously
  untestable -- a constant zone_height can't distinguish correct bucketing
  from a bug). Passes at atol=1e-6, and incidentally also confirms the
  armside fix is applied consistently between the batched and reference
  paths (they'd disagree otherwise).
- Added the Stuff+ outcome-proxy validation diagnostic recommended in the
  review, in notebooks/stuff.ipynb (not stuff.py -- diagnostic only, no
  change to how Stuff+ is fit): correlated stuff_plus against CSW%
  (called-strike-plus-whiff rate) per pitch type and pooled, on 2023-2024
  data (>=100 pitches of outcome data as the reliability bar). Real result,
  verified against an equivalent standalone run of stuff.py's actual
  add_stuff_plus (not reimplemented logic): pooled Spearman rho +0.078
  (p=2e-6, n=3,721 reliable pitcher-pitch_type-seasons) -- real but weak.
  Per pitch type, most are positive (ST +0.219 p=0.0001, SL +0.155 p=0.0001,
  FF +0.085 p=0.007), but CH (-0.056) and FS (-0.141) are negative point
  estimates, though neither reaches significance at these sample sizes. Read:
  the physics-only composite's "more outlier = harder to hit" assumption
  holds, but weakly and not uniformly -- changeups/splitters show no
  positive signal here. Doesn't change the outcome-independent design
  (architecture + stability + joint-model evidence still favor it), but is
  real evidence against treating a high Stuff+ score as a strong bat-missing
  guarantee, especially for off-speed types. NOTE: the notebook cells were
  not executed in-session (would require rerunning the full V1/V2 feature
  engineering pipeline on 3.55M rows, expensive); the numbers above are
  verified correct via an equivalent standalone script using the actual
  production add_stuff_plus, not fabricated.
- Cleanup: removed pitching.py's dead PITCHING_SCALE_K constant (a comment
  now clarifies Pitching+'s ratio scale is tied to location.py's
  LOCATION_SCALE_K, not independently tunable). Trimmed stuff.py's
  `_build_v1_features` to only compute what STUFF_FEATURES actually reads
  (acceleration_mag, horizontal_acceleration, time_30ft) instead of also
  building spin_axis_sin/cos, velocity_mag, horizontal_velocity, and full
  x/z/vx/vy/vz/speed trajectories at 10/20/30/40ft for every in-scope pitch;
  replaced TRAJ_DISTANCES with REACTION_DISTANCE_FT=30. Dropped
  PLAYER_NAME_COL from stuff.py's pitcher_agg groupby key (unused downstream,
  was a latent merge-fanout risk). Added a cache-staleness warning to
  location.py: `_cache_fingerprint()` (LOCATION_FEATURES + the now-factored-
  out HGB_PARAMS + N_FOLDS + MIN_GROUP_SIZE_FOR_MODEL) is saved alongside
  the model cache; a mismatch on load warns instead of silently reusing a
  stale cache (a cache saved before this existed has no fingerprint file and
  is treated as "unknown," not forced to retrain). Gave full_pipeline.py's
  `run_pipeline` models_dir/retrain parameters (previously hardcoded to the
  real production cache with no way to point at an isolated one) and added
  tests/test_full_pipeline.py, the first test to actually call it end-to-end
  rather than only piecewise through each module's own tests.
- Deliberately NOT changed: MIN_PITCHES_FOR_SCORE=20, despite the review's
  finding that Stuff+'s own R^2 keeps improving well past that bar (8/27
  entry) -- picking a real replacement value needs the kind of reliability
  sweep that finding came from, re-run and re-decided, not a quick constant
  bump that would silently shift every reported Stuff+/Pitching+ number.
- Full suite: 25 tests pass (18 baseline + 7 new: the location sigma
  regression test, the pitching dispatch-effect test, the bestpitch arsenal-
  invariant test, the bestpitch smoothed-vs-pinpoint mechanism test, the
  bestpitch batched-equivalence test, the location cache-staleness-warning
  test, and test_full_pipeline.py's end-to-end test).

# 9/2 (cont'd): rebuilt notebooks/bestpitch.ipynb, dropped an off-scope exploration cell
- notebooks/bestpitch.ipynb never actually made it onto this branch (or its
  parent, feat/pitching-plus-v2): the 8/31 porting entry above only carried
  over bestpitch.py/pitching.py (scripts) from feat/fangraphs-style-pitching,
  not the exploration notebook. Rebuilt it fresh, in the same style/section
  order as the original (score the full pipeline, zone-reference sanity
  check, bestPitch+ sign check, closest-to-optimal leaderboard, synopsis),
  reran against the current code rather than ported verbatim, so its numbers
  reflect this branch's actual fixes (armside, calibration sigma, arsenal
  invariant), not the old branch's.
- Added, then removed, a "Potential+" (Pitching+ + bestPitch+) exploration
  cell (top-10 SP/RP leaderboard) -- an idle "what if" look at combining
  realized quality with room-for-improvement, not something purpose.md
  defines or any script computes. Cut since it's not currently used anywhere
  and doesn't belong in an exploration notebook that's supposed to mirror
  what's actually shipped; the idea can come back as a real proposal (with
  its own purpose.md entry and a decision on what the combined number is
  supposed to mean) if it turns out to be useful later.
- Verification pass before considering this branch mergeable: reread
  purpose.md, README.md, and all four scripts end-to-end against the current
  dev_log's claims (sign conventions, calibration levels, cache fingerprinting,
  the arsenal-invariant fmax fix) -- all confirmed matching, full 25-test
  suite still green.
- Found and removed leftover temporary profiling instrumentation in
  bestpitch.py's `_search_best_pitching_plus` (the build/predict/score-assign
  wall-clock timers and the `[profile]` print, added during the
  counterfactual-search vectorization work and explicitly marked "temporary"
  in its own comment). That investigation is done -- the batched/deduplicated
  rewrite it was measuring is already reviewed, tested, and documented above
  -- so the instrumentation no longer earns its keep as shipped code. No
  behavior change: `verbose=True`'s per-candidate progress line is untouched.
- Checked feat/pitching-plus-v2 and feat/fangraphs-style-pitching for anything
  substantive not yet reflected here: feat/pitching-plus-v2 is a strict
  ancestor of this branch (nothing left to port). feat/fangraphs-style-pitching
  (unmerged, still pushed to origin) is fully superseded -- every real fix
  and script it introduced was already ported and improved on 8/31 and 9/2
  (see the provenance note above); its own bestpitch.ipynb has the identical
  section structure to the one rebuilt above, confirming nothing was lost in
  the redo. Its consolidated tests/test_pitching_plus.py (one file for all
  four modules) versus this branch's one-file-per-module layout is a style
  difference, not a missing capability -- not worth reverting the recent
  tests/ reorg for.

# 9/4/2026: retooled Stuff+ from a PCA composite to a trained model

- New branch rework/stuff-weighting, off main. User reviewed FanGraphs'
  Stuff+/Location+/Pitching+ primer specifically on Stuff+'s methodology:
  their real Stuff+ is trained against run value via a decision-tree model
  capturing nonlinear physics-to-run-value relationships ("isn't only
  outlierness, it's how specific outlier characteristics affect run value
  generation"), not a variance-maximizing composite. This supersedes the
  8/26 and 9/2 entries' "keep Stuff+ outcome-independent" verdict -- both
  entries stay as history, not deleted; this entry documents the reversal
  and why. The per-pitch score stays independent of THAT pitch's own
  outcome throughout (out-of-fold CV, never a same-data fit), matching the
  original ask ("the 'outcome' Stuff+ would still be independent of
  result") -- what changed is that the WEIGHTS are now informed by run
  value in aggregate, not that any single pitch's own result leaks into its
  own score.
- New design mirrors location.py's architecture: one
  HistGradientBoostingRegressor per pitch type (same HGB_PARAMS as
  location.py: max_depth=6, learning_rate=0.05, max_iter=300,
  l2_regularization=1.0), 5-fold out-of-fold cross_val_predict against
  delta_pitcher_run_exp, cached under pitching_plus/models/
  (stuff_models.joblib, stuff_historical_scores.joblib,
  stuff_cache_fingerprint.joblib) exactly like Location+'s cache.
- Added a new feature, axis_differential: a seam-shifted-wake proxy, the
  angular gap (wrapped to [0,180]) between Statcast's measured spin_axis and
  the direction implied by observed movement (pfx_x/pfx_z, both already in
  the raw data, already gravity-adjusted -- no trajectory re-derivation
  needed): movement_angle = atan2(pfx_x, -pfx_z) mod 360; gap = |spin_axis -
  movement_angle| wrapped to [0,180]. Under pure Magnus physics the two
  should point the same way; a large gap means the ball's actual break
  isn't explained by its bulk spin. Several naive atan2 sign/axis-order
  combinations were tried first and did NOT produce a sane pattern -- the
  `-pfx_z` sign flip was the one that did. Validated directly against the
  full 3,565,743-row 2021-2025 dataset (read-only script, not the shipped
  code path, as a first check): median gap by pitch type is FF 8.7 deg
  (n=1,164,954), CU 9.3, KC 10.3, CH 12.4, SI 18.6, FS 20.6, FC 26.3,
  SL 27.6, SV 29.1, ST 31.1. Matches the expected ordering: four-seamers and
  curveballs (known highest spin efficiency, closest to pure Magnus) show
  the tightest alignment; sinkers/splitters (known for real SSW) and
  cutters/sliders/sweepers (known gyro-heavy) show larger gaps. Holds
  separately by pitcher handedness (FF: RHP 8.5 / LHP 9.3; SI: RHP 19.5 /
  LHP 16.5), so no per-hand mirroring is needed. Re-confirmed identically
  through the actual shipped `add_stuff_plus` -> `_build_v1_features` path
  during end-to-end validation below, not just the standalone check.
  STUFF_FEATURES is now: release_speed, release_spin_rate,
  release_extension, acceleration_mag, horizontal_acceleration,
  movement_per_reaction_time, axis_differential.
- Fixed the aggregate-vs-pitch-level calibration sigma bug proactively
  (the same bug class fixed in location.py and pitching.py on 9/2 and 8/30,
  flagged as unfixed here in the 9/2 review, and noted then as "low-impact
  so far" only because physics variance is much smaller than run-value
  variance relative to between-pitcher spread -- a justification that stops
  holding once Stuff+ predicts delta_pitcher_run_exp directly, exactly as
  noisy as location_run_value already proved to be, 7x too narrow before
  its fix). stuff.py's new `_calibrate` fits agg_calibration (aggregate-
  level, from the spread of pitcher-season means) and pitch_calibration
  (pitch-level, from the real per-pitch stuff_run_value spread among
  reliable arsenals) as two separate fits, mirroring location.py's
  now-correct `_calibrate` pattern -- own local reimplementation in
  stuff.py, not an import (location.py imports FROM stuff.py, so the
  reverse would be circular). Confirmed working on real data below: pitch-
  level std (9.70) tracks the aggregate std (8.7-10.3 by type) closely,
  not the ~7x-too-narrow failure mode location.py had before its own fix.
- REQUIRED_COLS grew to include TARGET_COL (delta_pitcher_run_exp) and
  KEY_COLS (game_pk, at_bat_number, pitch_number, for OOF cache keying,
  duplicated from location.py's constant of the same name/values -- can't
  import it, same circular-import reason), plus pfx_x/pfx_z (for
  axis_differential). Real, measured scope narrowing versus the old
  physics-only REQUIRED_COLS, checked directly against the full dataset:
  3,526,127 pitches were physics-complete under the old scope; 3,525,964
  clear the new scope, a loss of 163 rows (0.005%) -- 143 missing pfx_x/
  pfx_z, 20 missing delta_pitcher_run_exp (some overlap). Negligible.
- Fixed two call sites that needed to pass models_dir/retrain through to the
  now-caching add_stuff_plus: full_pipeline.py's `add_stuff_plus(df)` call
  and pitching.py's `add_stuff_plus(raw_df)` call inside add_pitching_plus.
  The second wasn't part of the original ask but is just as necessary --
  bestpitch.py -> add_pitching_plus -> add_stuff_plus is a real call chain,
  and every test exercising it passes an isolated tmp_path models_dir;
  without this fix those tests would have silently trained against and
  written into the real production pitching_plus/models/ cache using
  synthetic test data. Also updated both scripts' `--retrain` CLI help text
  (previously said "Retrain Location+ models," now mentions Stuff+ too).
- bestpitch.py needed zero changes: it never scores a counterfactual
  Stuff+ value, only reads a pitcher's real, already-computed stuff_plus/
  stuff_plus_reliable from `_arsenal_wide`. location.py needed zero changes.
- Test suite: rewrote test_stuff.py's PCA-sign-anchor assertion (no PCA
  sign exists for a supervised model) into a check grounded in the
  fixture's own engineered signal (release_speed's coefficient in the
  fixture's delta_pitcher_run_exp formula) instead; added cache-roundtrip,
  stale-fingerprint, and pitch-vs-aggregate-sigma regression tests mirroring
  test_location.py's equivalents; added a direct axis_differential unit
  test. conftest.py's shared fixture gained pfx_x/pfx_z (previously
  absent), spin_axis is now a noisy function of pfx_x/pfx_z instead of
  fully independent random (so axis_differential has real content to test
  against, computed from a SEPARATE portion of the RNG stream added after
  the existing balls/strikes/location/on-base draws so plate_x/plate_z/
  noise's exact realized values are unaffected), and delta_pitcher_run_exp
  now also depends on release_speed and horizontal_acceleration (small
  terms relative to the existing location term) so stuff.py's GBM has
  genuine signal to fit in tests, not just noise. Also fixed
  small_stuff_thresholds' monkeypatch target (MIN_GROUP_SIZE_FOR_PCA ->
  MIN_GROUP_SIZE_FOR_MODEL) and three test_pitching.py call sites that
  called `stuff.add_stuff_plus(raw)` with no models_dir/retrain -- same
  real-cache-pollution risk as the full_pipeline.py/pitching.py fixes
  above, just in test code instead of production code. One of the three was
  missed on the first pass (a `replace_all` edit matched two
  `scored = location.add_location_plus(stuff.add_stuff_plus(raw), ...)`
  call sites but not the third, assigned to `pre_scored` instead of
  `scored`) and only surfaced as a UserWarning in the full suite's output,
  not a failure -- the real production cache already existed (from this
  entry's own end-to-end validation run below) and the missed call site's
  retrain=False default meant it loaded that real cache rather than
  training over it, so no corruption occurred (verified directly: cache
  file mtimes/fingerprint unchanged and correct afterward), but it was a
  live bug for the seconds it ran and a real gap in the first fix pass.
  Fixed and reconfirmed warning-free.
  Adding the new fixture terms shifted the RNG draw sequence enough to flip
  test_location.py::test_add_location_plus_row_alignment_and_scope's
  |plate_x|-vs-location_run_value correlation from a real (if weak, ~-0.1
  to -0.3 depending on draw) negative relationship to a spurious +0.02 --
  not a location.py regression, just that the location signal's original
  magnitude (-0.05 coefficient against 0.3 noise, n=300) was only
  marginally robust to begin with and a different random draw pushed it
  over. Fixed by strengthening all three of the fixture's signal terms
  (location coefficients 0.05 -> 0.12, release_speed 0.01 -> 0.025,
  horizontal_acceleration 0.002 -> 0.004) rather than chasing bit-identical
  RNG-stream reproduction across an evolving shared fixture. Full suite
  (29 tests: 24 baseline + 5 new stuff.py tests) passes after the fix.
- Full end-to-end validation against the real 3,565,743-row 2021-2025
  dataset (via the actual `add_stuff_plus`/`add_location_plus` entry
  points, not a standalone reimplementation):
    - Training: loading the raw CSV took ~78-79s (I/O, unchanged by this
      retool); `add_stuff_plus(raw, retrain=True)` itself took 79.2s --
      faster than location.py's own 94-340s baseline on the same dataset
      (7 features vs. Location+'s 13, otherwise identical N_FOLDS/
      HGB_PARAMS/row counts per type). 3,525,964 / 3,565,743 pitches scored
      (98.9%).
    - Calibration sanity: reliable-population stuff_plus lands on exactly
      100.0 for every pitch type (by construction), with std 8.7-10.3
      across types (CH 10.1, CU 8.7, FC 10.3, FF 10.0, FS 9.8, KC 9.8,
      SI 10.0, SL 9.6, ST 9.7, SV 9.5) -- a sane "+"-stat spread matching
      STUFF_SCALE_K=0.10's "roughly +/-10 points per SD" design intent.
      pitch_stuff_plus overall: mean 99.97, std 9.70; stuff_plus
      (broadcast) overall: mean 101.46, std 10.11 -- pitch-level and
      aggregate-level spreads track closely, confirming the proactive
      sigma-split fix above works on real data, not just the
      synthetic test fixture.
    - pitching.py's blend, refit fresh against the new stuff_plus
      distribution (no code change needed there): beta_stuff=+0.0041,
      beta_loc=+0.0055 -- both positive (higher Stuff+/Location+ still
      predicts better run value for the pitcher, same sign convention as
      before) and comparable in magnitude, not degenerate. Weighted
      aggregate-level R^2 = 0.123 (stuff-alone 0.055, location-alone
      0.061, on the same 14,343 reliable pitcher-pitch-type-season rows
      the 8/26 entry's PCA-era comparison used). This is an improvement
      over the old PCA composite's own numbers from that entry (stuff-alone
      R^2 ~0.02, combined ~0.08, location-alone ~0.05): the new run-value-
      trained Stuff+ is ~2.7x more predictive of run value on its own, and
      the combined blend's R^2 improved by roughly 54%. Directly supports
      the retool's premise -- weighting physics characteristics by how they
      actually relate to run value, instead of by variance alone, produces
      a more informative signal, not just a differently-shaped one.
  Not yet done: rebuilding notebooks/stuff.ipynb to match (still describes
  the PCA-era design; would need a full rebuild like bestpitch.ipynb got on
  9/2, not attempted here) and re-running fg_pitching.ipynb's joint-model
  comparison against the new Stuff+ (that comparison's conclusion --
  "the blend beats a jointly-trained model" -- was about Pitching+'s own
  architecture, not Stuff+'s internal design, so it isn't invalidated by
  this change, but hasn't been re-verified against the new numbers either).

# 9/4/2026 (cont'd): brought the notebooks up to date, and found a real leak doing it

- User asked to bring notebook documentation up to date with the retool, then
  check the writing with the avoid-ai-writing skill. Three notebooks directly
  depend on Stuff+ and needed real work, not just a find-replace: stuff.ipynb
  (rebuilt), pitching.ipynb (re-executed, one stale claim fixed), and
  fg_pitching.ipynb (re-executed, and a real leak found and fixed along the
  way -- see below). bestpitch.ipynb was checked and needs no text changes
  (it never describes Stuff+'s internals, just consumes the stuff_plus
  column), but its own numbers are technically stale too; re-running it
  wasn't attempted here (its counterfactual search alone took 910s on the
  full dataset per the 9/2 entry -- deliberately deferred, not an oversight).
- stuff.ipynb: rebuilt from scratch rather than patched. The old version
  reimplemented the full V1/V2 feature engineering and hand-rolled PCA
  inline; none of that logic exists anymore, so patching it in place would
  have meant rewriting nearly the whole notebook anyway. The new version
  calls the real `add_stuff_plus` directly (mirroring how pitching.ipynb/
  bestpitch.ipynb already work) instead of reimplementing the model, and
  keeps only what's still notebook-shaped: the axis_differential
  derivation/validation, leaderboards, and a rerun of the CSW% outcome
  diagnostic. Executed end-to-end (`jupyter nbconvert --execute`, ~32s
  total: 25s load + 7s score from cache) rather than left with
  standalone-script-verified-but-unexecuted numbers, since the cache made
  that cheap enough this time (the 9/2 CSW% diagnostic wasn't executed
  in-notebook for exactly the opposite reason -- retraining was too
  expensive before caching existed).
  Real results: axis_differential's median-gap ordering matches the
  standalone check already in this file's 9/4 entry above (FF 8.7 ... ST
  31.1). The CSW% rerun is a genuine improvement, not just a reshuffling:
  pooled Spearman rho +0.136 (p=9.6e-17, n=3,721, same population as the old
  PCA-era check) vs. the old +0.078 (p=2e-6) -- and every one of the nine
  pitch types now shows a positive point estimate, where the old composite
  had two negative ones (CH -0.056, FS -0.141, both now positive: +0.169,
  +0.036). This is the diagnostic that originally flagged the old design's
  weak spot, and it lines up with the R^2 improvement already measured
  above (stuff-alone 0.02 -> 0.055 combined 0.08 -> 0.123).
- pitching.ipynb: re-executed end-to-end (38 cells, ~26s to load +
  score). Found one stale factual claim in the process, not just stale
  numbers: cell 5's sanity check said "`stuff.py` sign-anchors its PCA
  composite," a real property of the old design with no equivalent for a
  supervised model. Fixed the markdown and the print statement's own
  conclusion text before re-running (own release_speed vs. own
  pitch_stuff_plus correlation is still strongly positive everywhere, 0.07
  to 0.51 by pitch type, now framed as an empirical expectation for a real
  predictor of run value, not a built-in guarantee). Every downstream
  number moved (expected, since they're all computed from the new
  stuff_plus), but the qualitative story didn't: Location+ still dominates
  at the pitch level, Stuff+ still needs a much larger sample to show its
  full relationship (its R^2 now leads Location+'s starting from the lowest
  threshold tested, 20 pitches, versus needing several hundred before under
  the old design) and is still the far stickier season-to-season skill
  (0.84 vs. 0.42, versus the old ~0.89/~0.41). What did change materially:
  Stuff+'s own standalone aggregate R^2 roughly tripled (0.02 -> 0.059,
  matching the standalone check above) and its standardized beta closed
  most of the gap with Location+'s (was roughly half Location+'s magnitude,
  now 0.166 vs. 0.206, within 20%). Rewrote the closing "Reconciling"
  synopsis cell with the real numbers rather than leaving 8/27-era figures
  in place next to a freshly-run notebook.
- fg_pitching.ipynb: this is the one that mattered. Before executing,
  read every markdown cell for stale claims (the same check that caught
  pitching.ipynb's sign-anchoring line) and found a real methodological
  problem, not just stale prose: cell 5's own caveat said Stuff+ "keeps one
  small asymmetry" in this notebook's train/2021-2024-test/2025 holdout
  design -- its loadings are fit pooled across all five seasons, called "a
  much weaker form of leakage than a supervised model training on
  individual future outcomes (a shared direction-of-variance across years,
  not memorized results)." That description was accurate for the PCA-era
  Stuff+ (never touches outcome data, so pooling seasons couldn't leak
  results) and is no longer accurate for the retooled one: `add_stuff_plus`
  now trains directly against `delta_pitcher_run_exp`, and the notebook's
  "fair comparison" cell was scoring the blend's Stuff+ input from the
  pooled production cache -- meaning 2025's own outcomes were leaking into
  the blend's 2025 score in a comparison whose whole point is a strict,
  leak-free holdout.
  Fixed by giving Stuff+ the identical train-only treatment Location+
  already had in this same notebook: refit `stuff_mod._train_models`
  on `train` (2021-2024) only, score `test` (2025) by direct prediction.
  Simplified along the way -- realized the blend's own z-scoring step
  already standardizes per-(pitch_type, season) before pooling across
  types, so raw `stuff_run_value` (already the same units as
  `location_run_value`: a direct run-value prediction) works as the blend
  input without needing stuff.py's 100+ ratio-scale/log round trip at all;
  removed `log_stuff` throughout in favor of `stuff_run_value`.
  The fix reverses the notebook's headline conclusion. Old (leaky)
  result: blend holdout R^2=0.0523 beat the joint model's 0.0386. New
  (leak-fixed) result: the joint model's holdout R^2=0.0452 beats the
  blend's 0.0353 -- and every joint-model variant tested in this notebook
  (raw features, three regularized configs, and the two-calibrated-inputs
  version) now beats the blend too, where before the blend beat all of
  them. The joint model's own numbers didn't move at all between the two
  runs (it was never fed the leaky cache); only the blend's score changed,
  which is exactly what confirms the leak was the cause, not
  re-run-to-rerun noise from unrelated changes.
  Rewrote every downstream section that was built to explain the old
  result (the regularization sweep, the calibrated-inputs experiment, the
  closing synopsis) rather than just refreshing numbers in place, since
  their original framing ("does X help the joint model close the gap to
  the blend") no longer has a gap to close. The individual sub-findings
  mostly still hold and were kept, reframed: regularizing the joint model
  here still makes it worse, not better (every regularized variant holds
  up worse on the 2025 holdout than the original untuned config); the
  two-calibrated-inputs variant still falls short of the raw-feature
  joint model's absolute holdout R^2, though it now has the single best
  (lowest) shrinkage of any variant tested, blend included.
  This is flagged as an open question for the user, not resolved here:
  `pitching.py`'s production Pitching+ is currently the weighted blend,
  and that choice was partly justified by this exact notebook's original
  (leaky) comparison. The honest, leak-fixed version of that comparison no
  longer supports it -- a jointly-trained model now generalizes better in
  this specific rerun. That changes what the data recommends, but deciding
  whether to rebuild `pitching.py` around a joint model is a
  production-architecture decision, not something to change as a side
  effect of a notebook-documentation pass. No code in `pitching_plus/
  scripts/` was touched by this finding.
- Full pytest suite unaffected by any of this (no scripts changed in this
  entry, only notebooks); last verified green in the entry above.

# 9/4/2026 (cont'd): leave-one-season-out check -- the joint model wins in every fold

- The prior entry's finding (jointly-trained model beats the weighted blend once
  the Stuff+ leak is fixed) rested on exactly one train/test split: train
  2021-2024, test 2025, the only holdout fg_pitching.ipynb has ever used, out of
  only 5 available seasons. Real risk that a single split is a 2025-specific
  artifact rather than a robust property. Before touching production code
  (`pitching.py`/`bestpitch.py`), ran a leave-one-season-out (LOSO) check:
  rotate the held-out season across all 5 (2021-2025), retraining the joint
  model and both blend inputs (Stuff+, Location+, both train-only per the
  leak fix) fresh on each fold's 4-season train set, scoring on the held-out
  season. Scoped to the core comparison only (original hyperparameters, no
  regularization or calibrated-inputs variants -- those weren't part of what
  needed re-validating).
- Pre-committed pass bar, decided before running: joint model beats the blend's
  holdout R^2 in >= 4/5 folds AND the mean holdout R^2 gap across all 5 folds is
  positive. New cells appended to fg_pitching.ipynb (after cell 21) rather than
  a separate script, reusing `engineered`/`JOINT_FEATURES`/`weighted_r2` and the
  exact train-only-refit/aggregate/WLS pattern the existing cells already
  established. Fold logic kept in a pure function taking `engineered` and
  `held_out_season`, returning a plain dict -- deliberately does not touch the
  module-level `train`/`test` globals the rest of the notebook depends on, so
  an out-of-order rerun of earlier cells can't get silently corrupted by the
  loop's own state.
- Result: **5/5 folds**, joint model beats the blend in every one. Per-season
  holdout R^2 (blend / joint): 2021 0.0937/0.1009, 2022 0.0966/0.1096, 2023
  0.1097/0.1211, 2024 0.0834/0.0939, 2025 0.0353/0.0452. Gap is consistently
  positive (+0.0072 to +0.0130, mean +0.0104) -- no season is an outlier in the
  gap itself, though 2025's absolute R^2 is unusually low for both approaches
  compared to the other four seasons (a real, separate observation: 2025 looks
  like a harder season to predict generally, not a reason the joint model's
  edge happened to show up there specifically). Total runtime: 756s for all 5
  folds (~74s/fold for the joint model's training, ~78s/fold for the blend
  inputs' train-only refits), on top of the ~34s data load already paid
  earlier in the notebook. Full re-execution of the whole notebook (all 27
  cells, fresh kernel) took under 15 minutes total.
- **Verdict: passes cleanly, at the strongest possible outcome.** The original
  single-split result was not a 2025-specific artifact. **Stage B of the
  migration plan is authorized to proceed**: rearchitecting `pitching.py` and
  `bestpitch.py` around a per-pitch-type jointly-trained model
  (`HistGradientBoostingRegressor` on `STUFF_FEATURES + LOCATION_FEATURES`
  against `delta_pitcher_run_exp`, mirroring `stuff.py`'s own architecture),
  replacing the WLS blend and its `_apply_blend`/`_fit_pitching_plus_model`
  internals. See the migration plan drafted for this entry for the full
  design, including a real correctness risk identified during planning:
  `bestpitch.py`'s `_build_base_array` zone-dedup key is currently a bijection
  over game-state + handedness only, which stops being true once
  pitcher-varying physics features are folded into the same array -- the dedup
  key needs to also discriminate on `(pitcher, season)` or counterfactual
  scoring would silently combine different pitchers' physics.
- No code in `pitching_plus/scripts/` touched by this entry -- only
  `fg_pitching.ipynb` (the new LOSO cells) and this dev_log entry. Full pytest
  suite unaffected, still green as of the entry above.

# 9/4/2026 (cont'd): Stage B -- Pitching+/bestPitch+ rearchitected as a joint model

- Stage A passed 5/5 -- this entry is Stage B, rearchitecting `pitching.py`
  and `bestpitch.py` around the jointly-trained model per the migration
  plan. Sequenced `pitching.py` first (tested fully in isolation), then
  `bestpitch.py` (depends on `pitching.py`'s new cache/calibration).

- **`pitching.py` rewrite**: replaced `_apply_blend`/`_fit_pitching_plus_model`
  with a `_train_models`/`load_cached_models`/`_cache_fingerprint`/
  `_load_or_score`/`_calibrate` architecture mirroring `stuff.py`'s retool
  exactly -- one `HistGradientBoostingRegressor` per pitch type on
  `JOINT_FEATURES = STUFF_FEATURES + LOCATION_FEATURES` (20 features: 7
  physics + 13 location/count) against `delta_pitcher_run_exp`, 5-fold OOF
  CV, cached under `pitching_plus/models/` (`pitching_models.joblib`,
  `pitching_historical_scores.joblib`, `pitching_cache_fingerprint.joblib`)
  with the same fingerprint-staleness warning. Reused `location.py`'s
  `_ratio_calibration`/`_to_100_scale` directly via import instead of
  duplicating them (no circular-import constraint here, unlike `stuff.py`).
  `_score_pitching_plus` was split into two composable pieces:
  `_calibrate_raw_value` (apply an already-fitted 100+ calibration to a raw
  value, real or hypothetical) and `_score_pitching_plus` (predict via
  per-pitch-type models, then calibrate) -- `bestpitch.py`'s target-averaging
  machinery computes its own raw predictions and only needs the calibration
  half. `_calibrate` now returns `pitch_calibration` itself (not just
  applies it), since `bestpitch.py` needs the table, not just already-scored
  rows. Public contract unchanged: same `add_pitching_plus` signature, same
  three output columns.
- Real-data validation: full retrain on the 3,565,743-row dataset --
  `add_stuff_plus` 176s, `add_location_plus` 427s, `add_pitching_plus` 539s
  (includes a real, accepted redundancy: `add_pitching_plus` re-runs
  `stuff_mod._build_v1_features`/`location_mod._build_features` on its own
  in-scope population rather than threading through what `add_stuff_plus`/
  `add_location_plus` already built, mirroring this codebase's existing
  convention of each module building its own working frame independently).
  Calibration lands exactly on 100.0 for every pitch type (std 9.0-9.9,
  matching the "~10 points per SD" design intent); pitch-level std (9.17)
  tracks the aggregate-level std (8.49) closely -- the dual-level sigma
  split was implemented correctly from day one here, unlike the PCA-era
  `stuff.py`/pre-9/2 `location.py`, so there was no bug to find, only a
  clean confirmation.
- All 6 `test_pitching.py` tests pass: the 3 architecture-independent ones
  survive with `small_pitching_thresholds` added (a **new fixture**,
  required because `pitching.py` now has its own `MIN_GROUP_SIZE_FOR_MODEL`
  gate -- and because it imports `MIN_PITCHES_FOR_SCORE` from `stuff.py` via
  `from module import name`, a one-time binding at import that
  `monkeypatch.setattr(stuff, ...)` cannot reach after the fact, so
  `pitching.py`'s own constants need their own fixture regardless of what
  `stuff.py`'s says); `test_pitch_level_calibration_uses_its_own_spread_not_the_aggregates`
  and `test_apply_blend_level_dispatch_actually_uses_the_right_sigma`
  (blend-internals tests with nothing left to test) were replaced by
  `test_pitch_pitching_plus_calibration_uses_pitch_level_spread_not_the_aggregates`,
  mirroring `stuff.py`'s own sigma-split regression test; 2 new tests added
  (cache-roundtrip determinism, stale-fingerprint warning), mirroring
  `stuff.py`'s suite.

- **`bestpitch.py` rewrite**: `_arsenal_wide` (a `cand_stuff_<type>` scalar
  per pitcher-season) replaced by `_arsenal_physics_avg` (a full
  `STUFF_FEATURES` vector per pitcher-pitch_type-season, renamed to
  `cand_<feature>` to avoid colliding with a row's own real physics columns
  on the same working frame). `_build_base_array` now builds a
  `JOINT_FEATURES`-wide array and takes a `physics_cols` parameter selecting
  which columns to read physics from (a row's own real `STUFF_FEATURES` for
  `_actual_smoothed_pitching_plus`, or a candidate's `cand_<feature>`
  columns for `_search_best_pitching_plus`).
- **Real bug found and fixed during planning, before any code was written**:
  `_build_base_array`'s `center_key` dedup was a genuine bijection over
  game-state + handedness alone (`re288_state * stand_R * p_throws_R`,
  <=1,152 keys) under the old location-only model, because every
  non-location feature besides the 3 varying ones was constant across
  pitchers for a fixed game-state. That stops being true once
  pitcher-varying physics are folded into the same array -- two rows
  sharing a game-state but belonging to different pitchers are NOT
  interchangeable, and deduping them together would silently score one
  pitcher's candidate with another pitcher's physics. Fixed by extending
  `center_key` (and the `zh_key` derived from it) to also discriminate on
  `(pitcher, season)` via `dedup_by_pitcher_season=True` -- a compact id
  factorized from `pitcher_season_gamestate`, folded into the key before any
  row ever gets deduplicated. `_actual_smoothed_pitching_plus` uses
  `dedup_by_pitcher_season=False` instead (a trivial per-row key): physics
  there is each pitch's own real, continuously-varying measurement, not a
  per-pitcher-season average, so no two rows are genuinely interchangeable
  and there's no real dedup benefit to chase -- `np.unique` on an
  already-unique key is a no-op split, not a bug, letting both callers share
  the identical downstream averaging code. Added
  `test_batched_target_averaging_matches_unbatched_reference_with_varying_pitcher_physics`,
  which builds a fixture with multiple pitchers sharing a game-state, gives
  each distinct candidate physics, and confirms the batched path matches an
  unbatched per-row reference -- a reverted, game-state-only key fails this
  by borrowing one pitcher's physics for another's row.

- **Second real bug, found only by running on the full dataset, not caught
  by any test or by planning**: the existing `_all_zones_target_averaged_*`
  optimization (batch every candidate zone into 2 `model.predict()` calls
  per pitch type instead of 2 per zone) was safe under the old location-only
  key because its dedup ceiling was a hard <=1,152, so the `n_zones`
  multiplier (13x) never produced more than a few tens of thousands of rows
  per predict() call regardless of population size. Once `center_key` also
  discriminates by pitcher-season, that ceiling disappears: a pitch type
  thrown selectively by many pitchers in a narrow, scattered set of
  situations (a show-me curveball, not a bread-and-butter fastball) barely
  dedupes at all, so `n_zones * 3 * k` (or `* 2 * m`) rows can reach the tens
  of millions. Measured directly: on the real 2021-2025 dataset, the
  candidate-population for CU (curveball) is 1,687,435 rows across 1,360
  pitcher-seasons -- *smaller* than FF's 3,235,576 rows across 3,381
  pitcher-seasons -- yet CU's single build-and-predict step ran over 3 hours
  and consumed 15+ GB of RAM (confirmed live via `Get-Process`: 15,567
  CPU-seconds accumulated against ~52 minutes of wall-clock time, i.e.
  genuinely running hot across ~5 cores, not merely slow), before being
  killed, while FF's own step added only ~800s. The likely reason CU fared
  worse despite a smaller population: a pitch thrown narrowly by many
  pitchers in a scattered set of situations has far less (pitcher,
  game-state) repetition than a bread-and-butter pitch thrown by fewer,
  heavier-volume arms in nearly every count -- a much higher unique-key
  ratio relative to row count, which is exactly what the `n_zones` multiplier
  then amplifies. Fixed by removing the all-zones batching from
  `_search_best_pitching_plus` entirely, reverting to a per-zone loop
  (mirroring `_actual_smoothed_pitching_plus`'s already-adopted pattern) --
  `_all_zones_target_averaged_pitching_run_value` is now dead code and was
  deleted rather than left unused. Validated the fix in isolation before
  re-running the full dataset: CU's entire 13-zone search (the pathological
  case) dropped from >10,800s (killed before finishing) to 341s.
- **Caveat on the full-dataset re-timing run**: after the fix, the first
  108/130 candidates completed in a clean ~2,709s (~45 min, matching the
  isolated CU test's per-candidate rate). A single gap then appeared between
  candidates 108 and 109 (`ST` zone 4 to zone 5 -- an unremarkable, small
  pitch type, not a repeat of the CU pathology) of 18,321s, coinciding with
  an overnight period during which the session's own clock rolled over a
  calendar day while waiting on this exact run. Every candidate immediately
  before and after that gap completed in single-digit-to-low-double-digit
  seconds, with no corroborating evidence of heavy computation during the
  gap (unlike the CU bug, which was directly confirmed via live process
  memory/CPU inspection before it was fixed) -- this has the signature of
  the machine sleeping mid-run, not a second code-level bug, but is reported
  as an open, not fully certain, attribution rather than dismissed outright.
  Subtracting that one gap from the reported 21,330s total gives **~3,009s
  (~50 min)** as the real, corrected search time -- about 3.3x the
  pre-migration 910s baseline, a real and measurable slowdown but the kind
  the migration plan explicitly anticipated (each candidate now costs a real
  `model.predict()` call plus a weakened dedup ceiling, not a closed-form
  blend over two scalars), not the >20x catastrophic regression the
  unfixed bug actually produced.
- Distribution sanity, same checklist as every other stage of this
  migration: `pitch_bestpitch_plus` mean 9.16, median 7.48, std 8.36, max
  100.09, **100.00% non-negative** (n=3,521,210) -- meets or exceeds the
  "positive or close to 0 for nearly every pitch" contract even more
  cleanly than any prior version of this metric (99.1%-99.95% in earlier,
  pre-migration runs per the 8/30 and 8/31 entries). Leaderboard eyeballed
  for plausibility: top `pitching_plus` names are deGrom, Glasnow, Treinen,
  Burnes (appearing twice, different seasons/pitch types), Cole -- real,
  known plus-stuff arms, not small-sample noise; the closest-to-optimal
  `bestPitch+` list surfaces known-command names (deGrom again, among
  others), consistent with the pattern the 8/27 entry found for the
  original location-based command leaderboard.
- Full pytest suite (32 tests, including the new
  `test_batched_target_averaging_matches_unbatched_reference_with_varying_pitcher_physics`
  and `test_arsenal_physics_avg_only_uses_reliable_rows`) passes after both
  fixes.

- **Documentation**: `purpose.md`'s Pitching+ section rewritten to describe
  the joint-model design (mirroring the phrasing already used for Stuff+'s
  own retooled section). `pitching.py`'s module docstring rewritten from
  scratch -- removes the stale 0.052/0.039 blend-won numbers, describes the
  joint model, and explicitly supersedes (not deletes) the 8/26/8/31
  framing, pointing to this entry and the two before it for the reversal's
  full history. `README.md` checked, needs no change (its one relevant line
  is generic enough to remain accurate).
- Not done in this entry: no new branch was cut for Stage B (stayed on
  `rework/stuff-weighting`, per the plan's own framing of a separate branch
  as a suggestion, not a requirement, for a branch that hasn't been merged
  yet). `notebooks/bestpitch.ipynb` and `notebooks/pitching.ipynb` were not
  re-executed against the new Pitching+ architecture -- both notebooks'
  own numbers are now stale relative to what's shipped in
  `pitching_plus/scripts/`, a known gap in the same spirit as the 9/4
  "brought the notebooks up to date" entry's own deliberately-deferred
  `bestpitch.ipynb` rerun, now compounded by this migration. Worth a future
  pass, not blocking.

# 9/8/2026: avoid-ai-writing audit + code-simplification pass on the retool

- Ran two pre-ship checks against this branch's staged diff (the Stuff+/
  Pitching+/bestPitch+ retool above) using Claude Code's `avoid-ai-writing`
  and `code-simplification` skills: an audit of the new comments/docstrings,
  and a simplification pass on the four changed scripts.
- avoid-ai-writing: swept `stuff.py`, `pitching.py`, `bestpitch.py`,
  `full_pipeline.py`, `purpose.md`, and this file's new 9/4 entries against
  the skill's word list (delve, leverage, robust, seamless, testament to,
  etc.), confidence-calibration filler ("notably," "worth noting"), and
  vague attributions. Zero hits in the four scripts and `purpose.md`; the
  only near-hits (this file's own "robust"/"harness"/"leverage" uses)
  turned out to be legitimate statistical/ML/baseball terms of art (a
  regression coefficient's robustness, an evaluation harness, count
  leverage), not filler. The one stylistic pattern actually present --
  heavy `--` parenthetical-dash use -- is a pre-existing, consistent
  convention across the whole codebase (confirmed at a comparable rate,
  roughly 7-9 per 1,000 words, in the untouched `location.py`), not
  something this diff introduced, so left alone rather than de-dashing only
  the touched files and making the codebase's voice inconsistent. No edits
  made for this half of the pass.
- Code simplification: read all four changed scripts against the skill's
  structural/duplication/naming checklist. Two candidates considered and
  rejected under Chesterton's Fence:
    - The near-identical `_train_models`/`_cache_fingerprint`/
      `_load_or_score` trio duplicated across `stuff.py`/`location.py`/
      `pitching.py` (~80 lines each) is a real DRY violation on its face,
      but it's a deliberate, repeatedly-documented convention -- each
      module's own docstrings say it mirrors the others' architecture so it
      can be read standalone, `stuff.py` explains why it can't import from
      `location.py` (circular), and `pitching.py` already reuses
      `location.py`'s `_ratio_calibration`/`_to_100_scale` directly
      wherever no such constraint applies. Sharing this trio would also
      require touching `location.py`, outside this diff's scope. Left as-is.
    - `stuff.py`'s own `load_cached_models` is, by its own docstring's
      admission, also currently uncalled ("included for architectural
      parity"), the same shape as the item actually removed below. Kept
      anyway: it's a 7-line pure wrapper, `pitching.py`'s copy of the same
      function IS called (by `bestpitch.py`), and `location.py`'s copy is
      called from a notebook -- cheap enough, and consistent enough with
      real usage elsewhere, not to be worth removing.
  Two changes applied:
    - Removed `pitching.py`'s `_score_pitching_plus`: dead code introduced
      by this same retool. Its own docstring already admitted bestPitch+'s
      real search path calls `_calibrate_raw_value` directly instead of it;
      confirmed via grep across `scripts/`, `tests/`, and `notebooks/` that
      nothing calls it (the one hit, `bestpitch.ipynb`'s markdown, names a
      pre-retool function with a completely different signature and was
      already stale before this pass). Also fixed `load_cached_models`'s
      docstring, which still claimed bestPitch+ scores through it.
    - `bestpitch.py`'s `_build_base_array` did a repeated
      `physics_cols[STUFF_FEATURES.index(col)]` list search per
      STUFF_FEATURES column inside its per-pitch-type setup loop. Replaced
      with one `dict(zip(STUFF_FEATURES, physics_cols))` built up front,
      matching the cleaner zip-based pattern `_predict_at_point` (a few
      lines above it in the same file) already uses for the identical
      mapping. No behavior change -- 7 items, built once per pitch type, so
      this was never a real cost, just a needless indirection to read.
  Also fixed, found during the same pass, not really a "simplification":
  `full_pipeline.py`'s `--retrain` CLI help text still said "Retrain
  Stuff+/Location+ models," stale since this retool made Pitching+ (and by
  extension bestPitch+, which calls `add_pitching_plus`) retrain under that
  same flag too. The other three scripts' help text was already corrected
  during the retool itself; this one copy was missed.
- Full pytest suite (32 tests) reconfirmed green after both edits.
  Separately, a real-data smoke run of the full pipeline was underway from
  the prior verification pass (stuff/location/pitching via cache, ~40s
  each on the full 3.57M-row dataset; bestPitch+'s counterfactual search
  in progress) and unaffected by these edits, since neither touches scoring
  logic or column names.

# 9/21/2026: bestPitch+ reported as a percent of value captured

- User asked whether "{pitcher} achieved X% of the max possible value" would
  read more easily than "the gap between the best and actual pitch was 10
  Pitching+ points." Added it as a reporting layer, not a retool: the
  counterfactual search, the joint model, the calibration, and
  best_pitching_plus / pitch_bestpitch_plus are unchanged (worked on main,
  no branch). Three new columns from add_bestpitch_plus:
  pitch_bestpitch_plus_pct (100 * actual_smoothed / best_pitching_plus),
  bestpitch_plus_pct (pitcher x pitch_type x season), and
  bestpitch_plus_pct_reliable (>= MIN_PITCHES_FOR_SCORE in-scope pitches
  behind the ratio, same bar as the other *_reliable flags).
- Design choices, all confirmed with the user first. The ratio uses the two
  100+ scores, not raw run value: the 100+ scale is 100*exp(k*z)/raw_ratio_mean,
  strictly positive, so the ratio is always defined, while pitching_run_value
  can be negative or near zero. It means "share of the achievable calibrated
  score," not "share of runs." The season figure is 100 * mean(actual) /
  mean(best) per group, not the mean of per-pitch percentages: the two differ
  whenever best_pitching_plus varies within a group, and the mean of ratios
  overweights pitches whose best score is small (same aggregate-first
  principle as _calibrate in location.py/pitching.py). Not clipped at 100,
  matching pitch_bestpitch_plus not being clipped at 0. A 100+ rescaling of
  the percentage itself was left out on purpose.
- add_bestpitch_plus now restores the input's index after the aggregate
  merge (merge() resets it). Added 4 tests to test_bestpitch.py (ratio and
  bounds, NaN scope identical to pitch_bestpitch_plus, ratio-of-means versus
  mean-of-ratios on a group where they differ, reliability gate) and the
  three columns to test_full_pipeline.py's column list. One trap in the
  reliability test: bestpitch.py binds MIN_PITCHES_FOR_SCORE at import, so it
  has to be patched on bestpitch, not stuff. 36 tests pass.
- Real-data run on all 3,565,743 pitches: add_bestpitch_plus took 2,578s
  (stuff/location/pitching from cache; the counterfactual search is nearly
  all of it). 3,521,210 pitches get a percentage, and the NaN mask is
  identical to pitch_bestpitch_plus. Pitch level: min 31.6, median 93.1,
  mean 91.8, max 100.0; every value > 0, none above 100, and 3.74% sit at
  ~100 (the actual pitch was the best candidate). Season level: 17,545
  groups, 14,343 reliable, median 92.0, middle half 90.7-93.2, range
  68.7-97.6. Across 3,590 pitcher-seasons (all pitch types, >= 100 pitches)
  the Spearman correlation between mean point gap and percent captured is
  -0.997, so the two versions rank pitchers almost identically.
- Finding: the range is compressed. Both scores sit on a 100-centered scale,
  so a typical pitch already captures about 92% and pitcher-season values
  have a standard deviation of 1.0 (82.8 to 95.2). A difference of one or two
  points is meaningful, and 92% should not be read as "missed 8% of the
  value." The percentage is easier to say aloud but spreads pitchers less
  than the point gap does. Measuring capture relative to 100 instead (edge
  over average) would spread it out but breaks when the actual score is
  below 100, and rescaling the percentage to 100 = league average is the
  natural follow-up; neither is built.
- Not checked: the bottom of the pitcher-season list (Lugo 2023 at 82.8%,
  Newcomb, Fairbanks) may partly reflect the max-over-candidates effect from
  the 8/27 and 8/30 entries, where a pitcher with more candidates gets a
  higher best_pitching_plus regardless of decision quality. Untested. The top
  (Campbell 2025, deGrom 2024, Pomeranz 2025, Hill 2025) is plausible.
  bestpitch.ipynb was not rerun and does not show the new columns.