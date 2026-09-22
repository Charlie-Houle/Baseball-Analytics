# 9/2/2026
- Created slippery_slope/ project folder with docs, notebooks, scripts, and tests subfolders, mirroring pitching_plus's layout
- Added purpose.md: how an at-bat's trajectory shifts after early pitch results, starting with count-state outcome rates and the first-pitch-ball effect, then extending to pitch-type strategy (Pitcher Plinko-style) and eventually location strategy
- Scaffolding only; no data work, notebooks, or scripts started yet

# 9/3/2026: notebooks/count_state.ipynb, purpose.md's workflow item 1
- Built count_state.ipynb: baseline count-state outcome rates (K rate, BB rate, AVG) by
  ball-strike count, loading only game_pk/at_bat_number/pitch_number/balls/strikes/events/
  description/game_year from data/MLB_2021-2025.csv (usecols, no full-width read).
- Core design decision, called out explicitly by the user before writing any code: a naive
  "outcome rate by count" is ambiguous between two different questions that don't share an
  answer. Immediate pitch outcome (did THIS pitch, thrown at this count, directly end the
  at-bat) is mechanically constrained by the rules, e.g. strikeout rate at 0-1 is exactly 0%
  by construction, since a strikeout needs 2 strikes already banked. Resulting at-bat outcome
  (of every at-bat that ever passed through this count, how does it eventually end) is the
  one that's actually informative. The notebook computes both and reports the resulting
  version as the headline number, with the immediate version kept only as an explicit
  contrast (built as its own cell/section, not just mentioned in passing).
- At-bat-level results built by collapsing each (game_pk, at_bat_number) group to its one
  non-null `events` value (verified this is 1:1 on the full 3,565,743-row dataset before
  writing the collapse logic: 914,304/914,791 at-bats have exactly one recorded final event;
  487 don't, mostly suspended/cutoff games, and are dropped since there's no eventual result
  to attribute to any count they passed through). Results bucketed into strikeout, walk
  (incl. intentional), hit_by_pitch, hit, and out_or_other; `is_official_at_bat` flags the
  traditional AVG denominator (excludes walks/HBP/sacrifices).
- Resulting rates need a dedupe step immediate rates don't: the same (balls, strikes) can
  recur within one at-bat (fouls at 2 strikes don't advance the count), so at-bat x count
  pairs are deduplicated before joining to the at-bat's eventual result, otherwise an at-bat
  with 4 two-strike fouls would get counted 4x at that count state.
- Ran end-to-end (`jupyter nbconvert --execute --inplace`) rather than just eyeballing the
  code, confirmed no errors and a real plot renders. Pooled across every at-bat: 22.6%
  strikeout, 21.9% hit, 8.4% walk, 1.1% HBP, 45.9% out_or_other. 0-1 is the clean illustration
  of the immediate/resulting split: immediate K rate 0% (impossible by rule) vs. resulting K
  rate 30.7% of the 459,559 at-bats that ever reach 0-1, well above the 22.6% pooled rate.
  3-2 is the only count where both immediate rates are nonzero: 20.2% immediate K, 22.9%
  immediate BB.
- This is purpose.md workflow item 1 only (baseline count-state outcome rates). Item 2
  (first-pitch-ball effect: 1-0 vs. 0-1 downstream comparison) is the next notebook, not
  started yet.

# 9/3/2026 (cont'd): count_plinko_artifact.html, a result-based pitcher-plinko diagram
- User asked for a Baseball Savant "Pitcher Plinko"-style diagram, but with the pie slices
  as pitch results (hit/walk/strikeout) instead of pitch-type mix, pooled leaguewide (not
  aggregated by pitcher/pitch type like the reference image), plus hover tooltips. Built as
  a standalone published Artifact (slippery_slope/notebooks/count_plinko_artifact.html) since
  the hover interactivity and custom node-link layout aren't a notebook-native deliverable.
- Data prep (scratchpad script, not checked in): loaded game_pk/at_bat_number/pitch_number/
  balls/strikes/events from the full 3,565,743-row CSV. Classified every pitch by whether it
  ends the at-bat (`events` non-null: strikeout/walk incl. intentional/hit_by_pitch/hit/
  out_or_other, the last bucket also folding in the ~487 truncated_pa/catcher_interf rows,
  negligible volume) or continues it (`events` null). Node volume/composition = value_counts
  of that category per (balls, strikes). Edge volume = pitch-to-pitch count transitions
  within the same at-bat (groupby+shift(-1) on sorted pitch_number, self-transitions from
  2-strike fouls excluded from the drawn edges since the reference diagram doesn't show
  them either, though they still count toward that node's own pitch total).
- Cross-validated against count_state.ipynb's numbers before trusting the new script: e.g.
  strikeout/total at (0,2) and (3,2) match count_state's immediate_k_rate exactly, walk/total
  at the three balls=3 nodes matches immediate_bb_rate, and node totals sum to exactly
  3,565,743 (every pitch counted once, at its own entering count, no rows dropped this time
  since even the excluded-event rows still get bucketed rather than dropped).
- Layout: nodes positioned by row = balls+strikes (0 to 5, matching the reference image's
  6-row hexagon shape exactly), evenly spaced within each row sorted by balls ascending,
  reproducing the reference's left-to-right/top-to-bottom arrangement without hardcoding
  pixel positions. Node radius and edge stroke width both sqrt-scaled (area/thickness
  proportional to actual pitch volume, not the raw value) between an observed min/max.
- Followed the dataviz skill for the categorical palette: "Continues" (the dominant,
  least-interesting slice) gets a neutral chrome gray rather than a hue, so the five real
  outcome categories are what visually pop; the five hues are a validated adjacent-safe
  sub-sequence of the skill's default 8-slot order (ran scripts/validate_palette.js against
  both light and dark, all checks pass). Slice/legend colors are set via CSS var() rather
  than a resolved hex, so a live theme toggle re-colors the SVG without a JS re-render.
- Light mode's contrast check came back WARN for 3 of the 5 hues (below 3:1 against the
  chart surface), which the skill treats as non-dismissable: added a "Table" toggle
  (full node x category % breakdown) alongside the diagram as the required relief, on top
  of the hover tooltips and aria-labels already planned for interactivity.
- Ran the full pipeline end-to-end and checked the resulting JSON by hand (12 nodes, 17
  edges, matching the count-state graph's actual structure: every balls<3 node has a ball
  edge, every strikes<2 node has a strike edge) before embedding it in the artifact; did not
  get a rendered screenshot of the published page in this session.

# 9/3/2026 (cont'd): plinko moved into count_state.ipynb, plus a resulting/rolling version
- Two follow-up requests: (1) the same diagram but on a "resulting" basis (% of at-bats at
  this count that eventually end in each result, mirroring count_state.ipynb's immediate-vs-
  resulting split rather than the pitch-level immediate-only version built earlier), and
  (2) move the whole thing into the notebook instead of a standalone artifact, since the
  published artifact's look/interaction was already confirmed correct.
- Resulting version's node volume is at-bats reaching that count (n_abs_through, not pitch
  count), and its pie has no "Continues" slice, since every at-bat that passes through a
  count eventually has some final result; the 5 terminal categories alone sum to 100%.
  Realized while building it that edges don't need separate at-bat-deduped logic: a
  (from, to) count transition can only happen once per at-bat, since balls/strikes are
  monotonic non-decreasing (the graph is a DAG with no way back to an earlier count), so
  the same pitch-to-pitch transition tally already used for the immediate version is
  correct at the at-bat level too, edges only needed re-deriving because the resulting
  version restricts to ab_outcomes-eligible pitches (drops the same handful of truncated/
  interference at-bats count_state.ipynb already excludes) rather than every pitch.
- Cross-validated the new resulting numbers against cells already in the notebook before
  trusting them: resulting node total at 0-0 (912,449) matches n_abs_through(0,0) computed
  earlier in the notebook exactly, and resulting strikeout rate at 0-1 (30.73%) matches the
  synopsis's already-reported number exactly.
- For "isolated into the notebook": refactored the standalone artifact's CSS/HTML/JS into a
  single parameterized `render_plinko(chart_id, data, categories, title, subtitle, note,
  unit)` Python function (same template, unchanged logic) that returns an HTML string for
  `IPython.display.HTML`. Every CSS selector and element id is namespaced by `chart_id`
  (`plinko-root-imm` / `plinko-root-res`, `imm-chart` / `res-chart`, etc.), which the
  standalone single-page artifact didn't need but two independent chart outputs sharing one
  notebook DOM do, otherwise the second chart's styles/getElementById calls would collide
  with the first's. Slice/legend colors still set via CSS var() (from the artifact's dataviz-
  skill-validated palette work) so a live theme toggle re-colors both without a JS re-render.
- Data prep now lives in the notebook itself as real pandas cells (not a scratchpad script
  embedding a precomputed JSON blob like the first version): reuses classify_result,
  the *_EVENTS sets, `data`, `pitch_level`, `ab_outcomes`, `visits_with_result`, and `AB_KEY`
  already defined earlier in the notebook rather than recomputing or reloading anything.
- Ran the full notebook end-to-end (`jupyter nbconvert --execute --inplace`, all 23 cells,
  no errors) and diffed the DATA JSON embedded in each rendered HTML output against the
  scratchpad numbers used to build the original artifact: identical (12 nodes/17 edges each,
  same totals). Deleted count_plinko_artifact.html from the repo now that the notebook is
  the canonical version; the previously published claude.ai artifact link is unaffected
  (publishing and the repo file are independent) but won't be kept in sync with future
  notebook changes.

# 9/3/2026 (cont'd): swapped the hand-rolled HTML/CSS/JS plinko for Plotly
- User doesn't know HTML and wants to be able to read/maintain this themselves, so the
  IPython.display.HTML(custom CSS/SVG/JS) version from the previous entry is gone. Rebuilt
  `plot_plinko` as a plain Python function returning a `plotly.graph_objects.Figure`;
  displayed with `fig.show()`, no markup anywhere in the notebook.
- plotly wasn't installed in this project's .venv (only in an unrelated system Python that
  happened to satisfy an earlier `python3 -c "import plotly"` sanity check from a differently-
  resolved interpreter, which briefly gave a false read on availability). Installed
  plotly==7.0.0 into .venv and added it to requirements.txt.
- The hard part porting this to Plotly: `go.Pie` doesn't take a data-space (x, y, radius) the
  way the old SVG arcs did, only a `domain` (a rectangle in figure-fraction space, [0,1] on
  each axis). Solved by keeping the exact same pixel-based row/column layout math as the SVG
  version (unchanged), then converting each node's pixel position and radius to a domain
  rectangle by dividing by the figure's own pixel width/height. Put the edges (`go.Scatter`
  lines) and labels (`fig.add_annotation`) on a matching hidden 0-to-1 x/y axis pair (not the
  default data range) so both coordinate systems agree without any manual pixel/margin
  reconciliation. First draft set `scaleanchor`/`scaleratio` on the y-axis to force a 1:1
  aspect ratio; removed it after realizing domains and a plain (unlinked) 0-to-1 axis already
  scale independently to the figure's actual width/height by default, which is what the
  position math assumes, so the scaleanchor constraint was fighting the layout rather than
  helping it.
- Hover is now native Plotly, not custom code: pie slices get a `hovertemplate` (percent,
  raw count, and the node's total, via `customdata`) instead of a hand-built tooltip div, and
  edges are real `go.Scatter(mode="lines")` traces (not static SVG lines) so hovering one
  shows its own pitch count for free. The category legend is built from small invisible
  dummy scatter traces (`x=[None]`), a standard Plotly pattern for a custom legend when the
  real data traces (12 separate Pie traces sharing the same categories) would otherwise each
  contribute their own redundant entries.
- Table fallback is a plain pandas `Styler` (`.style.format("{:.1%}")` on the already-computed
  `immediate_node_counts` / `resulting_node_counts` DataFrames) instead of a hand-built HTML
  table, one line each.
- Verified via the real venv interpreter, not the system one that caused the plotly
  false-positive above: ran a minimal Plotly figure through `jupyter nbconvert --execute`
  first to confirm `fig.show()` embeds as a self-contained `application/vnd.plotly.v1+json`
  Jupyter output (no CDN fetch needed to render, unlike `include_plotlyjs="cdn"` static HTML
  exports) before wiring it into the real notebook. Then ran count_state.ipynb end-to-end
  (25 cells total now, no errors) and confirmed both figures' embedded JSON: 35 traces
  (17 edges + 12 pies + 6 legend entries) for the immediate version, 34 (no "Continues" swatch)
  for resulting, pie domains ordered top-to-bottom matching the count-progression rows, same
  as the deleted HTML version. Cannot visually screenshot a rendered Plotly widget in this
  session either, same caveat as the HTML version before it.

# 9/8/2026: notebooks/fastball_location.ipynb, purpose.md's workflow item 4 (fastball cut, out of order)
- Built `fastball_location.ipynb`, testing whether pitchers locate fastballs (FF/SI/FC --
  Baseball Savant's public "Fastball" umbrella) more aggressively in counts that favor them
  and more centrally in counts that favor the hitter. This is purpose.md's item 4, done
  ahead of item 3 (pitch-type-mix Plinko, still not started) at the user's request --
  answering "does location shift with count" doesn't need the pitch-type-mix analysis to
  exist first, only a later "how much of this is pitch-type mix vs. location" refinement
  would. Recorded here as a deliberate reordering, not an oversight.
- Reused rather than reimplemented: `pitching_plus/scripts/location.py`'s `_build_features`
  (zone-relative `plate_z_rel`, handedness-normalized `plate_x_armside`, `re288_state`),
  `add_location_plus`, and `load_cached_models`; `bestpitch.py`'s `CANDIDATE_ZONE_CODES`
  convention (zone 5 = heart, 2/4/6/8 = edge, 1/3/7/9 = corner, 11-14/other = chase) and its
  `_predict_at_point` held-situation-fixed scoring pattern; `count_state.ipynb`'s
  `classify_result`, `AB_KEY`, and resulting-vs-immediate distinction.
- Count-leverage ranking (Stage 0): during planning, the user flagged that 2-1's leverage
  isn't obvious from a naive ball/strike read (2-2's strikeout rate is much higher than
  3-1's, despite both being one pitch from 2-1), so bucket boundaries needed to come from
  data, not an assumed label. First attempt: sum each pitch's actual
  `delta_pitcher_run_exp` forward from a count's first-arrival pitch to the end of the
  at-bat (a per-count run-expectancy table, telescoping to `RE(end of AB) - RE(just before
  this count)` by construction). Mathematically sound but produced a result that inverts
  known sabermetric literature -- 3-0 ranked as the single MOST pitcher-favorable count,
  ahead of 0-2. Root cause, confirmed by decomposing the 3-0 population: RE288 deltas scale
  up with count depth (a resolved full-count strikeout/out swings run expectancy far more
  than an equivalent early-count one), so the ~42% of 3-0 arrivals that get resolved by a
  ball in play or a deep-count strikeout contribute large positive outliers that outweigh
  the more numerous, but individually smaller-magnitude, walks in a straight mean (median
  was correctly negative at -0.103; only the mean was inverted). Real effect, not a bug, but
  too counterintuitive to use as the basis for hexbin bucket boundaries here.
- Replaced with a much simpler, already-validated metric: resulting BB% minus resulting K%
  per count (pure reuse of `count_state.ipynb`'s own `classify_result`/dedup logic, just a
  different aggregation of numbers already proven correct there -- cross-checked
  `n_abs_through` at 0-0 (912,449) and 0-1 (459,559) against count_state.ipynb's own
  numbers as an in-notebook assert). This ranking matches the standard sabermetric count-
  value ordering closely (3-0 most hitter-favorable, 0-2 most pitcher-favorable). 2-1 lands
  at rank 8 of 12 -- essentially tied with 1-0 (-3.16% vs -3.36%) right on the boundary
  between the "Even" and "Hitter-ahead" terciles, confirming it's a pivot rather than a
  clean "even" count.
- Buckets (equal terciles by leverage rank, 4 counts each): Pitcher-ahead = 0-2/1-2/0-1/2-2,
  Even = 1-1/0-0/1-0/2-1, Hitter-ahead = 3-2/2-0/3-1/3-0.
- Stage A (hexbin density): matplotlib, not Plotly, for this one chart specifically --
  true hexagonal binning is a matplotlib feature Plotly has no native equivalent for (same
  reasoning `count_state.ipynb`'s original bar chart used matplotlib). Paired with a numeric
  dispersion table (mean |plate_x_armside|, mean distance from mid-zone height, % within a
  +/-0.83ft heart-width band) per bucket, since a chart's shape isn't something to assert on
  without seeing it rendered. Pitcher-ahead fastballs are the most spread out on both axes
  (0.66ft mean |x|, 0.44 mean z-distance, 68.3% in the heart-width band); hitter-ahead are
  the tightest cluster (0.57ft, 0.35, 75.2%).
- Stage B (zone-code %): reusing `bestpitch.py`'s zone grouping directly. Found a real
  nuance on the "getting cute vs. just get it over" framing: Heart% and Corner% both roughly
  *double* from pitcher-ahead to hitter-ahead counts (5.2%/15.8% at 0-2 vs. 10.6%/21.4% at
  3-0) -- pitchers aren't specifically aiming for dead-center over corners under pressure.
  The real swing is Chase/Ball% collapsing (60.5% at 0-2 down to 37.4% at 3-0): two-strike
  counts are where pitchers can afford to leave the zone; hitter's counts are where they
  compress toward anywhere in-or-near it, heart or corner alike.
- Stage C ("meatball gap"): for every real fastball, score a counterfactual dead-center
  pitch (`plate_x_armside=0, plate_z_rel=0.5`) through the same cached per-pitch-type model,
  same count/situation held fixed (same pattern as `bestpitch.py`'s `_predict_at_point`,
  adapted to one fixed reference point instead of a full candidate search -- avoids that
  module's documented multi-hour cost at full scale). Gap = actual `location_run_value` -
  meatball prediction. Positive in 0-2 (+0.029) and 1-2 (+0.024): real locations beat a
  hypothetical grooved pitch there. Increasingly negative toward 3-1 (-0.060) and 3-0
  (-0.061): real execution falls short of even a dead-center pitch's expected value once a
  walk becomes costly enough that center-cut itself scores well in the model.
- 2-1 fork: split at-bats reaching 2-1 by whether the next pitch goes to 3-1 or 2-2 (the
  only two live continuations; balls/strikes can't revisit 2-1 within an at-bat). Sharp
  reversal: the 3-1 branch resolves 44.7% walk / 14.0% strikeout; the 2-2 branch resolves
  38.3% strikeout / 13.3% walk. Concrete confirmation of the Stage 0 boundary placement.
- Ran end-to-end via `jupyter nbconvert --execute --inplace` on the real 2021-2025 data
  (3,565,743 pitches; Location+ scoring reused the existing `pitching_plus/models/` cache,
  no retrain needed), confirmed zero error outputs across all cells, then read back every
  cell's actual output before writing the two data-dependent discussion markdown cells
  (Stage A dispersion, 2-1 fork) so their numbers are the real executed values, not
  estimates written ahead of the run.

# 9/22/2026: scripts/app.py -- an interactive version of fastball_location.ipynb

- User asked for a way to toggle pitch type and filter to one pitcher on top of
  `fastball_location.ipynb`'s fastball-only, leaguewide charts, framed as the app idea
  from the portfolio review. Also asked for the app to load its own sample via pybaseball
  rather than pushing `data/MLB_2021-2025.csv` into the repo, since a small live pull is
  faster for anyone trying the app for the first time and the data's already public.
- Scope cut before writing anything: only Stage A (location by count-leverage bucket) and
  Stage B (zone-code share by count) carry over. Stage C (the Location+ "meatball gap")
  needs `pitching_plus`'s cached models, which need a full run over the 3.5M-row dataset to
  build first -- too heavy a dependency for something meant to load in a couple of minutes.
  Not built; purpose.md's Tooling section names it as deferred rather than leaving it
  unmentioned.
- Four new modules under `scripts/`, mirroring `pitching_plus/scripts`'s one-module-one-job
  convention rather than one large app file:
    - `count_leverage.py`: the (balls, strikes) -> Pitcher-ahead/Even/Hitter-ahead mapping
      and the Heart/Edge/Corner/Chase-Ball zone grouping. Both are fixed lookups ported
      from `fastball_location.ipynb`'s Stage 0/B results (full 2021-2025 data), not
      recomputed from whatever the app happens to load -- a week's live sample is a few
      thousand pitches, nowhere near enough to re-rank the near-tied counts (1-0 and 2-1
      differ by 0.002 in bb_minus_k) without the bucket assignment flipping week to week.
    - `sample_data.py`: `load_sample(start_date, end_date)`, a thin wrapper on
      `pybaseball.statcast()` trimmed to the columns the app needs and sorted the same way
      this repo's other raw-data loads are. `pybaseball.cache.enable()` at import so a
      second run of the same date range skips the network entirely.
    - `location_view.py`: the actual data engineering and figure-building
      (`engineer_pitches`, `location_figure`, `zone_share_figure`), reusing
      `pitching_plus/scripts/location.py`'s `_build_features` for the zone-relative,
      arm-side-adjusted coordinates instead of re-deriving them. No Streamlit import here,
      so it's unit-testable without a running app and app.py stays UI-only.
    - `app.py`: the Streamlit page itself -- sidebar date range and pitch-type/pitcher
      filters, `st.cache_data` around the sample load so changing a filter doesn't re-fetch,
      then the two figures from `location_view.py`. Cross-project import of
      `pitching_plus/scripts/location.py` uses the same `sys.path.insert` pattern the
      notebooks already use, since `streamlit run` executes this file directly rather than
      as part of a package.
    - `location_view.py`/`app.py` both try a package-relative import first
      (`pitching_plus.scripts.location`, `.count_leverage`) and fall back to a bare one,
      matching `location.py`'s own dual-import convention -- the first branch resolves
      under pytest (repo root on sys.path via `slippery_slope/__init__.py`, added this
      session to mirror `pitching_plus`'s package layout), the second under `streamlit run`.
- Added `slippery_slope/__init__.py`, `scripts/__init__.py`, and `tests/__init__.py` --
  this project had docs/ and notebooks/ only until now, even though the 9/2 entry described
  scaffolding all four subfolders when the project was created. First real content in
  scripts/ and tests/.
- 13 new tests (`test_count_leverage.py`, `test_sample_data.py`, `test_location_view.py`):
  the count/zone lookups, `load_sample`'s column-trim/sort/error paths (pybaseball's
  `statcast` monkeypatched, no real network call in the suite), and `engineer_pitches`
  plus both figure builders against a small synthetic sample. `pitching_plus/tests` (36
  tests) reconfirmed green in the same environment afterward -- run once against a
  mismatched, unpinned global Python install first (10 failures, all environment noise:
  wrong pandas/numpy/scikit-learn versions), then again against `.venv`'s pinned versions,
  where all 36 pass. Worth a note here since it looked like a real regression at first.
- Real-data checks, not just synthetic ones: `load_sample` against the actual default
  range (2025-06-02 to 2025-06-08) returned 27,228 pitches, 421 pitchers, 94 games in about
  20 seconds. `engineer_pitches` kept 27,004 of those; the fastball subset (FF/SI/FC) was
  14,812 pitches across 417 pitchers. Spot-checked the pitcher filter on the single busiest
  fastball arm in that window (Drew Rasmussen, 143 pitches: 75 Even/52 Pitcher-ahead/16
  Hitter-ahead) -- enough per bucket for the hexbin to show something, not just noise.
  Ran the full app through `streamlit.testing.v1.AppTest` (executes the real script, not a
  browser simulation) against this same live sample, including simulated pitch-type and
  pitcher-filter interactions: zero exceptions in every case. Caught one real issue this
  way -- `st.plotly_chart`'s `use_container_width` argument is deprecated as of this
  streamlit version and slated for removal; switched to `width="stretch"` before it became
  a problem.
- Added `streamlit==1.64.0` to requirements.txt (pinned, matching every other dependency
  here) and a Setup/Slippery-Slope note in README.md on how to run the app.

# 9/22/2026 (cont'd): comparison modes, season-expanded pitcher samples, better titling

User feedback on the first version, in order: a bulk Fastball/Breaking/Offspeed selector
instead of hand-picking pitch types one at a time; a bigger sample-size option, since the
default week was thin; a per-pitch-type comparison row instead of only pooling selected
types together; a pitcher-vs-league-or-pitcher comparison instead of a plain single-pitcher
filter; clearer axis/chart titling; and, once a pitcher is selected, coarser charts plus a
bigger sample for that pitcher specifically, with an option to narrow to their own arsenal.
Landed as one combined pass rather than five separate ones, since the later asks changed
how the earlier ones needed to be built (see below).

- `pitch_groups.py` (new): `PITCH_TYPE_CATEGORY`, Baseball Savant's own Fastball/Breaking/
  Offspeed grouping, same fixed-lookup reasoning as `count_leverage.py`'s bucket map --
  these are established pitch-classification groups, not something to infer from a sample.
  Wired into a "Bulk select by category" multiselect in app.py whose `on_change` callback
  overwrites the pitch-type multiselect's `session_state` value; the two stay linked one
  direction only (category -> types), so a manual edit to the pitch-type list afterward
  doesn't get overwritten until the category selector is touched again.
- Sample-size presets: a "Sample window" selectbox (1 day / 1 week / ~100K pitches / custom)
  replacing the old two bare date pickers, computing the actual start date from the end date
  and a day-count. The ~100K option's day-count (26) was derived from the measured rate in
  the last entry (27,228 pitches / 7 days = ~3,890/day) and confirmed live: the 26-day range
  returned 101,217 pitches. Labels are explicit that these are ballpark, not guaranteed,
  since the real count depends on how many games are actually on the slate.
- Pitch-type comparison: `location_view.build_comparison_rows` splits the selected pitch
  types into either one pooled row or one row per type (a "Compare pitch types side by
  side" checkbox, shown only once 2+ types are selected), and `location_figure` was
  rewritten to take a list of rows instead of one dataframe, rendering an N-row x 3-bucket
  grid. Verified live: FF+SL together came back as 6 axes (2 rows x 3 buckets).
- Pitcher-vs-pitcher comparison: `build_comparison_rows` no longer filters an engineered
  frame by player name internally -- it now takes `background_source`/`overlay_source` as
  two already-engineered frames the caller resolves, so the background can be the league
  window, a comparison pitcher's own data, or (see below) a season pull, without the
  function needing to know which. app.py adds a "Compare against: League average / Another
  pitcher" radio once a pitcher is selected, plus a "Comparison pitcher" selectbox; the
  overlay checkbox from the last entry is unchanged in behavior, just resolved against
  whichever background was picked instead of always the league.
- Season-expanded pitcher samples: `sample_data.load_pitcher_season(pitcher_id, season)`
  wraps `pybaseball.statcast_pitcher()`, which filters to one pitcher server-side rather
  than pulling the whole league and filtering locally -- confirmed live, a full 2025 season
  for one pitcher (3,443 pitches) loaded in 7.1s, versus load_sample's per-day cost for the
  whole league. Whenever a specific pitcher is selected, app.py fetches this once (cached
  by pitcher_id+season) and uses it for that pitcher's overlay, their zone-share chart, and
  their own pitch-type list -- the small league window stays the background/comparison
  source unless the user explicitly picks another pitcher to compare against (who then also
  gets their own season pull). `sample_data.py`'s `_trim_and_sort` helper was pulled out of
  `load_sample` so both loaders share the same column-trim/sort/validation logic instead of
  duplicating it.
- Adaptive hexbin granularity: fixed `gridsize=25` looked like real density over a league
  background (thousands of points per bucket) and like meaningless speckle over a single
  pitcher's own pitches (tens to a couple hundred per bucket, even across a full season).
  `location_view._hexbin_gridsize(n)` scales `gridsize` with `sqrt(n)`, floored at 6 and
  capped at 25, so a hexagon covers roughly the same number of points either way.
- "Limit to `{pitcher}`'s own pitch types" button: on click, sets the pitch-type
  multiselect to that pitcher's actual thrown types (from their season pull, so it's their
  real arsenal, not just whatever the small league window happened to catch). Needed the
  pitcher selectbox to stop depending on the currently selected pitch types (it's now built
  from every pitcher in the whole loaded sample, not just those throwing the current
  selection) -- otherwise picking a pitcher to find their own arsenal would have required
  already having picked pitch types they throw, a circular requirement. The pitch-type
  multiselect's own option list is now the union of the small window's types and the
  focused pitcher's season types, so a type their season shows but the window missed is
  still selectable.
- Better titling and explanations: `zone_share_figure`'s x-axis now says "Count
  (balls-strikes), sorted pitcher-ahead to hitter-ahead -- not alphabetical" instead of
  leaving the ordering unexplained, and its title changed from "Zone-code share by count"
  to "Where in the zone pitches go, by count." The location hexbin's axis labels spell out
  what they mean (arm-side sign convention, zone-relative height) instead of using the raw
  column names. Added a "What am I looking at?" expander to the app itself covering the
  count-leverage buckets (built from `count_leverage.counts_by_bucket()`, a new function
  that lists each bucket's counts from `COUNT_BUCKET` directly rather than a separately
  hand-typed string, so the two can't drift apart), the zone groups, the location axes, the
  comparison-mode background/overlay convention, and the season-expansion behavior.
- 8 new tests (`test_pitch_groups.py`, `counts_by_bucket` coverage in
  `test_count_leverage.py`, `build_comparison_rows`/`location_figure`/`_hexbin_gridsize`
  coverage and `load_pitcher_season` coverage in the existing test files) -- 27 total, all
  passing.
- Verified live end-to-end via `streamlit.testing.v1.AppTest`, not just unit tests: category
  bulk-select actually repopulating the pitch-type list, the pitcher selectbox and its
  season-pull spinner, the "limit to own types" button, switching to "Another pitcher," and
  the ~100K sample window, all with zero exceptions. Built the full combined case directly
  (not through AppTest) to inspect the actual figure: Drew Rasmussen's complete arsenal
  (CH/CU/FC/FF/SI/SL/ST) as 7 comparison rows, background = Paul Skenes's season, overlay =
  Rasmussen's season. The comparison surfaced something concrete on the first try: Skenes
  threw zero cutters in pitcher-ahead counts all season (n=0 in that panel) against
  Rasmussen's 295 -- a sign the feature does what it's for, not just that it runs without
  error.

# 9/22/2026 (cont'd): overlay opacity slider

User feedback: the overlay points (open circles, no fill, just a colored edge) read as too
heavy once there's more than a handful of them layered over the hexbin. `location_figure`
takes a new `overlay_alpha` argument (default 1.0, unchanged for any other caller) applied
to the overlay scatter's `alpha`; app.py adds a slider (0.1-1.0, default 0.6) next to the
overlay checkbox, only shown when the overlay is actually on. Verified the slider actually
changes the rendered collection's alpha via a new location_figure test, and end-to-end via
`streamlit.testing.v1.AppTest` against live data (select a pitcher, move the slider, zero
exceptions). 28 tests passing.

# 9/22/2026 (cont'd): dashed comparison lines on the zone-share chart

Zone-share chart previously only ever showed one side (the focused pitcher's own zone mix,
or the league's if none was selected) -- the comparison feature added earlier only reached
the location hexbin. User asked for the comparison group's lines to show up there too: same
color per zone group, dashed, with an on/off toggle.

- `location_view.zone_share_figure` takes new `comparison`/`comparison_label` arguments. The
  comparison group's own zone-share-by-count is computed with the same logic as the main
  series (pulled out into `_zone_share_by_count` so it isn't duplicated), then drawn as a
  dashed line in that category's existing color rather than a second color scale -- four
  colors already carry the zone-group meaning, so "who" needed a different visual channel,
  not a fifth-through-eighth color. Two legend-only key traces (`x=[None]`, plain black,
  solid vs. dashed) name which side is which, since the real dashed traces themselves are
  `showlegend=False` to avoid doubling the four-item legend into eight near-duplicate
  entries. Skips the comparison entirely (no dashed traces, no key) when it's empty rather
  than dividing by zero.
- app.py reuses `background_in_scope`/`background_desc`, the same two values already
  computed for the location hexbin's background -- whatever's being compared against there
  (league average or a chosen pitcher) is the same thing dashed in here, not a separate
  comparison target with its own selection.
- New checkbox, "Show comparison lines on the rate chart" (default on, next to the other
  pitcher-comparison controls), since a busy 8-trace chart isn't always wanted.
- 3 new tests (dashed traces get the right color/dash/showlegend, the key traces are named
  correctly, an empty comparison is skipped cleanly) -- 30 total, all passing. Verified live:
  built the real figure for Rasmussen vs. league average directly and printed every trace's
  name/dash/color/showlegend to confirm the actual rendered output, and exercised the
  checkbox toggle through `streamlit.testing.v1.AppTest` against live data with zero
  exceptions.

# 9/22/2026 (cont'd): "First Last" display names, still sorted by last name

Statcast's player_name is "Last, First" -- which is exactly why sorting the raw strings
alphabetically already sorts by last name, but reads worse in a dropdown or a chart title
than "First Last" does. User asked for the display fixed without losing that sort order.

- `names.py` (new): `to_first_last(player_name)`, a plain `", "`-split. Suffixes (Jr., II,
  III, IV) stay attached to the last name since Statcast already puts them before the comma
  ("Leiter Jr., Mark" -> "Mark Leiter Jr."); a name with no comma passes through unchanged
  (covers the "All pitchers" sentinel, which isn't a real player_name).
- Every dataframe filter, dict lookup, and `_load_pitcher_engineered` call in app.py still
  keys on the raw "Last, First" string -- nothing about the underlying data or matching
  logic changed. Only the text actually shown to the user was touched: the two pitcher
  selectboxes now pass `format_func=to_first_last` (Streamlit's own built-in for exactly
  this -- display one thing, keep the widget's real value as the option string, so the
  options list itself, and therefore the sort order, is untouched), and every other
  display site (checkbox/button labels, chart titles, the overlay's legend label, the
  zone-share chart's title and comparison key, the "no pitches" info message, and the
  season-pull spinner/warning inside `_load_pitcher_engineered`) got an explicit
  `to_first_last(...)` call.
- One easy-to-miss spot: `build_comparison_rows`'s `overlay_label` argument flows straight
  into `location_figure`'s legend text, so it needed converting at the call site in app.py
  even though nothing about `build_comparison_rows` or `location_figure` itself changed --
  those two stay name-format-agnostic, just displaying whatever label string they're given.
- 4 new tests (`test_names.py`) -- 34 total, all passing. Verified live via `AppTest` and a
  direct figure build: dropdown options came back as `['All pitchers', 'Andrew Abbott',
  'Mick Abel', 'Bryan Abreu', 'Jason Adam', 'Zach Agnos']` (still last-name order: Abbott <
  Abel < Abreu < Adam < Agnos), and a real location/zone-share figure for Drew Rasmussen
  showed "Drew Rasmussen" in the suptitle, legend, chart title, and comparison key --
  nowhere left showing "Rasmussen, Drew".

# 9/22/2026 (cont'd): opaque legend background on the zone-share chart

User feedback: the plotly legend was hard to read wherever it landed over a line. Added
`legend=dict(bgcolor=..., bordercolor=..., borderwidth=1)` to `zone_share_figure`'s layout
(translucent white, light gray border) -- unconditional, not tied to the comparison feature,
since the same legibility problem exists with just the four base zone-group lines. 1 new
test (legend has a non-null bgcolor); 35 total, all passing.

# 9/22/2026 (cont'd): legend text was still white, plus "League Average" capitalization

User follow-up: the new legend background showed up, but its text stayed white -- unreadable
against the new white box. Root cause: `st.plotly_chart` defaults to `theme="streamlit"`,
which overlays Streamlit's own (here, dark-mode) font color on top of the figure without
touching an already-set background, since bgcolor and font color are set independently.
- Two fixes, not one, since either alone leaves a gap: `zone_share_figure`'s legend now sets
  an explicit `font=dict(color="#111111")`, guaranteeing readable text regardless of theme.
  Separately, its `st.plotly_chart` call passes `theme=None`, so Streamlit stops overlaying
  its own theme on this chart at all and the figure's own `plotly_white` styling (already
  chosen deliberately) governs everywhere, not just the one spot that was visibly broken.
- Also asked for "League Average" (title case) in the legend. `LEAGUE_LABEL` was already
  "League average"; fixed the constant itself to "League Average" and dropped the `.lower()`
  call that had been building `background_desc` from it, so the same capitalization now
  shows consistently in the "Compare against" radio, the hexbin's title_suffix, and the
  zone-share legend key, rather than three different casings for the same word.
- Verified directly against a real figure: legend bgcolor/font color both confirmed set, and
  the key trace names read `['Drew Rasmussen', 'League Average']`. 35 tests still passing
  (no behavior this touches was under test beyond the existing legend-bgcolor check).

# 9/22/2026 (cont'd): no gridlines on the zone-share chart

User asked for the gridlines off. `showgrid=False` on both axes in `zone_share_figure`'s
layout (folded `xaxis_title`/`yaxis_title`/`yaxis_tickformat` into explicit `xaxis=dict(...)`/
`yaxis=dict(...)` blocks to add it cleanly rather than mixing shorthand and dict forms). 1
new test; 36 total, all passing.
