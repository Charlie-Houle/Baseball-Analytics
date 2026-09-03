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
