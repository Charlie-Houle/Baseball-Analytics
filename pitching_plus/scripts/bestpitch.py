"""
bestPitch+: per docs/purpose.md, "given pitcher arsenal ... calculate
hypothetical Pitching+ score (bestPitch)" and compare it to the pitch actually
thrown: `bestPitch - Pitching+ = bestPitch+`, a decision-quality metric
(was this the right pitch/location "idea"?), not another "stuff" metric.

For each in-scope pitch, holds the real game situation fixed (count, outs,
base state, batter/pitcher handedness, everything but pitch type and
location) and searches every combination of:
  - a candidate pitch type from that pitcher's own RELIABLE arsenal that
    season (stuff_plus_reliable == True), never a pitch type they don't
    actually throw. Since Pitching+ is retooled 9/4/2026 into a joint model
    trained on physics + location together (docs/dev_log.md), a candidate's
    "stuff" contribution is now that pitch type's own average physics
    (STUFF_FEATURES) for this pitcher-season, not a precomputed scalar --
    see _arsenal_physics_avg.
  - a candidate zone, using Statcast's own `zone` 1-9 (in-strike) plus 11-14
    (chase/waste corners just outside the zone: real "up-and-in chase"
    type locations, not just heart-of-the-zone spots) as purpose.md's "zone,
    not pinpoint", represented by that zone's dataset-wide average
    (plate_x, plate_z_rel), not a synthetic point.

Each candidate's location quality is the model's prediction *averaged over a
small "target" area* around that reference point (a 5-point sample within
TARGET_RADIUS_FT, not a single pinpoint query): purpose.md's "zone, not
pinpoint" taken literally. No pitcher hits an exact spot, and scoring only the
single reference point let bestPitch+'s max-over-candidates search exploit
any local spike in the fitted regression surface, substantially inflating the
result (see docs/dev_log.md). The candidate's fixed physics (that pitch
type's own arsenal-average STUFF_FEATURES) and varying location together form
one joint feature row scored directly by pitching.py's own trained model for
that pitch type, then calibrated to the identical pitch-level 100+ scale as
real pitches (pitching._calibrate_raw_value). The max over all candidates is
`best_pitching_plus`.

`best_pitching_plus` is scored with target-averaging (smoothed, area-based);
the real `pitch_pitching_plus` is a pinpoint value (the pitch's own exact
location, no averaging). Comparing a smoothed quantity against a pinpoint one
directly is an apples-to-oranges comparison: a specific real pitch can
easily beat a smoothed area estimate just from its own single-point noise,
which showed up empirically as bestPitch+ losing its expected sign (see
docs/dev_log.md). Fixed by scoring the actual pitch's own (pitch type, zone)
combination through the *identical* target-averaging machinery
(`_actual_smoothed_pitching_plus`) before comparing.

The counterfactual search itself only considers candidate pitch types from
the pitcher's own RELIABLE arsenal that season, so a pitch type thrown too
rarely to be reliable never enters the search -- including for scoring that
type's own pitches. `add_bestpitch_plus` explicitly folds the actual
(smoothed) score into the running max as a floor (`np.fmax`) after the
search, so the actual combination is always effectively a candidate even
when its own type falls outside the reliable-arsenal filter, restoring
`best_pitching_plus >= actual (smoothed)` as an (almost) guaranteed property
for every row a search ran for at all (see docs/dev_log.md's 9/2 entry).

Public entry point is `add_bestpitch_plus`, which takes a raw Statcast
dataframe and returns it with:
  - `best_pitching_plus`   : best achievable Pitching+ (100+, target-averaged)
                             from this pitcher's own arsenal/zone options,
                             same situation as the actual pitch.
  - `pitch_bestpitch_plus` : best_pitching_plus minus the actual pitch's own
                             (pitch type, zone) scored the same
                             target-averaged way; positive or close to 0 for
                             nearly every pitch, since the actual
                             combination is always one of the candidates
                             considered (see notebooks/bestpitch.ipynb).

Calls add_stuff_plus/add_location_plus/add_pitching_plus itself if their
columns aren't already present. Requires a cached Pitching+ model (via
pitching.load_cached_models); run add_pitching_plus at least once first.
"""

import time

import numpy as np
import pandas as pd

try:
    from . import pitching as pitching_mod
    from . import stuff as stuff_mod
    from .location import (
        BASE_STATE_COLS, DEFAULT_MODELS_DIR, LOCATION_FEATURES,
        _build_features,
    )
    from .pitching import JOINT_FEATURES, add_pitching_plus
    from .stuff import JUNK_PITCH_TYPES, PITCH_TYPE_COL, PITCHER_COL, SEASON_COL, STUFF_FEATURES
except ImportError:
    import pitching as pitching_mod
    import stuff as stuff_mod
    from location import (
        BASE_STATE_COLS, DEFAULT_MODELS_DIR, LOCATION_FEATURES,
        _build_features,
    )
    from pitching import JOINT_FEATURES, add_pitching_plus
    from stuff import JUNK_PITCH_TYPES, PITCH_TYPE_COL, PITCHER_COL, SEASON_COL, STUFF_FEATURES

# ============================================================
# CONFIGURATION
# ============================================================

ZONE_COL = "zone"
# Statcast's own zone codes: 1-9 are the in-strike-zone regions, 11-14 are the
# four "chase/waste" corners just outside the zone (e.g. up-and-in). Both are
# real, meaningful pitch-location "ideas" a pitcher might aim for.
CANDIDATE_ZONE_CODES = list(range(1, 10)) + [11, 12, 13, 14]

# "Target" radius for averaging a candidate zone's location prediction over a
# small area rather than a single pinpoint. purpose.md's "zone, not
# pinpoint," and no pitcher hits an exact spot. ~3 baseballs across (a
# baseball is ~2.9 inches in diameter).
BALL_DIAMETER_IN = 2.9
TARGET_DIAMETER_BALLS = 3.0
TARGET_RADIUS_FT = (TARGET_DIAMETER_BALLS * BALL_DIAMETER_IN / 2) / 12

# `zone_height` (sz_top - sz_bot) is Statcast's per-PITCH strike-zone
# estimate, not a fixed per-batter constant -- it's a near-continuous float
# (on the real dataset, 1.73M distinct values out of 3.57M rows), so keying
# the +z/-z dedup on its raw value collapses almost nothing (unlike
# center_key, which cleanly collapses to a bounded number of values).
# Rounding it to the nearest ZONE_HEIGHT_DEDUP_ROUND_FT before using it as a
# dedup key groups pitches with near-identical strike zones into the same
# predict() call. Error this introduces: the +z/-z offset is
# TARGET_RADIUS_FT / zone_height, so d(offset)/d(zone_height) =
# -TARGET_RADIUS_FT / zone_height^2 -- at a typical zone_height of ~1.8ft,
# that's about -0.11 ft of offset error per ft of zone_height error, so a
# 0.02ft (~1/4 inch) rounding step induces at most ~0.002ft (~0.03 inch) of
# error in the queried plate_z_rel point, on a target radius that's already
# ~4.35 inches (TARGET_RADIUS_FT) -- roughly 0.5% of the target radius
# itself, well inside the "not a pinpoint, a small target area"
# approximation the design already accepts.
ZONE_HEIGHT_DEDUP_ROUND_FT = 0.02

# LOCATION_FEATURES minus the three location-derived columns that get
# overwritten per candidate point; everything else a row already has.
NON_LOCATION_FEATURES = [c for c in LOCATION_FEATURES if c not in ("plate_x", "plate_z_rel", "plate_x_armside")]

# Column positions of every JOINT_FEATURES entry, so the batched search path
# (_build_base_array / _batched_target_averaged_pitching_run_value) can
# overwrite the 3 location-derived columns in a plain numpy array without
# going through pandas per candidate.
_JOINT_COL_INDEX = {col: i for i, col in enumerate(JOINT_FEATURES)}
_PLATE_X_IDX = _JOINT_COL_INDEX["plate_x"]
_PLATE_Z_REL_IDX = _JOINT_COL_INDEX["plate_z_rel"]
_PLATE_X_ARMSIDE_IDX = _JOINT_COL_INDEX["plate_x_armside"]
_STUFF_FEATURE_SET = set(STUFF_FEATURES)

# A candidate pitch type's own arsenal-average physics, distinctly named so
# they don't collide with a row's own real STUFF_FEATURES columns when both
# are present on the same working frame (see _arsenal_physics_avg).
STUFF_FEATURE_CAND_COLS = [f"cand_{f}" for f in STUFF_FEATURES]

# NOTE: BASE_STATE_COLS (on_1b/on_2b/on_3b) is deliberately NOT folded into
# REQUIRED_COLS. NaN there means "base empty" (a real game state), not
# missing data, exactly as in location.py. Must be present as columns, but
# not dropna-gated. REQUIRED_COLS now needs pitching.py's own union (stuff's
# physics columns + location's columns), not just location's, since scoring
# a counterfactual candidate needs a full joint feature row.
REQUIRED_COLS = sorted(set(pitching_mod.REQUIRED_COLS) | {ZONE_COL})


# ============================================================
# ZONE REFERENCE LOCATIONS
# ============================================================

def _zone_reference(engineered):
    """
    Dataset-wide average (plate_x, plate_z_rel) per candidate `zone` code:
    purpose.md's "zone, not pinpoint" representative location, pooled across
    all pitch types/seasons rather than per-pitch-type (a pitch type's own
    typical spot within a zone is a targeting choice already captured by
    scoring it through that pitch type's own location model).
    """

    ref = (
        engineered[engineered[ZONE_COL].isin(CANDIDATE_ZONE_CODES)]
        .groupby(ZONE_COL)[["plate_x", "plate_z_rel"]]
        .mean()
    )
    return ref


# ============================================================
# ARSENAL: WHICH PITCH TYPES CAN THIS PITCHER ACTUALLY THROW,
# AND WHAT DOES THEIR PHYSICS LOOK LIKE
# ============================================================

def _arsenal_physics_avg(df):
    """
    One row per (pitcher, pitch_type, season), with that pitcher's own
    average STUFF_FEATURES for pitch types that are a reliable part of their
    arsenal that season (stuff_plus_reliable == True) -- the fixed physics
    half of a counterfactual joint-feature row. Replaces the old
    _arsenal_wide (which held a single cand_stuff_<type> scalar): the joint
    model needs the full physics vector, not a precomputed 100+ score, and
    this table doubles as the arsenal-membership gate, exactly like
    cand_stuff_<type> being NaN used to (a pitch type absent from this table
    for a given pitcher-season is automatically excluded from the search).
    Requires `df` to already have STUFF_FEATURES engineered
    (stuff_mod._build_v1_features already run on it) and stuff_plus_reliable
    attached.
    """

    reliable = df[df["stuff_plus_reliable"].fillna(False)]
    avg = (
        reliable
        .groupby([PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], observed=True)[STUFF_FEATURES]
        .mean()
        .reset_index()
        .rename(columns=dict(zip(STUFF_FEATURES, STUFF_FEATURE_CAND_COLS)))
    )
    return avg


# ============================================================
# COUNTERFACTUAL SEARCH: BEST (PITCH TYPE, ZONE) PER PITCH
# ============================================================

def _predict_at_point(base, model, physics_cols, non_location_features, plate_x, plate_z_rel):
    X = base[non_location_features].copy()
    for feat, src_col in zip(STUFF_FEATURES, physics_cols):
        X[feat] = base[src_col]
    X["plate_x"] = plate_x
    X["plate_z_rel"] = plate_z_rel
    # Arm-side sign is keyed on the PITCHER's throwing hand (p_throws), not
    # batter stand -- see location.py's plate_x_armside comment for the
    # derivation. Fixed 9/2 (docs/dev_log.md); was previously keyed on stand.
    X["plate_x_armside"] = np.where(base["p_throws"] == "R", -plate_x, plate_x)
    return model.predict(X[JOINT_FEATURES].to_numpy())


def _build_base_array(base, physics_cols, dedup_by_pitcher_season):
    """
    Prebuilds `base`'s non-location features as one numpy array, in
    JOINT_FEATURES column order (the 3 location columns get overwritten per
    candidate point by _batched_target_averaged_pitching_run_value), plus
    the per-row quantities every candidate point needs: p_throws == 'R'
    (arm-side sign), strike-zone height (for converting TARGET_RADIUS_FT to
    plate_z_rel units), and `center_key`.

    `physics_cols` selects which columns to read STUFF_FEATURES values from
    -- the row's own real STUFF_FEATURES columns when scoring the actual
    pitch (_actual_smoothed_pitching_plus, physics_cols=STUFF_FEATURES), or
    a candidate pitch type's cand_<feature> arsenal-average columns when
    scoring a counterfactual pitch-type candidate
    (_search_best_pitching_plus, physics_cols=STUFF_FEATURE_CAND_COLS).
    Either way, physics values are FIXED across every candidate zone for a
    given row -- only the 3 location columns vary.

    `center_key` identifies rows that are indistinguishable to the model for
    any candidate point that doesn't depend on strike-zone height (center,
    +x, -x -- 3 of the 5 target-averaging points). For the location-only
    model this used to be a genuine bijection over game-state + handedness
    alone (re288_state * stand_R * p_throws_R, <=1,152 distinct keys),
    because every non-location LOCATION_FEATURES column besides the 3
    varying ones is constant across pitchers for a fixed game-state. That
    stopped being true once physics (which varies by pitcher, and by
    pitch type/season for the candidate case) got folded into the same
    array: two rows sharing a game-state but different pitcher-seasons are
    NOT interchangeable anymore, and deduping them together would silently
    score one pitcher's candidate with another pitcher's physics -- a real
    bug found during this retool's planning (docs/dev_log.md).

    `dedup_by_pitcher_season=True` (the candidate-search case) extends the
    key with a compact (pitcher, season) id, so only rows sharing BOTH a
    game-state AND a pitcher-season collapse together -- still a real
    reduction versus the raw row count (many real pitches share both), just
    a smaller one than the location-only ceiling. `dedup_by_pitcher_season=
    False` (the actual-pitch case) uses a trivial per-row-unique key instead:
    physics there is each pitch's own real, continuously-varying
    measurement, not a per-pitcher-season average, so no two rows are
    genuinely interchangeable and there is no real dedup benefit to chase --
    `np.unique` on an already-unique key is a no-op split, not a bug,
    letting both callers share the identical downstream averaging code.
    """

    physics_col_by_feature = dict(zip(STUFF_FEATURES, physics_cols))

    array = np.zeros((len(base), len(JOINT_FEATURES)), dtype=float)
    for col, idx in _JOINT_COL_INDEX.items():
        if col in _STUFF_FEATURE_SET:
            array[:, idx] = base[physics_col_by_feature[col]].to_numpy(dtype=float)
        elif col not in ("plate_x", "plate_z_rel", "plate_x_armside"):
            array[:, idx] = base[col].to_numpy(dtype=float)

    p_throws_is_R = (base["p_throws"] == "R").to_numpy()
    zone_height = (base["sz_top"] - base["sz_bot"]).to_numpy()

    if dedup_by_pitcher_season:
        stand_is_R = (base["stand"] == "R").to_numpy()
        re288_state = base["re288_state"].to_numpy(dtype=np.int64)
        p_throws_R = base["p_throws_R"].to_numpy(dtype=np.int64)
        game_state_key = (re288_state * 2 + stand_is_R.astype(np.int64)) * 2 + p_throws_R
        composite = (
            base[PITCHER_COL].astype(str) + "_" + base[SEASON_COL].astype(str)
            + "_" + pd.Series(game_state_key, index=base.index).astype(str)
        )
        center_key, _ = pd.factorize(composite)
    else:
        center_key = np.arange(len(base), dtype=np.int64)

    return array, p_throws_is_R, zone_height, center_key


def _batched_target_averaged_pitching_run_value(model, array, p_throws_is_R, zone_height, center_key, center_x, center_z_rel):
    """
    Averages the model's raw pitching_run_value prediction over the same
    5-point "target" (center + N/S/E/W at TARGET_RADIUS_FT) as
    _predict_at_point, but deduplicated: rows that would get an identical
    (or, for +z/-z, a near-identical -- see below) model input for a given
    point are predicted once, not once each (see _build_base_array's
    docstring for `center_key`). Every row gets the same value it would from
    predicting on it directly (center/+x/-x), or a value within a tiny,
    bounded tolerance of it (+z/-z, see ZONE_HEIGHT_DEDUP_ROUND_FT),
    computed once and broadcast back via `inverse` instead of recomputed per
    duplicate row. No logic here changed when this was extended from a
    location-only model to the joint model -- `array` simply carries more
    columns now, and `model.predict()` returns the target's raw joint
    prediction directly instead of a location-only value needing a separate
    blend step afterward.

    Center/+x/-x (indices 0-2) don't depend on strike-zone height, so they
    dedupe exactly on `center_key` alone. +z/-z (indices 3-4) additionally
    depend on each row's own zone_height (a near-continuous per-pitch float
    -- see ZONE_HEIGHT_DEDUP_ROUND_FT's comment -- raw zone_height alone
    would barely dedupe at all), so they dedupe on (center_key, zone_height
    rounded to ZONE_HEIGHT_DEDUP_ROUND_FT) instead. The z-offset itself
    still uses each dedup group's representative row's exact (unrounded)
    zone_height, so the rounding only affects which rows share a predict()
    call, not the offset's own precision. The z-offset is converted from
    physical feet to plate_z_rel units per row, since strike zone height
    varies by batter.
    """

    uniq_center, first_center, inverse_center = np.unique(center_key, return_index=True, return_inverse=True)
    k = uniq_center.shape[0]
    center_rows = np.tile(array[first_center], (3, 1))
    center_p_throws_R = np.tile(p_throws_is_R[first_center], 3)

    center_x_stack = np.empty((3, k), dtype=float)
    center_x_stack[0] = center_x
    center_x_stack[1] = center_x + TARGET_RADIUS_FT
    center_x_stack[2] = center_x - TARGET_RADIUS_FT
    center_x_flat = center_x_stack.reshape(-1)
    center_z_flat = np.full(3 * k, center_z_rel, dtype=float)

    center_rows[:, _PLATE_X_IDX] = center_x_flat
    center_rows[:, _PLATE_Z_REL_IDX] = center_z_flat
    center_rows[:, _PLATE_X_ARMSIDE_IDX] = np.where(center_p_throws_R, -center_x_flat, center_x_flat)

    center_preds = model.predict(center_rows).reshape(3, k)[:, inverse_center]

    zone_height_bucket = np.round(zone_height / ZONE_HEIGHT_DEDUP_ROUND_FT)
    zh_key = np.column_stack([center_key, zone_height_bucket])
    uniq_zh, first_zh, inverse_zh = np.unique(zh_key, axis=0, return_index=True, return_inverse=True)
    m = uniq_zh.shape[0]
    ns_rows = np.tile(array[first_zh], (2, 1))
    ns_p_throws_R = np.tile(p_throws_is_R[first_zh], 2)
    ns_z_offset = TARGET_RADIUS_FT / zone_height[first_zh]

    ns_z_stack = np.empty((2, m), dtype=float)
    ns_z_stack[0] = center_z_rel + ns_z_offset
    ns_z_stack[1] = center_z_rel - ns_z_offset
    ns_z_flat = ns_z_stack.reshape(-1)
    ns_x_flat = np.full(2 * m, center_x, dtype=float)

    ns_rows[:, _PLATE_X_IDX] = ns_x_flat
    ns_rows[:, _PLATE_Z_REL_IDX] = ns_z_flat
    ns_rows[:, _PLATE_X_ARMSIDE_IDX] = np.where(ns_p_throws_R, -ns_x_flat, ns_x_flat)

    ns_preds = model.predict(ns_rows).reshape(2, m)[:, inverse_zh]

    predictions = np.concatenate([center_preds, ns_preds], axis=0)
    return predictions.mean(axis=0)


def _actual_smoothed_pitching_plus(engineered, models, zone_ref, pitch_calibration, verbose=False):
    """
    Scores each pitch's own (actual pitch type, actual zone) combination
    through the identical target-averaging machinery as the candidate
    search, so bestPitch+'s comparison is smoothed-vs-smoothed, not
    smoothed-vs-pinpoint (see module docstring). Uses the row's own real
    STUFF_FEATURES (physics_cols=STUFF_FEATURES, dedup_by_pitcher_season=
    False -- see _build_base_array's docstring for why no dedup benefit
    exists here). Pitches whose own zone isn't one of CANDIDATE_ZONE_CODES,
    or whose own pitch type has no trained model, get NaN (excluded from
    bestPitch+ entirely, same as any other out-of-scope pitch).

    With verbose=True, prints one progress line per pitch type (coverage is
    checked per-ptype here rather than per-zone since each pitch type only
    visits the zones it was actually thrown in, an a priori unknown count).
    """

    result = pd.Series(np.nan, index=engineered.index, dtype=float)
    total_rows = len(engineered)
    start = time.time()

    for ptype_num, (ptype, model) in enumerate(models.items(), start=1):
        rows = (engineered[PITCH_TYPE_COL] == ptype) & engineered[ZONE_COL].isin(zone_ref.index)
        if not rows.any():
            continue
        base = engineered.loc[rows]

        for zone in base[ZONE_COL].unique():
            zone_rows = base[base[ZONE_COL] == zone]
            ref_row = zone_ref.loc[zone]
            zone_array, zone_p_throws_is_R, zone_height, zone_center_key = _build_base_array(
                zone_rows, physics_cols=STUFF_FEATURES, dedup_by_pitcher_season=False,
            )
            raw_pitching_value_hyp = _batched_target_averaged_pitching_run_value(
                model, zone_array, zone_p_throws_is_R, zone_height, zone_center_key,
                ref_row["plate_x"], ref_row["plate_z_rel"],
            )
            score = pitching_mod._calibrate_raw_value(
                raw_pitching_value_hyp, np.full(len(zone_rows), ptype), zone_rows[SEASON_COL].to_numpy(),
                pitch_calibration,
            )
            result.loc[zone_rows.index] = score

        if verbose:
            covered = result.notna().sum()
            print(
                f"  [actual {ptype_num}/{len(models)}] pitch_type={ptype} -- "
                f"{covered:,}/{total_rows:,} rows scored ({covered / total_rows:.0%}) -- "
                f"{time.time() - start:.0f}s elapsed"
            )

    return result


def _search_best_pitching_plus(engineered, arsenal_physics, models, zone_ref, pitch_calibration, verbose=False):
    """
    For every (candidate pitch type with a trained model, candidate zone)
    pair, scores the whole in-scope population at once (vectorized model
    .predict(), averaged over each zone's target area), restricted to rows
    whose pitcher-season has that pitch type as a reliable arsenal member
    (via an inner join against `arsenal_physics`, which also supplies that
    pitch type's fixed candidate physics -- see _arsenal_physics_avg).
    Returns the running max across all candidates, `best_pitching_plus`,
    aligned to engineered's index. Uses `pitch_calibration` since every
    candidate here is a single-pitch score, same as the real
    pitch_pitching_plus it's compared against.

    Scores one zone at a time via _batched_target_averaged_pitching_run_value
    (base_array/p_throws_is_R/zone_height/center_key are built once per pitch
    type, outside this loop, since they don't vary by zone -- only the
    predict() call itself repeats per zone). An earlier version batched every
    candidate zone into a single predict() call per pitch type
    (n_zones * 3 and n_zones * 2 rows for the two point sets), which was safe
    under the pre-9/4 location-only key (<=1,152 dedup groups regardless of
    population size). It is NOT safe now that `center_key` also discriminates
    by pitcher-season (see _build_base_array's docstring): a pitch type
    thrown selectively by many pitchers (e.g. a show-me curveball used in a
    narrow set of counts per pitcher) barely dedupes at all, so the batched
    version's n_zones multiplier turned a merely-large per-zone predict()
    call into a tens-of-millions-of-rows one -- measured directly on the full
    2021-2025 dataset: one pitch type's build step alone ran for over 3
    hours and consumed 15+ GB of RAM before being killed, versus a few
    minutes for the per-zone version below. See docs/dev_log.md's Pitching+
    migration entry.

    With verbose=True, prints one progress line per (pitch type, zone)
    candidate: which candidate just finished out of the total, what fraction
    of in-scope rows have a best_pitching_plus value so far (every row that's
    seen at least one candidate from its own pitcher-season arsenal), and
    elapsed time -- this is the dominant cost of add_bestpitch_plus, so this
    is where visibility matters most.
    """

    best = pd.Series(np.nan, index=engineered.index, dtype=float)
    total_rows = len(engineered)
    total_candidates = len(models) * len(zone_ref)
    candidate_num = 0
    start = time.time()

    # Carry the original row identity through the per-pitch-type inner join
    # below (a plain .merge() resets the index) -- built once here, not per
    # pitch type.
    engineered = engineered.copy()
    engineered["_orig_idx"] = engineered.index

    for ptype, model in models.items():
        ptype_arsenal = arsenal_physics.loc[
            arsenal_physics[PITCH_TYPE_COL] == ptype, [PITCHER_COL, SEASON_COL] + STUFF_FEATURE_CAND_COLS
        ]
        if ptype_arsenal.empty:
            candidate_num += len(zone_ref)
            continue

        base = engineered.merge(ptype_arsenal, on=[PITCHER_COL, SEASON_COL], how="inner")
        if base.empty:
            candidate_num += len(zone_ref)
            continue
        base = base.set_index("_orig_idx")
        base.index.name = None

        # base, and everything derived from it below, is identical across
        # every candidate zone for this pitch type -- build it once here
        # rather than inside the zone loop (see _build_base_array's
        # docstring).
        base_array, p_throws_is_R, zone_height, center_key = _build_base_array(
            base, physics_cols=STUFF_FEATURE_CAND_COLS, dedup_by_pitcher_season=True,
        )
        pitch_types = np.full(len(base), ptype)
        seasons = base[SEASON_COL].to_numpy()

        for zone, ref_row in zone_ref.iterrows():
            raw_pitching_value_hyp = _batched_target_averaged_pitching_run_value(
                model, base_array, p_throws_is_R, zone_height, center_key,
                ref_row["plate_x"], ref_row["plate_z_rel"],
            )
            pitching_plus_hyp = pitching_mod._calibrate_raw_value(
                raw_pitching_value_hyp, pitch_types, seasons, pitch_calibration,
            )
            best.loc[base.index] = np.fmax(best.loc[base.index].to_numpy(), pitching_plus_hyp)

            candidate_num += 1
            if verbose:
                covered = best.notna().sum()
                print(
                    f"  [search {candidate_num}/{total_candidates}] pitch_type={ptype} zone={zone:g} -- "
                    f"{covered:,}/{total_rows:,} rows covered ({covered / total_rows:.0%}) -- "
                    f"{time.time() - start:.0f}s elapsed"
                )

    return best


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_bestpitch_plus(raw_df, models_dir=DEFAULT_MODELS_DIR, retrain=False, verbose=False):
    """
    Takes a raw Statcast dataframe and returns a copy with `best_pitching_plus`
    and `pitch_bestpitch_plus` columns added. Row count and order match the
    input; pitches out of scope (no Pitching+ score, or no reliable arsenal
    for that pitcher-season) get NaN in the new columns.

    The counterfactual search (see _search_best_pitching_plus) is the
    expensive part of this call -- minutes on the full dataset. Pass
    verbose=True for progress prints (candidates completed, row coverage,
    elapsed time) instead of a single opaque wait.
    """

    missing = [col for col in REQUIRED_COLS + BASE_STATE_COLS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required columns: {missing}")

    if "pitch_pitching_plus" not in raw_df.columns:
        raw_df = add_pitching_plus(raw_df, models_dir=models_dir, retrain=retrain)

    models = pitching_mod.load_cached_models(models_dir)

    fit_scope = raw_df.dropna(subset=["pitching_run_value"])
    _, _, pitch_calibration = pitching_mod._calibrate(fit_scope)

    in_scope = (
        raw_df.loc[
            ~raw_df[PITCH_TYPE_COL].isin(JUNK_PITCH_TYPES),
            REQUIRED_COLS + BASE_STATE_COLS + ["pitch_pitching_plus", "stuff_plus", "stuff_plus_reliable"],
        ]
        .dropna(subset=REQUIRED_COLS + ["stuff_plus"])
        .copy()
    )
    # Joint-model scoring needs the full physics + location feature set
    # engineered directly, same as pitching.py's own add_pitching_plus, not
    # just the location columns the old location-only search needed.
    engineered = stuff_mod._build_v1_features(in_scope)
    engineered = _build_features(engineered)
    engineered[ZONE_COL] = in_scope[ZONE_COL]
    engineered["stuff_plus"] = in_scope["stuff_plus"]
    engineered["stuff_plus_reliable"] = in_scope["stuff_plus_reliable"]

    arsenal_physics = _arsenal_physics_avg(engineered)

    zone_ref = _zone_reference(engineered)
    if verbose:
        total_candidates = len(models) * len(zone_ref)
        print(
            f"bestPitch+: {len(engineered):,} in-scope rows, {len(models)} pitch-type models, "
            f"{len(zone_ref)} candidate zones ({total_candidates} candidates in the search)"
        )
        print("bestPitch+: counterfactual search (best_pitching_plus) ...")
    best_pitching_plus = _search_best_pitching_plus(
        engineered, arsenal_physics, models, zone_ref, pitch_calibration, verbose=verbose
    )
    if verbose:
        print("bestPitch+: scoring actual (pitch type, zone) combinations for comparison ...")
    actual_smoothed = _actual_smoothed_pitching_plus(
        engineered, models, zone_ref, pitch_calibration, verbose=verbose
    )

    result = raw_df.copy()
    result["best_pitching_plus"] = np.nan
    result.loc[best_pitching_plus.index, "best_pitching_plus"] = best_pitching_plus.to_numpy()

    actual_col = pd.Series(np.nan, index=result.index, dtype=float)
    actual_col.loc[actual_smoothed.index] = actual_smoothed.to_numpy()

    # The candidate search above is restricted to each pitcher-season's
    # RELIABLE arsenal (stuff_plus_reliable), so a pitch type thrown too
    # rarely that season to be reliable never enters the search at all --
    # including for scoring that type's own pitches. Without this, a pitch
    # whose own type isn't reliable that season could score
    # best_pitching_plus purely from the pitcher's OTHER reliable types,
    # breaking this module's own documented invariant that the actual
    # combination is always among the candidates considered (see docs/
    # dev_log.md's 9/2 entry). Taking the max against the actual (smoothed)
    # score restores that invariant for every row a search actually ran
    # for, without loosening the reliable-arsenal filter for genuine
    # alternative-pitch-type candidates. Rows with no reliable arsenal at
    # all (best_pitching_plus still NaN -- no candidates were ever
    # considered) are left NaN, matching the documented out-of-scope
    # contract; np.fmax also leaves a row's own value untouched wherever
    # actual_col is NaN (own zone/pitch type out of scope for the
    # comparison itself).
    has_candidate = result["best_pitching_plus"].notna()
    result.loc[has_candidate, "best_pitching_plus"] = np.fmax(
        result.loc[has_candidate, "best_pitching_plus"].to_numpy(),
        actual_col.loc[has_candidate].to_numpy(),
    )

    result["pitch_bestpitch_plus"] = result["best_pitching_plus"] - actual_col

    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    default_input = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
    default_output = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_bestpitch.csv"

    parser = argparse.ArgumentParser(description="Add bestPitch+ columns to raw Statcast data.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--retrain", action="store_true", help="Retrain Stuff+/Location+/Pitching+ models instead of using the cache")
    parser.add_argument("--verbose", action="store_true", help="Print counterfactual-search progress")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_bestpitch_plus(raw_df, models_dir=args.models_dir, retrain=args.retrain, verbose=args.verbose)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["pitch_bestpitch_plus"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a bestPitch+ score.")
