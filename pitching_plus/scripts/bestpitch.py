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
    actually throw.
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
result (see docs/dev_log.md). Combined with that pitcher's own real stuff_plus
for the candidate pitch type, run through pitching.py's already-fitted blend +
its pitch-level 100+ calibration (_score_pitching_plus, level="pitch"), the
exact same scale as the real pitch_pitching_plus, not a separately-derived
one. The max over all candidates is `best_pitching_plus`.

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
columns aren't already present. Requires a cached Location+ model (via
location.load_cached_models); run add_location_plus at least once first.
"""

import time

import numpy as np
import pandas as pd

try:
    from . import pitching as pitching_mod
    from .location import (
        BASE_STATE_COLS, DEFAULT_MODELS_DIR, LOCATION_FEATURES,
        REQUIRED_COLS as LOCATION_REQUIRED_COLS, TARGET_COL,
        _build_features, load_cached_models,
    )
    from .pitching import add_pitching_plus
    from .stuff import JUNK_PITCH_TYPES, PITCH_TYPE_COL, PITCHER_COL, SEASON_COL
except ImportError:
    import pitching as pitching_mod
    from location import (
        BASE_STATE_COLS, DEFAULT_MODELS_DIR, LOCATION_FEATURES,
        REQUIRED_COLS as LOCATION_REQUIRED_COLS, TARGET_COL,
        _build_features, load_cached_models,
    )
    from pitching import add_pitching_plus
    from stuff import JUNK_PITCH_TYPES, PITCH_TYPE_COL, PITCHER_COL, SEASON_COL

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
# center_key, which cleanly collapses to <=1,152 values). Rounding it to the
# nearest ZONE_HEIGHT_DEDUP_ROUND_FT before using it as a dedup key groups
# pitches with near-identical strike zones into the same predict() call.
# Error this introduces: the +z/-z offset is TARGET_RADIUS_FT / zone_height,
# so d(offset)/d(zone_height) = -TARGET_RADIUS_FT / zone_height^2 -- at a
# typical zone_height of ~1.8ft, that's about -0.11 ft of offset error per ft
# of zone_height error, so a 0.02ft (~1/4 inch) rounding step induces at most
# ~0.002ft (~0.03 inch) of error in the queried plate_z_rel point, on a
# target radius that's already ~4.35 inches (TARGET_RADIUS_FT) -- roughly
# 0.5% of the target radius itself, well inside the "not a pinpoint, a small
# target area" approximation the design already accepts.
ZONE_HEIGHT_DEDUP_ROUND_FT = 0.02

# LOCATION_FEATURES minus the three location-derived columns _predict_at_point
# overwrites per candidate point; everything else a row already has.
NON_LOCATION_FEATURES = [c for c in LOCATION_FEATURES if c not in ("plate_x", "plate_z_rel", "plate_x_armside")]

# Column positions of the three location-derived features within
# LOCATION_FEATURES, so the batched search path (_build_base_array /
# _batched_target_averaged_location_run_value) can overwrite them in a plain
# numpy array without going through pandas per candidate.
_LOCATION_COL_INDEX = {col: i for i, col in enumerate(LOCATION_FEATURES)}
_PLATE_X_IDX = _LOCATION_COL_INDEX["plate_x"]
_PLATE_Z_REL_IDX = _LOCATION_COL_INDEX["plate_z_rel"]
_PLATE_X_ARMSIDE_IDX = _LOCATION_COL_INDEX["plate_x_armside"]

# NOTE: BASE_STATE_COLS (on_1b/on_2b/on_3b) is deliberately NOT folded into
# REQUIRED_COLS. NaN there means "base empty" (a real game state), not
# missing data, exactly as in location.py. Must be present as columns, but
# not dropna-gated.
REQUIRED_COLS = LOCATION_REQUIRED_COLS + [ZONE_COL]


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
# ARSENAL: WHICH PITCH TYPES CAN THIS PITCHER ACTUALLY THROW
# ============================================================

def _arsenal_wide(df):
    """
    One row per (pitcher, season), one column per candidate pitch type
    (`cand_stuff_<type>`) holding that pitcher's own real stuff_plus for it,
    NaN wherever that pitch type isn't a reliable part of their arsenal that
    season (stuff_plus_reliable == True), so it's automatically excluded from
    the candidate search rather than needing a separate mask.
    """

    arsenal = (
        df[df["stuff_plus_reliable"].fillna(False)]
        [[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, "stuff_plus"]]
        .drop_duplicates()
    )
    wide = arsenal.pivot_table(index=[PITCHER_COL, SEASON_COL], columns=PITCH_TYPE_COL, values="stuff_plus")
    wide.columns = [f"cand_stuff_{c}" for c in wide.columns]
    return wide.reset_index()


# ============================================================
# COUNTERFACTUAL SEARCH: BEST (PITCH TYPE, ZONE) PER PITCH
# ============================================================

def _predict_at_point(base, model, non_location_features, plate_x, plate_z_rel):
    X = base[non_location_features].copy()
    X["plate_x"] = plate_x
    X["plate_z_rel"] = plate_z_rel
    # Arm-side sign is keyed on the PITCHER's throwing hand (p_throws), not
    # batter stand -- see location.py's plate_x_armside comment for the
    # derivation. Fixed 9/2 (docs/dev_log.md); was previously keyed on stand.
    X["plate_x_armside"] = np.where(base["p_throws"] == "R", -plate_x, plate_x)
    return model.predict(X[LOCATION_FEATURES].to_numpy())


def _build_base_array(base):
    """
    Prebuilds `base`'s non-location features as one numpy array, in
    LOCATION_FEATURES column order (the 3 location columns start at 0 and get
    overwritten per candidate point by _batched_target_averaged_location_run_value),
    plus the per-row quantities every candidate point needs: p_throws == 'R'
    (arm-side sign is keyed on the PITCHER's throwing hand, not batter
    stand -- see location.py's plate_x_armside comment), strike-zone height
    (for converting TARGET_RADIUS_FT to plate_z_rel units), and `center_key`.

    `center_key` identifies rows that are indistinguishable to the location
    model for any candidate point that doesn't depend on strike-zone height
    (center, +x, -x -- 3 of the 5 target-averaging points): `re288_state`
    already bijectively encodes balls/strikes/outs_when_up/on-base state (see
    location.py's _build_features), so together with stand_R and p_throws_R
    it fully determines every LOCATION_FEATURES column except the 3 location
    ones. There are at most 288 * 2 * 2 = 1,152 distinct keys, versus up to
    millions of rows in `base` -- _batched_target_averaged_location_run_value
    predicts once per unique key for those 3 points instead of once per row.
    Note `center_key` itself still needs batter stand (stand_R is a real
    model feature, independent of the arm-side sign convention), even though
    the arm-side sign returned separately below does not.
    """

    array = np.zeros((len(base), len(LOCATION_FEATURES)), dtype=float)
    for col, idx in _LOCATION_COL_INDEX.items():
        if col not in ("plate_x", "plate_z_rel", "plate_x_armside"):
            array[:, idx] = base[col].to_numpy(dtype=float)

    stand_is_R = (base["stand"] == "R").to_numpy()
    p_throws_is_R = (base["p_throws"] == "R").to_numpy()
    zone_height = (base["sz_top"] - base["sz_bot"]).to_numpy()
    re288_state = base["re288_state"].to_numpy(dtype=np.int64)
    p_throws_R = base["p_throws_R"].to_numpy(dtype=np.int64)
    center_key = (re288_state * 2 + stand_is_R.astype(np.int64)) * 2 + p_throws_R

    return array, p_throws_is_R, zone_height, center_key


def _batched_target_averaged_location_run_value(model, array, p_throws_is_R, zone_height, center_key, center_x, center_z_rel):
    """
    Averages the model's location_run_value prediction over the same 5-point
    "target" (center + N/S/E/W at TARGET_RADIUS_FT) as _predict_at_point, but
    deduplicated: rows that would get an identical (or, for +z/-z, a
    near-identical -- see below) model input for a given point are predicted
    once, not once each (see _build_base_array's docstring for
    `center_key`). Every row gets the same value it would from predicting on
    it directly (center/+x/-x), or a value within a tiny, bounded tolerance
    of it (+z/-z, see ZONE_HEIGHT_DEDUP_ROUND_FT), computed once and
    broadcast back via `inverse` instead of recomputed per duplicate row.

    Center/+x/-x (indices 0-2) don't depend on strike-zone height, so they
    dedupe exactly on `center_key` alone (<=1,152 distinct rows to predict
    on, regardless of how large `array` is). +z/-z (indices 3-4)
    additionally depend on each row's own zone_height (a near-continuous
    per-pitch float -- see ZONE_HEIGHT_DEDUP_ROUND_FT's comment -- raw
    zone_height alone would barely dedupe at all), so they dedupe on
    (center_key, zone_height rounded to ZONE_HEIGHT_DEDUP_ROUND_FT) instead.
    The z-offset itself still uses each dedup group's representative row's
    exact (unrounded) zone_height, so the rounding only affects which rows
    share a predict() call, not the offset's own precision. The z-offset is
    converted from physical feet to plate_z_rel units per row, since strike
    zone height varies by batter.
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


def _all_zones_target_averaged_location_run_value(model, array, p_throws_is_R, zone_height, center_key, zone_ref):
    """
    Same 5-point target-averaged prediction as
    _batched_target_averaged_location_run_value, but for every candidate zone
    in `zone_ref` at once instead of one zone per call. Used by
    _search_best_pitching_plus, which needs every zone's prediction for the
    same pitch-type population; _actual_smoothed_pitching_plus still uses the
    per-zone version since its per-zone `base` subset (rows actually thrown
    in that zone) differs by zone, so there's no shared population to batch
    across.

    _search_best_pitching_plus's zone loop used to call
    _batched_target_averaged_location_run_value once per zone (13x per pitch
    type), even though `array`/`p_throws_is_R`/`zone_height`/`center_key` are
    identical across all 13 zones for a given pitch type (see
    _search_best_pitching_plus's own comment on `base`). Each of those 13
    calls redundantly reran np.unique(center_key) and np.unique(zh_key), and
    issued its own 2 model.predict() calls -- 26 predict() calls per pitch
    type, 260 for the full search. This version dedupes once and stacks all
    13 zones' query points into 2 total predict() calls per pitch type: one
    for the center/+x/-x set (n_zones * 3 * k rows), one for the +z/-z set
    (n_zones * 2 * m rows), where k/m are the same <=1,152 / <=unique-zone-
    height dedup counts as the per-zone version.

    Same output as calling the per-zone version once per zone (verified
    against it on synthetic data with a stub model, using the same
    ZONE_HEIGHT_DEDUP_ROUND_FT bucketing on both sides -- max abs diff
    ~1e-13, floating-point summation-order noise; see that constant's
    comment for the bounded approximation this introduces relative to a
    truly unrounded zone_height key). Returns an (n_zones, len(array))
    array, row order matching zone_ref.index order; row i of the result is
    what _batched_target_averaged_location_run_value would return for
    zone_ref.iloc[i].
    """

    uniq_center, first_center, inverse_center = np.unique(center_key, return_index=True, return_inverse=True)
    k = uniq_center.shape[0]
    zone_height_bucket = np.round(zone_height / ZONE_HEIGHT_DEDUP_ROUND_FT)
    zh_key = np.column_stack([center_key, zone_height_bucket])
    uniq_zh, first_zh, inverse_zh = np.unique(zh_key, axis=0, return_index=True, return_inverse=True)
    m = uniq_zh.shape[0]

    n_zones = len(zone_ref)
    zone_x = zone_ref["plate_x"].to_numpy()
    zone_z = zone_ref["plate_z_rel"].to_numpy()

    # --- center / +x / -x set: one predict() call for all zones ---
    center_rows = np.tile(array[first_center], (n_zones * 3, 1))
    center_p_throws_R = np.tile(p_throws_is_R[first_center], n_zones * 3)

    x_offsets = np.array([0.0, TARGET_RADIUS_FT, -TARGET_RADIUS_FT])
    center_x_flat = np.repeat(zone_x, 3 * k) + np.tile(np.repeat(x_offsets, k), n_zones)
    center_z_flat = np.repeat(zone_z, 3 * k)

    center_rows[:, _PLATE_X_IDX] = center_x_flat
    center_rows[:, _PLATE_Z_REL_IDX] = center_z_flat
    center_rows[:, _PLATE_X_ARMSIDE_IDX] = np.where(center_p_throws_R, -center_x_flat, center_x_flat)

    center_preds = model.predict(center_rows).reshape(n_zones, 3, k)

    # --- +z / -z set: one predict() call for all zones ---
    ns_rows = np.tile(array[first_zh], (n_zones * 2, 1))
    ns_p_throws_R = np.tile(p_throws_is_R[first_zh], n_zones * 2)
    ns_z_offset = TARGET_RADIUS_FT / zone_height[first_zh]

    ns_x_flat = np.repeat(zone_x, 2 * m)
    z_signed_offsets = np.concatenate([ns_z_offset, -ns_z_offset])
    ns_z_flat = np.repeat(zone_z, 2 * m) + np.tile(z_signed_offsets, n_zones)

    ns_rows[:, _PLATE_X_IDX] = ns_x_flat
    ns_rows[:, _PLATE_Z_REL_IDX] = ns_z_flat
    ns_rows[:, _PLATE_X_ARMSIDE_IDX] = np.where(ns_p_throws_R, -ns_x_flat, ns_x_flat)

    ns_preds = model.predict(ns_rows).reshape(n_zones, 2, m)

    n_rows = array.shape[0]
    result = np.empty((n_zones, n_rows), dtype=float)
    for zi in range(n_zones):
        c = center_preds[zi][:, inverse_center]
        n = ns_preds[zi][:, inverse_zh]
        result[zi] = np.concatenate([c, n], axis=0).mean(axis=0)
    return result


def _actual_smoothed_pitching_plus(engineered, models, zone_ref, blend_params, pitch_calibration, verbose=False):
    """
    Scores each pitch's own (actual pitch type, actual zone) combination
    through the identical target-averaging machinery as the candidate
    search, so bestPitch+'s comparison is smoothed-vs-smoothed, not
    smoothed-vs-pinpoint (see module docstring). Pitches whose own zone
    isn't one of CANDIDATE_ZONE_CODES, or whose own pitch type has no
    trained model, get NaN (excluded from bestPitch+ entirely, same as any
    other out-of-scope pitch).

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
            zone_array, zone_p_throws_is_R, zone_height, zone_center_key = _build_base_array(zone_rows)
            location_run_value_hyp = _batched_target_averaged_location_run_value(
                model, zone_array, zone_p_throws_is_R, zone_height, zone_center_key,
                ref_row["plate_x"], ref_row["plate_z_rel"],
            )
            score = pitching_mod._score_pitching_plus(
                zone_rows["stuff_plus"].to_numpy(), location_run_value_hyp,
                np.full(len(zone_rows), ptype), zone_rows[SEASON_COL].to_numpy(),
                blend_params, pitch_calibration, level="pitch",
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


def _search_best_pitching_plus(engineered, models, zone_ref, blend_params, pitch_calibration, verbose=False):
    """
    For every (candidate pitch type with a trained model, candidate zone)
    pair, scores the whole in-scope population at once (vectorized model
    .predict(), averaged over each zone's target area), restricted to rows
    where that pitch type is actually in the row's own pitcher-season
    arsenal. Returns the running max across all candidates,
    `best_pitching_plus`, aligned to engineered's index. Uses `pitch_calibration`
    (not `calibration`) since every candidate here is a single-pitch score,
    same as the real pitch_pitching_plus it's compared against.

    The location prediction for all 13 zones of a given pitch type is
    computed together via _all_zones_target_averaged_location_run_value (2
    model.predict() calls per pitch type instead of 2 per zone); the
    per-zone loop below only does the cheap part (blend + calibration merge,
    running max).

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

    # PROFILING (temporary, for the bestpitch-v2 investigation into where the
    # ~1000s is actually going now that predict() calls are batched -- see
    # docs/dev_log.md / the plan for this branch). Cumulative time in each
    # phase across all 130 candidates, printed once at the end.
    build_phase_time = 0.0
    predict_phase_time = 0.0
    score_assign_phase_time = 0.0

    for ptype, model in models.items():
        cand_col = f"cand_stuff_{ptype}"
        if cand_col not in engineered.columns:
            candidate_num += len(zone_ref)
            continue
        valid = engineered[cand_col].notna()
        if not valid.any():
            candidate_num += len(zone_ref)
            continue
        base = engineered.loc[valid]
        # base, and everything derived from it below, is identical across all
        # 13 zones for this pitch type -- build it once here rather than
        # inside the zone loop (see _build_base_array's docstring).
        t0 = time.time()
        base_array, p_throws_is_R, zone_height, center_key = _build_base_array(base)
        cand_stuff = base[cand_col].to_numpy()
        pitch_types = np.full(len(base), ptype)
        seasons = base[SEASON_COL].to_numpy()
        build_phase_time += time.time() - t0

        # All 13 zones' location predictions in 2 model.predict() calls total
        # (not 2 per zone -- see _all_zones_target_averaged_location_run_value's
        # docstring), since base_array/p_throws_is_R/zone_height/center_key
        # don't vary by zone either.
        t0 = time.time()
        location_run_value_hyp_all = _all_zones_target_averaged_location_run_value(
            model, base_array, p_throws_is_R, zone_height, center_key, zone_ref
        )
        predict_phase_time += time.time() - t0

        for zi, (zone, ref_row) in enumerate(zone_ref.iterrows()):
            t0 = time.time()
            location_run_value_hyp = location_run_value_hyp_all[zi]
            pitching_plus_hyp = pitching_mod._score_pitching_plus(
                cand_stuff, location_run_value_hyp, pitch_types, seasons,
                blend_params, pitch_calibration, level="pitch",
            )
            best.loc[base.index] = np.fmax(best.loc[base.index].to_numpy(), pitching_plus_hyp)
            score_assign_phase_time += time.time() - t0

            candidate_num += 1
            if verbose:
                covered = best.notna().sum()
                print(
                    f"  [search {candidate_num}/{total_candidates}] pitch_type={ptype} zone={zone:g} -- "
                    f"{covered:,}/{total_rows:,} rows covered ({covered / total_rows:.0%}) -- "
                    f"{time.time() - start:.0f}s elapsed"
                )

    if verbose:
        print(
            f"  [profile] build={build_phase_time:.0f}s predict={predict_phase_time:.0f}s "
            f"score+assign={score_assign_phase_time:.0f}s (of {time.time() - start:.0f}s total)"
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

    models = load_cached_models(models_dir)

    fit_scope = raw_df.dropna(subset=["pitch_stuff_plus", "stuff_plus", "location_run_value", TARGET_COL])
    blend_params, _, pitch_calibration, _ = pitching_mod._fit_pitching_plus_model(fit_scope)

    in_scope = (
        raw_df.loc[
            ~raw_df[PITCH_TYPE_COL].isin(JUNK_PITCH_TYPES),
            REQUIRED_COLS + BASE_STATE_COLS + ["pitch_pitching_plus", "stuff_plus", "stuff_plus_reliable"],
        ]
        .dropna(subset=REQUIRED_COLS + ["stuff_plus"])
        .copy()
    )
    engineered = _build_features(in_scope)
    engineered[ZONE_COL] = in_scope[ZONE_COL]
    engineered["stuff_plus"] = in_scope["stuff_plus"]

    arsenal = _arsenal_wide(raw_df)
    engineered = engineered.merge(arsenal, on=[PITCHER_COL, SEASON_COL], how="left")
    engineered.index = in_scope.index

    zone_ref = _zone_reference(engineered)
    if verbose:
        total_candidates = len(models) * len(zone_ref)
        print(
            f"bestPitch+: {len(engineered):,} in-scope rows, {len(models)} pitch-type models, "
            f"{len(zone_ref)} candidate zones ({total_candidates} candidates in the search)"
        )
        print("bestPitch+: counterfactual search (best_pitching_plus) ...")
    best_pitching_plus = _search_best_pitching_plus(
        engineered, models, zone_ref, blend_params, pitch_calibration, verbose=verbose
    )
    if verbose:
        print("bestPitch+: scoring actual (pitch type, zone) combinations for comparison ...")
    actual_smoothed = _actual_smoothed_pitching_plus(
        engineered, models, zone_ref, blend_params, pitch_calibration, verbose=verbose
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
    parser.add_argument("--retrain", action="store_true", help="Retrain Location+ models instead of using the cache")
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
