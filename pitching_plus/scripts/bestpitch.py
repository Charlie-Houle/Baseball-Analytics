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
(`_actual_smoothed_pitching_plus`) before comparing. Since that
combination is itself always one of the candidates the search considers,
`best_pitching_plus` is a true max over a set that includes it, restoring
`best_pitching_plus >= actual (smoothed)` as an (almost) guaranteed property.

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

# LOCATION_FEATURES minus the three location-derived columns _predict_at_point
# overwrites per candidate point; everything else a row already has.
NON_LOCATION_FEATURES = [c for c in LOCATION_FEATURES if c not in ("plate_x", "plate_z_rel", "plate_x_armside")]

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
    X["plate_x_armside"] = np.where(base["stand"] == "R", plate_x, -plate_x)
    return model.predict(X[LOCATION_FEATURES].to_numpy())


def _target_averaged_location_run_value(base, model, non_location_features, center_x, center_z_rel):
    """
    Averages the model's location_run_value prediction over a 5-point
    "target" (center + N/S/E/W at TARGET_RADIUS_FT) around (center_x,
    center_z_rel), instead of querying the single reference point. See
    module docstring for why. The z-offset is converted from physical feet
    to plate_z_rel units per row, since strike zone height varies by batter.
    """

    zone_height = (base["sz_top"] - base["sz_bot"]).to_numpy()
    z_offset = TARGET_RADIUS_FT / zone_height

    points = [
        (center_x, center_z_rel),
        (center_x + TARGET_RADIUS_FT, center_z_rel),
        (center_x - TARGET_RADIUS_FT, center_z_rel),
        (center_x, center_z_rel + z_offset),
        (center_x, center_z_rel - z_offset),
    ]
    predictions = [_predict_at_point(base, model, non_location_features, px, pz) for px, pz in points]
    return np.mean(predictions, axis=0)


def _actual_smoothed_pitching_plus(engineered, models, zone_ref, blend_params, pitch_calibration):
    """
    Scores each pitch's own (actual pitch type, actual zone) combination
    through the identical target-averaging machinery as the candidate
    search, so bestPitch+'s comparison is smoothed-vs-smoothed, not
    smoothed-vs-pinpoint (see module docstring). Pitches whose own zone
    isn't one of CANDIDATE_ZONE_CODES, or whose own pitch type has no
    trained model, get NaN (excluded from bestPitch+ entirely, same as any
    other out-of-scope pitch).
    """

    result = pd.Series(np.nan, index=engineered.index, dtype=float)

    for ptype, model in models.items():
        rows = (engineered[PITCH_TYPE_COL] == ptype) & engineered[ZONE_COL].isin(zone_ref.index)
        if not rows.any():
            continue
        base = engineered.loc[rows]

        for zone in base[ZONE_COL].unique():
            zone_rows = base[base[ZONE_COL] == zone]
            ref_row = zone_ref.loc[zone]
            location_run_value_hyp = _target_averaged_location_run_value(
                zone_rows, model, NON_LOCATION_FEATURES, ref_row["plate_x"], ref_row["plate_z_rel"]
            )
            score = pitching_mod._score_pitching_plus(
                zone_rows["stuff_plus"].to_numpy(), location_run_value_hyp,
                np.full(len(zone_rows), ptype), zone_rows[SEASON_COL].to_numpy(),
                blend_params, pitch_calibration, level="pitch",
            )
            result.loc[zone_rows.index] = score

    return result


def _search_best_pitching_plus(engineered, models, zone_ref, blend_params, pitch_calibration):
    """
    For every (candidate pitch type with a trained model, candidate zone)
    pair, scores the whole in-scope population at once (vectorized model
    .predict(), averaged over each zone's target area), restricted to rows
    where that pitch type is actually in the row's own pitcher-season
    arsenal. Returns the running max across all candidates,
    `best_pitching_plus`, aligned to engineered's index. Uses `pitch_calibration`
    (not `calibration`) since every candidate here is a single-pitch score,
    same as the real pitch_pitching_plus it's compared against.
    """

    best = pd.Series(np.nan, index=engineered.index, dtype=float)

    for ptype, model in models.items():
        cand_col = f"cand_stuff_{ptype}"
        if cand_col not in engineered.columns:
            continue
        valid = engineered[cand_col].notna()
        if not valid.any():
            continue
        base = engineered.loc[valid]

        for zone, ref_row in zone_ref.iterrows():
            location_run_value_hyp = _target_averaged_location_run_value(
                base, model, NON_LOCATION_FEATURES, ref_row["plate_x"], ref_row["plate_z_rel"]
            )
            pitching_plus_hyp = pitching_mod._score_pitching_plus(
                base[cand_col].to_numpy(), location_run_value_hyp,
                np.full(len(base), ptype), base[SEASON_COL].to_numpy(),
                blend_params, pitch_calibration, level="pitch",
            )
            best.loc[base.index] = np.fmax(best.loc[base.index].to_numpy(), pitching_plus_hyp)

    return best


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_bestpitch_plus(raw_df, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    """
    Takes a raw Statcast dataframe and returns a copy with `best_pitching_plus`
    and `pitch_bestpitch_plus` columns added. Row count and order match the
    input; pitches out of scope (no Pitching+ score, or no reliable arsenal
    for that pitcher-season) get NaN in the new columns.
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
    best_pitching_plus = _search_best_pitching_plus(engineered, models, zone_ref, blend_params, pitch_calibration)
    actual_smoothed = _actual_smoothed_pitching_plus(engineered, models, zone_ref, blend_params, pitch_calibration)

    result = raw_df.copy()
    result["best_pitching_plus"] = np.nan
    result.loc[best_pitching_plus.index, "best_pitching_plus"] = best_pitching_plus.to_numpy()

    actual_col = pd.Series(np.nan, index=result.index, dtype=float)
    actual_col.loc[actual_smoothed.index] = actual_smoothed.to_numpy()
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
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_bestpitch_plus(raw_df, models_dir=args.models_dir, retrain=args.retrain)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["pitch_bestpitch_plus"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a bestPitch+ score.")
