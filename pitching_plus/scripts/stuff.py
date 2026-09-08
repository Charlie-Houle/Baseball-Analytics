"""
Stuff+: a trained model that predicts a pitch's expected run value from its
own physics alone -- release speed/spin/extension, acceleration,
reaction-time-adjusted movement, and a seam-shifted-wake proxy
(axis_differential). Unlike Location+/Pitching+, it never sees location,
count, situational, or arsenal-comparison features. See
notebooks/stuff.ipynb for the original PCA-era derivation and
docs/dev_log.md's 9/4 entry for this retool's derivation/validation.

Retooled 9/4/2026 from an earlier unsupervised PCA-composite design (physics
"outlierness," deliberately no outcome data -- see the 8/26 and 9/2 entries)
to a supervised model trained against delta_pitcher_run_exp, mirroring
location.py's architecture, per FanGraphs' Stuff+/Location+/Pitching+ primer's
description of how their own Stuff+ works. The 8/26 and 9/2 "keep Stuff+
outcome-independent" verdicts are superseded by this change, not deleted --
see dev_log.md for the full history of both designs.

Public entry point is `add_stuff_plus`, which takes a raw Statcast dataframe
(as loaded by data/load_data.py) and returns it with:
  - `stuff_run_value`     : pitch-level expected run value (runs), positive
                            favors the pitcher, from out-of-fold CV
                            prediction (never the pitch's own realized
                            outcome or an in-sample fit)
  - `pitch_stuff_plus`    : pitch-level, 100+ scaled (per pitch_type/season)
  - `stuff_plus`          : pitcher x pitch_type x season aggregate, 100+
                            scaled, broadcast onto every pitch in that
                            group -- the headline figure (purpose.md's
                            "aggregated to pitch type")
  - `stuff_plus_reliable` : whether that (pitcher, pitch_type, season) met
                            the pitch-count bar (>= MIN_PITCHES_FOR_SCORE)
Row count and order are unchanged; pitches out of scope (junk pitch types,
incomplete physics/outcome data, or pitch types too rare to train/calibrate)
get NaN in the new columns.

Training the per-pitch-type gradient-boosted models with 5-fold
cross-validation is the expensive step, so trained models plus the
out-of-fold historical scores are cached under pitching_plus/models/
(joblib, gitignored), exactly like location.py. A normal `add_stuff_plus`
call loads the cache instead of retraining. Pass retrain=True (or run
`python stuff.py --retrain`) to rebuild it after the underlying data or
STUFF_FEATURES/HGB_PARAMS change.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

# ============================================================
# CONFIGURATION
# ============================================================

PITCHER_COL = "pitcher"
PITCH_TYPE_COL = "pitch_type"
SEASON_COL = "game_year"
PLAYER_NAME_COL = "player_name"

# Duplicated from location.py, not imported: location.py imports FROM
# stuff.py (JUNK_PITCH_TYPES, MIN_PITCHES_FOR_SCORE, PITCH_TYPE_COL,
# PITCHER_COL, SEASON_COL), so stuff.py importing back would be circular.
# Same literal values by convention, not by shared code.
KEY_COLS = ["game_pk", "at_bat_number", "pitch_number"]

TARGET_COL = "delta_pitcher_run_exp"

# Non-standard / misc pitch codes out of scope for a Stuff+ model
# (pitchouts, unknowns, eephus, forkball, knuckleball, slow curve, screwball, generic "fastball")
JUNK_PITCH_TYPES = {"FA", "EP", "FO", "KN", "CS", "SC", "PO", "UN"}

PHYSICS_COLS = [
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "pfx_x",
    "pfx_z",
]

# Now includes TARGET_COL + KEY_COLS: the model trains against TARGET_COL,
# and out-of-fold scores are cached/reattached by KEY_COLS, exactly like
# location.py. Real (small) scope narrowing vs. the old PCA version: a
# physically-complete pitch missing delta_pitcher_run_exp used to still get
# a Stuff+ score and no longer will (checked against real data: negligible,
# see docs/dev_log.md's 9/4 entry).
REQUIRED_COLS = [
    PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, PLAYER_NAME_COL, TARGET_COL,
] + KEY_COLS + PHYSICS_COLS

# Distance (feet from release) used for "reaction time" in
# movement_per_reaction_time -- purpose.md's "Reaction x Movement" concept.
REACTION_DISTANCE_FT = 30

# Features fed to the per-pitch-type GBM. Same six as the PCA era plus
# axis_differential (seam-shifted-wake proxy). The PCA-era dedup rationale
# ("reaction time is r=0.99 with velocity, inflates PC1 weights") doesn't
# apply to a tree model -- HistGradientBoostingRegressor splits on whichever
# correlated feature is locally most informative and isn't harmed by
# redundant inputs the way a linear composite is.
STUFF_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "acceleration_mag",
    "horizontal_acceleration",
    "movement_per_reaction_time",
    "axis_differential",
]

# A pitch type needs at least this many pitches (pooled across all
# pitchers/seasons) before a model is trained for it at all.
MIN_GROUP_SIZE_FOR_MODEL = 5000

N_FOLDS = 5

HGB_PARAMS = dict(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42)

# Minimum pitches in a (pitcher, pitch_type, season) group for its average to
# set the league reference distribution / be reported as "reliable."
MIN_PITCHES_FOR_SCORE = 20

# Controls how many points one SD of stuff is worth on the final scale
# (0.10 -> roughly +/-10 points per SD, in line with typical "+" stat spreads).
STUFF_SCALE_K = 0.10

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================
# V1: PHYSICAL PITCH REPRESENTATION + TRAJECTORY FEATURES
# ============================================================

def _solve_time_to_y_vectorized(y0, vy0, ay, distance):
    """
    Vectorized solve for the smallest positive time at which a pitch
    (starting at y0, with velocity vy0 and acceleration ay) reaches
    `distance` feet from the release point:

        target_y = y0 - distance
        target_y = y0 + vy0*t + 0.5*ay*t^2

    Since target_y = y0 - distance, the "c" term of the quadratic
    (y0 - target_y) simplifies to `distance` directly.
    """

    a = 0.5 * ay
    b = vy0
    c = float(distance)

    with np.errstate(invalid="ignore", divide="ignore"):
        discriminant = b**2 - 4 * a * c
        valid_disc = discriminant >= 0
        sqrt_disc = np.sqrt(np.clip(discriminant, 0, None))

        is_linear = np.abs(a) < 1e-10
        b_nonzero = np.abs(b) >= 1e-10

        t_linear = np.where(b_nonzero, -c / np.where(b_nonzero, b, 1.0), np.nan)
        t_linear = np.where(t_linear > 0, t_linear, np.nan)

        safe_a = np.where(is_linear, 1.0, a)
        t1 = (-b + sqrt_disc) / (2 * safe_a)
        t2 = (-b - sqrt_disc) / (2 * safe_a)
        t1 = np.where(t1 > 0, t1, np.inf)
        t2 = np.where(t2 > 0, t2, np.inf)
        t_quad = np.minimum(t1, t2)
        t_quad = np.where(np.isinf(t_quad), np.nan, t_quad)

        t = np.where(is_linear, t_linear, t_quad)
        t = np.where(valid_disc, t, np.nan)

    return t


def _build_v1_features(df):
    """
    Adds derived-physics and reaction-time features. Expects `df` to already
    be filtered to in-scope pitch types with complete PHYSICS_COLS (no NaNs).
    """

    df = df.copy()

    df["acceleration_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    df["horizontal_acceleration"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2)

    df["time_30ft"] = _solve_time_to_y_vectorized(
        df["release_pos_y"].to_numpy(), df["vy0"].to_numpy(), df["ay"].to_numpy(), REACTION_DISTANCE_FT
    )

    # "Reaction x Movement": how much the ball deviates within the reaction
    # window available (purpose.md's explicit concept).
    df["movement_per_reaction_time"] = df["horizontal_acceleration"] / df["time_30ft"]

    # Seam-shifted-wake proxy: angular gap between the ball's measured spin
    # axis (Statcast's spin_axis, clock-face degrees) and the direction
    # implied by its OBSERVED movement (pfx_x/pfx_z, already gravity-
    # adjusted -- no trajectory re-derivation needed). Under pure Magnus
    # physics the two should point the same way; a large gap means the
    # ball's actual break isn't explained by its bulk spin -- the SSW
    # signature.
    #
    # Formula validated empirically against the full 2021-2025 dataset (see
    # docs/dev_log.md's 9/4 entry): movement_angle = atan2(pfx_x, -pfx_z),
    # compared directly against spin_axis (both in the same 0-360 clock
    # convention once pfx_z is negated). Median gap by pitch type: FF 8.7
    # deg, CU 9.3, KC 10.3, CH 12.4, SI 18.6, FS 20.6, FC 26.3, SL 27.6,
    # SV 29.1, ST 31.1 -- matches known spin-efficiency ordering
    # (four-seamers/curveballs closest to pure Magnus; sinkers/splitters
    # known for real SSW; cutters/sliders/sweepers gyro-heavy). Consistent
    # across RHP/LHP splits, so no per-handedness mirroring is needed.
    movement_angle_deg = np.degrees(np.arctan2(df["pfx_x"], -df["pfx_z"])) % 360
    raw_gap = np.abs(df["spin_axis"] - movement_angle_deg) % 360
    df["axis_differential"] = np.minimum(raw_gap, 360 - raw_gap)

    return df


# ============================================================
# MODEL TRAINING (expensive, cached)
# ============================================================

def _train_models(df):
    """
    Fits one gradient-boosted regressor per pitch type against TARGET_COL.
    Returns (models, historical_scores), identical structure/contract to
    location.py's _train_models: models are refit on ALL in-scope rows per
    type (for scoring pitches outside the training set later);
    historical_scores holds KEY_COLS + stuff_run_value from 5-fold
    out-of-fold CV, never the same-data fit, so a pitch's own outcome never
    leaks into its own score.
    """

    models = {}
    oof = pd.Series(np.nan, index=df.index, dtype=float)

    for ptype, type_group in df.groupby(PITCH_TYPE_COL, observed=True):

        if len(type_group) < MIN_GROUP_SIZE_FOR_MODEL:
            continue

        X = type_group[STUFF_FEATURES].to_numpy()
        y = type_group[TARGET_COL].to_numpy()

        model = HistGradientBoostingRegressor(**HGB_PARAMS)

        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof.loc[type_group.index] = cross_val_predict(model, X, y, cv=kfold, n_jobs=-1)

        model.fit(X, y)
        models[ptype] = model

    scored_mask = oof.notna()
    historical_scores = df.loc[scored_mask, KEY_COLS].copy()
    historical_scores["stuff_run_value"] = oof.loc[scored_mask].to_numpy()

    return models, historical_scores


def load_cached_models(models_dir=DEFAULT_MODELS_DIR):
    """
    Loads just the cached per-pitch-type models, mirroring location.py's
    function of the same name. Nothing in this codebase currently calls it
    (bestpitch.py never scores a counterfactual Stuff+ value -- it only
    reads a pitcher's real, already-computed stuff_plus). Included for
    architectural parity with location.py's public surface.
    """

    models_path = Path(models_dir) / "stuff_models.joblib"
    if not models_path.exists():
        raise FileNotFoundError(
            f"No cached Stuff+ models at {models_path}. Run add_stuff_plus "
            "(or `python stuff.py --retrain`) at least once first."
        )
    return joblib.load(models_path)


def _cache_fingerprint():
    """
    Everything that changes what a cached model/score actually means: the
    feature set and the per-pitch-type training config. Saved alongside the
    cache so a stale cache (built under a different STUFF_FEATURES or
    HGB_PARAMS) can be detected instead of silently reused. A cache saved
    before this fingerprint existed has no sidecar file; that's treated as
    "unknown," not "stale."
    """

    return {
        "stuff_features": list(STUFF_FEATURES),
        "hgb_params": dict(HGB_PARAMS),
        "n_folds": N_FOLDS,
        "min_group_size_for_model": MIN_GROUP_SIZE_FOR_MODEL,
    }


def _load_or_score(engineered, models_dir, retrain):
    models_dir = Path(models_dir)
    models_path = models_dir / "stuff_models.joblib"
    scores_path = models_dir / "stuff_historical_scores.joblib"
    fingerprint_path = models_dir / "stuff_cache_fingerprint.joblib"

    if not retrain and models_path.exists() and scores_path.exists():
        if fingerprint_path.exists() and joblib.load(fingerprint_path) != _cache_fingerprint():
            warnings.warn(
                f"Stuff+ cache at {models_dir} was built under a different "
                "STUFF_FEATURES/HGB_PARAMS configuration than the current code. "
                "Reusing it anyway since retrain=False; pass retrain=True to rebuild "
                "it under the current configuration.",
                stacklevel=2,
            )
        models = joblib.load(models_path)
        historical_scores = joblib.load(scores_path)
    else:
        models, historical_scores = _train_models(engineered)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, models_path, compress=3)
        joblib.dump(historical_scores, scores_path, compress=3)
        joblib.dump(_cache_fingerprint(), fingerprint_path, compress=3)

    scored = engineered.merge(historical_scores, on=KEY_COLS, how="left")
    scored.index = engineered.index

    # Pitches not found in the historical cache (e.g. new data since the
    # models were last trained) fall back to direct model scoring. The
    # normal, correct thing to do for a pitch the model hasn't seen before.
    missing = scored["stuff_run_value"].isna()
    for ptype, model in models.items():
        rows = missing & (scored[PITCH_TYPE_COL] == ptype)
        if rows.any():
            X = scored.loc[rows, STUFF_FEATURES].to_numpy()
            scored.loc[rows, "stuff_run_value"] = model.predict(X)

    return scored


# ============================================================
# 100+ SCALE CALIBRATION
# ============================================================

def _ratio_calibration(reliable, group_cols, value_col):
    """
    Local copy of location.py's _ratio_calibration pattern -- stuff.py can't
    import it (location.py imports from stuff.py; importing back would be
    circular). Same genuine-ratio-scale approach: league reference (agg_mu,
    agg_sigma) plus the ratio-mean anchor, computed only from the reliable
    rows: 100*exp(k*z), renormalized so the reliable population's mean is
    exactly 100.
    """

    calibration = (
        reliable.groupby(group_cols)[value_col]
        .agg(agg_mu="mean", agg_sigma="std")
        .reset_index()
    )
    merged = reliable.merge(calibration, on=group_cols, how="left")
    z = (merged[value_col] - merged["agg_mu"]) / merged["agg_sigma"]
    merged["raw_ratio"] = np.exp(STUFF_SCALE_K * z)

    raw_ratio_mean = (
        merged.groupby(group_cols)["raw_ratio"]
        .mean()
        .rename("raw_ratio_mean")
        .reset_index()
    )
    return calibration.merge(raw_ratio_mean, on=group_cols, how="left")


def _to_100_scale(value, calibrated_frame):
    return 100 * np.exp(
        STUFF_SCALE_K * (value - calibrated_frame["agg_mu"]) / calibrated_frame["agg_sigma"]
    ) / calibrated_frame["raw_ratio_mean"]


def _calibrate(scored):
    """
    Two calibrations, each fit on its OWN level's real spread -- an
    aggregate-vs-pitch-level sigma split that the PCA-era code got wrong
    (fit a single sigma from the spread of per-pitcher-season MEANS, then
    applied it to raw per-pitch values too; identical to a bug fixed in
    location.py/pitching.py on 9/2 and 8/30, flagged but left unfixed here
    at the time -- see docs/dev_log.md's 9/2 and 9/4 entries). agg_calibration
    fits agg_mu/agg_sigma from the spread of per-(pitcher, pitch_type,
    season) MEANS; pitch_calibration fits its own agg_mu/agg_sigma from the
    real per-pitch stuff_run_value spread among reliable-arsenal pitches.

    Returns (pitch_level, pitcher_agg):
      pitch_level : one row per in-scope pitch, with pitch_stuff_plus.
      pitcher_agg : one row per (pitcher, pitch_type, season), with
                    stuff_plus and a `reliable` flag (>= MIN_PITCHES_FOR_SCORE).
    """

    has_score = scored.dropna(subset=["stuff_run_value"]).copy()

    pitcher_agg = (
        has_score
        .groupby([PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], observed=True)["stuff_run_value"]
        .agg(mean_stuff_value="mean", n_pitches="count")
        .reset_index()
    )
    reliable_agg = pitcher_agg[pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE].copy()

    agg_calibration = _ratio_calibration(reliable_agg, [PITCH_TYPE_COL, SEASON_COL], "mean_stuff_value")
    pitcher_agg = pitcher_agg.merge(agg_calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    pitcher_agg["stuff_plus"] = _to_100_scale(pitcher_agg["mean_stuff_value"], pitcher_agg)
    pitcher_agg["reliable"] = pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE

    # Pitch-level calibration must be fit on the spread of raw, per-pitch
    # stuff_run_value among pitches belonging to a reliable (pitcher,
    # pitch_type, season) -- NOT the spread of that group's own mean above,
    # which is far narrower (an aggregate is a mean over many pitches).
    reliable_keys = reliable_agg[[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL]]
    pitch_level_reliable = has_score.merge(reliable_keys, on=[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], how="inner")
    pitch_calibration = _ratio_calibration(pitch_level_reliable, [PITCH_TYPE_COL, SEASON_COL], "stuff_run_value")

    # merge() resets the index, but add_stuff_plus reattaches pitch_stuff_plus
    # to `result` positionally via has_score.index. Restore it (left merge on
    # a unique key preserves row order, so this is a straight relabel, not a
    # reshuffle) -- the exact bug class fixed 9/2 in the PCA-era code.
    original_index = has_score.index
    has_score = has_score.merge(pitch_calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    has_score.index = original_index
    has_score["pitch_stuff_plus"] = _to_100_scale(has_score["stuff_run_value"], has_score)

    return has_score, pitcher_agg


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_stuff_plus(raw_df, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    """
    Takes a raw Statcast dataframe and returns a copy with `stuff_run_value`
    (pitch-level, out-of-fold expected run value), `pitch_stuff_plus`
    (pitch-level, 100+ scaled), `stuff_plus` (pitcher x pitch_type x season
    aggregate, 100+ scaled, broadcast onto every pitch in that group), and
    `stuff_plus_reliable` columns added. Row count and order match the
    input; pitches out of scope get NaN in the new columns. Uses cached
    trained models under `models_dir` unless retrain=True or no cache
    exists yet.
    """

    missing = [col for col in REQUIRED_COLS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required columns: {missing}")

    # Select only the columns feature engineering/scoring needs before
    # filtering rows. raw_df has ~119 Statcast columns, and carrying all of
    # them through every intermediate step multiplies peak memory many times
    # over for no benefit, since only the new columns get reattached to
    # raw_df at the end.
    in_scope = (
        raw_df.loc[~raw_df[PITCH_TYPE_COL].isin(JUNK_PITCH_TYPES), REQUIRED_COLS]
        .dropna(subset=REQUIRED_COLS)
        .copy()
    )

    engineered = _build_v1_features(in_scope)
    scored = _load_or_score(engineered, models_dir, retrain)
    pitch_level, pitcher_agg = _calibrate(scored)

    result = raw_df.copy()

    for col in ["stuff_run_value", "pitch_stuff_plus"]:
        result[col] = np.nan
        result.loc[pitch_level.index, col] = pitch_level[col].to_numpy()

    agg_cols = pitcher_agg[
        [PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, "stuff_plus", "reliable"]
    ].rename(columns={"reliable": "stuff_plus_reliable"})

    result = result.merge(agg_cols, on=[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], how="left")

    return result


if __name__ == "__main__":
    import argparse

    default_input = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
    default_output = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_stuff.csv"

    parser = argparse.ArgumentParser(description="Add Stuff+ columns to raw Statcast data.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--retrain", action="store_true", help="Retrain Stuff+ models instead of using the cache")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_stuff_plus(raw_df, models_dir=args.models_dir, retrain=args.retrain)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["pitch_stuff_plus"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a Stuff+ score.")
