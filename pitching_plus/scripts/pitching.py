"""
Pitching+: a jointly-trained model predicting a pitch's expected run value
directly from its combined physics AND location/count/situational features
(one HistGradientBoostingRegressor per pitch type on
STUFF_FEATURES + LOCATION_FEATURES against delta_pitcher_run_exp), not a
weighted blend of separately-computed Stuff+ and Location+ scores.

Retooled 9/4/2026 from an earlier weighted-blend design (an intercept and
two z-scored WLS slopes on log(stuff_plus) and mean location_run_value --
see docs/dev_log.md's 8/26 and 8/31 entries for its original derivation).
That design was itself empirically justified at the time:
notebooks/fg_pitching.ipynb found the blend generalized better than a
jointly-trained alternative on a strict 2021-2024-train/2025-test holdout
(blend holdout R^2 0.052 vs. joint 0.039). That comparison turned out to be
leaking 2025 outcome data into the blend's Stuff+ input once Stuff+ was
itself retooled into a trained model (stuff.py's own 9/4 retool; see
docs/dev_log.md's second 9/4 entry) -- fixing the leak reversed the result
(blend 0.035 vs. joint 0.045), and a follow-up leave-one-season-out check
confirmed the reversal holds in every one of the 5 available seasons (joint
model wins 5/5 folds, mean holdout R^2 gap +0.0104; see docs/dev_log.md's
third 9/4 entry for the full validation). The 8/26/8/31 "blend wins" verdict
is superseded by this change, not deleted -- see dev_log.md for the full
history of both designs.

Public entry point is `add_pitching_plus`, which takes a raw Statcast
dataframe and returns it with:
  - `pitching_run_value`    : pitch-level expected run value (runs), positive
                              favors the pitcher, from out-of-fold CV
                              prediction (never the pitch's own realized
                              outcome or an in-sample fit)
  - `pitch_pitching_plus`   : pitch-level, 100+ scaled (per pitch_type/season)
  - `pitching_plus`         : pitcher x pitch_type x season aggregate, 100+
                              scaled (the headline figure, matching Stuff+'s
                              per-pitch-type reporting level)
  - `pitching_plus_reliable`: whether that pitcher-pitch-type-season met
                              the MIN_PITCHES_FOR_SCORE bar
Row count and order are unchanged; pitches out of scope get NaN in the new
columns.

Calls add_stuff_plus/add_location_plus itself if their columns aren't already
present -- still needed even though this module no longer blends their
outputs, since bestpitch.py's arsenal-reliability gate depends on
stuff_plus/stuff_plus_reliable independent of how Pitching+ itself is
architected, and full_pipeline.py/other callers may rely on
location_run_value/pitch_location_plus being present as a side effect of
calling add_pitching_plus standalone.

Training the per-pitch-type gradient-boosted models with 5-fold
cross-validation is the expensive step, so trained models plus the
out-of-fold historical scores are cached under pitching_plus/models/
(joblib, gitignored), exactly like stuff.py/location.py. A normal
add_pitching_plus call loads the cache instead of retraining. Pass
retrain=True (or run `python pitching.py --retrain`) to rebuild it after the
underlying data or JOINT_FEATURES/HGB_PARAMS change.

_ratio_calibration/_to_100_scale are reused directly from location.py, not
duplicated: pitching.py sits downstream of both stuff.py and location.py
with no circular-import constraint, unlike stuff.py itself (which had to
keep a local copy). This also means Pitching+'s ratio-scale steepness is
tied to Location+'s hardcoded LOCATION_SCALE_K, not independently tunable
(a dead PITCHING_SCALE_K constant of that name was removed 9/2, see
docs/dev_log.md).
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

try:
    from . import location as location_mod
    from . import stuff as stuff_mod
    from .location import (
        DEFAULT_MODELS_DIR, LOCATION_FEATURES, BASE_STATE_COLS, KEY_COLS,
        TARGET_COL, _ratio_calibration, _to_100_scale, add_location_plus,
    )
    from .stuff import (
        JUNK_PITCH_TYPES, MIN_PITCHES_FOR_SCORE, PITCHER_COL, PITCH_TYPE_COL,
        SEASON_COL, STUFF_FEATURES, add_stuff_plus,
    )
except ImportError:
    import location as location_mod
    import stuff as stuff_mod
    from location import (
        DEFAULT_MODELS_DIR, LOCATION_FEATURES, BASE_STATE_COLS, KEY_COLS,
        TARGET_COL, _ratio_calibration, _to_100_scale, add_location_plus,
    )
    from stuff import (
        JUNK_PITCH_TYPES, MIN_PITCHES_FOR_SCORE, PITCHER_COL, PITCH_TYPE_COL,
        SEASON_COL, STUFF_FEATURES, add_stuff_plus,
    )

# ============================================================
# CONFIGURATION
# ============================================================

# Union of both modules' own REQUIRED_COLS: the joint model's training input
# is the raw engineered physics + location/count columns (STUFF_FEATURES +
# LOCATION_FEATURES), not either module's calibrated scalar output, so this
# module needs everything both feature-engineering steps need.
REQUIRED_COLS = sorted(set(stuff_mod.REQUIRED_COLS) | set(location_mod.REQUIRED_COLS))

JOINT_FEATURES = STUFF_FEATURES + LOCATION_FEATURES

MIN_GROUP_SIZE_FOR_MODEL = 5000
N_FOLDS = 5

HGB_PARAMS = dict(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42)


# ============================================================
# MODEL TRAINING (expensive, cached)
# ============================================================

def _train_models(df):
    """
    Fits one gradient-boosted regressor per pitch type against TARGET_COL,
    on the combined JOINT_FEATURES set. Identical structure/contract to
    stuff.py's/location.py's own _train_models: models are refit on ALL
    in-scope rows per type (for scoring pitches outside the training set
    later, including bestpitch.py's counterfactual candidates);
    historical_scores holds KEY_COLS + pitching_run_value from 5-fold
    out-of-fold CV, never the same-data fit, so a pitch's own outcome never
    leaks into its own score.
    """

    models = {}
    oof = pd.Series(np.nan, index=df.index, dtype=float)

    for ptype, type_group in df.groupby(PITCH_TYPE_COL, observed=True):

        if len(type_group) < MIN_GROUP_SIZE_FOR_MODEL:
            continue

        X = type_group[JOINT_FEATURES].to_numpy()
        y = type_group[TARGET_COL].to_numpy()

        model = HistGradientBoostingRegressor(**HGB_PARAMS)

        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof.loc[type_group.index] = cross_val_predict(model, X, y, cv=kfold, n_jobs=-1)

        model.fit(X, y)
        models[ptype] = model

    scored_mask = oof.notna()
    historical_scores = df.loc[scored_mask, KEY_COLS].copy()
    historical_scores["pitching_run_value"] = oof.loc[scored_mask].to_numpy()

    return models, historical_scores


def load_cached_models(models_dir=DEFAULT_MODELS_DIR):
    """
    Loads just the cached per-pitch-type joint models, mirroring
    stuff.py's/location.py's function of the same name. Used by
    bestpitch.py's counterfactual search to score hypothetical
    (pitch_type, zone) combinations: it predicts from these models directly
    (with its own batching/dedup logic) and calibrates the result via
    _calibrate_raw_value.
    """

    models_path = Path(models_dir) / "pitching_models.joblib"
    if not models_path.exists():
        raise FileNotFoundError(
            f"No cached Pitching+ models at {models_path}. Run add_pitching_plus "
            "(or `python pitching.py --retrain`) at least once first."
        )
    return joblib.load(models_path)


def _cache_fingerprint():
    """
    Everything that changes what a cached model/score actually means: the
    joint feature set and the per-pitch-type training config. Saved
    alongside the cache so a stale cache (built under a different
    JOINT_FEATURES or HGB_PARAMS) can be detected instead of silently
    reused. A cache saved before this fingerprint existed has no sidecar
    file; that's treated as "unknown," not "stale."
    """

    return {
        "joint_features": list(JOINT_FEATURES),
        "hgb_params": dict(HGB_PARAMS),
        "n_folds": N_FOLDS,
        "min_group_size_for_model": MIN_GROUP_SIZE_FOR_MODEL,
    }


def _load_or_score(engineered, models_dir, retrain):
    models_dir = Path(models_dir)
    models_path = models_dir / "pitching_models.joblib"
    scores_path = models_dir / "pitching_historical_scores.joblib"
    fingerprint_path = models_dir / "pitching_cache_fingerprint.joblib"

    if not retrain and models_path.exists() and scores_path.exists():
        if fingerprint_path.exists() and joblib.load(fingerprint_path) != _cache_fingerprint():
            warnings.warn(
                f"Pitching+ cache at {models_dir} was built under a different "
                "JOINT_FEATURES/HGB_PARAMS configuration than the current code. "
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
    missing = scored["pitching_run_value"].isna()
    for ptype, model in models.items():
        rows = missing & (scored[PITCH_TYPE_COL] == ptype)
        if rows.any():
            X = scored.loc[rows, JOINT_FEATURES].to_numpy()
            scored.loc[rows, "pitching_run_value"] = model.predict(X)

    return scored


# ============================================================
# 100+ SCALE CALIBRATION
# ============================================================

def _calibrate(scored):
    """
    Two calibrations, each fit on its OWN level's real spread -- the same
    aggregate-vs-pitch-level split stuff.py's/location.py's own _calibrate
    already gets right (a bug class fixed in location.py/pitching.py on 9/2
    and 8/30 respectively, implemented correctly here from day one since
    this is a from-scratch rewrite). agg_calibration fits agg_mu/agg_sigma
    from the spread of per-(pitcher, pitch_type, season) MEANS;
    pitch_calibration fits its own agg_mu/agg_sigma from the real per-pitch
    pitching_run_value spread among reliable arsenals.

    Returns (pitch_level, pitcher_agg, pitch_calibration):
      pitch_level : one row per in-scope pitch, with pitch_pitching_plus.
      pitcher_agg : one row per (pitcher, pitch_type, season), with
                    pitching_plus and a `reliable` flag (>= MIN_PITCHES_FOR_SCORE).
      pitch_calibration : the pitch-level 100+ calibration table itself (not
                    just applied) -- bestpitch.py calls this function
                    directly on already-scored real data to obtain it, then
                    reuses it via _calibrate_raw_value to score hypothetical
                    (pitch_type, zone) combinations on the identical scale
                    as real pitches.
    """

    has_score = scored.dropna(subset=["pitching_run_value"]).copy()

    pitcher_agg = (
        has_score
        .groupby([PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], observed=True)["pitching_run_value"]
        .agg(mean_pitching_value="mean", n_pitches="count")
        .reset_index()
    )
    reliable_agg = pitcher_agg[pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE].copy()

    agg_calibration = _ratio_calibration(reliable_agg, [PITCH_TYPE_COL, SEASON_COL], "mean_pitching_value")
    pitcher_agg = pitcher_agg.merge(agg_calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    pitcher_agg["pitching_plus"] = _to_100_scale(pitcher_agg["mean_pitching_value"], pitcher_agg)
    pitcher_agg["reliable"] = pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE

    # Pitch-level calibration must be fit on the spread of raw, per-pitch
    # pitching_run_value among pitches belonging to a reliable (pitcher,
    # pitch_type, season) -- NOT the spread of that group's own mean above,
    # which is far narrower (an aggregate is a mean over many pitches).
    reliable_keys = reliable_agg[[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL]]
    pitch_level_reliable = has_score.merge(reliable_keys, on=[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], how="inner")
    pitch_calibration = _ratio_calibration(pitch_level_reliable, [PITCH_TYPE_COL, SEASON_COL], "pitching_run_value")

    # merge() resets the index, but add_pitching_plus reattaches
    # pitch_pitching_plus to `result` positionally via has_score.index.
    # Restore it (left merge on a unique key preserves row order, so this is
    # a straight relabel, not a reshuffle).
    original_index = has_score.index
    has_score = has_score.merge(pitch_calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    has_score.index = original_index
    has_score["pitch_pitching_plus"] = _to_100_scale(has_score["pitching_run_value"], has_score)

    return has_score, pitcher_agg, pitch_calibration


def _calibrate_raw_value(raw_pitching_value, pitch_type, season, calibration):
    """
    Applies an already-fitted 100+ calibration to raw pitching_run_value
    predictions, real or hypothetical (e.g. bestpitch.py's counterfactual
    target-averaged predictions, which compute the raw value themselves via
    their own batched model.predict() calls and only need this calibration
    step). `calibration` must be whichever of _calibrate's two tables
    matches the level being scored (the aggregate-level table for
    pitcher-season scoring, the pitch-level table for single-pitch
    scoring). Returns NaN wherever a (pitch_type, season) has no
    calibration (too few reliable pitcher-seasons to set a reference).

    Deliberately a plain pd.merge, not a
    `calibration.set_index([...]).reindex(keys)` MultiIndex lookup -- that
    looked like a reasonable optimization for bestpitch.py's counterfactual
    search but hung for 30+ minutes on the full dataset (see the identical
    note in the pre-9/4 version of this function, which this one carries
    forward unchanged).
    """

    frame = pd.DataFrame({
        PITCH_TYPE_COL: np.asarray(pitch_type),
        SEASON_COL: np.asarray(season),
        "raw_pitching_value": np.asarray(raw_pitching_value),
    })
    frame = frame.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    return _to_100_scale(frame["raw_pitching_value"], frame).to_numpy()


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_pitching_plus(raw_df, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    """
    Takes a raw Statcast dataframe and returns a copy with
    `pitching_run_value`, `pitch_pitching_plus`, `pitching_plus`, and
    `pitching_plus_reliable` columns added. Row count and order match the
    input; pitches out of scope get NaN in the new columns. Uses cached
    trained models under `models_dir` unless retrain=True or no cache
    exists yet. Calls add_stuff_plus/add_location_plus if their columns
    aren't already present (see module docstring for why that's still
    needed even though this module no longer blends their outputs).
    """

    missing = [col for col in REQUIRED_COLS + BASE_STATE_COLS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required columns: {missing}")

    if "pitch_stuff_plus" not in raw_df.columns or "stuff_plus" not in raw_df.columns:
        raw_df = add_stuff_plus(raw_df, models_dir=models_dir, retrain=retrain)
    if "location_run_value" not in raw_df.columns:
        raw_df = add_location_plus(raw_df, models_dir=models_dir, retrain=retrain)

    # Select only the columns feature engineering/scoring needs before
    # filtering rows, same rationale as stuff.py/location.py: raw_df has
    # ~119 Statcast columns, and carrying all of them through every
    # intermediate step multiplies peak memory many times over for no
    # benefit, since only the new columns get reattached to raw_df at the end.
    in_scope = (
        raw_df.loc[~raw_df[PITCH_TYPE_COL].isin(JUNK_PITCH_TYPES), REQUIRED_COLS + BASE_STATE_COLS]
        .dropna(subset=REQUIRED_COLS)
        .copy()
    )

    # Joint-model training input is the raw engineered physics + location
    # columns (JOINT_FEATURES), not either module's own calibrated scalar
    # output, so both feature-engineering steps run directly here rather
    # than going through add_stuff_plus/add_location_plus's own pipelines.
    engineered = stuff_mod._build_v1_features(in_scope)
    engineered = location_mod._build_features(engineered)

    scored = _load_or_score(engineered, models_dir, retrain)
    pitch_level, pitcher_agg, _pitch_calibration = _calibrate(scored)

    result = raw_df.copy()

    for col in ["pitching_run_value", "pitch_pitching_plus"]:
        result[col] = np.nan
        result.loc[pitch_level.index, col] = pitch_level[col].to_numpy()

    agg_cols = pitcher_agg[
        [PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, "pitching_plus", "reliable"]
    ].rename(columns={"reliable": "pitching_plus_reliable"})

    result = result.merge(agg_cols, on=[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], how="left")

    return result


if __name__ == "__main__":
    import argparse

    default_input = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
    default_output = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_pitching.csv"

    parser = argparse.ArgumentParser(description="Add Pitching+ columns to raw Statcast data.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--retrain", action="store_true", help="Retrain Stuff+/Location+/Pitching+ models instead of using the cache")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_pitching_plus(raw_df, models_dir=args.models_dir, retrain=args.retrain)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["pitch_pitching_plus"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a Pitching+ score.")
