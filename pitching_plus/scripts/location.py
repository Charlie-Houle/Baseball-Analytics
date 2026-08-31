"""
Location+: an actual trained model (unlike Stuff+, a standardized index with
no outcome data) that predicts a pitch's expected run value from where/when
it was thrown -- pitch type, location (batter-zone-relative, arm-side-
adjusted), RE288 situation (base-out state x count), and handedness. See
pitching_plus/notebooks/location.ipynb for the full derivation and validation.

Public entry point is `add_location_plus`, which takes a raw Statcast
dataframe and returns it with:
  - `location_run_value`     -- pitch-level expected run value (runs),
                                 positive favors the pitcher
  - `pitch_location_plus`    -- pitch-level, 100+ scaled (per pitch_type/season)
  - `location_plus`          -- pitcher x season aggregate across the WHOLE
                                 arsenal, 100+ scaled. This is the "final"
                                 command score, not the per-pitch-type
                                 breakdown -- a pitcher's Location+ for one
                                 pitch type off a handful of pitches is too
                                 easily dominated by a small sample (see
                                 docs/dev_log.md 8/26).
  - `location_plus_reliable` -- whether that pitcher-season met the season
                                 pitch-count bar (>= MIN_PITCHES_FOR_SEASON_SCORE)

Training the per-pitch-type gradient-boosted models with 5-fold
cross-validation is the expensive step (minutes on the full 2021-2025 data),
so trained models plus the out-of-fold historical scores are cached under
pitching_plus/models/ (joblib, gitignored). A normal `add_location_plus` call
loads the cache instead of retraining. Pass retrain=True (or run
`python location.py --retrain`) to rebuild it after the underlying data changes.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

try:
    # Package-relative import: works when this module is imported as
    # pitching_plus.scripts.location (e.g. from tests) without needing
    # scripts/ on sys.path.
    from .stuff import JUNK_PITCH_TYPES, MIN_PITCHES_FOR_SCORE, PITCH_TYPE_COL, PITCHER_COL, SEASON_COL
except ImportError:
    # Bare top-level import: works when this module is run directly
    # (`python location.py`) or loaded via sys.path.insert(scripts_dir), as
    # the notebooks do.
    from stuff import JUNK_PITCH_TYPES, MIN_PITCHES_FOR_SCORE, PITCH_TYPE_COL, PITCHER_COL, SEASON_COL

# ============================================================
# CONFIGURATION
# ============================================================

PLAYER_NAME_COL = "player_name"

# Uniquely identifies a single pitch -- used to reattach cached out-of-fold
# historical scores to a freshly loaded raw dataframe without retraining.
KEY_COLS = ["game_pk", "at_bat_number", "pitch_number"]

REQUIRED_COLS = [
    PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, PLAYER_NAME_COL,
    "balls", "strikes", "outs_when_up", "stand", "p_throws",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "delta_pitcher_run_exp",
] + KEY_COLS

# Must be present, but NOT required to be non-null -- NaN here means "base
# empty," a real game state, not a missing-data problem, so these are checked
# separately from REQUIRED_COLS (which gates the dropna below).
BASE_STATE_COLS = ["on_1b", "on_2b", "on_3b"]

LOCATION_FEATURES = [
    "plate_x", "plate_z_rel", "plate_x_armside",
    "balls", "strikes",
    "on_1b_occupied", "on_2b_occupied", "on_3b_occupied", "outs_when_up",
    "re288_state",
    "stand_R", "p_throws_R", "platoon_matchup",
]

TARGET_COL = "delta_pitcher_run_exp"

MIN_GROUP_SIZE_FOR_MODEL = 5000  # same scope gate as Stuff+
N_FOLDS = 5

LOCATION_SCALE_K = 0.10

# Season-overall reliability needs a much bigger sample than the per-pitch-type
# gate (MIN_PITCHES_FOR_SCORE) -- a few dozen pitches total in a season is too
# thin to call a real read on a pitcher's command.
MIN_PITCHES_FOR_SEASON_SCORE = 100

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _build_features(df):
    """
    Adds the RE288 situation state, batter-zone-relative and arm-side-adjusted
    location, and handedness features. Expects `df` to already be filtered to
    in-scope pitch types with complete REQUIRED_COLS (on_1b/on_2b/on_3b NaN
    means "base empty," a real game state, and is left untouched here).
    """

    df = df.copy()

    df["on_1b_occupied"] = df["on_1b"].notna().astype(int)
    df["on_2b_occupied"] = df["on_2b"].notna().astype(int)
    df["on_3b_occupied"] = df["on_3b"].notna().astype(int)

    # 8 base combos x 3 out counts = 24 base-out states;
    # 4 ball counts x 3 strike counts = 12 count states; 24 x 12 = 288 states.
    base_state = df["on_1b_occupied"] * 4 + df["on_2b_occupied"] * 2 + df["on_3b_occupied"]
    count_state = df["balls"] * 3 + df["strikes"]
    df["re288_state"] = (df["outs_when_up"] * 8 + base_state) * 12 + count_state

    # 0 = bottom of this batter's strike zone, 1 = top.
    df["plate_z_rel"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])

    # Positive = arm-side, negative = glove-side, regardless of batter stand.
    df["plate_x_armside"] = np.where(df["stand"] == "R", df["plate_x"], -df["plate_x"])

    df["stand_R"] = (df["stand"] == "R").astype(int)
    df["p_throws_R"] = (df["p_throws"] == "R").astype(int)
    df["platoon_matchup"] = (df["stand"] != df["p_throws"]).astype(int)

    return df


# ============================================================
# MODEL TRAINING (expensive -- cached)
# ============================================================

def _train_models(df):
    """
    Fits one gradient-boosted regressor per pitch type. Returns:
      models             -- {pitch_type: fitted HistGradientBoostingRegressor},
                             fit on ALL in-scope rows for that type (used to
                             score pitches outside this training set later).
      historical_scores  -- KEY_COLS + location_run_value for every in-scope
                             row, using out-of-fold (5-fold CV) predictions --
                             NOT the final model's own fit -- so a pitch's
                             "expected" value never leaks its own realized
                             outcome.
    """

    models = {}
    oof = pd.Series(np.nan, index=df.index, dtype=float)

    for ptype, type_group in df.groupby(PITCH_TYPE_COL, observed=True):

        if len(type_group) < MIN_GROUP_SIZE_FOR_MODEL:
            continue

        X = type_group[LOCATION_FEATURES].to_numpy()
        y = type_group[TARGET_COL].to_numpy()

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            random_state=42,
        )

        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof.loc[type_group.index] = cross_val_predict(model, X, y, cv=kfold, n_jobs=-1)

        model.fit(X, y)
        models[ptype] = model

    scored_mask = oof.notna()
    historical_scores = df.loc[scored_mask, KEY_COLS].copy()
    historical_scores["location_run_value"] = oof.loc[scored_mask].to_numpy()

    return models, historical_scores


def _load_or_score(engineered, models_dir, retrain):
    models_dir = Path(models_dir)
    models_path = models_dir / "location_models.joblib"
    scores_path = models_dir / "location_historical_scores.joblib"

    if not retrain and models_path.exists() and scores_path.exists():
        models = joblib.load(models_path)
        historical_scores = joblib.load(scores_path)
    else:
        models, historical_scores = _train_models(engineered)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, models_path, compress=3)
        joblib.dump(historical_scores, scores_path, compress=3)

    scored = engineered.merge(historical_scores, on=KEY_COLS, how="left")
    scored.index = engineered.index

    # Pitches not found in the historical cache (e.g. new data since the
    # models were last trained) fall back to direct model scoring -- the
    # normal, correct thing to do for a pitch the model hasn't seen before.
    missing = scored["location_run_value"].isna()
    for ptype, model in models.items():
        rows = missing & (scored[PITCH_TYPE_COL] == ptype)
        if rows.any():
            X = scored.loc[rows, LOCATION_FEATURES].to_numpy()
            scored.loc[rows, "location_run_value"] = model.predict(X)

    return scored


# ============================================================
# 100+ SCALE CALIBRATION
# ============================================================

def _ratio_calibration(reliable, group_cols, value_col):
    """
    league reference (agg_mu, agg_sigma) plus the ratio-mean anchor, computed
    only from the reliable rows -- same genuine-ratio-scale approach as
    Stuff+ (100 * exp(k*z), renormalized so the reliable population's mean is
    exactly 100).
    """

    calibration = (
        reliable.groupby(group_cols)[value_col]
        .agg(agg_mu="mean", agg_sigma="std")
        .reset_index()
    )
    merged = reliable.merge(calibration, on=group_cols, how="left")
    z = (merged[value_col] - merged["agg_mu"]) / merged["agg_sigma"]
    merged["raw_ratio"] = np.exp(LOCATION_SCALE_K * z)

    raw_ratio_mean = (
        merged.groupby(group_cols)["raw_ratio"]
        .mean()
        .rename("raw_ratio_mean")
        .reset_index()
    )
    return calibration.merge(raw_ratio_mean, on=group_cols, how="left")


def _to_100_scale(value, calibrated_frame):
    return 100 * np.exp(
        LOCATION_SCALE_K * (value - calibrated_frame["agg_mu"]) / calibrated_frame["agg_sigma"]
    ) / calibrated_frame["raw_ratio_mean"]


def _calibrate(scored):
    has_score = scored.dropna(subset=["location_run_value"]).copy()

    # ---- pitch-level scale, calibrated per (pitch_type, season) ----
    pitcher_type_agg = (
        has_score
        .groupby([PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], observed=True)["location_run_value"]
        .agg(mean_location_value="mean", n_pitches="count")
        .reset_index()
    )
    reliable_type = pitcher_type_agg[pitcher_type_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE].copy()
    type_calibration = _ratio_calibration(reliable_type, [PITCH_TYPE_COL, SEASON_COL], "mean_location_value")

    # merge() resets the index, but add_location_plus reattaches this frame's
    # columns to `result` positionally via has_score.index -- restore it (left
    # merge on a unique key preserves row order, so this is a straight
    # relabel, not a reshuffle).
    original_index = has_score.index
    has_score = has_score.merge(type_calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    has_score.index = original_index
    has_score["pitch_location_plus"] = _to_100_scale(has_score["location_run_value"], has_score)
    has_score = has_score.drop(columns=["agg_mu", "agg_sigma", "raw_ratio_mean"])

    # ---- season-overall scale, across the whole arsenal ----
    pitcher_season_agg = (
        has_score
        .groupby([PITCHER_COL, SEASON_COL], observed=True)["location_run_value"]
        .agg(mean_location_value="mean", n_pitches="count")
        .reset_index()
    )
    reliable_season = pitcher_season_agg[pitcher_season_agg["n_pitches"] >= MIN_PITCHES_FOR_SEASON_SCORE].copy()
    season_calibration = _ratio_calibration(reliable_season, [SEASON_COL], "mean_location_value")

    pitcher_season_agg = pitcher_season_agg.merge(season_calibration, on=[SEASON_COL], how="left")
    pitcher_season_agg["location_plus"] = _to_100_scale(pitcher_season_agg["mean_location_value"], pitcher_season_agg)
    pitcher_season_agg["location_plus_reliable"] = pitcher_season_agg["n_pitches"] >= MIN_PITCHES_FOR_SEASON_SCORE

    return has_score, pitcher_season_agg


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_location_plus(raw_df, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    """
    Takes a raw Statcast dataframe and returns a copy with `location_run_value`,
    `pitch_location_plus`, `location_plus`, and `location_plus_reliable`
    columns added. Row count and order match the input; pitches out of scope
    get NaN in the new columns. Uses cached trained models under `models_dir`
    unless retrain=True or no cache exists yet.
    """

    missing = [col for col in REQUIRED_COLS + BASE_STATE_COLS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required columns: {missing}")

    # Select only the columns feature engineering/scoring needs before
    # filtering rows -- raw_df has ~119 Statcast columns, and carrying all of
    # them through every intermediate step (rather than just REQUIRED_COLS +
    # BASE_STATE_COLS) multiplies peak memory many times over for no benefit,
    # since only the four new columns get reattached to raw_df at the end.
    in_scope = (
        raw_df.loc[~raw_df[PITCH_TYPE_COL].isin(JUNK_PITCH_TYPES), REQUIRED_COLS + BASE_STATE_COLS]
        .dropna(subset=REQUIRED_COLS)
        .copy()
    )

    engineered = _build_features(in_scope)
    scored = _load_or_score(engineered, models_dir, retrain)
    pitch_level, pitcher_season = _calibrate(scored)

    result = raw_df.copy()

    for col in ["location_run_value", "pitch_location_plus"]:
        result[col] = np.nan
        result.loc[pitch_level.index, col] = pitch_level[col].to_numpy()

    season_cols = pitcher_season[[PITCHER_COL, SEASON_COL, "location_plus", "location_plus_reliable"]]
    result = result.merge(season_cols, on=[PITCHER_COL, SEASON_COL], how="left")

    return result


if __name__ == "__main__":
    import argparse

    default_input = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
    default_output = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_location.csv"

    parser = argparse.ArgumentParser(description="Add Location+ columns to raw Statcast data.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--retrain", action="store_true", help="Retrain models instead of using the cache")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_location_plus(raw_df, models_dir=args.models_dir, retrain=args.retrain)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["location_run_value"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a Location+ score.")
