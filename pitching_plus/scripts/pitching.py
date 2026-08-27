"""
Pitching+: a weighted blend of Stuff+ and Location+ (see docs/purpose.md's
"reward maximized Location+ w/ weight for Stuff+"), not a jointly-trained
model on raw physics/location features.

This choice is empirically justified, not just purpose.md's original framing:
notebooks/pitching.ipynb found the blend a real, generalizing (season-holdout
validated) relationship, and notebooks/fg_pitching.ipynb head-to-head compared
it against FanGraphs' actual approach (one HistGradientBoostingRegressor per
pitch type on the combined feature set) on a strict 2021-2024-train/2025-test
split -- the blend generalized better (holdout R^2 0.052 vs. 0.039), and a
follow-up regularization sweep on the joint model narrowed but did not close
that gap. See docs/dev_log.md's 8/26-8/27 entries for the full derivation.

Public entry point is `add_pitching_plus`, which takes a raw Statcast
dataframe and returns it with:
  - `pitch_pitching_plus` -- pitch-level, 100+ scaled (per pitch_type/season)
  - `pitching_plus`       -- pitcher x pitch_type x season aggregate, 100+
                             scaled (the headline figure, matching Stuff+'s
                             per-pitch-type reporting level)
  - `pitching_plus_reliable` -- whether that pitcher-pitch-type-season met
                                 the MIN_PITCHES_FOR_SCORE bar

Calls add_stuff_plus/add_location_plus itself if their columns aren't already
present, so it can run standalone; full_pipeline.py's existing stuff+location
calls are reused as-is (no recomputation) when chained together.

The blend weights (an intercept and two z-scored slopes on log(Stuff+) and
mean Location+ run value) are fit fresh on every call via WLS -- like
stuff.py's PCA, this is cheap enough (a few thousand aggregate rows) not to
need location.py's model-caching machinery.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from .location import TARGET_COL, DEFAULT_MODELS_DIR, _ratio_calibration, _to_100_scale, add_location_plus
    from .stuff import MIN_PITCHES_FOR_SCORE, PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, add_stuff_plus
except ImportError:
    from location import TARGET_COL, DEFAULT_MODELS_DIR, _ratio_calibration, _to_100_scale, add_location_plus
    from stuff import MIN_PITCHES_FOR_SCORE, PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, add_stuff_plus

# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_COLS = [PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, TARGET_COL]

PITCHING_SCALE_K = 0.10


# ============================================================
# BLEND: FIT WEIGHTS, SCORE BOTH LEVELS
#
# Split into a fit step (_fit_pitching_plus_model) and a pure scoring step
# (_score_pitching_plus) -- not just for add_pitching_plus's own two levels,
# but so bestpitch.py can score *hypothetical* (pitch_type, location)
# combinations on the exact same fitted blend/calibration, not a re-derived
# one.
# ============================================================

def _apply_blend(mean_stuff_plus, mean_location_run_value, blend_params):
    stuff_z = (np.log(mean_stuff_plus) - blend_params["log_stuff_mu"]) / blend_params["log_stuff_sigma"]
    loc_z = (mean_location_run_value - blend_params["loc_mu"]) / blend_params["loc_sigma"]
    return blend_params["intercept"] + blend_params["beta_stuff"] * stuff_z + blend_params["beta_loc"] * loc_z


def _fit_pitching_plus_model(df):
    """
    Fits one pooled WLS blend (not per pitch type -- notebooks/pitching.ipynb
    confirmed the same relationship holds within every pitch type individually,
    so pooling is a stability choice, not a pattern-hiding one) of z-scored
    log(stuff_plus) and mean location_run_value against realized run value, on
    the reliable (pitcher, pitch_type, season) aggregate population. Also fits
    the 100+ ratio-scale calibration on that same population's raw_pitching_value.

    Returns (blend_params, calibration, pitcher_agg):
      blend_params -- dict of fitted WLS coefficients + z-scoring mu/sigma,
                       sufficient to score any (mean_stuff_plus,
                       mean_location_run_value) pair, real or hypothetical.
      calibration   -- per (pitch_type, season) 100+ scale reference, from
                        location.py's own _ratio_calibration.
      pitcher_agg   -- one row per (pitcher, pitch_type, season), with
                        raw_pitching_value/pitching_plus and a `reliable` flag.
    """

    pitcher_agg = (
        df.groupby([PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], observed=True)
        .agg(
            mean_rv=(TARGET_COL, "mean"),
            mean_stuff_plus=("stuff_plus", "mean"),
            mean_location_run_value=("location_run_value", "mean"),
            n_pitches=(TARGET_COL, "count"),
        )
        .reset_index()
    )
    pitcher_agg["reliable"] = pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE
    reliable = pitcher_agg[pitcher_agg["reliable"]].copy()

    blend_params = {
        "log_stuff_mu": np.log(reliable["mean_stuff_plus"]).mean(),
        "log_stuff_sigma": np.log(reliable["mean_stuff_plus"]).std(),
        "loc_mu": reliable["mean_location_run_value"].mean(),
        "loc_sigma": reliable["mean_location_run_value"].std(),
        "intercept": 0.0,
        "beta_stuff": 0.0,
        "beta_loc": 0.0,
    }
    reliable["stuff_z"] = (np.log(reliable["mean_stuff_plus"]) - blend_params["log_stuff_mu"]) / blend_params["log_stuff_sigma"]
    reliable["loc_z"] = (reliable["mean_location_run_value"] - blend_params["loc_mu"]) / blend_params["loc_sigma"]

    model = sm.WLS(
        reliable["mean_rv"],
        sm.add_constant(reliable[["stuff_z", "loc_z"]]),
        weights=reliable["n_pitches"],
    ).fit()
    blend_params["intercept"] = model.params["const"]
    blend_params["beta_stuff"] = model.params["stuff_z"]
    blend_params["beta_loc"] = model.params["loc_z"]

    pitcher_agg["raw_pitching_value"] = _apply_blend(
        pitcher_agg["mean_stuff_plus"], pitcher_agg["mean_location_run_value"], blend_params
    )
    reliable = pitcher_agg[pitcher_agg["reliable"]]
    calibration = _ratio_calibration(reliable, [PITCH_TYPE_COL, SEASON_COL], "raw_pitching_value")

    pitcher_agg = pitcher_agg.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    pitcher_agg["pitching_plus"] = _to_100_scale(pitcher_agg["raw_pitching_value"], pitcher_agg)

    return blend_params, calibration, pitcher_agg


def _score_pitching_plus(mean_stuff_plus, mean_location_run_value, pitch_type, season, blend_params, calibration):
    """
    Applies an already-fitted blend + 100+ calibration to arbitrary
    (mean_stuff_plus, mean_location_run_value) inputs -- real or hypothetical
    (e.g. bestpitch.py's counterfactual pitch-type/zone combinations) --
    keyed by pitch_type/season for calibration lookup. Returns NaN wherever
    that (pitch_type, season) has no calibration (too few reliable
    pitcher-seasons to set a reference).
    """

    raw_pitching_value = _apply_blend(mean_stuff_plus, mean_location_run_value, blend_params)
    frame = pd.DataFrame(
        {PITCH_TYPE_COL: np.asarray(pitch_type), SEASON_COL: np.asarray(season), "raw_pitching_value": np.asarray(raw_pitching_value)}
    )
    frame = frame.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    return _to_100_scale(frame["raw_pitching_value"], frame).to_numpy()


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_pitching_plus(raw_df, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    """
    Takes a raw Statcast dataframe and returns a copy with `pitch_pitching_plus`,
    `pitching_plus`, and `pitching_plus_reliable` columns added. Row count and
    order match the input; pitches out of scope (no Stuff+ or Location+ score)
    get NaN in the new columns. Calls add_stuff_plus/add_location_plus itself
    if their columns aren't already present.
    """

    missing = [col for col in REQUIRED_COLS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required columns: {missing}")

    if "pitch_stuff_plus" not in raw_df.columns or "stuff_plus" not in raw_df.columns:
        raw_df = add_stuff_plus(raw_df)
    if "location_run_value" not in raw_df.columns:
        raw_df = add_location_plus(raw_df, models_dir=models_dir, retrain=retrain)

    in_scope = raw_df.dropna(subset=["pitch_stuff_plus", "stuff_plus", "location_run_value", TARGET_COL]).copy()

    blend_params, calibration, pitcher_agg = _fit_pitching_plus_model(in_scope)

    pitch_level = in_scope
    pitch_level["pitch_pitching_plus"] = _score_pitching_plus(
        pitch_level["stuff_plus"], pitch_level["location_run_value"],
        pitch_level[PITCH_TYPE_COL], pitch_level[SEASON_COL],
        blend_params, calibration,
    )

    result = raw_df.copy()

    result["pitch_pitching_plus"] = np.nan
    result.loc[pitch_level.index, "pitch_pitching_plus"] = pitch_level["pitch_pitching_plus"].to_numpy()

    agg_cols = pitcher_agg[[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, "pitching_plus", "reliable"]].rename(
        columns={"reliable": "pitching_plus_reliable"}
    )
    result = result.merge(agg_cols, on=[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], how="left")

    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    default_input = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
    default_output = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_pitching.csv"

    parser = argparse.ArgumentParser(description="Add Pitching+ columns to raw Statcast data.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--retrain", action="store_true", help="Retrain Location+ models instead of using the cache")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_pitching_plus(raw_df, models_dir=args.models_dir, retrain=args.retrain)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["pitch_pitching_plus"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a Pitching+ score.")
