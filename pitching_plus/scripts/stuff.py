"""
Stuff+: a standardized physics "outlierness" index for individual pitches,
not a trained model (see docs/purpose.md); no outcome data involved.

Public entry point is `add_stuff_plus`, which takes a raw Statcast dataframe
(as loaded by data/load_data.py) and returns it with `pitch_stuff_plus`
(pitch-level), `stuff_plus` (pitcher x pitch_type x season aggregate,
broadcast onto every pitch in that group), and `stuff_plus_reliable` columns
added. Row count and order are unchanged; pitches out of scope (junk pitch
types, incomplete physics data, or pitch types too rare to calibrate) get
NaN in the new columns.

Ported from notebooks/stuff.ipynb. See that notebook and docs/dev_log.md for
the exploration behind these choices, including why the V2 arsenal-comparison
features (pairwise "_vs_" columns) are NOT used here: they don't change
Stuff+ rankings meaningfully (Spearman rho 0.998 overall, checked against
the full 2021-2025 dataset) but cost ~50s/13GB extra to build.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ============================================================
# CONFIGURATION
# ============================================================

PITCHER_COL = "pitcher"
PITCH_TYPE_COL = "pitch_type"
SEASON_COL = "game_year"
PLAYER_NAME_COL = "player_name"

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
]

REQUIRED_COLS = [PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, PLAYER_NAME_COL] + PHYSICS_COLS

# Distance (feet from release) used for "reaction time" in
# movement_per_reaction_time -- purpose.md's "Reaction x Movement" concept.
# stuff.ipynb's exploration notebook solves full trajectories (position,
# velocity, spin_axis_sin/cos, etc.) at several distances for its V2
# arsenal-comparison features; only this one distance's TIME is read by
# STUFF_FEATURES here, so that's all the production script builds.
REACTION_DISTANCE_FT = 30

# Features that go into the per-pitch PCA composite. Deduped to avoid
# redundant features inflating PCA weights (e.g. reaction time is r=0.99
# with velocity); see docs/dev_log.md 8/26 entry.
STUFF_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "acceleration_mag",
    "horizontal_acceleration",
    "movement_per_reaction_time",
]

# A pitch type needs at least this many pitches (pooled across all
# pitchers/seasons) before PCA is fit for it at all.
MIN_GROUP_SIZE_FOR_PCA = 5000

# Minimum pitches in a (pitcher, pitch_type, season) group for its average to
# set the league reference distribution / be reported as "reliable."
MIN_PITCHES_FOR_SCORE = 20

# Controls how many points one SD of stuff is worth on the final scale
# (0.10 -> roughly +/-10 points per SD, in line with typical "+" stat spreads).
STUFF_SCALE_K = 0.10


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
    # window available (purpose.md's explicit concept), and the single
    # highest-loading PC1 feature across every pitch type.
    df["movement_per_reaction_time"] = df["horizontal_acceleration"] / df["time_30ft"]

    return df


# ============================================================
# STUFF+ SCORING: PCA COMPOSITE -> 100+ SCALE
# ============================================================

def _score_stuff_plus(df):
    """
    Per pitch type: z-score STUFF_FEATURES within (pitch_type, season), let
    PCA pick weights (PC1, sign-anchored so higher release_speed loads
    positively), then convert to a genuine ratio scale (exp(k*z), renormalized
    so the league average for that pitch type/season is exactly 100).

    Returns (stuff_df, pitcher_agg):
      stuff_df    : one row per in-scope pitch, with pitch_stuff_plus.
      pitcher_agg : one row per (pitcher, pitch_type, season), with
                    stuff_plus and a `reliable` flag (>= MIN_PITCHES_FOR_SCORE).
    """

    stuff_df = df[np.isfinite(df["movement_per_reaction_time"])].copy()
    stuff_df["pitch_composite"] = np.nan

    for ptype, type_group in stuff_df.groupby(PITCH_TYPE_COL, observed=True):

        if len(type_group) < MIN_GROUP_SIZE_FOR_PCA:
            continue

        # z-score within (pitch_type, season) so "average" means that season's average
        season_mu = type_group.groupby(SEASON_COL)[STUFF_FEATURES].transform("mean")
        season_sigma = type_group.groupby(SEASON_COL)[STUFF_FEATURES].transform("std")
        Z = (type_group[STUFF_FEATURES] - season_mu) / season_sigma

        # PCA weights fit pooled across seasons: more stable than fitting per season
        pca = PCA(n_components=len(STUFF_FEATURES))
        pca.fit(Z.to_numpy())

        loadings = pca.components_[0]
        if loadings[STUFF_FEATURES.index("release_speed")] < 0:
            loadings = -loadings

        stuff_df.loc[type_group.index, "pitch_composite"] = Z.to_numpy() @ loadings

    # Aggregate to (pitcher, pitch_type, season). Stuff+ is reported at
    # this level, matching how real "+" stats (wRC+, ERA-) work.
    # PLAYER_NAME_COL deliberately isn't in the groupby key (or read at all
    # downstream -- add_stuff_plus's final merge never selects it): folding
    # it in added a latent risk of duplicating rows in that merge if any
    # (pitcher, pitch_type, season) ever had more than one distinct
    # player_name string in the raw data (encoding variants, mid-season name
    # corrections), for no benefit. See docs/dev_log.md's 9/2 entry.
    pitcher_agg = (
        stuff_df
        .dropna(subset=["pitch_composite"])
        .groupby([PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], observed=True)["pitch_composite"]
        .agg(mean_composite="mean", n_pitches="count")
        .reset_index()
    )

    # Only "reliable" (large enough sample) pitcher-seasons set the reference:
    # a tiny sample's own mean is noisy and would distort the league average/SD.
    reliable = pitcher_agg[pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE].copy()

    calibration = (
        reliable
        .groupby([PITCH_TYPE_COL, SEASON_COL])["mean_composite"]
        .agg(agg_mu="mean", agg_sigma="std")
        .reset_index()
    )
    reliable = reliable.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    reliable["agg_z"] = (reliable["mean_composite"] - reliable["agg_mu"]) / reliable["agg_sigma"]
    reliable["raw_ratio"] = np.exp(STUFF_SCALE_K * reliable["agg_z"])

    # raw_ratio_mean anchors the scale to exactly 100, computed from the SAME
    # reliable pitcher-seasons only, so unreliable small samples can't skew it.
    raw_ratio_mean = (
        reliable
        .groupby([PITCH_TYPE_COL, SEASON_COL])["raw_ratio"]
        .mean()
        .rename("raw_ratio_mean")
        .reset_index()
    )
    calibration = calibration.merge(raw_ratio_mean, on=[PITCH_TYPE_COL, SEASON_COL], how="left")

    # Apply the same calibration to both levels: shared scale, shared "100".
    pitcher_agg = pitcher_agg.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    pitcher_agg["stuff_plus"] = 100 * np.exp(
        STUFF_SCALE_K * (pitcher_agg["mean_composite"] - pitcher_agg["agg_mu"]) / pitcher_agg["agg_sigma"]
    ) / pitcher_agg["raw_ratio_mean"]
    pitcher_agg["reliable"] = pitcher_agg["n_pitches"] >= MIN_PITCHES_FOR_SCORE

    # merge() resets the index, but add_stuff_plus reattaches pitch_stuff_plus to
    # `result` positionally via stuff_df.index. Restore it (left merge on a
    # unique key preserves row order, so this is a straight relabel, not a
    # reshuffle).
    original_index = stuff_df.index
    stuff_df = stuff_df.merge(calibration, on=[PITCH_TYPE_COL, SEASON_COL], how="left")
    stuff_df.index = original_index
    stuff_df["pitch_stuff_plus"] = 100 * np.exp(
        STUFF_SCALE_K * (stuff_df["pitch_composite"] - stuff_df["agg_mu"]) / stuff_df["agg_sigma"]
    ) / stuff_df["raw_ratio_mean"]

    return stuff_df, pitcher_agg


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def add_stuff_plus(raw_df):
    """
    Takes a raw Statcast dataframe and returns a copy with `pitch_stuff_plus`
    (pitch-level score), `stuff_plus` (pitcher x pitch_type x season
    aggregate, broadcast onto every pitch in that group), and
    `stuff_plus_reliable` columns added. Row count and order match the input;
    pitches out of scope get NaN in the new columns.
    """

    missing = [col for col in REQUIRED_COLS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required columns: {missing}")

    # Select only the columns feature engineering needs before filtering rows.
    # raw_df has ~119 Statcast columns, and carrying all of them through
    # every intermediate step (rather than just REQUIRED_COLS) multiplies
    # peak memory many times over for no benefit, since only pitch_stuff_plus
    # gets reattached to raw_df at the end anyway.
    in_scope = (
        raw_df.loc[~raw_df[PITCH_TYPE_COL].isin(JUNK_PITCH_TYPES), REQUIRED_COLS]
        .dropna(subset=REQUIRED_COLS)
        .copy()
    )

    engineered = _build_v1_features(in_scope)
    stuff_df, pitcher_agg = _score_stuff_plus(engineered)

    result = raw_df.copy()

    result["pitch_stuff_plus"] = np.nan
    result.loc[stuff_df.index, "pitch_stuff_plus"] = stuff_df["pitch_stuff_plus"].to_numpy()

    pitcher_agg_cols = pitcher_agg[
        [PITCHER_COL, PITCH_TYPE_COL, SEASON_COL, "stuff_plus", "reliable"]
    ].rename(columns={"reliable": "stuff_plus_reliable"})

    result = result.merge(pitcher_agg_cols, on=[PITCHER_COL, PITCH_TYPE_COL, SEASON_COL], how="left")

    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    default_input = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
    default_output = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_stuff.csv"

    parser = argparse.ArgumentParser(description="Add Stuff+ columns to raw Statcast data.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    print(f"Scoring {len(raw_df):,} pitches ...")
    result = add_stuff_plus(raw_df)

    print(f"Writing {args.output} ...")
    result.to_csv(args.output, index=False)

    n_scored = result["pitch_stuff_plus"].notna().sum()
    print(f"Done. {n_scored:,} / {len(result):,} pitches received a Stuff+ score.")
