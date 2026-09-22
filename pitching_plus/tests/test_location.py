import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import location


def test_re288_state_known_combinations():
    df = pd.DataFrame(
        {
            "on_1b": [np.nan, 1.0, 1.0],
            "on_2b": [np.nan, 1.0, 1.0],
            "on_3b": [np.nan, 1.0, 1.0],
            "outs_when_up": [0, 2, 2],
            "balls": [0, 3, 3],
            "strikes": [0, 2, 2],
            "stand": ["R", "R", "R"],
            "plate_z": [2.5, 2.5, 2.5],
            "sz_top": [3.5, 3.5, 3.5],
            "sz_bot": [1.5, 1.5, 1.5],
            "p_throws": ["R", "R", "R"],
            "plate_x": [0.0, 0.0, 0.0],
        }
    )
    engineered = location._build_features(df)
    # bases empty, 0 outs, 0-0 count -> state 0
    assert engineered.loc[0, "re288_state"] == 0
    # bases loaded, 2 outs, 3-2 count -> the maximum state, 287
    assert engineered.loc[1, "re288_state"] == 287
    assert engineered.loc[2, "re288_state"] == 287


def test_plate_z_rel_and_armside():
    # Regression test for a real sign-convention bug (docs/dev_log.md 9/2
    # entry): plate_x_armside used to be keyed on batter `stand`, which only
    # gave the documented sign for same-handed matchups and was inverted for
    # platoon ones. Arm-side is defined by the PITCHER's throwing hand: a
    # RHP's own arm-side run tails toward a RHH (who stands on the
    # third-base/negative-plate_x side), so a RHP's arm side is negative
    # plate_x regardless of the batter. Four rows here hold plate_x fixed
    # and vary both stand and p_throws independently, so a test that only
    # checks one platoon matchup (as the pre-fix version of this test did)
    # can't tell a stand-keyed bug from a p_throws-keyed fix -- this can.
    df = pd.DataFrame(
        {
            "on_1b": [np.nan] * 4, "on_2b": [np.nan] * 4, "on_3b": [np.nan] * 4,
            "outs_when_up": [0] * 4, "balls": [0] * 4, "strikes": [0] * 4,
            "plate_z": [2.5] * 4, "sz_top": [3.5] * 4, "sz_bot": [1.5] * 4,
            "plate_x": [0.7, 0.7, 0.7, 0.7],
            "stand": ["L", "R", "L", "R"],
            "p_throws": ["R", "R", "L", "L"],
        }
    )
    engineered = location._build_features(df)
    np.testing.assert_allclose(engineered["plate_z_rel"].to_numpy(), 0.5)
    # Keyed only on p_throws (R, R, L, L) -- identical across the two
    # different `stand` values in rows 0-1 and rows 2-3, confirming it's
    # independent of batter stand, not just differently signed.
    np.testing.assert_allclose(engineered["plate_x_armside"].to_numpy(), [-0.7, -0.7, 0.7, 0.7])
    assert engineered["platoon_matchup"].tolist() == [1, 0, 0, 1]


def test_ratio_calibration_and_to_100_scale_mean_100():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "pitch_type": ["FF"] * 100,
            "game_year": [2024] * 100,
            "value": rng.normal(0.0, 1.0, size=100),
        }
    )
    calibration = location._ratio_calibration(df, ["pitch_type", "game_year"], "value")
    merged = df.merge(calibration, on=["pitch_type", "game_year"])
    scaled = location._to_100_scale(merged["value"], merged)
    assert scaled.mean() == pytest.approx(100.0, abs=1e-6)


def test_add_location_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        location.add_location_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_location_plus_row_alignment_and_scope(make_raw_df, small_location_thresholds, tmp_path):
    # Same class of index-alignment regression as stuff.py's equivalent test;
    # `_calibrate` had (and dev_log.md 8/26 fixed) the identical bug.
    raw = make_raw_df(
        n_per_type=300, pitch_types=("FF",), junk_pitch_types=("KN",),
        n_pitchers=10, shuffled_index=True,
    )
    result = location.add_location_plus(raw, models_dir=tmp_path, retrain=True)

    # Same note as stuff.py's equivalent test: the final merge resets the
    # index to a fresh RangeIndex; row count/order is the actual contract.
    assert len(result) == len(raw)
    np.testing.assert_array_equal(result["plate_x"].to_numpy(), raw["plate_x"].to_numpy())

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, "location_run_value"].isna().all()

    # Pitches closer to the plate center (lower |plate_x|) were constructed
    # to have higher expected run value for the pitcher; the model should
    # recover a real (negative) relationship, not noise.
    scored = result.dropna(subset=["location_run_value"])
    assert scored["plate_x"].abs().corr(scored["location_run_value"]) < -0.1


def test_pitch_location_plus_calibration_uses_pitch_level_spread_not_the_aggregates(
    make_raw_df, small_location_thresholds, tmp_path
):
    # Regression test for the aggregate-vs-pitch-level calibration mismatch
    # (docs/dev_log.md 9/2 entry): `_calibrate`'s pitch-level type_calibration
    # used to be fit on the spread of per-(pitcher, pitch_type, season)
    # MEANS, far narrower than the raw per-pitch spread, so pitch_location_plus
    # came out badly overdispersed (observed on real data: mean ~121, std
    # ~65 instead of a sane ~100-centered scale). Mean landing on 100 is NOT
    # a sufficient check on its own -- the ratio-scale renormalization
    # forces mean == 100 regardless of which sigma was used (the identical
    # trap documented for pitching.py's own level-separation test) -- so
    # this checks the real magnitude/spread the bug actually broke.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10)
    result = location.add_location_plus(raw, models_dir=tmp_path, retrain=True)
    has_score = result.dropna(subset=["location_run_value"])

    pitcher_type_agg = (
        has_score.groupby(["pitcher", "pitch_type", "game_year"], observed=True)["location_run_value"]
        .agg(mean_location_value="mean", n_pitches="count").reset_index()
    )
    reliable_type = pitcher_type_agg[pitcher_type_agg["n_pitches"] >= location.MIN_PITCHES_FOR_SCORE]
    assert len(reliable_type) > 0

    # Confirm this fixture actually has the property the bug depends on:
    # the spread of per-pitcher-season MEANS is meaningfully narrower than
    # the real pitch-level spread (an aggregate is a mean over many
    # pitches, which suppresses noise) -- otherwise this test couldn't
    # distinguish a correct fix from the bug at all.
    aggregate_of_means_sigma = reliable_type.groupby(["pitch_type", "game_year"])["mean_location_value"].std().mean()
    reliable_pitches = has_score.merge(
        reliable_type[["pitcher", "pitch_type", "game_year"]], on=["pitcher", "pitch_type", "game_year"], how="inner"
    )
    pitch_level_sigma = reliable_pitches.groupby(["pitch_type", "game_year"])["location_run_value"].std().mean()
    assert pitch_level_sigma > aggregate_of_means_sigma * 1.3

    # Regression guard: with the bug, pitch_location_plus's spread inflates
    # roughly in proportion to (pitch_level_sigma / aggregate_of_means_sigma,
    # confirmed >1.3x above); with the fix it should track a normal
    # "+"-stat spread (LOCATION_SCALE_K=0.10 is documented as "roughly
    # +/-10 points per SD").
    reliable_scored = result.merge(
        reliable_type[["pitcher", "pitch_type", "game_year"]], on=["pitcher", "pitch_type", "game_year"], how="inner"
    ).dropna(subset=["pitch_location_plus"])
    assert len(reliable_scored) > 0
    assert reliable_scored["pitch_location_plus"].std() < 30


def test_add_location_plus_cache_roundtrip_is_deterministic(make_raw_df, small_location_thresholds, tmp_path):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)

    trained = location.add_location_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "location_models.joblib").exists()

    cached = location.add_location_plus(raw, models_dir=tmp_path, retrain=False)

    pd.testing.assert_series_equal(
        trained["location_run_value"], cached["location_run_value"]
    )


def test_load_or_score_warns_on_stale_cache_fingerprint(make_raw_df, small_location_thresholds, monkeypatch, tmp_path):
    # Regression coverage for the cache-staleness guard added 9/2 (docs/
    # dev_log.md): previously a cache was reused purely based on file
    # existence, so editing LOCATION_FEATURES/HGB_PARAMS and calling with
    # retrain=False would silently score against a stale model.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    location.add_location_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "location_cache_fingerprint.joblib").exists()

    # Simulate a code change to the training config between the cache being
    # built and this call, without retraining.
    monkeypatch.setattr(location, "HGB_PARAMS", dict(location.HGB_PARAMS, max_depth=3))
    with pytest.warns(UserWarning, match="different LOCATION_FEATURES/HGB_PARAMS"):
        location.add_location_plus(raw, models_dir=tmp_path, retrain=False)
