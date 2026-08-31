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
    df = pd.DataFrame(
        {
            "on_1b": [np.nan], "on_2b": [np.nan], "on_3b": [np.nan],
            "outs_when_up": [0], "balls": [0], "strikes": [0],
            "plate_z": [2.5], "sz_top": [3.5], "sz_bot": [1.5],
            "plate_x": [0.7], "stand": ["L"], "p_throws": ["R"],
        }
    )
    engineered = location._build_features(df)
    assert engineered.loc[0, "plate_z_rel"] == pytest.approx(0.5)
    # Left-handed batter: arm-side is the mirror of raw plate_x.
    assert engineered.loc[0, "plate_x_armside"] == pytest.approx(-0.7)
    assert engineered.loc[0, "platoon_matchup"] == 1


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
    # Same class of index-alignment regression as stuff.py's equivalent test --
    # `_calibrate` had (and dev_log.md 8/26 fixed) the identical bug.
    raw = make_raw_df(
        n_per_type=300, pitch_types=("FF",), junk_pitch_types=("KN",),
        n_pitchers=10, shuffled_index=True,
    )
    result = location.add_location_plus(raw, models_dir=tmp_path, retrain=True)

    # Same note as stuff.py's equivalent test: the final merge resets the
    # index to a fresh RangeIndex -- row count/order is the actual contract.
    assert len(result) == len(raw)
    np.testing.assert_array_equal(result["plate_x"].to_numpy(), raw["plate_x"].to_numpy())

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, "location_run_value"].isna().all()

    # Pitches closer to the plate center (lower |plate_x|) were constructed
    # to have higher expected run value for the pitcher -- the model should
    # recover a real (negative) relationship, not noise.
    scored = result.dropna(subset=["location_run_value"])
    assert scored["plate_x"].abs().corr(scored["location_run_value"]) < -0.1


def test_add_location_plus_cache_roundtrip_is_deterministic(make_raw_df, small_location_thresholds, tmp_path):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)

    trained = location.add_location_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "location_models.joblib").exists()

    cached = location.add_location_plus(raw, models_dir=tmp_path, retrain=False)

    pd.testing.assert_series_equal(
        trained["location_run_value"], cached["location_run_value"]
    )
