import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import stuff


def test_solve_time_to_y_vectorized_matches_manual_roots():
    y0, vy0, ay, distance = 54.0, -130.0, 6.0, 30.0
    a, b, c = 0.5 * ay, vy0, distance
    roots = np.roots([a, b, c])
    expected = min(r for r in roots if np.isreal(r) and r.real > 0).real

    t = stuff._solve_time_to_y_vectorized(
        y0=np.array([y0]), vy0=np.array([vy0]), ay=np.array([ay]), distance=distance
    )
    assert t[0] == pytest.approx(expected, rel=1e-6)


def test_solve_time_to_y_vectorized_no_real_solution_is_nan():
    t = stuff._solve_time_to_y_vectorized(
        y0=np.array([54.0]), vy0=np.array([50.0]), ay=np.array([0.0]), distance=30
    )
    assert np.isnan(t[0])


def test_build_v1_features_movement_per_reaction_time(make_raw_df):
    raw = make_raw_df(n_per_type=20, pitch_types=("FF",))
    engineered = stuff._build_v1_features(raw)

    finite = np.isfinite(engineered["movement_per_reaction_time"])
    assert finite.any()
    np.testing.assert_allclose(
        engineered.loc[finite, "movement_per_reaction_time"],
        engineered.loc[finite, "horizontal_acceleration"] / engineered.loc[finite, "time_30ft"],
    )


def test_build_v1_features_axis_differential_matches_manual_angle(make_raw_df):
    raw = make_raw_df(n_per_type=20, pitch_types=("FF",))
    engineered = stuff._build_v1_features(raw)

    movement_angle = np.degrees(np.arctan2(raw["pfx_x"], -raw["pfx_z"])) % 360
    raw_gap = np.abs(raw["spin_axis"] - movement_angle) % 360
    expected = np.minimum(raw_gap, 360 - raw_gap)

    np.testing.assert_allclose(engineered["axis_differential"].to_numpy(), expected.to_numpy())
    assert (engineered["axis_differential"] >= 0).all()
    assert (engineered["axis_differential"] <= 180).all()


def test_add_stuff_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        stuff.add_stuff_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_stuff_plus_junk_pitch_types_get_nan(make_raw_df, small_stuff_thresholds, tmp_path):
    raw = make_raw_df(n_per_type=200, pitch_types=("FF",), junk_pitch_types=("KN",))
    result = stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True)

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, ["pitch_stuff_plus", "stuff_plus"]].isna().all().all()


def test_add_stuff_plus_row_alignment_and_reliable_calibration_mean_100(
    make_raw_df, small_stuff_thresholds, tmp_path
):
    # Regression test for the index-misalignment bug (docs/dev_log.md 8/26):
    # `_calibrate`'s calibration merge used to silently reset the working
    # frame's index, so add_stuff_plus's positional `.loc[...]` reattachment
    # landed pitch_stuff_plus on the wrong rows whenever the input's index
    # wasn't a clean default RangeIndex.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10, shuffled_index=True)
    result = stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True)

    # add_stuff_plus's final merge legitimately resets the index to a fresh
    # RangeIndex (normal pandas merge behavior); what's actually promised
    # (and what matters here) is row count and order, not index labels.
    assert len(result) == len(raw)
    np.testing.assert_array_equal(result["release_speed"].to_numpy(), raw["release_speed"].to_numpy())

    # The fixture's delta_pitcher_run_exp is constructed to depend positively
    # on release_speed (conftest.py), so a correctly row-aligned, correctly
    # trained model should recover a real positive relationship between a
    # pitch's own release_speed and its own pitch_stuff_plus -- the same
    # diagnostic that caught the original misalignment bug (it measured ~0
    # correlation when scores were misaligned), reframed for a supervised
    # model rather than the old PCA's sign-anchoring.
    scored = result.dropna(subset=["pitch_stuff_plus"])
    assert scored["release_speed"].corr(scored["pitch_stuff_plus"]) > 0.1

    # raw_ratio_mean anchors the scale so the mean over exactly the reliable
    # *pitcher-pitch_type-season* population is 100, by construction. Note
    # this must be deduplicated first, since `result` broadcasts stuff_plus
    # across every pitch in the group (a pitch-weighted mean over the raw
    # pitch-level rows is a different, unrelated quantity).
    reliable = result[result["stuff_plus_reliable"].fillna(False)].drop_duplicates(
        subset=["pitcher", "pitch_type", "game_year"]
    )
    means = reliable.groupby(["pitch_type", "game_year"])["stuff_plus"].mean()
    assert all(m == pytest.approx(100.0, abs=1e-6) for m in means)


def test_pitch_stuff_plus_calibration_uses_pitch_level_spread_not_the_aggregates(
    make_raw_df, small_stuff_thresholds, tmp_path
):
    # Regression test for the aggregate-vs-pitch-level calibration mismatch
    # this retool fixed proactively (docs/dev_log.md 9/2 and 9/4 entries):
    # a calibration sigma fit from the spread of per-(pitcher, pitch_type,
    # season) MEANS is far narrower than the raw per-pitch spread, and
    # applying it to per-pitch values overdisperses pitch_stuff_plus badly
    # (the same bug, unfixed, made location.py's pitch_location_plus come
    # out mean ~121/std ~65 on real data instead of a sane ~100-centered
    # scale). Mean landing on 100 is NOT a sufficient check on its own --
    # the ratio-scale renormalization forces mean == 100 regardless of which
    # sigma was used -- so this checks the real magnitude/spread the bug
    # actually breaks.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10)
    result = stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True)
    has_score = result.dropna(subset=["stuff_run_value"])

    pitcher_type_agg = (
        has_score.groupby(["pitcher", "pitch_type", "game_year"], observed=True)["stuff_run_value"]
        .agg(mean_stuff_value="mean", n_pitches="count").reset_index()
    )
    reliable_type = pitcher_type_agg[pitcher_type_agg["n_pitches"] >= stuff.MIN_PITCHES_FOR_SCORE]
    assert len(reliable_type) > 0

    # Confirm this fixture actually has the property the bug depends on:
    # the spread of per-pitcher-season MEANS is meaningfully narrower than
    # the real pitch-level spread (an aggregate is a mean over many
    # pitches, which suppresses noise) -- otherwise this test couldn't
    # distinguish a correct fix from the bug at all.
    aggregate_of_means_sigma = reliable_type.groupby(["pitch_type", "game_year"])["mean_stuff_value"].std().mean()
    reliable_pitches = has_score.merge(
        reliable_type[["pitcher", "pitch_type", "game_year"]], on=["pitcher", "pitch_type", "game_year"], how="inner"
    )
    pitch_level_sigma = reliable_pitches.groupby(["pitch_type", "game_year"])["stuff_run_value"].std().mean()
    assert pitch_level_sigma > aggregate_of_means_sigma * 1.3

    # Regression guard: with the bug, pitch_stuff_plus's spread inflates
    # roughly in proportion to (pitch_level_sigma / aggregate_of_means_sigma,
    # confirmed >1.3x above); with the fix it should track a normal
    # "+"-stat spread (STUFF_SCALE_K=0.10 is documented as "roughly +/-10
    # points per SD").
    reliable_scored = result.merge(
        reliable_type[["pitcher", "pitch_type", "game_year"]], on=["pitcher", "pitch_type", "game_year"], how="inner"
    ).dropna(subset=["pitch_stuff_plus"])
    assert len(reliable_scored) > 0
    assert reliable_scored["pitch_stuff_plus"].std() < 30


def test_add_stuff_plus_cache_roundtrip_is_deterministic(make_raw_df, small_stuff_thresholds, tmp_path):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)

    trained = stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "stuff_models.joblib").exists()

    cached = stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=False)

    pd.testing.assert_series_equal(
        trained["stuff_run_value"], cached["stuff_run_value"]
    )


def test_load_or_score_warns_on_stale_cache_fingerprint(make_raw_df, small_stuff_thresholds, monkeypatch, tmp_path):
    # Regression coverage for the cache-staleness guard: previously (in
    # location.py, before this retool ported the same pattern to stuff.py) a
    # cache was reused purely based on file existence, so editing
    # STUFF_FEATURES/HGB_PARAMS and calling with retrain=False would
    # silently score against a stale model.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "stuff_cache_fingerprint.joblib").exists()

    # Simulate a code change to the training config between the cache being
    # built and this call, without retraining.
    monkeypatch.setattr(stuff, "HGB_PARAMS", dict(stuff.HGB_PARAMS, max_depth=3))
    with pytest.warns(UserWarning, match="different STUFF_FEATURES/HGB_PARAMS"):
        stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=False)
