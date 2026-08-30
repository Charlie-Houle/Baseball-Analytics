import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import bestpitch, location, pitching, stuff

# ============================================================
# stuff.py
# ============================================================


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


def test_add_stuff_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        stuff.add_stuff_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_stuff_plus_junk_pitch_types_get_nan(make_raw_df, monkeypatch):
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)

    raw = make_raw_df(n_per_type=200, pitch_types=("FF",), junk_pitch_types=("KN",))
    result = stuff.add_stuff_plus(raw)

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, ["pitch_stuff_plus", "stuff_plus"]].isna().all().all()


def test_add_stuff_plus_row_alignment_and_reliable_calibration_mean_100(make_raw_df, monkeypatch):
    # Regression test for the index-misalignment bug (docs/dev_log.md 8/26):
    # `_score_stuff_plus`'s calibration merge used to silently reset
    # stuff_df's index, so add_stuff_plus's positional `.loc[stuff_df.index]`
    # reattachment landed pitch_stuff_plus on the wrong rows whenever the
    # input's index wasn't a clean default RangeIndex.
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)

    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10, shuffled_index=True)
    result = stuff.add_stuff_plus(raw)

    # add_stuff_plus's final merge legitimately resets the index to a fresh
    # RangeIndex (normal pandas merge behavior) -- what's actually promised
    # (and what matters here) is row count and order, not index labels.
    assert len(result) == len(raw)
    np.testing.assert_array_equal(result["release_speed"].to_numpy(), raw["release_speed"].to_numpy())

    # The PCA composite is sign-anchored on release_speed, so within a pitch
    # type a pitch's own release_speed should correlate positively with its
    # own pitch_stuff_plus -- the diagnostic that caught the original bug
    # (it measured ~0 correlation when scores were misaligned).
    # Threshold is modest because this fixture's STUFF_FEATURES are drawn
    # independently (unlike real pitch data, where velocity/spin/movement are
    # naturally correlated) -- the point is a clear positive correlation, not
    # the ~0 the original misalignment bug actually produced.
    scored = result.dropna(subset=["pitch_stuff_plus"])
    assert scored["release_speed"].corr(scored["pitch_stuff_plus"]) > 0.15

    # raw_ratio_mean anchors the scale so the mean over exactly the reliable
    # *pitcher-pitch_type-season* population is 100, by construction -- note
    # this must be deduplicated first, since `result` broadcasts stuff_plus
    # across every pitch in the group (a pitch-weighted mean over the raw
    # pitch-level rows is a different, unrelated quantity).
    reliable = result[result["stuff_plus_reliable"].fillna(False)].drop_duplicates(
        subset=["pitcher", "pitch_type", "game_year"]
    )
    means = reliable.groupby(["pitch_type", "game_year"])["stuff_plus"].mean()
    assert all(m == pytest.approx(100.0, abs=1e-6) for m in means)


# ============================================================
# location.py
# ============================================================


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


def test_add_location_plus_row_alignment_and_scope(make_raw_df, monkeypatch, tmp_path):
    # Same class of index-alignment regression as stuff.py above --
    # `_calibrate` had (and dev_log.md 8/26 fixed) the identical bug.
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)

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


def test_add_location_plus_cache_roundtrip_is_deterministic(make_raw_df, monkeypatch, tmp_path):
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)

    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)

    trained = location.add_location_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "location_models.joblib").exists()

    cached = location.add_location_plus(raw, models_dir=tmp_path, retrain=False)

    pd.testing.assert_series_equal(
        trained["location_run_value"], cached["location_run_value"]
    )


# ============================================================
# pitching.py
# ============================================================


def test_add_pitching_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        pitching.add_pitching_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_pitching_plus_junk_and_reliable_calibration_mean_100(make_raw_df, monkeypatch, tmp_path):
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)

    raw = make_raw_df(
        n_per_type=300, pitch_types=("FF", "SL"), junk_pitch_types=("KN",),
        n_pitchers=10, shuffled_index=True,
    )
    result = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)

    assert len(result) == len(raw)

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, ["pitch_pitching_plus", "pitching_plus"]].isna().all().all()

    # raw_ratio_mean anchors the scale so the mean over exactly the reliable
    # pitcher-pitch_type-season population is 100, same convention as
    # stuff_plus/location_plus.
    reliable = result[result["pitching_plus_reliable"].fillna(False)].drop_duplicates(
        subset=["pitcher", "pitch_type", "game_year"]
    )
    assert len(reliable) > 0
    means = reliable.groupby(["pitch_type", "game_year"])["pitching_plus"].mean()
    assert all(m == pytest.approx(100.0, abs=1e-6) for m in means)


def test_pitch_level_calibration_uses_its_own_spread_not_the_aggregates(make_raw_df, monkeypatch, tmp_path):
    # Regression test for the aggregate-vs-pitch-level calibration mismatch:
    # location_run_value has much more spread at the pitch level than at the
    # (pitcher, pitch_type, season) aggregate level (an aggregate is a mean
    # over many pitches). Using the aggregate's spread to z-score pitch-level
    # values would inflate every pitch-level score's distance from 100 --
    # confirmed by checking loc_sigma_pitch > loc_sigma_agg here, and that
    # the pitch-level population (not just the aggregate one) also
    # calibrates to a mean of ~100 under its own dedicated calibration.
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)

    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10)
    scored = location.add_location_plus(stuff.add_stuff_plus(raw), models_dir=tmp_path, retrain=True)
    in_scope = scored.dropna(subset=["pitch_stuff_plus", "stuff_plus", "location_run_value", "delta_pitcher_run_exp"])

    blend_params, calibration, pitch_calibration, pitcher_agg = pitching._fit_pitching_plus_model(in_scope)

    assert blend_params["loc_sigma_pitch"] > blend_params["loc_sigma_agg"]

    # The pitch-level calibration's own reference population (every pitch
    # belonging to a reliable pitcher-pitch-type-season) should itself
    # average to ~100 per (pitch_type, season), same anchoring convention as
    # every other level in this codebase.
    reliable_keys = pitcher_agg[pitcher_agg["reliable"]][["pitcher", "pitch_type", "game_year"]]
    reliable_pitches = in_scope.merge(reliable_keys, on=["pitcher", "pitch_type", "game_year"], how="inner")
    scores = pitching._score_pitching_plus(
        reliable_pitches["stuff_plus"], reliable_pitches["location_run_value"],
        reliable_pitches["pitch_type"], reliable_pitches["game_year"],
        blend_params, pitch_calibration, level="pitch",
    )
    means = pd.Series(scores).groupby([reliable_pitches["pitch_type"].to_numpy(), reliable_pitches["game_year"].to_numpy()]).mean()
    assert all(m == pytest.approx(100.0, abs=1e-6) for m in means)


def test_add_pitching_plus_reuses_precomputed_stuff_and_location_columns(make_raw_df, monkeypatch, tmp_path):
    # add_pitching_plus should not recompute stuff/location scores that are
    # already present -- full_pipeline.py relies on this to avoid redundant work.
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)

    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    pre_scored = location.add_location_plus(stuff.add_stuff_plus(raw), models_dir=tmp_path, retrain=True)

    call_count = {"n": 0}
    real_add_location_plus = pitching.add_location_plus

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return real_add_location_plus(*args, **kwargs)

    monkeypatch.setattr(pitching, "add_location_plus", spy)
    pitching.add_pitching_plus(pre_scored, models_dir=tmp_path, retrain=False)
    assert call_count["n"] == 0


# ============================================================
# bestpitch.py
# ============================================================


def test_add_bestpitch_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        bestpitch.add_bestpitch_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_bestpitch_plus_junk_and_best_meets_or_exceeds_actual(make_raw_df, monkeypatch, tmp_path):
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)

    raw = make_raw_df(
        n_per_type=300, pitch_types=("FF", "SL", "CU"), junk_pitch_types=("KN",),
        n_pitchers=10, shuffled_index=True,
    )
    result = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=True)

    assert len(result) == len(raw)

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, ["best_pitching_plus", "pitch_bestpitch_plus"]].isna().all().all()

    in_scope = result.dropna(subset=["pitch_bestpitch_plus"])
    assert len(in_scope) > 0
    # best_pitching_plus is a max over real candidates including (usually) the
    # pitcher's own actual pitch type/zone, so it should meet or exceed the
    # realized pitch_pitching_plus for the large majority of pitches -- not
    # all, since the candidate uses a zone-average location while the actual
    # pitch's own precise spot within its zone can occasionally beat it.
    assert (in_scope["pitch_bestpitch_plus"] >= -1e-6).mean() > 0.9
