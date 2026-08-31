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


def test_add_stuff_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        stuff.add_stuff_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_stuff_plus_junk_pitch_types_get_nan(make_raw_df, small_stuff_thresholds):
    raw = make_raw_df(n_per_type=200, pitch_types=("FF",), junk_pitch_types=("KN",))
    result = stuff.add_stuff_plus(raw)

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, ["pitch_stuff_plus", "stuff_plus"]].isna().all().all()


def test_add_stuff_plus_row_alignment_and_reliable_calibration_mean_100(make_raw_df, small_stuff_thresholds):
    # Regression test for the index-misalignment bug (docs/dev_log.md 8/26):
    # `_score_stuff_plus`'s calibration merge used to silently reset
    # stuff_df's index, so add_stuff_plus's positional `.loc[stuff_df.index]`
    # reattachment landed pitch_stuff_plus on the wrong rows whenever the
    # input's index wasn't a clean default RangeIndex.
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
