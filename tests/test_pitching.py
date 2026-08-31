import pandas as pd
import pytest

from pitching_plus.scripts import location, pitching, stuff


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
