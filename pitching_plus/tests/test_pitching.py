import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import pitching


def test_add_pitching_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        pitching.add_pitching_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_pitching_plus_junk_and_reliable_calibration_mean_100(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
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


def test_pitch_pitching_plus_calibration_uses_pitch_level_spread_not_the_aggregates(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    # Regression test for the aggregate-vs-pitch-level calibration mismatch
    # this rewrite fixed from day one (docs/dev_log.md's 9/2, 8/30, and 9/4
    # entries): a calibration sigma fit from the spread of per-(pitcher,
    # pitch_type, season) MEANS is far narrower than the raw per-pitch
    # spread, and applying it to per-pitch values overdisperses
    # pitch_pitching_plus badly. Mean landing on 100 is NOT a sufficient
    # check on its own -- the ratio-scale renormalization forces mean == 100
    # regardless of which sigma was used -- so this checks the real
    # magnitude/spread the bug actually breaks.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10)
    result = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)
    has_score = result.dropna(subset=["pitching_run_value"])

    pitcher_type_agg = (
        has_score.groupby(["pitcher", "pitch_type", "game_year"], observed=True)["pitching_run_value"]
        .agg(mean_pitching_value="mean", n_pitches="count").reset_index()
    )
    reliable_type = pitcher_type_agg[pitcher_type_agg["n_pitches"] >= pitching.MIN_PITCHES_FOR_SCORE]
    assert len(reliable_type) > 0

    # Confirm this fixture actually has the property the bug depends on:
    # the spread of per-pitcher-season MEANS is meaningfully narrower than
    # the real pitch-level spread (an aggregate is a mean over many
    # pitches, which suppresses noise) -- otherwise this test couldn't
    # distinguish a correct fix from the bug at all.
    aggregate_of_means_sigma = reliable_type.groupby(["pitch_type", "game_year"])["mean_pitching_value"].std().mean()
    reliable_pitches = has_score.merge(
        reliable_type[["pitcher", "pitch_type", "game_year"]], on=["pitcher", "pitch_type", "game_year"], how="inner"
    )
    pitch_level_sigma = reliable_pitches.groupby(["pitch_type", "game_year"])["pitching_run_value"].std().mean()
    assert pitch_level_sigma > aggregate_of_means_sigma * 1.3

    reliable_scored = result.merge(
        reliable_type[["pitcher", "pitch_type", "game_year"]], on=["pitcher", "pitch_type", "game_year"], how="inner"
    ).dropna(subset=["pitch_pitching_plus"])
    assert len(reliable_scored) > 0
    assert reliable_scored["pitch_pitching_plus"].std() < 30


def test_add_pitching_plus_reuses_precomputed_stuff_and_location_columns(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, monkeypatch, tmp_path
):
    # add_pitching_plus should not recompute stuff/location scores that are
    # already present; full_pipeline.py relies on this to avoid redundant work.
    from pitching_plus.scripts import location, stuff

    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    pre_scored = location.add_location_plus(
        stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True), models_dir=tmp_path, retrain=True
    )

    call_count = {"n": 0}
    real_add_location_plus = pitching.add_location_plus

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return real_add_location_plus(*args, **kwargs)

    monkeypatch.setattr(pitching, "add_location_plus", spy)
    pitching.add_pitching_plus(pre_scored, models_dir=tmp_path, retrain=False)
    assert call_count["n"] == 0


def test_add_pitching_plus_cache_roundtrip_is_deterministic(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)

    trained = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "pitching_models.joblib").exists()

    cached = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=False)

    pd.testing.assert_series_equal(trained["pitching_run_value"], cached["pitching_run_value"])


def test_load_or_score_warns_on_stale_cache_fingerprint(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, monkeypatch, tmp_path
):
    # Regression coverage for the cache-staleness guard: previously (in
    # location.py, before stuff.py/pitching.py ported the same pattern) a
    # cache was reused purely based on file existence, so editing
    # JOINT_FEATURES/HGB_PARAMS and calling with retrain=False would
    # silently score against a stale model.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)
    assert (tmp_path / "pitching_cache_fingerprint.joblib").exists()

    # Simulate a code change to the training config between the cache being
    # built and this call, without retraining.
    monkeypatch.setattr(pitching, "HGB_PARAMS", dict(pitching.HGB_PARAMS, max_depth=3))
    with pytest.warns(UserWarning, match="different JOINT_FEATURES/HGB_PARAMS"):
        pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=False)
