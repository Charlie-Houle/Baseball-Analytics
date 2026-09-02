import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import bestpitch, location, pitching


def test_add_bestpitch_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        bestpitch.add_bestpitch_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_bestpitch_plus_junk_and_best_meets_or_exceeds_actual(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, tmp_path
):
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
    # realized pitch_pitching_plus for the large majority of pitches, not
    # all, since the candidate uses a zone-average location while the actual
    # pitch's own precise spot within its zone can occasionally beat it.
    assert (in_scope["pitch_bestpitch_plus"] >= -1e-6).mean() > 0.9


def test_add_bestpitch_plus_own_pitch_always_a_candidate_even_if_unreliable(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, tmp_path
):
    # Regression test for the "reliable arsenal" gate breaking the module's
    # own documented invariant (docs/dev_log.md 9/2 entry): a pitch type a
    # pitcher throws too rarely that season to be stuff_plus_reliable used
    # to be excluded from the counterfactual candidate search entirely --
    # including for scoring that type's own pitches -- so best_pitching_plus
    # for those pitches wasn't guaranteed to dominate their own actual
    # score. Give pitcher 0 a handful of CU pitches below the (lowered)
    # reliability bar, mixed in with everyone else's reliable CU, so CU is a
    # real, scoreable pitch type overall but not a reliable part of pitcher
    # 0's own arsenal specifically.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10, seed=1)
    cu = make_raw_df(n_per_type=300, pitch_types=("CU",), n_pitchers=10, seed=2)

    pitcher0_cu = cu[cu["pitcher"] == 0]
    assert len(pitcher0_cu) >= 3
    cu = cu.drop(index=pitcher0_cu.index[3:])

    combined = pd.concat([raw, cu], ignore_index=True)
    combined["game_pk"] = np.arange(len(combined))  # keep KEY_COLS unique across the two builds

    result = bestpitch.add_bestpitch_plus(combined, models_dir=tmp_path, retrain=True)

    cu_rows = result[(result["pitcher"] == 0) & (result["pitch_type"] == "CU")]
    assert len(cu_rows) == 3
    assert cu_rows["pitch_bestpitch_plus"].notna().all()
    assert (cu_rows["pitch_bestpitch_plus"] >= -1e-6).all()


def test_batched_target_averaging_matches_unbatched_reference_with_varying_zone_height(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, tmp_path
):
    # Equivalence test for the batched, deduplicated target-averaging path
    # (_build_base_array / _batched_target_averaged_location_run_value)
    # against the original per-row reference (_predict_at_point). The
    # batched path's docstrings claim this equivalence was "verified... on
    # synthetic data with a stub model," but that check was never a
    # checked-in test, and tests/conftest.py used to hold sz_top/sz_bot
    # constant across every row, so the ZONE_HEIGHT_DEDUP_ROUND_FT bucketing
    # -- the part of the batching that actually depends on zone height
    # varying -- was never exercised (docs/dev_log.md 9/2 entry).
    # conftest.py's fixture now draws sz_top/sz_bot per row, so this test
    # has real variation to check against.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    scored = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)

    in_scope = (
        scored.loc[:, bestpitch.REQUIRED_COLS + bestpitch.BASE_STATE_COLS + ["stuff_plus"]]
        .dropna(subset=bestpitch.REQUIRED_COLS + ["stuff_plus"])
        .copy()
    )
    engineered = bestpitch._build_features(in_scope)
    engineered[bestpitch.ZONE_COL] = in_scope[bestpitch.ZONE_COL].to_numpy()

    models = location.load_cached_models(tmp_path)
    model = models["FF"]
    base = engineered[engineered[bestpitch.PITCH_TYPE_COL] == "FF"].head(40)
    # Confirm the fixture actually varies zone height -- otherwise this
    # test couldn't distinguish correct dedup bucketing from a bug at all.
    assert (base["sz_top"] - base["sz_bot"]).nunique() > 1

    array, p_throws_is_R, zone_height, center_key = bestpitch._build_base_array(base)
    center_x, center_z = base["plate_x"].mean(), base["plate_z_rel"].mean()
    batched = bestpitch._batched_target_averaged_location_run_value(
        model, array, p_throws_is_R, zone_height, center_key, center_x, center_z
    )

    z_offset = bestpitch.TARGET_RADIUS_FT / zone_height
    points = [
        (center_x, center_z),
        (center_x + bestpitch.TARGET_RADIUS_FT, center_z),
        (center_x - bestpitch.TARGET_RADIUS_FT, center_z),
        (center_x, center_z + z_offset),
        (center_x, center_z - z_offset),
    ]
    reference = np.mean(
        [bestpitch._predict_at_point(base, model, bestpitch.NON_LOCATION_FEATURES, px, pz) for px, pz in points],
        axis=0,
    )

    np.testing.assert_allclose(batched, reference, atol=1e-6)


def test_actual_smoothed_differs_from_pinpoint_pitch_pitching_plus(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, tmp_path
):
    # Regression test for the pinpoint-vs-smoothed comparison bug (docs/
    # dev_log.md 8/30 entry): bestPitch+'s comparison must use the SAME
    # target-averaged scoring on both sides (best_pitching_plus vs.
    # _actual_smoothed_pitching_plus), never the real pitch's own pinpoint
    # pitch_pitching_plus. The existing end-to-end best-meets-or-exceeds-
    # actual test above doesn't have the power to catch a regression to the
    # pinpoint comparison on this fixture's gentle noise (confirmed
    # empirically: reproducing the pre-fix formula on this exact fixture
    # still clears its >0.9 threshold, at 95.6% -- see docs/dev_log.md's
    # 9/2 entry). This test instead checks the mechanism directly.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10)
    result = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=True)

    in_scope = result.dropna(subset=["pitch_bestpitch_plus", "pitch_pitching_plus", "best_pitching_plus"])
    assert len(in_scope) > 0
    # pitch_bestpitch_plus = best_pitching_plus - actual_smoothed by
    # construction, so this recovers actual_smoothed from the public output.
    actual_smoothed = in_scope["best_pitching_plus"] - in_scope["pitch_bestpitch_plus"]

    # The smoothed actual score and the pinpoint pitch_pitching_plus are the
    # same underlying pitch, but smoothing over a small target area changes
    # the location input -- and therefore the score -- for virtually every
    # real pitch. If bestpitch.py ever again subtracted pitch_pitching_plus
    # instead of the smoothed score, actual_smoothed reconstructed this way
    # would come back IDENTICAL to pitch_pitching_plus.
    assert not np.allclose(actual_smoothed.to_numpy(), in_scope["pitch_pitching_plus"].to_numpy())
    diffs = (actual_smoothed - in_scope["pitch_pitching_plus"]).abs()
    assert (diffs > 1e-6).mean() > 0.5
