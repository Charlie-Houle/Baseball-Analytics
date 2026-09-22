import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import bestpitch, pitching


def test_add_bestpitch_plus_missing_columns_raises():
    with pytest.raises(ValueError):
        bestpitch.add_bestpitch_plus(pd.DataFrame({"pitcher": [1]}))


def test_add_bestpitch_plus_junk_and_best_meets_or_exceeds_actual(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
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
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
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


def _game_state_key(df):
    return (
        df["re288_state"].to_numpy() * 2 + (df["stand"] == "R").to_numpy().astype(int)
    ) * 2 + df["p_throws_R"].to_numpy()


def test_batched_target_averaging_matches_unbatched_reference_with_varying_zone_height(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    # Equivalence test for the batched, deduplicated target-averaging path
    # (_build_base_array / _batched_target_averaged_pitching_run_value)
    # against the original per-row reference (_predict_at_point), for the
    # actual-pitch scoring path (dedup_by_pitcher_season=False -- physics
    # here is each row's own real, per-pitch value, so the dedup key is
    # trivial/per-row, exercising only the zone-height bucketing logic, not
    # the pitcher-season discrimination the candidate-search path needs;
    # see the test below for that).
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    scored = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)

    in_scope = (
        scored.loc[:, bestpitch.REQUIRED_COLS + bestpitch.BASE_STATE_COLS + ["stuff_plus"]]
        .dropna(subset=bestpitch.REQUIRED_COLS + ["stuff_plus"])
        .copy()
    )
    engineered = bestpitch.stuff_mod._build_v1_features(in_scope)
    engineered = bestpitch._build_features(engineered)
    engineered[bestpitch.ZONE_COL] = in_scope[bestpitch.ZONE_COL].to_numpy()

    models = pitching.load_cached_models(tmp_path)
    model = models["FF"]
    base = engineered[engineered[bestpitch.PITCH_TYPE_COL] == "FF"].head(40)
    # Confirm the fixture actually varies zone height -- otherwise this
    # test couldn't distinguish correct dedup bucketing from a bug at all.
    assert (base["sz_top"] - base["sz_bot"]).nunique() > 1

    array, p_throws_is_R, zone_height, center_key = bestpitch._build_base_array(
        base, physics_cols=bestpitch.STUFF_FEATURES, dedup_by_pitcher_season=False,
    )
    center_x, center_z = base["plate_x"].mean(), base["plate_z_rel"].mean()
    batched = bestpitch._batched_target_averaged_pitching_run_value(
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
        [
            bestpitch._predict_at_point(base, model, bestpitch.STUFF_FEATURES, bestpitch.NON_LOCATION_FEATURES, px, pz)
            for px, pz in points
        ],
        axis=0,
    )

    np.testing.assert_allclose(batched, reference, atol=1e-6)


def test_batched_target_averaging_matches_unbatched_reference_with_varying_pitcher_physics(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    # Regression test for the pitcher-season dedup-key fix this migration
    # made (docs/dev_log.md's Pitching+ retool entry): _build_base_array's
    # center_key used to be a bijection over game-state + handedness ALONE,
    # valid only because every non-location feature besides the 3 varying
    # ones was constant across pitchers for a fixed game-state. That broke
    # once pitcher-varying physics got folded into the same array for the
    # candidate-search path (dedup_by_pitcher_season=True) -- two different
    # pitchers sharing a game-state would silently get deduped together and
    # one would score with the other's physics. This builds a fixture with
    # multiple pitchers sharing at least one game-state, gives each pitcher
    # distinct candidate physics, and confirms the batched path matches an
    # unbatched per-row reference -- a reverted, game-state-only key would
    # fail this by borrowing one pitcher's physics for another's row.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    scored = pitching.add_pitching_plus(raw, models_dir=tmp_path, retrain=True)

    in_scope = (
        scored.loc[:, bestpitch.REQUIRED_COLS + bestpitch.BASE_STATE_COLS + ["stuff_plus"]]
        .dropna(subset=bestpitch.REQUIRED_COLS + ["stuff_plus"])
        .copy()
    )
    engineered = bestpitch.stuff_mod._build_v1_features(in_scope)
    engineered = bestpitch._build_features(engineered)
    engineered[bestpitch.ZONE_COL] = in_scope[bestpitch.ZONE_COL].to_numpy()

    base = engineered[engineered[bestpitch.PITCH_TYPE_COL] == "FF"].head(80).copy()

    game_state_key = _game_state_key(base)
    shared_counts = pd.Series(game_state_key, index=base.index).value_counts()
    multi_pitcher_group = None
    for gk in shared_counts[shared_counts > 1].index:
        candidate_rows = base[game_state_key == gk]
        if candidate_rows["pitcher"].nunique() > 1:
            multi_pitcher_group = candidate_rows
            break
    assert multi_pitcher_group is not None, (
        "fixture needs >1 pitcher sharing a game-state to test the pitcher-season dedup-key fix"
    )

    # Give each pitcher's rows distinct, deterministic candidate physics
    # (not the fixture's own real per-pitch values), so a dedup bug that
    # collapses different pitchers together produces a detectably wrong
    # prediction rather than a coincidentally-correct one.
    pitcher_ids = sorted(base["pitcher"].unique())
    for i, feat in enumerate(bestpitch.STUFF_FEATURES):
        cand_col = f"cand_{feat}"
        base[cand_col] = base["pitcher"].map({p: 90.0 + 10 * i + p for p in pitcher_ids})

    models = pitching.load_cached_models(tmp_path)
    model = models["FF"]

    array, p_throws_is_R, zone_height, center_key = bestpitch._build_base_array(
        base, physics_cols=bestpitch.STUFF_FEATURE_CAND_COLS, dedup_by_pitcher_season=True,
    )
    # The whole point of this test: confirm the key actually discriminates
    # by pitcher within a shared game-state, not just by game-state.
    same_state_rows = base.loc[multi_pitcher_group.index]
    same_state_keys = pd.Series(center_key, index=base.index).loc[same_state_rows.index]
    assert same_state_keys.nunique() > 1, "center_key must not collapse different pitchers sharing a game-state"

    center_x, center_z = base["plate_x"].mean(), base["plate_z_rel"].mean()
    batched = bestpitch._batched_target_averaged_pitching_run_value(
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
        [
            bestpitch._predict_at_point(
                base, model, bestpitch.STUFF_FEATURE_CAND_COLS, bestpitch.NON_LOCATION_FEATURES, px, pz
            )
            for px, pz in points
        ],
        axis=0,
    )

    np.testing.assert_allclose(batched, reference, atol=1e-6)


def test_actual_smoothed_differs_from_pinpoint_pitch_pitching_plus(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
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


def test_arsenal_physics_avg_only_uses_reliable_rows(make_raw_df, small_stuff_thresholds, small_pitching_thresholds, tmp_path):
    from pitching_plus.scripts import stuff

    raw = make_raw_df(n_per_type=300, pitch_types=("FF",), n_pitchers=10)
    scored = stuff.add_stuff_plus(raw, models_dir=tmp_path, retrain=True)
    engineered = bestpitch.stuff_mod._build_v1_features(
        scored.dropna(subset=stuff.REQUIRED_COLS)
    )

    avg = bestpitch._arsenal_physics_avg(engineered)

    assert set(avg.columns) == {"pitcher", "pitch_type", "game_year"} | set(bestpitch.STUFF_FEATURE_CAND_COLS)
    # Every (pitcher, pitch_type, season) row in the output must itself be
    # reliable -- a row built from unreliable (few-pitch) rows would be a
    # noisy candidate physics estimate feeding the counterfactual search.
    reliable_keys = set(
        map(tuple, engineered.loc[engineered["stuff_plus_reliable"].fillna(False), ["pitcher", "pitch_type", "game_year"]].drop_duplicates().to_numpy())
    )
    output_keys = set(map(tuple, avg[["pitcher", "pitch_type", "game_year"]].to_numpy()))
    assert output_keys <= reliable_keys
    assert len(output_keys) > 0


def _actual_smoothed(df):
    # pitch_bestpitch_plus is defined as best_pitching_plus - actual_smoothed,
    # so the actual score is recoverable exactly without exposing it as a column.
    return df["best_pitching_plus"] - df["pitch_bestpitch_plus"]


def test_pitch_bestpitch_plus_pct_matches_ratio_and_is_bounded(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL", "CU"), n_pitchers=10)
    result = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=True)

    in_scope = result.dropna(subset=["pitch_bestpitch_plus_pct"])
    assert len(in_scope) > 0

    expected = 100 * _actual_smoothed(in_scope) / in_scope["best_pitching_plus"]
    np.testing.assert_allclose(in_scope["pitch_bestpitch_plus_pct"].to_numpy(), expected.to_numpy(), atol=1e-6)

    # The 100+ scale is strictly positive, so the ratio is always > 0. It is
    # <= 100 for the same reason pitch_bestpitch_plus >= 0 (the fmax floor),
    # so it gets the same >90%-within-tolerance check rather than an exact one.
    assert (in_scope["pitch_bestpitch_plus_pct"] > 0).all()
    assert (in_scope["pitch_bestpitch_plus_pct"] <= 100 + 1e-4).mean() > 0.9


def test_pitch_bestpitch_plus_pct_scope_matches_pitch_bestpitch_plus(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    raw = make_raw_df(
        n_per_type=300, pitch_types=("FF", "SL", "CU"), junk_pitch_types=("KN",),
        n_pitchers=10, shuffled_index=True,
    )
    result = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=True)

    assert len(result) == len(raw)
    assert result["pitch_bestpitch_plus_pct"].isna().equals(result["pitch_bestpitch_plus"].isna())

    junk_rows = result["pitch_type"] == "KN"
    assert junk_rows.any()
    assert result.loc[junk_rows, "pitch_bestpitch_plus_pct"].isna().all()


def test_bestpitch_plus_pct_aggregate_is_ratio_of_means_not_mean_of_ratios(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL", "CU"), n_pitchers=10)
    result = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=True)

    in_scope = result.dropna(subset=["pitch_bestpitch_plus_pct"]).copy()
    in_scope["actual"] = _actual_smoothed(in_scope)
    keys = ["pitcher", "pitch_type", "game_year"]
    grouped = in_scope.groupby(keys, observed=True)

    ratio_of_means = 100 * grouped["actual"].mean() / grouped["best_pitching_plus"].mean()
    mean_of_ratios = grouped["pitch_bestpitch_plus_pct"].mean()
    reported = grouped["bestpitch_plus_pct"].first()

    np.testing.assert_allclose(reported.to_numpy(), ratio_of_means.to_numpy(), atol=1e-6)
    # The two aggregations only differ when best_pitching_plus varies within a
    # group; require that at least one group actually does, so this test can
    # tell the designs apart instead of passing on a fixture where they agree.
    assert (reported - mean_of_ratios).abs().max() > 1e-6


def test_bestpitch_plus_pct_reliable_gate(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path, monkeypatch
):
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL", "CU"), n_pitchers=10)

    # bestpitch.py binds MIN_PITCHES_FOR_SCORE at import time (from-import), so
    # patching stuff's copy can't reach it; patch bestpitch's own name.
    monkeypatch.setattr(bestpitch, "MIN_PITCHES_FOR_SCORE", 1)
    baseline = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=True)
    counts = (
        baseline.dropna(subset=["pitch_bestpitch_plus_pct"])
        .groupby(["pitcher", "pitch_type", "game_year"], observed=True)
        .size()
    )
    threshold = int(counts.median())
    assert (counts < threshold).any() and (counts >= threshold).any()

    monkeypatch.setattr(bestpitch, "MIN_PITCHES_FOR_SCORE", threshold)
    result = bestpitch.add_bestpitch_plus(raw, models_dir=tmp_path, retrain=False)

    flags = result.groupby(["pitcher", "pitch_type", "game_year"], observed=True)["bestpitch_plus_pct_reliable"].first()
    flags = flags.dropna().astype(bool)
    expected = (counts >= threshold).reindex(flags.index)
    assert (flags[expected.notna()] == expected.dropna()).all()

    # Unreliable is not missing: the value is still reported, just flagged.
    unreliable_keys = counts.index[counts < threshold]
    unreliable_rows = result.set_index(["pitcher", "pitch_type", "game_year"]).loc[unreliable_keys]
    assert unreliable_rows["bestpitch_plus_pct"].notna().all()
