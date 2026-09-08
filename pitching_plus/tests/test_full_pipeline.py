import pandas as pd

from pitching_plus.scripts import full_pipeline


def test_run_pipeline_end_to_end(
    make_raw_df, small_stuff_thresholds, small_location_thresholds, small_pitching_thresholds, tmp_path
):
    # The four-stage chain (Stuff+ -> Location+ -> Pitching+ -> bestPitch+)
    # was previously only exercised piecewise, via each module's own tests
    # assembling the chain manually -- nothing called run_pipeline itself
    # (docs/dev_log.md 9/2 entry). run_pipeline now accepts models_dir/
    # retrain so it can be pointed at an isolated tmp cache instead of the
    # real production one.
    raw = make_raw_df(n_per_type=300, pitch_types=("FF", "SL"), n_pitchers=10)
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "out.csv"
    raw.to_csv(input_path, index=False)

    result = full_pipeline.run_pipeline(
        input_path=input_path, output_path=output_path, models_dir=tmp_path / "models", retrain=True
    )

    assert output_path.exists()
    from_disk = pd.read_csv(output_path)
    assert len(from_disk) == len(raw)

    expected_cols = [
        "pitch_stuff_plus", "stuff_plus",
        "location_run_value", "pitch_location_plus", "location_plus",
        "pitch_pitching_plus", "pitching_plus",
        "best_pitching_plus", "pitch_bestpitch_plus",
    ]
    for col in expected_cols:
        assert col in result.columns
        assert col in from_disk.columns

    assert result["pitch_bestpitch_plus"].notna().any()
