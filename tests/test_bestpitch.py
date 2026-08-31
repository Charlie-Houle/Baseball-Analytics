import pandas as pd
import pytest

from pitching_plus.scripts import bestpitch, location, stuff


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
