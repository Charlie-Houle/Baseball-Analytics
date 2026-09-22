import pandas as pd
import pytest

from slippery_slope.scripts import sample_data


def _fake_statcast_df():
    # Two pitches, deliberately out of game/at-bat/pitch-number order, so a
    # correct load_sample has to actually sort rather than pass the raw
    # pybaseball order straight through.
    return pd.DataFrame(
        {
            "game_pk": [2, 1],
            "at_bat_number": [1, 1],
            "pitch_number": [1, 1],
            "pitcher": [100, 200],
            "player_name": ["Pitcher B", "Pitcher A"],
            "pitch_type": ["FF", "SL"],
            "pitch_name": ["4-Seam Fastball", "Slider"],
            "game_year": [2025, 2025],
            "balls": [0, 1],
            "strikes": [0, 2],
            "outs_when_up": [0, 1],
            "stand": ["R", "L"],
            "p_throws": ["R", "L"],
            "plate_x": [0.1, -0.2],
            "plate_z": [2.5, 1.8],
            "sz_top": [3.5, 3.4],
            "sz_bot": [1.5, 1.6],
            "zone": [5, 2],
            "on_1b": [None, 1],
            "on_2b": [None, None],
            "on_3b": [None, None],
            "extra_column_not_needed_by_the_app": ["x", "y"],
        }
    )


def test_load_sample_trims_columns_and_sorts_by_game_at_bat_pitch(monkeypatch):
    monkeypatch.setattr(sample_data.pb, "statcast", lambda **kwargs: _fake_statcast_df())

    result = sample_data.load_sample("2025-06-02", "2025-06-02")

    assert list(result.columns) == sample_data.SAMPLE_COLUMNS
    # game_pk 1 sorts before game_pk 2, the reverse of the fake data's order.
    assert result["game_pk"].tolist() == [1, 2]


def test_load_sample_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(sample_data.pb, "statcast", lambda **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="No Statcast pitches found"):
        sample_data.load_sample("2025-01-01", "2025-01-01")


def test_load_sample_raises_on_missing_columns(monkeypatch):
    incomplete = _fake_statcast_df().drop(columns=["zone"])
    monkeypatch.setattr(sample_data.pb, "statcast", lambda **kwargs: incomplete)

    with pytest.raises(ValueError, match="missing expected columns"):
        sample_data.load_sample("2025-06-02", "2025-06-02")


def test_load_pitcher_season_trims_columns_and_sorts(monkeypatch):
    monkeypatch.setattr(sample_data.pb, "statcast_pitcher", lambda *args: _fake_statcast_df())

    result = sample_data.load_pitcher_season(100, 2025)

    assert list(result.columns) == sample_data.SAMPLE_COLUMNS
    assert result["game_pk"].tolist() == [1, 2]


def test_load_pitcher_season_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(sample_data.pb, "statcast_pitcher", lambda *args: pd.DataFrame())

    with pytest.raises(ValueError, match="No Statcast pitches found"):
        sample_data.load_pitcher_season(100, 2025)
