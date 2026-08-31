"""
Shared fixtures for the pitching_plus test suite.

`make_raw_df` builds a small, synthetic Statcast-shaped dataframe covering
every column required by both stuff.py and location.py, with just enough
physical/statistical structure (real variance tied to release_speed and
plate location) that calibration/regression logic has something meaningful
to work on, without needing MIN_GROUP_SIZE_FOR_PCA/MIN_GROUP_SIZE_FOR_MODEL's
production-sized (5000+) row counts.

`small_stuff_thresholds`/`small_location_thresholds` lower those (and
MIN_PITCHES_FOR_SCORE/MIN_PITCHES_FOR_SEASON_SCORE) to values this fixture's
row counts can actually satisfy; request whichever module(s) a test scores.
"""

import numpy as np
import pandas as pd
import pytest

from pitching_plus.scripts import location, stuff


def _build_raw_df(
    n_per_type=300,
    pitch_types=("FF", "SL"),
    junk_pitch_types=(),
    seasons=(2024,),
    n_pitchers=4,
    seed=0,
    shuffled_index=False,
):
    rng = np.random.default_rng(seed)

    all_types = list(pitch_types) + list(junk_pitch_types)
    rows = []

    for ptype in all_types:
        for season in seasons:
            n = n_per_type
            pitcher_ids = rng.integers(0, n_pitchers, size=n)

            release_speed = rng.normal(92, 3, size=n)
            release_spin_rate = rng.normal(2300, 150, size=n)
            release_extension = rng.normal(6.3, 0.3, size=n)
            spin_axis = rng.uniform(0, 360, size=n)
            release_pos_x = rng.normal(-1.5, 0.3, size=n)
            release_pos_y = np.full(n, 54.0)
            release_pos_z = rng.normal(5.9, 0.2, size=n)

            # Roughly physical: horizontal velocity small vs. vy0 (toward plate).
            vx0 = rng.normal(0, 4, size=n)
            vy0 = -(release_speed * 1.467)  # mph -> ft/s, toward home plate
            vz0 = rng.normal(-4, 2, size=n)

            ax = rng.normal(0, 6, size=n)
            ay = rng.normal(30, 3, size=n)  # deceleration (drag), same sign as needed for a real solution
            az = rng.normal(-16, 2, size=n)  # gravity + Magnus, downward

            balls = rng.integers(0, 4, size=n)
            strikes = rng.integers(0, 3, size=n)
            outs_when_up = rng.integers(0, 3, size=n)
            stand = rng.choice(["L", "R"], size=n)
            p_throws = rng.choice(["L", "R"], size=n)

            sz_top = np.full(n, 3.5)
            sz_bot = np.full(n, 1.5)
            plate_x = rng.normal(0, 0.8, size=n)
            plate_z = rng.normal(2.5, 0.6, size=n)

            # Give location a real (if noisy) relationship to run value, so
            # location.py's model has genuine signal to find.
            noise = rng.normal(0, 0.3, size=n)
            delta_pitcher_run_exp = (
                -0.05 * np.abs(plate_x) - 0.05 * np.abs(plate_z - 2.5) + noise
            )

            # A rough 3x3 in-zone grid from plate_x/plate_z, matching
            # Statcast's real 1-9 `zone` numbering closely enough for tests
            # that exercise bestpitch.py's zone-grid logic (not testing zone
            # semantics themselves).
            plate_z_rel = (plate_z - sz_bot) / (sz_top - sz_bot)
            zone_row = np.where(plate_z_rel > 2 / 3, 0, np.where(plate_z_rel > 1 / 3, 1, 2))
            zone_col = np.where(plate_x < -0.28, 0, np.where(plate_x < 0.28, 1, 2))
            zone = zone_row * 3 + zone_col + 1

            on_1b = np.where(rng.random(n) < 0.3, 1.0, np.nan)
            on_2b = np.where(rng.random(n) < 0.3, 1.0, np.nan)
            on_3b = np.where(rng.random(n) < 0.3, 1.0, np.nan)

            df = pd.DataFrame(
                {
                    "pitcher": pitcher_ids,
                    "player_name": [f"Pitcher {i}" for i in pitcher_ids],
                    "pitch_type": ptype,
                    "game_year": season,
                    "release_speed": release_speed,
                    "release_spin_rate": release_spin_rate,
                    "spin_axis": spin_axis,
                    "release_pos_x": release_pos_x,
                    "release_pos_y": release_pos_y,
                    "release_pos_z": release_pos_z,
                    "release_extension": release_extension,
                    "vx0": vx0,
                    "vy0": vy0,
                    "vz0": vz0,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "balls": balls,
                    "strikes": strikes,
                    "outs_when_up": outs_when_up,
                    "stand": stand,
                    "p_throws": p_throws,
                    "plate_x": plate_x,
                    "plate_z": plate_z,
                    "sz_top": sz_top,
                    "sz_bot": sz_bot,
                    "zone": zone,
                    "on_1b": on_1b,
                    "on_2b": on_2b,
                    "on_3b": on_3b,
                    "delta_pitcher_run_exp": delta_pitcher_run_exp,
                }
            )
            rows.append(df)

    result = pd.concat(rows, ignore_index=True)

    # KEY_COLS: only needs to be unique per row for test purposes.
    result["game_pk"] = np.arange(len(result))
    result["at_bat_number"] = 0
    result["pitch_number"] = 0

    if shuffled_index:
        result = result.sample(frac=1.0, random_state=seed)

    return result


@pytest.fixture
def make_raw_df():
    return _build_raw_df


@pytest.fixture
def small_stuff_thresholds(monkeypatch):
    monkeypatch.setattr(stuff, "MIN_GROUP_SIZE_FOR_PCA", 50)
    monkeypatch.setattr(stuff, "MIN_PITCHES_FOR_SCORE", 5)


@pytest.fixture
def small_location_thresholds(monkeypatch):
    monkeypatch.setattr(location, "MIN_GROUP_SIZE_FOR_MODEL", 50)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SCORE", 5)
    monkeypatch.setattr(location, "MIN_PITCHES_FOR_SEASON_SCORE", 10)
