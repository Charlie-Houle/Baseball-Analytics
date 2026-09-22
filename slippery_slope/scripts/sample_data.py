"""
Small league-wide Statcast sample for the pitch-location app, pulled live via
pybaseball instead of requiring this repo's full data/MLB_2021-2025.csv. A
five-season, 3.5M-row CSV isn't something someone trying the app for the
first time should have to fetch and load first; a week of one season is
enough pitches to fill out the count-leverage buckets for most pitchers and
loads in a minute or two.

Public entry points are `load_sample` (a date range, every pitcher) and
`load_pitcher_season` (one pitcher's whole season, for when a selected
pitcher's own slice of `load_sample`'s small window isn't enough pitches to
say anything stable about them -- see docs/dev_log.md).
"""

import pybaseball as pb

pb.cache.enable()  # keeps a second run of the same request off the network

# A plain, unremarkable week (no opening day, trade deadline, or postseason
# quirks) from the 2025 season, the most recent year this repo's checked-in
# analysis covers. Not special beyond "known to have real completed data" --
# any range works, this is just a sane default for the app's date picker.
DEFAULT_START_DATE = "2025-06-02"
DEFAULT_END_DATE = "2025-06-08"

SAMPLE_COLUMNS = [
    "game_pk", "at_bat_number", "pitch_number", "pitcher", "player_name",
    "pitch_type", "pitch_name", "game_year", "balls", "strikes",
    "outs_when_up", "stand", "p_throws", "plate_x", "plate_z",
    "sz_top", "sz_bot", "zone", "on_1b", "on_2b", "on_3b",
]


def _trim_and_sort(raw, empty_message):
    if raw is None or raw.empty:
        raise ValueError(empty_message)

    missing = [col for col in SAMPLE_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"pybaseball's response is missing expected columns: {missing}")

    sample = raw[SAMPLE_COLUMNS].copy()
    return sample.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def load_sample(start_date, end_date):
    """
    Every pitch thrown league-wide between start_date and end_date
    (YYYY-MM-DD, inclusive), trimmed to SAMPLE_COLUMNS and sorted the same
    way this repo's other notebooks/scripts sort raw Statcast data (by
    game/at-bat/pitch number).
    """

    raw = pb.statcast(start_dt=start_date, end_dt=end_date, verbose=False)
    return _trim_and_sort(raw, f"No Statcast pitches found between {start_date} and {end_date}.")


def load_pitcher_season(pitcher_id, season):
    """
    Every pitch one pitcher (MLBAM id) threw in one calendar year. Unlike
    load_sample, this hits pybaseball's statcast_pitcher(), which filters to
    that one pitcher server-side rather than pulling the whole league and
    filtering locally -- a full season for one pitcher is a few thousand
    rows and loads in seconds, nowhere near load_sample's per-day cost.
    """

    raw = pb.statcast_pitcher(f"{season}-01-01", f"{season}-12-31", pitcher_id)
    return _trim_and_sort(raw, f"No Statcast pitches found for pitcher_id {pitcher_id} in {season}.")
