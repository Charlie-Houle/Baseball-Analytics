"""
Count-leverage buckets and zone-code groupings for the pitch-location app.

Both are fixed lookups, not something the app recomputes from whatever
sample it happens to load. COUNT_BUCKET is notebooks/fastball_location.ipynb's
Stage 0 result (resulting BB% minus resulting K% per count, ranked over the
full 2021-2025 dataset, 3.5M+ pitches); a few thousand pitches from one
week's live Statcast pull isn't enough to re-rank the near-tied counts
(1-0 and 2-1 differ by 0.002 in bb_minus_k) without the rank order jumping
around week to week. See that notebook for the full derivation and
docs/dev_log.md for why a resulting-outcome measure was used over a raw
cumulative run-value one.
"""

BUCKET_LABELS = ["Pitcher-ahead", "Even", "Hitter-ahead"]

# (balls, strikes) -> bucket. Source: fastball_location.ipynb's Stage 0
# leverage_rank table, full 2021-2025 Statcast.
COUNT_BUCKET = {
    (0, 2): "Pitcher-ahead",
    (1, 2): "Pitcher-ahead",
    (0, 1): "Pitcher-ahead",
    (2, 2): "Pitcher-ahead",
    (1, 1): "Even",
    (0, 0): "Even",
    (1, 0): "Even",
    (2, 1): "Even",
    (3, 2): "Hitter-ahead",
    (2, 0): "Hitter-ahead",
    (3, 1): "Hitter-ahead",
    (3, 0): "Hitter-ahead",
}

# Same 12 counts, ordered pitcher-favorable to hitter-favorable, kept apart
# from COUNT_BUCKET so chart x-axes can follow that order instead of sorting
# count labels alphabetically.
COUNT_ORDER = ["0-2", "1-2", "0-1", "2-2", "1-1", "0-0", "1-0", "2-1", "3-2", "2-0", "3-1", "3-0"]

HEART_ZONES = {5}
EDGE_ZONES = {2, 4, 6, 8}
CORNER_ZONES = {1, 3, 7, 9}


def count_bucket(balls, strikes):
    """Pitcher-ahead/Even/Hitter-ahead bucket for a (balls, strikes) count, or None if unrecognized."""
    return COUNT_BUCKET.get((balls, strikes))


def counts_by_bucket():
    """
    BUCKET_LABELS -> its counts as "b-s" strings, ordered by COUNT_ORDER.
    Built from COUNT_BUCKET rather than written out separately, so app.py's
    on-screen explanation of what each bucket contains can't drift out of
    sync with the actual mapping.
    """
    by_bucket = {label: [] for label in BUCKET_LABELS}
    for count_label in COUNT_ORDER:
        balls, strikes = (int(x) for x in count_label.split("-"))
        by_bucket[COUNT_BUCKET[(balls, strikes)]].append(count_label)
    return by_bucket


def zone_bucket(zone):
    """
    Groups Statcast's own zone codes (1-9 in-zone, 11-14 chase/waste
    corners) into Heart/Edge/Corner/Chase-Ball, matching
    fastball_location.ipynb's Stage B and the zone scope
    pitching_plus/scripts/bestpitch.py's CANDIDATE_ZONE_CODES searches.
    """
    if zone in HEART_ZONES:
        return "Heart"
    if zone in EDGE_ZONES:
        return "Edge"
    if zone in CORNER_ZONES:
        return "Corner"
    return "Chase/Ball"
