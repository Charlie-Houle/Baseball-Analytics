from slippery_slope.scripts.count_leverage import (
    BUCKET_LABELS,
    COUNT_BUCKET,
    COUNT_ORDER,
    count_bucket,
    counts_by_bucket,
    zone_bucket,
)


def test_count_bucket_covers_every_real_count():
    # Every (balls, strikes) pair a real at-bat can reach: 4 ball counts x 3
    # strike counts. A plate appearance ends the instant it would reach 4
    # balls or 3 strikes, so those combinations never come up.
    for balls in range(4):
        for strikes in range(3):
            assert count_bucket(balls, strikes) in BUCKET_LABELS


def test_count_bucket_unknown_count_returns_none():
    assert count_bucket(4, 3) is None


def test_count_order_matches_count_bucket_keys():
    assert set(COUNT_ORDER) == {f"{b}-{s}" for b, s in COUNT_BUCKET}


def test_zone_bucket_known_codes():
    assert zone_bucket(5) == "Heart"
    assert zone_bucket(2) == "Edge"
    assert zone_bucket(4) == "Edge"
    assert zone_bucket(1) == "Corner"
    assert zone_bucket(9) == "Corner"


def test_zone_bucket_chase_and_unknown_codes_fall_back_to_chase_ball():
    assert zone_bucket(11) == "Chase/Ball"
    assert zone_bucket(14) == "Chase/Ball"
    assert zone_bucket(99) == "Chase/Ball"


def test_counts_by_bucket_matches_count_bucket_and_covers_every_count():
    by_bucket = counts_by_bucket()
    assert set(by_bucket) == set(BUCKET_LABELS)

    # Round-tripping every listed count through count_bucket() should land back
    # in the bucket it was filed under -- this is app.py's on-screen source of
    # truth for what each bucket means, so it can't disagree with count_bucket().
    for bucket, count_labels in by_bucket.items():
        for count_label in count_labels:
            balls, strikes = (int(x) for x in count_label.split("-"))
            assert count_bucket(balls, strikes) == bucket

    all_listed = [count_label for count_labels in by_bucket.values() for count_label in count_labels]
    assert sorted(all_listed) == sorted(COUNT_ORDER)
