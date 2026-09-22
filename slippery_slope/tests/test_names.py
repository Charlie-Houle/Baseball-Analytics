from slippery_slope.scripts.names import to_first_last


def test_to_first_last_simple_name():
    assert to_first_last("Rasmussen, Drew") == "Drew Rasmussen"


def test_to_first_last_keeps_a_suffix_with_the_last_name():
    assert to_first_last("Leiter Jr., Mark") == "Mark Leiter Jr."
    assert to_first_last("Trivino III, Lou") == "Lou Trivino III"


def test_to_first_last_passes_through_a_name_with_no_comma():
    assert to_first_last("All pitchers") == "All pitchers"


def test_to_first_last_preserves_mixed_case_last_names():
    assert to_first_last("deGrom, Jacob") == "Jacob deGrom"
