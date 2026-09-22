from slippery_slope.scripts.pitch_groups import PITCH_CATEGORIES, PITCH_TYPE_CATEGORY, types_for_categories


def test_every_pitch_type_maps_to_a_listed_category():
    assert set(PITCH_TYPE_CATEGORY.values()) == set(PITCH_CATEGORIES)


def test_types_for_categories_single_category():
    assert types_for_categories(["Fastball"]) == {"FF", "SI", "FC"}


def test_types_for_categories_unions_multiple_categories():
    fastball = types_for_categories(["Fastball"])
    breaking = types_for_categories(["Breaking"])
    assert types_for_categories(["Fastball", "Breaking"]) == fastball | breaking


def test_types_for_categories_empty_selection_returns_empty_set():
    assert types_for_categories([]) == set()
