import matplotlib.figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from slippery_slope.scripts import location_view


@pytest.fixture
def sample_pitches():
    n = 20
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "pitch_type": ["FF"] * (n // 2) + ["KN"] * (n // 2),  # KN is a JUNK_PITCH_TYPES entry
            "player_name": ["Test Pitcher"] * n,
            "balls": ([0] * 5 + [3] * 5) * 2,
            "strikes": ([0] * 5 + [2] * 5) * 2,
            "outs_when_up": [0] * n,
            "stand": ["R"] * n,
            "p_throws": ["R"] * n,
            "plate_x": rng.normal(0, 0.5, size=n),
            "plate_z": rng.normal(2.5, 0.5, size=n),
            "sz_top": [3.5] * n,
            "sz_bot": [1.5] * n,
            "zone": [5, 11] * (n // 2),  # alternating, so the surviving FF rows mix Heart and Chase/Ball
            "on_1b": [np.nan] * n,
            "on_2b": [np.nan] * n,
            "on_3b": [np.nan] * n,
        }
    )


def test_engineer_pitches_drops_junk_pitch_types(sample_pitches):
    engineered = location_view.engineer_pitches(sample_pitches)
    assert set(engineered["pitch_type"]) == {"FF"}


def test_engineer_pitches_assigns_bucket_and_zone_bucket(sample_pitches):
    engineered = location_view.engineer_pitches(sample_pitches)
    # 0-0 -> Even, 3-2 -> Hitter-ahead (count_leverage's own lookup, checked separately in
    # test_count_leverage.py; this just confirms engineer_pitches actually wires it up).
    assert set(engineered["bucket"].dropna()) <= {"Even", "Hitter-ahead"}
    assert set(engineered["zone_bucket"]) <= {"Heart", "Chase/Ball"}


def test_engineer_pitches_drops_rows_missing_a_required_field(sample_pitches):
    sample_pitches.loc[0, "plate_x"] = np.nan
    engineered = location_view.engineer_pitches(sample_pitches)
    assert len(engineered) == len(sample_pitches[sample_pitches["pitch_type"] == "FF"]) - 1


@pytest.fixture
def two_pitcher_two_type_pitches():
    def rows(n, pitch_type, player_name, balls, strikes, zone):
        return {
            "pitch_type": [pitch_type] * n,
            "player_name": [player_name] * n,
            "balls": [balls] * n,
            "strikes": [strikes] * n,
            "outs_when_up": [0] * n,
            "stand": ["R"] * n,
            "p_throws": ["R"] * n,
            "plate_x": [0.1] * n,
            "plate_z": [2.5] * n,
            "sz_top": [3.5] * n,
            "sz_bot": [1.5] * n,
            "zone": [zone] * n,
            "on_1b": [np.nan] * n,
            "on_2b": [np.nan] * n,
            "on_3b": [np.nan] * n,
        }

    groups = [
        rows(4, "FF", "Test Pitcher", 0, 0, 5),
        rows(4, "FF", "Other Pitcher", 0, 0, 11),
        rows(4, "SL", "Test Pitcher", 3, 2, 5),
        rows(4, "KN", "Test Pitcher", 0, 0, 5),  # junk, should never survive engineer_pitches
    ]
    combined = {key: sum((group[key] for group in groups), []) for key in groups[0]}
    return pd.DataFrame(combined)


def test_build_comparison_rows_pools_selected_types_by_default(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    rows = location_view.build_comparison_rows(["FF", "SL"], False, engineered)
    assert len(rows) == 1
    assert rows[0]["label"] == "FF/SL"
    assert len(rows[0]["background"]) == 12  # 4 FF/Test + 4 FF/Other + 4 SL/Test
    assert rows[0]["overlay"] is None


def test_build_comparison_rows_splits_one_row_per_type_when_comparing(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    rows = location_view.build_comparison_rows(["FF", "SL"], True, engineered)
    assert [row["label"] for row in rows] == ["FF", "SL"]
    assert len(rows[0]["background"]) == 8
    assert len(rows[1]["background"]) == 4


def test_build_comparison_rows_uses_the_given_background_source_not_the_full_pool(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    other_pitcher_only = engineered[engineered["player_name"] == "Other Pitcher"]
    rows = location_view.build_comparison_rows(["FF", "SL"], True, other_pitcher_only)
    assert len(rows[0]["background"]) == 4  # Other Pitcher's own FF rows
    assert len(rows[1]["background"]) == 0  # Other Pitcher never threw a slider in this fixture


def test_build_comparison_rows_overlay_source_carries_into_every_row(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    test_pitcher_only = engineered[engineered["player_name"] == "Test Pitcher"]
    rows = location_view.build_comparison_rows(
        ["FF", "SL"], True, engineered, overlay_source=test_pitcher_only, overlay_label="Test Pitcher",
    )
    assert all(row["overlay_label"] == "Test Pitcher" for row in rows)
    assert len(rows[0]["overlay"]) == 4
    assert len(rows[1]["overlay"]) == 4


def test_build_comparison_rows_no_overlay_source_means_no_overlay(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    rows = location_view.build_comparison_rows(["FF"], False, engineered)
    assert rows[0]["overlay"] is None
    assert rows[0]["overlay_label"] is None


def test_location_figure_returns_one_row_of_three_panels_per_comparison_row(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    rows = location_view.build_comparison_rows(["FF", "SL"], True, engineered)
    fig = location_view.location_figure(rows, "league average (background)")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 6  # 2 comparison rows x 3 count-leverage buckets


def test_location_figure_with_overlay_labels_the_legend(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    test_pitcher_only = engineered[engineered["player_name"] == "Test Pitcher"]
    rows = location_view.build_comparison_rows(
        ["FF"], False, engineered, overlay_source=test_pitcher_only, overlay_label="Test Pitcher",
    )
    fig = location_view.location_figure(rows, "league average (background) vs Test Pitcher (points)")
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert legend.texts[0].get_text() == "Test Pitcher"


def test_location_figure_applies_overlay_alpha_to_the_overlay_points(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    test_pitcher_only = engineered[engineered["player_name"] == "Test Pitcher"]
    rows = location_view.build_comparison_rows(
        ["FF"], False, engineered, overlay_source=test_pitcher_only, overlay_label="Test Pitcher",
    )
    fig = location_view.location_figure(rows, "league average (background) vs Test Pitcher (points)", overlay_alpha=0.35)
    overlay_collection = fig.axes[0].collections[-1]  # hexbin (if any) is added first, the overlay scatter last
    assert overlay_collection.get_alpha() == pytest.approx(0.35)


def test_hexbin_gridsize_scales_with_n_between_its_floor_and_ceiling():
    assert location_view._hexbin_gridsize(4) == location_view.MIN_HEXBIN_GRIDSIZE
    assert location_view._hexbin_gridsize(10_000) == location_view.MAX_HEXBIN_GRIDSIZE
    assert location_view.MIN_HEXBIN_GRIDSIZE < location_view._hexbin_gridsize(200) < location_view.MAX_HEXBIN_GRIDSIZE


def test_zone_share_figure_returns_a_trace_per_zone_bucket_present(sample_pitches):
    engineered = location_view.engineer_pitches(sample_pitches)
    fig = location_view.zone_share_figure(engineered, "FF, Test Pitcher")
    assert isinstance(fig, go.Figure)
    # Only Heart and Chase/Ball are present in the fixture (see sample_pitches' zone column).
    assert {trace.name for trace in fig.data} == {"Heart", "Chase/Ball"}


def test_zone_share_figure_gives_the_legend_an_opaque_background(sample_pitches):
    engineered = location_view.engineer_pitches(sample_pitches)
    fig = location_view.zone_share_figure(engineered, "FF, Test Pitcher")
    assert fig.layout.legend.bgcolor is not None


def test_zone_share_figure_has_no_gridlines(sample_pitches):
    engineered = location_view.engineer_pitches(sample_pitches)
    fig = location_view.zone_share_figure(engineered, "FF, Test Pitcher")
    assert fig.layout.xaxis.showgrid is False
    assert fig.layout.yaxis.showgrid is False


def test_zone_share_figure_with_comparison_adds_dashed_lines_and_a_key(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    test_pitcher_only = engineered[engineered["player_name"] == "Test Pitcher"]
    other_pitcher_only = engineered[engineered["player_name"] == "Other Pitcher"]

    fig = location_view.zone_share_figure(
        test_pitcher_only, "Test Pitcher", comparison=other_pitcher_only, comparison_label="Other Pitcher",
    )

    key_traces = [trace for trace in fig.data if trace.line.color == "black"]
    assert {trace.name for trace in key_traces} == {"Test Pitcher", "Other Pitcher"}
    assert {trace.line.dash for trace in key_traces} == {None, "dash"}

    # Test Pitcher's FF rows are all zone 5 (Heart); Other Pitcher's are all zone 11 (Chase/Ball).
    data_traces = [trace for trace in fig.data if trace.line.color != "black"]
    dashed_data_traces = [trace for trace in data_traces if trace.line.dash == "dash"]
    assert len(dashed_data_traces) == 1
    assert dashed_data_traces[0].showlegend is False


def test_zone_share_figure_skips_comparison_when_it_is_empty(two_pitcher_two_type_pitches):
    engineered = location_view.engineer_pitches(two_pitcher_two_type_pitches)
    test_pitcher_only = engineered[engineered["player_name"] == "Test Pitcher"]
    empty = engineered.iloc[0:0]

    fig = location_view.zone_share_figure(test_pitcher_only, "Test Pitcher", comparison=empty, comparison_label="Nobody")

    assert all(trace.line.dash != "dash" for trace in fig.data)
    assert "Nobody" not in {trace.name for trace in fig.data}
