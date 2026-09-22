"""
Interactive expansion of notebooks/fastball_location.ipynb: pick any pitch
type (or a whole Fastball/Breaking/Offspeed group) and optionally one
pitcher, and see where those pitches land relative to the strike zone
across the three count-leverage buckets, plus each bucket's zone-code mix.
Loads its own small Statcast sample live via pybaseball (see sample_data.py)
rather than requiring this repo's full data/MLB_2021-2025.csv.

Run with `streamlit run slippery_slope/scripts/app.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# streamlit runs this file directly (not as part of the pitching_plus
# package), so pitching_plus/scripts needs to go on sys.path by hand, the
# same way the notebooks do, before location_view (below) imports from it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pitching_plus" / "scripts"))

from sample_data import DEFAULT_END_DATE, DEFAULT_START_DATE, load_pitcher_season, load_sample  # noqa: E402
from location_view import build_comparison_rows, engineer_pitches, location_figure, zone_share_figure  # noqa: E402
from count_leverage import counts_by_bucket  # noqa: E402
from names import to_first_last  # noqa: E402
from pitch_groups import PITCH_CATEGORIES, types_for_categories  # noqa: E402

LEAGUE_LABEL = "League Average"
ANOTHER_PITCHER_LABEL = "Another pitcher"
DEFAULT_OVERLAY_ALPHA = 0.6  # full opacity made the overlay points read as too heavy against the hexbin

# Day-count for each preset (inclusive of both endpoints). Pitch counts are
# approximate: measured once against the default 1-week range (27,228
# pitches, so ~3,890/day) and scaled -- the real count moves with how many
# games are actually on the slate (off days, doubleheaders, the All-Star
# break), so treat these as ballpark, not a guarantee.
SAMPLE_WINDOW_DAYS = {
    "1 day (~4K pitches)": 1,
    "1 week (~27K pitches)": 7,
    "~100K pitches (26 days)": 26,
}
CUSTOM_RANGE_LABEL = "Custom range"


@st.cache_data(show_spinner=False)
def _cached_sample(start_date, end_date):
    return load_sample(start_date, end_date)


@st.cache_data(show_spinner=False)
def _cached_pitcher_season(pitcher_id, season):
    return load_pitcher_season(pitcher_id, season)


def _load_pitcher_engineered(engineered, player_name, spinner_verb="Loading"):
    """
    A full season for one pitcher (see sample_data.load_pitcher_season),
    engineered the same way as the main sample -- used wherever a single
    pitcher is the actual subject (their overlay, their zone-share chart,
    their own arsenal), since the main sample's window is deliberately small
    and a specific pitcher's slice of it is too thin to read much into. Falls
    back to that pitcher's own rows from the already-loaded window if the
    season pull fails, so one bad request doesn't take down the whole page.
    """
    pitcher_rows = engineered.loc[engineered["player_name"] == player_name]
    pitcher_id = int(pitcher_rows["pitcher"].iloc[0])
    season = int(pitcher_rows["game_year"].iloc[0])
    display_name = to_first_last(player_name)
    with st.spinner(f"{spinner_verb} {display_name}'s full {season} season..."):
        try:
            season_raw = _cached_pitcher_season(pitcher_id, season)
            return engineer_pitches(season_raw)
        except ValueError as exc:
            st.warning(f"Couldn't load a season sample for {display_name} ({exc}); using only the loaded window instead.")
            return pitcher_rows


def _explanation():
    by_bucket = counts_by_bucket()
    with st.expander("What am I looking at?"):
        st.markdown(
            "**Count-leverage buckets** group the 12 ball-strike counts by how much the "
            "resulting at-bat favors the pitcher or the hitter (walk rate minus strikeout "
            "rate among every at-bat that ever reached that count, full 2021-2025 Statcast -- "
            "see `slippery_slope/notebooks/fastball_location.ipynb`'s Stage 0), not by the raw "
            "ball/strike numbers:\n"
            f"- **Pitcher-ahead**: {', '.join(by_bucket['Pitcher-ahead'])}\n"
            f"- **Even**: {', '.join(by_bucket['Even'])}\n"
            f"- **Hitter-ahead**: {', '.join(by_bucket['Hitter-ahead'])}\n\n"
            "**Location plot axes**: the horizontal axis is distance from the center of the "
            "plate, in feet, with positive always meaning the *pitcher's* arm side -- so a "
            "lefty's and a righty's arm-side misses land on the same side of the chart instead "
            "of mirrored. The vertical axis is relative to *that batter's own* strike zone "
            "(0 = bottom, 1 = top), so hitters of different heights line up too. The black "
            "outline is the strike zone.\n\n"
            "**Zone groups** (the second chart): Heart is the middle of the zone (Statcast "
            "zone 5), Edge is the four zone-code regions just inside the border (2/4/6/8), "
            "Corner is the four true corners (1/3/7/9), and Chase/Ball is everything outside "
            "the strike zone. That chart's x-axis is ordered by the same leverage ranking as "
            "the buckets above, not alphabetically -- reading left to right is pitcher-ahead "
            "to hitter-ahead.\n\n"
            "**Comparing pitch types or pitchers**: with a pitcher selected, the density map "
            "(the hexbin shading) is always the *background* -- league average by default, or "
            "another pitcher if you pick one under \"Compare against\" -- and the selected "
            "pitcher's own pitches, if the overlay checkbox is on, are the individual points "
            "drawn on top. \"Compare pitch types side by side\" gives each selected pitch type "
            "its own row of three buckets instead of pooling them into one.\n\n"
            "**A selected pitcher's own points come from their whole season**, not just the "
            "sample window above, since a specific pitcher's slice of a one-week league sample "
            "is too thin to plot on its own -- the league-average background still reflects "
            "just the loaded window.\n\n"
            "**On the zone-share chart**, the comparison group's own lines (league average, or "
            "another pitcher) are added dashed, in the same color per zone group as the "
            "selected pitcher's solid lines -- a small black solid/dashed key in the legend "
            "names which is which. Turn it off with \"Show comparison lines on the rate "
            "chart\" if it's too busy."
        )


def main():
    st.set_page_config(page_title="Slippery Slope: Pitch Location by Count", layout="wide")
    st.title("Pitch location by count leverage")
    st.caption(
        "Expands fastball_location.ipynb's fastball-only, league-wide analysis to any pitch "
        "type and an optional single-pitcher filter, over a small live Statcast sample."
    )
    _explanation()

    with st.sidebar:
        st.header("Sample")
        end_date_choice = st.date_input("End date", pd.Timestamp(DEFAULT_END_DATE))
        window_choice = st.selectbox("Sample window", list(SAMPLE_WINDOW_DAYS) + [CUSTOM_RANGE_LABEL], index=1)
        if window_choice == CUSTOM_RANGE_LABEL:
            start_date_choice = st.date_input("Start date", pd.Timestamp(DEFAULT_START_DATE))
        else:
            start_date_choice = end_date_choice - pd.Timedelta(days=SAMPLE_WINDOW_DAYS[window_choice] - 1)
        start_date, end_date = start_date_choice.isoformat(), end_date_choice.isoformat()
        st.caption("A bigger window fills out each bucket more but takes longer to load.")

    with st.spinner(f"Loading Statcast pitches, {start_date} to {end_date}..."):
        try:
            sample = _cached_sample(start_date, end_date)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    engineered = engineer_pitches(sample)
    st.caption(
        f"{len(sample):,} pitches loaded, {len(engineered):,} in scope after dropping junk "
        "pitch types and rows missing a required field."
    )

    with st.sidebar:
        st.header("Pitcher")
        # Statcast's own player_name is "Last, First," which is why sorting
        # these raw strings already sorts by last name; format_func displays
        # "First Last" without disturbing that order or the values these
        # widgets actually return (still the raw "Last, First" strings, so
        # every dataframe filter below keeps working unchanged).
        all_pitchers = sorted(engineered["player_name"].dropna().unique())
        pitcher_options = ["All pitchers"] + all_pitchers
        selected_pitcher = st.selectbox(
            "Pitcher", pitcher_options,
            format_func=lambda name: to_first_last(name) if name != "All pitchers" else name,
        )

        pitcher_engineered = None
        background_pitcher, background_engineered = None, None
        overlay_pitcher = None
        overlay_alpha = DEFAULT_OVERLAY_ALPHA
        show_rate_comparison = False

        if selected_pitcher != "All pitchers":
            pitcher_engineered = _load_pitcher_engineered(engineered, selected_pitcher)

            other_pitchers = [p for p in all_pitchers if p != selected_pitcher]
            compare_options = [LEAGUE_LABEL] + ([ANOTHER_PITCHER_LABEL] if other_pitchers else [])
            compare_against = st.radio("Compare against", compare_options, horizontal=True)
            if compare_against == ANOTHER_PITCHER_LABEL:
                background_pitcher = st.selectbox("Comparison pitcher", other_pitchers, format_func=to_first_last)
                background_engineered = _load_pitcher_engineered(
                    engineered, background_pitcher, spinner_verb="Loading comparison pitcher"
                )

            if st.checkbox(f"Overlay {to_first_last(selected_pitcher)}'s pitches", value=True):
                overlay_pitcher = selected_pitcher
                overlay_alpha = st.slider("Overlay point opacity", 0.1, 1.0, DEFAULT_OVERLAY_ALPHA, step=0.05)

            show_rate_comparison = st.checkbox(
                "Show comparison lines on the rate chart",
                value=True,
                help="Adds the comparison group's zone-share lines, dashed, in the same colors.",
            )

        st.header("Filter")
        pitch_types = sorted(engineered["pitch_type"].dropna().unique())
        if pitcher_engineered is not None:
            # A pitcher's full season can include a type this window's small
            # sample never happened to catch.
            pitch_types = sorted(set(pitch_types) | set(pitcher_engineered["pitch_type"].dropna().unique()))

        # session_state already holding a type no longer in `pitch_types`
        # (e.g. after changing the date range or the selected pitcher) would
        # make the multiselect below raise, since a selected value has to be
        # one of its current options.
        if "pitch_type_select" in st.session_state:
            st.session_state["pitch_type_select"] = [
                pt for pt in st.session_state["pitch_type_select"] if pt in pitch_types
            ]

        def _apply_category_selection():
            wanted = types_for_categories(st.session_state["pitch_category_select"])
            st.session_state["pitch_type_select"] = [pt for pt in pitch_types if pt in wanted]

        st.multiselect(
            "Bulk select by category", PITCH_CATEGORIES, default=["Fastball"],
            key="pitch_category_select", on_change=_apply_category_selection,
            help="Sets the pitch-type list below to every type in the chosen group(s); still editable by hand after.",
        )
        if "pitch_type_select" not in st.session_state:
            _apply_category_selection()  # seed pitch_type_select from the category default on first run

        if pitcher_engineered is not None:
            def _limit_to_pitcher_types():
                st.session_state["pitch_type_select"] = sorted(pitcher_engineered["pitch_type"].dropna().unique())

            st.button(f"Limit to {to_first_last(selected_pitcher)}'s own pitch types", on_click=_limit_to_pitcher_types)

        selected_types = st.multiselect("Pitch type(s)", pitch_types, key="pitch_type_select")

        compare_types = False
        if len(selected_types) >= 2:
            compare_types = st.checkbox(
                "Compare pitch types side by side",
                help="One row of buckets per pitch type instead of pooling them into one.",
            )

    if not selected_types:
        st.warning("Pick at least one pitch type.")
        st.stop()

    background_source = background_engineered if background_engineered is not None else engineered
    overlay_source = pitcher_engineered if overlay_pitcher else None

    background_in_scope = background_source[background_source["pitch_type"].isin(selected_types)]
    overlay_in_scope = overlay_source[overlay_source["pitch_type"].isin(selected_types)] if overlay_source is not None else None
    if background_in_scope.empty and (overlay_in_scope is None or overlay_in_scope.empty):
        st.warning("No pitches match this pitch-type selection for either side of the comparison. Try different pitch types or a wider sample.")
        st.stop()

    rows = build_comparison_rows(
        selected_types, compare_types, background_source,
        overlay_source=overlay_source, overlay_label=to_first_last(overlay_pitcher) if overlay_pitcher else None,
    )

    background_desc = to_first_last(background_pitcher) if background_pitcher else LEAGUE_LABEL
    title_suffix = f"{background_desc} (background)"
    if overlay_pitcher:
        title_suffix += f" vs {to_first_last(overlay_pitcher)} (points)"

    st.pyplot(location_figure(rows, title_suffix, overlay_alpha=overlay_alpha))

    zone_source = pitcher_engineered if selected_pitcher != "All pitchers" else engineered
    zone_title_suffix = to_first_last(selected_pitcher) if selected_pitcher != "All pitchers" else "all pitchers"
    scoped = zone_source[zone_source["pitch_type"].isin(selected_types)]

    zone_comparison, zone_comparison_label = None, None
    if show_rate_comparison:
        zone_comparison, zone_comparison_label = background_in_scope, background_desc

    if scoped.empty:
        st.info(f"{to_first_last(selected_pitcher)} has no pitches of the selected type(s) in this sample, so there's no zone-share chart to show.")
    else:
        st.plotly_chart(
            zone_share_figure(scoped, zone_title_suffix, comparison=zone_comparison, comparison_label=zone_comparison_label),
            width="stretch",
            theme=None,  # let the figure's own plotly_white styling stand, not Streamlit's (dark-mode) overlay
        )


if __name__ == "__main__":
    main()
