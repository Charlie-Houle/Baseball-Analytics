"""
Pure data-shaping and figure-building for the pitch-location app: takes a
raw pitch sample (see sample_data.py) and produces the same two views as
notebooks/fastball_location.ipynb's Stage A/B, generalized to any pitch
type and pitcher rather than fastballs league-wide. No Streamlit dependency
here on purpose, so app.py stays UI-only and this module stays unit-testable
without a running app.

Deferred, out of scope here: that notebook's Stage C ("meatball gap" against
Location+'s trained model) needs pitching_plus's cached models, which need a
full run over the 3.5M-row dataset to build -- too heavy a dependency for an
app meant to load its own data in a couple of minutes. See docs/dev_log.md.
"""

import matplotlib

matplotlib.use("Agg")  # figures are handed to Streamlit/tests, never shown from this process directly

import matplotlib.pyplot as plt
import plotly.graph_objects as go

try:
    # Works when imported as slippery_slope.scripts.location_view (tests,
    # or anything importing this as part of the package).
    from pitching_plus.scripts.location import JUNK_PITCH_TYPES, _build_features
    from .count_leverage import BUCKET_LABELS, COUNT_ORDER, count_bucket, zone_bucket
except ImportError:
    # Works when app.py runs this module as a plain script (streamlit run),
    # with pitching_plus/scripts and this file's own directory added to
    # sys.path -- see app.py.
    from location import JUNK_PITCH_TYPES, _build_features
    from count_leverage import BUCKET_LABELS, COUNT_ORDER, count_bucket, zone_bucket

REQUIRED_FOR_FEATURES = [
    "balls", "strikes", "outs_when_up", "stand", "p_throws",
    "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
]

ZONE_HALF_WIDTH_FT = 0.83  # standard plate-width-plus-ball-radius approximation, same as the notebook
ZONE_COLORS = {"Heart": "#c0392b", "Corner": "#eda100", "Edge": "#2a78d6", "Chase/Ball": "#c3c2b7"}

MIN_HEXBIN_GRIDSIZE = 6
MAX_HEXBIN_GRIDSIZE = 25


def _hexbin_gridsize(n):
    """
    A fixed gridsize=25 shows real texture over a league-wide background
    (thousands of points per bucket), but just speckle over a single
    pitcher's own pitches (tens to a couple hundred per bucket, even with a
    full season -- see sample_data.load_pitcher_season).
    Scaling gridsize down with sqrt(n) keeps each hexagon covering roughly
    the same number of points either way, floored so a small panel still
    shows more than one or two blobs.
    """
    return max(MIN_HEXBIN_GRIDSIZE, min(MAX_HEXBIN_GRIDSIZE, int(n ** 0.5)))


def engineer_pitches(sample):
    """
    Filters out junk pitch types and rows missing a required field, then adds
    the zone-relative/arm-side location features (pitching_plus's own
    location.py), the count-leverage bucket, and the zone-code bucket.
    """

    in_scope = sample[~sample["pitch_type"].isin(JUNK_PITCH_TYPES)].dropna(subset=REQUIRED_FOR_FEATURES).copy()
    engineered = _build_features(in_scope)
    engineered["bucket"] = [count_bucket(b, s) for b, s in zip(engineered["balls"], engineered["strikes"])]
    engineered["zone_bucket"] = engineered["zone"].apply(zone_bucket)
    engineered["count_label"] = [f"{b}-{s}" for b, s in zip(engineered["balls"], engineered["strikes"])]
    return engineered


def build_comparison_rows(selected_types, compare_types, background_source, overlay_source=None, overlay_label=None):
    """
    Builds location_figure's `rows` argument: one row per pitch type when
    `compare_types` is True, or a single row pooling every selected type
    together when it's False.

    `background_source` and `overlay_source` are already-engineered pitch
    frames (see engineer_pitches) -- not necessarily the same one, since a
    focused pitcher's own overlay may come from a full-season pull (see
    sample_data.load_pitcher_season) while the background stays a smaller
    league-wide window, or vice versa for a pitcher-vs-pitcher comparison.
    Each is filtered down to a row's pitch type(s) here; whichever rows
    (player, date range, etc.) belong in each source is the caller's call.
    """

    type_groups = [[pitch_type] for pitch_type in selected_types] if compare_types else [list(selected_types)]
    rows = []
    for types_in_row in type_groups:
        background = background_source[background_source["pitch_type"].isin(types_in_row)]
        overlay = overlay_source[overlay_source["pitch_type"].isin(types_in_row)] if overlay_source is not None else None
        rows.append({
            "label": "/".join(types_in_row),
            "background": background,
            "overlay": overlay,
            "overlay_label": overlay_label,
        })
    return rows


def location_figure(rows, title_suffix, overlay_alpha=1.0):
    """
    Hexbin of plate_x_armside/plate_z_rel, one row of three count-leverage-
    bucket panels per entry in `rows` (see build_comparison_rows), matching
    fastball_location.ipynb's Stage A chart (matplotlib, since Plotly has no
    native hexbin equivalent). A row's background is its comparison group's
    density; if that row also has an overlay, that pitcher's own pitches are
    drawn as individual points on top of it instead of replacing it, at
    `overlay_alpha` opacity (a solid point cloud reads as too heavy over a
    hexbin once there's more than a handful of overlay points).
    """

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.6 * n_rows), sharex=True, sharey=True, squeeze=False)
    has_overlay = any(row["overlay"] is not None for row in rows)

    for row_idx, row in enumerate(rows):
        background, overlay, overlay_label = row["background"], row["overlay"], row["overlay_label"]
        for col_idx, bucket in enumerate(BUCKET_LABELS):
            ax = axes[row_idx][col_idx]
            bg = background[background["bucket"] == bucket]
            if len(bg) >= 2:
                ax.hexbin(
                    bg["plate_x_armside"], bg["plate_z_rel"],
                    gridsize=_hexbin_gridsize(len(bg)), cmap="inferno", mincnt=1,
                )
            elif len(bg) == 1:
                ax.scatter(bg["plate_x_armside"], bg["plate_z_rel"], color="black", s=15)

            title = f"{bucket}\n(n={len(bg):,})"
            if overlay is not None:
                ov = overlay[overlay["bucket"] == bucket]
                ax.scatter(
                    ov["plate_x_armside"], ov["plate_z_rel"],
                    facecolors="none", edgecolors="#39d6ff", linewidths=1.1, s=32, zorder=5,
                    alpha=overlay_alpha,
                    label=overlay_label if row_idx == 0 and col_idx == 0 else None,
                )
                title += f", {overlay_label}: {len(ov):,}"

            ax.plot(
                [-ZONE_HALF_WIDTH_FT, ZONE_HALF_WIDTH_FT, ZONE_HALF_WIDTH_FT, -ZONE_HALF_WIDTH_FT, -ZONE_HALF_WIDTH_FT],
                [0, 0, 1, 1, 0],
                color="black", linewidth=1.2,
            )
            ax.set_title(title, fontsize=9 if n_rows > 1 else 10)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-1.2, 2.2)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Horizontal location (ft from plate center,\n+ = pitcher's arm side)")

        y_label = "Vertical location\n(0 = bottom of batter's zone, 1 = top)"
        if n_rows > 1:
            y_label = f"{row['label']}\n\n{y_label}"
        axes[row_idx][0].set_ylabel(y_label)

    if has_overlay:
        axes[0][0].legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.suptitle(f"Pitch location by count-leverage bucket (black box = strike zone) -- {title_suffix}")
    fig.tight_layout()
    return fig


def _zone_share_by_count(engineered):
    zone_dist = engineered.groupby(["count_label", "zone_bucket"]).size().unstack(fill_value=0)
    return zone_dist.div(zone_dist.sum(axis=1), axis=0).reindex(COUNT_ORDER)


def zone_share_figure(engineered, title_suffix, comparison=None, comparison_label=None):
    """
    Zone-code (Heart/Edge/Corner/Chase-Ball) share by count, matching Stage
    B's chart. If `comparison` is given (another already-engineered pitch
    frame, e.g. the same league/pitcher background shown in location_figure),
    its own zone-share lines are drawn dashed in the same per-category colors
    rather than a second color scale, so solid-vs-dashed reads as "who," and
    a small black solid/dashed key names which is which.
    """

    zone_dist_pct = _zone_share_by_count(engineered)

    fig = go.Figure()
    for category in ["Heart", "Corner", "Edge", "Chase/Ball"]:
        if category not in zone_dist_pct.columns:
            continue
        fig.add_trace(go.Scatter(
            x=COUNT_ORDER, y=zone_dist_pct[category], mode="lines+markers", name=category,
            line=dict(color=ZONE_COLORS[category], width=2.5),
        ))

    if comparison is not None and not comparison.empty:
        comparison_pct = _zone_share_by_count(comparison)
        for category in ["Heart", "Corner", "Edge", "Chase/Ball"]:
            if category not in comparison_pct.columns:
                continue
            fig.add_trace(go.Scatter(
                x=COUNT_ORDER, y=comparison_pct[category], mode="lines", showlegend=False,
                line=dict(color=ZONE_COLORS[category], width=2, dash="dash"),
                hovertemplate=f"{category} ({comparison_label})<br>%{{x}}: %{{y:.1%}}<extra></extra>",
            ))
        # Legend-only key lines (no real x/y data) naming which side is
        # solid vs. dashed, since the per-category colors already cover
        # the four zone groups and can't also encode "who."
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="black", width=2.5), name=title_suffix))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="black", width=2, dash="dash"), name=comparison_label))

    fig.update_layout(
        title=f"Where in the zone pitches go, by count -- {title_suffix}",
        xaxis=dict(title="Count (balls-strikes), sorted pitcher-ahead to hitter-ahead -- not alphabetical", showgrid=False),
        yaxis=dict(title="Share of pitches in this zone group", tickformat=".0%", showgrid=False),
        template="plotly_white",
        # A transparent legend was hard to read wherever it overlapped a line;
        # an opaque background with a light border keeps it legible anywhere
        # plotly places it, comparison key included. Font color is explicit
        # too -- Streamlit's own plotly theming can override a figure's
        # inherited text color (e.g. in dark mode) without touching an
        # already-set background, which left white text on the new white box.
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1,
            font=dict(color="#111111"),
        ),
    )
    return fig
