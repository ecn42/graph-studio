# app.py
# Streamlit Graph Studio – customizable charts with transparent export
# Author: Eduardo + T3 Chat
# License: MIT

from __future__ import annotations

import html
import io
import json
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from PIL import Image, ImageColor

# ----------------------------- Page config --------------------------------- #

st.set_page_config(
    page_title="Ceres Graph Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Kaleido Fix -------------------------------------------- #
# New API (replaces pio.kaleido.scope.*)
pio.defaults.image_engine = "kaleido"  # ensure Kaleido is used
pio.defaults.image_format = "png"  # default export format
pio.defaults.mathjax = None  # avoid external MathJax loads
pio.defaults.image_export_timeout = 30000  # ms timeout for export

# Chromium args for stability in containers/WSL/CI and some Linux/macOS setups
pio.defaults.chromium_args = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-software-rasterizer",
    "--headless=new",
]

# ----------------------------- Helpers ------------------------------------- #


def load_data(upload, excel_sheet: Optional[str] = None) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    name = upload.name.lower()
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            return pd.read_csv(upload)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            # Respect selected sheet if provided
            return pd.read_excel(upload, sheet_name=excel_sheet)
        if name.endswith(".parquet"):
            return pd.read_parquet(upload)
        if name.endswith(".json"):
            return pd.read_json(upload)
        st.error("Unsupported file type. Use CSV/Excel/Parquet/JSON.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return pd.DataFrame()


def get_excel_sheets(upload) -> List[str]:
    try:
        raw = upload.getvalue()
        with pd.ExcelFile(io.BytesIO(raw)) as xls:
            return xls.sheet_names
    except Exception as e:
        st.error(f"Could not inspect Excel sheets: {e}")
        return []


def make_sample_data(kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    if kind == "Stocks timeseries (synthetic)":
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        data = []
        for t in tickers:
            lvl = rng.uniform(80, 200)
            noise = rng.normal(0, 1.8, size=n)
            trend = np.linspace(0, rng.uniform(-20, 40), n)
            price = np.maximum(1, lvl + trend + noise).round(2)
            data.append(pd.DataFrame({"Date": dates, "Ticker": t, "Close": price}))
        df = pd.concat(data, ignore_index=True)
        wide = df.pivot(index="Date", columns="Ticker", values="Close").reset_index()
        return wide
    if kind == "Iris":
        n = 150
        species = np.array(["setosa", "versicolor", "virginica"])
        df = pd.DataFrame(
            {
                "sepal_length": np.round(np.random.normal(5.8, 0.8, n), 2),
                "sepal_width": np.round(np.random.normal(3.0, 0.4, n), 2),
                "petal_length": np.round(np.random.normal(3.7, 1.8, n), 2),
                "petal_width": np.round(np.random.normal(1.1, 0.7, n), 2),
                "species": species[np.random.randint(0, 3, n)],
            }
        )
        return df
    return pd.DataFrame()


def parse_color_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    tokens = re.split(r"[,\s]+", s)
    colors = []
    for t in tokens:
        if not t:
            continue
        if re.match(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$", t):
            colors.append(t)
        else:
            colors.append(t)
    return colors


def rgba_to_hex_and_alpha(color: str) -> Tuple[str, float]:
    """
    Accepts rgba(...) | rgb(...) | #hex | name.
    Returns (#RRGGBB, alpha). If cannot parse, returns default (#e1e5ea, 1.0).
    """
    if not isinstance(color, str):
        return "#e1e5ea", 1.0
    c = color.strip()
    try:
        if c.startswith("rgba("):
            nums = c[5:-1].split(",")
            r, g, b = [int(float(nums[i])) for i in range(3)]
            a = float(nums[3])
            return "#{:02X}{:02X}{:02X}".format(r, g, b), max(0.0, min(1.0, a))
        if c.startswith("rgb("):
            nums = c[4:-1].split(",")
            r, g, b = [int(float(nums[i])) for i in range(3)]
            return "#{:02X}{:02X}{:02X}".format(r, g, b), 1.0
        # hex or named
        rgb = ImageColor.getrgb(c)
        return "#{:02X}{:02X}{:02X}".format(*rgb), 1.0
    except Exception:
        return "#e1e5ea", 1.0


def hex_to_rgba(color_hex: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, float(alpha)))
    try:
        r, g, b = ImageColor.getrgb(color_hex)
    except Exception:
        r, g, b = (225, 229, 234)
    return f"rgba({int(r)},{int(g)},{int(b)},{alpha})"


def to_long_df(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    color_col: Optional[str],
) -> pd.DataFrame:
    id_vars = [x_col] + ([color_col] if color_col else [])
    long = df.melt(
        id_vars=id_vars,
        value_vars=y_cols,
        var_name="Series",
        value_name="Value",
    )

    # Create proper color grouping
    if color_col and len(y_cols) > 1:
        long["ColorGroup"] = (
            long["Series"].astype(str) + " • " + long[color_col].astype(str)
        )
    elif color_col and len(y_cols) == 1:
        long["ColorGroup"] = long[color_col].astype(str)
    elif len(y_cols) > 1:
        long["ColorGroup"] = long["Series"].astype(str)
    else:
        long["ColorGroup"] = None

    return long


def apply_rolling(
    long_df: pd.DataFrame,
    x_col: str,
    color_key: Optional[str],
    window: int,
) -> pd.DataFrame:
    if window <= 1:
        return long_df
    df = long_df.copy()
    if color_key and color_key in df.columns:
        df = (
            df.sort_values([color_key, x_col])
            .assign(
                Value=lambda d: d.groupby(color_key)["Value"].transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                )
            )
        )
    else:
        df = df.sort_values([x_col])
        df["Value"] = df["Value"].rolling(window, min_periods=1).mean()
    return df


def add_glow_traces(
    fig: go.Figure,
    chart_type: str,
    series_glow_settings: dict,
    series_display_map: dict,
) -> None:
    """
    For Line/Scatter charts, add a wider, semi-transparent duplicate line under
    each eligible trace to simulate a glow.
    """
    if chart_type not in ["Line", "Scatter"]:
        return

    # Build reverse map from display -> original y_col
    disp_to_y = {v: k for k, v in series_display_map.items()}

    original_traces = list(fig.data)
    fig.data = ()

    for tr in original_traces:
        # Determine base display name (strip color group suffix if present)
        name = getattr(tr, "name", None)
        base_name = name.split(" • ", 1)[0] if isinstance(name, str) else None

        # Figure out which y_col (if any) this trace maps to
        y_key = None
        if base_name and base_name in disp_to_y:
            y_key = disp_to_y[base_name]
        elif len(series_display_map) == 1:
            # Single-series case, tolerate missing names
            y_key = next(iter(series_display_map.keys()))

        # Only apply to line-capable scatter traces
        mode = getattr(tr, "mode", "") or ""
        is_liney = (tr.type == "scatter") and ("lines" in mode or chart_type == "Line")

        if (
            y_key
            and is_liney
            and series_glow_settings.get(y_key, {}).get("enabled", False)
        ):
            glow_cfg = series_glow_settings[y_key]
            boost = float(glow_cfg.get("width_boost", 8.0))
            glow_alpha = float(glow_cfg.get("opacity", 0.6))

            # Attempt to derive the base RGB from the trace
            base_color = None
            if hasattr(tr, "line") and tr.line and tr.line.color:
                base_color = tr.line.color
            elif hasattr(tr, "marker") and tr.marker and tr.marker.color:
                base_color = tr.marker.color

            # Normalize to rgba string with chosen opacity
            rgba_color = None
            if isinstance(base_color, str):
                hex_part, _ = rgba_to_hex_and_alpha(base_color)
                rgba_color = hex_to_rgba(hex_part, glow_alpha)
            else:
                rgba_color = hex_to_rgba("#000000", glow_alpha)

            base_width = 2.0
            try:
                if hasattr(tr, "line") and tr.line and tr.line.width:
                    base_width = float(tr.line.width)
            except Exception:
                pass

            glow_trace = go.Scatter(
                x=getattr(tr, "x", None),
                y=getattr(tr, "y", None),
                mode="lines",
                line=dict(
                    width=base_width + boost,
                    color=rgba_color,
                    dash=getattr(tr.line, "dash", None) if tr.line else None,
                    shape=getattr(tr.line, "shape", None) if tr.line else None,
                ),
                name=name,
                showlegend=False,
                hoverinfo="skip",
                legendgroup=getattr(tr, "legendgroup", None),
                xaxis=getattr(tr, "xaxis", None),
                yaxis=getattr(tr, "yaxis", None),
                opacity=1.0,
                connectgaps=getattr(tr, "connectgaps", False),
            )
            # Add glow first (so it stays behind)
            fig.add_trace(glow_trace)

        # Add the original trace on top
        fig.add_trace(tr)


def build_figure(
    chart_type: str,
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    y_col_names: dict,
    color_col: Optional[str],
    parse_dates: bool,
    sort_x: bool,
    rolling_window: int,
    colorway: Optional[List[str]],
    palette_theme: dict,
    use_palette_theme: bool,
    width_px: int,
    height_px: int,
    bg_transparent: bool,
    paper_bg: str,
    plot_bg: str,
    font_color: str,
    # Per-axis grid controls
    show_x_grid: bool,
    x_grid_color: str,
    x_grid_width: float,
    show_y_grid: bool,
    y_grid_color: str,
    y_grid_width: float,
    # Per-axis line controls
    show_x_line: bool,
    x_line_color: str,
    x_line_width: float,
    show_y_line: bool,
    y_line_color: str,
    y_line_width: float,
    show_legend: bool,
    legend_orientation: str,
    legend_x: float,
    legend_y: float,
    legend_anchor: str,
    legend_title: str,
    font_family: str,
    base_font_size: int,
    title: str,
    title_x: float,
    title_y: float,
    x_title: str,
    x_title_standoff: int,
    x_tickformat: Optional[str],
    x_tickprefix: str,
    x_ticksuffix: str,
    x_tickangle: int,
    y_title: str,
    y_title_standoff: int,
    # New Y-axis tick label controls
    y_tickformat: Optional[str],
    y_tickprefix: str,
    y_ticksuffix: str,
    y_tickangle: int,
    y_log: bool,
    y2_series: List[str],
    y2_title: str,
    y2_title_standoff: int,
    # New Y2 tick label controls
    y2_tickformat: Optional[str],
    y2_tickprefix: str,
    y2_ticksuffix: str,
    y2_tickangle: int,
    y2_log: bool,
    line_width: float,
    line_dash: str,
    markers: bool,
    marker_size: int,
    barmode: str,
    candlestick_cols: dict,
    series_avg_settings: dict,
    series_rolling_settings: dict,
    series_glow_settings: dict,
    avg_line_color: str,
    rolling_line_color: str,
    watermark_img,
    watermark_opacity: float,
    watermark_size: float,
    watermark_x: float,
    watermark_y: float,
    logo_img,
    logo_size: float,
    logo_x: float,
    logo_y: float,
    swap_axes: bool,
    # Pie-specific options (ignored for non-Pie)
    pie_labels_col: Optional[str] = None,
    pie_values_col: Optional[str] = None,
    pie_hole: float = 0.0,
    pie_text_template: Optional[str] = None,
    pie_text_position: str = "inside",
    pie_sort: bool = True,
    pie_direction: str = "counterclockwise",
    pie_start_angle: int = 0,
    # Pie/Bar small-category grouping
    pie_group_small: bool = False,
    pie_group_threshold: float = 0.0,
    pie_group_others_label: str = "Others",
    bar_group_small: bool = False,
    bar_group_threshold: float = 0.0,
    bar_group_by: str = "x",  # "x" or "color"
    bar_group_others_label: str = "Others",
) -> go.Figure:
    data = df.copy()
    y2_set = set(y2_series or [])

    if parse_dates:
        try:
            data[x_col] = pd.to_datetime(data[x_col], errors="coerce")
            # Ensure JS/Plotly-friendly precision (ms) and remove tz
            data[x_col] = data[x_col].dt.tz_localize(None).astype("datetime64[ms]")
        except Exception:
            pass

    if sort_x:
        data = data.sort_values(by=x_col)

    # Apply palette theme colors if enabled (but do not override grid_color)
    if use_palette_theme and palette_theme:
        if not bg_transparent:
            paper_bg = palette_theme.get("paper_bg", paper_bg)
            plot_bg = palette_theme.get("plot_bg", plot_bg)
        font_color = palette_theme.get("font_color", font_color)

    if chart_type == "Candlestick":
        if swap_axes:
            st.warning(
                "Axis swap is not supported for Candlestick; showing normal orientation."
            )
        o, h, l, c = (
            candlestick_cols["open"],
            candlestick_cols["high"],
            candlestick_cols["low"],
            candlestick_cols["close"],
        )
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data[x_col],
                    open=data[o],
                    high=data[h],
                    low=data[l],
                    close=data[c],
                    increasing_line_color="#16a34a",
                    decreasing_line_color="#dc2626",
                    increasing_fillcolor="rgba(22,163,74,0.7)",
                    decreasing_fillcolor="rgba(220,38,38,0.7)",
                )
            ]
        )
        fig.update_xaxes(
            title=dict(
                text=x_title, font=dict(color=font_color), standoff=x_title_standoff
            ),
            type="date",
            tickformat=x_tickformat if x_tickformat else None,
            tickprefix=x_tickprefix or None,
            ticksuffix=x_ticksuffix or None,
            tickangle=int(x_tickangle),
            showgrid=show_x_grid,
            gridcolor=x_grid_color,
            gridwidth=x_grid_width,
            tickfont=dict(color=font_color),
            showline=show_x_line,
            linecolor=x_line_color,
            linewidth=x_line_width,
        )
        fig.update_layout(
            yaxis=dict(
                title=dict(
                    text=y_title,
                    font=dict(color=font_color),
                    standoff=y_title_standoff,
                ),
                type="log" if y_log else "linear",
                tickformat=y_tickformat if y_tickformat else None,
                tickprefix=y_tickprefix or None,
                ticksuffix=y_ticksuffix or None,
                tickangle=int(y_tickangle),
                showgrid=show_y_grid,
                gridcolor=y_grid_color,
                gridwidth=y_grid_width,
                tickfont=dict(color=font_color),
                showline=show_y_line,
                linecolor=y_line_color,
                linewidth=y_line_width,
            )
        )
    elif chart_type == "Pie":
        # Determine label/value columns
        if not pie_labels_col or not pie_values_col:
            num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in data.columns if c not in num_cols]
            pie_labels_col = pie_labels_col or (
                cat_cols[0] if cat_cols else data.columns[0]
            )
            pie_values_col = pie_values_col or (
                num_cols[0] if num_cols else data.columns[-1]
            )

        df_pie = data[[pie_labels_col, pie_values_col]].copy()

        # Group small categories into "Others"
        if pie_group_small and pie_group_threshold > 0:
            totals = (
                df_pie.groupby(pie_labels_col, dropna=False)[pie_values_col]
                .sum()
                .sort_values(ascending=False)
            )
            small = set(totals[totals < float(pie_group_threshold)].index)
            if small:
                df_pie["_labels"] = df_pie[pie_labels_col].apply(
                    lambda v: pie_group_others_label if v in small else v
                )
                df_pie = (
                    df_pie.groupby("_labels", dropna=False)[pie_values_col]
                    .sum()
                    .reset_index()
                    .rename(columns={"_labels": pie_labels_col})
                )

        fig = px.pie(
            df_pie,
            names=pie_labels_col,
            values=pie_values_col,
            color_discrete_sequence=colorway,
            hole=pie_hole,
        )

        # Text content and placement
        if pie_text_template and pie_text_template.strip():
            fig.update_traces(
                texttemplate=pie_text_template,
                textinfo="none",
                textposition=pie_text_position,
                sort=pie_sort,
                direction=pie_direction,
                rotation=int(pie_start_angle),
                textfont=dict(color=font_color),
                insidetextfont=dict(color=font_color),
                outsidetextfont=dict(color=font_color),
            )
        else:
            fig.update_traces(
                textinfo="percent",
                textposition=pie_text_position,
                sort=pie_sort,
                direction=pie_direction,
                rotation=int(pie_start_angle),
                textfont=dict(color=font_color),
                insidetextfont=dict(color=font_color),
                outsidetextfont=dict(color=font_color),
            )
    else:
        # Create long format with custom names
        long = to_long_df(data, x_col, y_cols, color_col)

        # Apply custom series names
        if y_col_names:
            long["SeriesDisplay"] = long["Series"].map(y_col_names).fillna(
                long["Series"]
            )
        else:
            long["SeriesDisplay"] = long["Series"]

        # Decide color column used by Plotly
        color_col_to_use = None
        if color_col and len(y_cols) > 1:
            color_col_to_use = "ColorGroup"
        elif color_col and len(y_cols) == 1:
            color_col_to_use = color_col
        elif len(y_cols) > 1:
            color_col_to_use = "SeriesDisplay"

        # Bar-specific: group small categories before rolling/plotting
        if chart_type == "Bar" and bar_group_small and bar_group_threshold > 0:
            if bar_group_by == "x":
                totals = long.groupby(x_col, dropna=False)["Value"].sum()
                small = set(totals[totals < float(bar_group_threshold)].index)
                if small:
                    long[x_col] = long[x_col].apply(
                        lambda v: bar_group_others_label if v in small else v
                    )
            elif bar_group_by == "color" and color_col_to_use is not None:
                totals = long.groupby(color_col_to_use, dropna=False)["Value"].sum()
                small = set(totals[totals < float(bar_group_threshold)].index)
                if small:
                    long[color_col_to_use] = long[color_col_to_use].apply(
                        lambda v: bar_group_others_label if v in small else v
                    )

        # Apply smoothing after grouping
        long = apply_rolling(long, x_col, "ColorGroup", rolling_window)

        # Create the main chart
        if chart_type == "Line":
            if not swap_axes:
                fig = px.line(
                    long,
                    x=x_col,
                    y="Value",
                    color=color_col_to_use,
                    markers=markers,
                    color_discrete_sequence=colorway,
                )
            else:
                fig = px.line(
                    long,
                    x="Value",
                    y=x_col,
                    color=color_col_to_use,
                    markers=markers,
                    color_discrete_sequence=colorway,
                )
        elif chart_type == "Area":
            if not swap_axes:
                fig = px.area(
                    long,
                    x=x_col,
                    y="Value",
                    color=color_col_to_use,
                    color_discrete_sequence=colorway,
                )
            else:
                fig = px.area(
                    long,
                    x="Value",
                    y=x_col,
                    color=color_col_to_use,
                    color_discrete_sequence=colorway,
                )
        elif chart_type == "Bar":
            if not swap_axes:
                fig = px.bar(
                    long,
                    x=x_col,
                    y="Value",
                    color=color_col_to_use,
                    color_discrete_sequence=colorway,
                )
            else:
                # horizontal bars when swapped
                fig = px.bar(
                    long,
                    x="Value",
                    y=x_col,
                    color=color_col_to_use,
                    color_discrete_sequence=colorway,
                    orientation="h",
                )
        elif chart_type == "Scatter":
            if not swap_axes:
                fig = px.scatter(
                    long,
                    x=x_col,
                    y="Value",
                    color=color_col_to_use,
                    opacity=0.9,
                    color_discrete_sequence=colorway,
                )
            else:
                fig = px.scatter(
                    long,
                    x="Value",
                    y=x_col,
                    color=color_col_to_use,
                    opacity=0.9,
                    color_discrete_sequence=colorway,
                )
        else:
            fig = go.Figure()

        # Add individual series averages and rolling averages
        for i, y_col in enumerate(y_cols):
            series_name = y_col_names.get(y_col, y_col) if y_col_names else y_col
            axis_for_series = "y2" if y_col in y2_set else "y"
            xaxis_for_series = "x2" if y_col in y2_set else "x"

            # Average line
            if (
                series_avg_settings.get(y_col, False)
                and chart_type in ["Line", "Area", "Scatter"]
            ):
                avg_val = data[y_col].mean()
                if not swap_axes:
                    # horizontal line across the plot
                    fig.add_shape(
                        type="line",
                        xref="paper",
                        x0=0,
                        x1=1,
                        yref=axis_for_series,
                        y0=avg_val,
                        y1=avg_val,
                        line=dict(dash="dash", color=avg_line_color, width=2),
                    )
                    fig.add_annotation(
                        x=1.0,
                        xref="paper",
                        y=avg_val,
                        yref=axis_for_series,
                        text=f"Avg {series_name}: {avg_val:.2f}",
                        showarrow=False,
                        xanchor="right",
                        yanchor="bottom" if i == 0 else "top",
                        font=dict(color=font_color),
                    )
                else:
                    # vertical line across the plot when swapped
                    fig.add_shape(
                        type="line",
                        xref=xaxis_for_series,
                        x0=avg_val,
                        x1=avg_val,
                        yref="paper",
                        y0=0,
                        y1=1,
                        line=dict(dash="dash", color=avg_line_color, width=2),
                    )
                    fig.add_annotation(
                        x=avg_val,
                        xref=xaxis_for_series,
                        y=1.0,
                        yref="paper",
                        text=f"Avg {series_name}: {avg_val:.2f}",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                        font=dict(color=font_color),
                    )

            # Rolling average
            if (
                series_rolling_settings.get(y_col, {}).get("enabled", False)
                and chart_type in ["Line", "Area", "Scatter"]
            ):
                window = series_rolling_settings[y_col].get("window", 10)
                rolling_avg = data[y_col].rolling(window=window, min_periods=1).mean()
                if not swap_axes:
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_col],
                            y=rolling_avg,
                            mode="lines",
                            line=dict(color=rolling_line_color, width=2, dash="dot"),
                            name=f"Rolling Avg {series_name} ({window})",
                            opacity=0.8,
                            yaxis=axis_for_series,
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=rolling_avg,
                            y=data[x_col],
                            mode="lines",
                            line=dict(color=rolling_line_color, width=2, dash="dot"),
                            name=f"Rolling Avg {series_name} ({window})",
                            opacity=0.8,
                            xaxis=xaxis_for_series,
                        )
                    )

        # Base styling of traces (before glow)
        if chart_type in ["Line", "Area"]:
            fig.update_traces(line=dict(width=line_width, dash=line_dash))
        if chart_type in ["Line", "Scatter"]:
            fig.update_traces(marker=dict(size=marker_size))
        if chart_type == "Bar":
            fig.update_layout(barmode=barmode)

        # Assign traces to secondary axis if requested (before glow)
        series_display_map = {
            y: (y_col_names.get(y, y) if y_col_names else y) for y in y_cols
        }
        if y2_set:
            disp_to_y = {v: k for k, v in series_display_map.items()}
            for tr in fig.data:
                name = getattr(tr, "name", None)
                base_name = name.split(" • ", 1)[0] if isinstance(name, str) else None
                y_key = None
                if base_name and base_name in disp_to_y:
                    y_key = disp_to_y[base_name]
                elif len(series_display_map) == 1:
                    y_key = next(iter(series_display_map.keys()))
                if y_key and y_key in y2_set:
                    if not swap_axes:
                        tr.update(yaxis="y2")
                    else:
                        tr.update(xaxis="x2")

        # Apply glow per-series for Line/Scatter
        add_glow_traces(fig, chart_type, series_glow_settings, series_display_map)

        # Axes titles and type
        xaxis_title_text = y_title if swap_axes else x_title
        xaxis_kwargs = dict(
            title=dict(
                text=xaxis_title_text,
                font=dict(color=font_color),
                standoff=x_title_standoff,
            )
        )
        # Apply types and formats when not swapped (X is on xaxis)
        if not swap_axes:
            if parse_dates:
                xaxis_kwargs.update(type="date")
            if x_tickformat:
                xaxis_kwargs["tickformat"] = x_tickformat
            if x_tickprefix:
                xaxis_kwargs["tickprefix"] = x_tickprefix
            if x_ticksuffix:
                xaxis_kwargs["ticksuffix"] = x_ticksuffix
            xaxis_kwargs["tickangle"] = int(x_tickangle)

        fig.update_xaxes(**xaxis_kwargs)

    # Layout configuration
    legend_dict = {
        "orientation": legend_orientation,
        "x": legend_x,
        "y": legend_y,
        "title": dict(text=legend_title) if legend_title else None,
        "font": dict(color=font_color),
    }

    # Add legend anchor positioning
    if legend_anchor == "top-left":
        legend_dict.update({"xanchor": "left", "yanchor": "top"})
    elif legend_anchor == "top-right":
        legend_dict.update({"xanchor": "right", "yanchor": "top"})
    elif legend_anchor == "bottom-left":
        legend_dict.update({"xanchor": "left", "yanchor": "bottom"})
    elif legend_anchor == "bottom-right":
        legend_dict.update({"xanchor": "right", "yanchor": "bottom"})
    elif legend_anchor == "center":
        legend_dict.update({"xanchor": "center", "yanchor": "middle"})

    fig.update_layout(
        width=width_px,
        height=height_px,
        showlegend=show_legend,
        legend=legend_dict,
        font=dict(family=font_family, size=base_font_size, color=font_color),
        title=(
            dict(text=title, x=title_x, y=title_y, font=dict(color=font_color))
            if title
            else {}
        ),
        colorway=colorway or px.colors.qualitative.Set2,
    )

    # Grid and axis styling – not applicable to Pie
    if chart_type not in ["Pie", "Candlestick"]:
        fig.update_xaxes(
            showgrid=show_x_grid,
            gridcolor=x_grid_color,
            gridwidth=x_grid_width,
            tickfont=dict(color=font_color),
            showline=show_x_line,
            linecolor=x_line_color,
            linewidth=x_line_width,
        )
        fig.update_yaxes(
            showgrid=show_y_grid,
            gridcolor=y_grid_color,
            gridwidth=y_grid_width,
            tickfont=dict(color=font_color),
            showline=show_y_line,
            linecolor=y_line_color,
            linewidth=y_line_width,
        )

    # Finalize primary/secondary axis configuration (titles, scales, tick labels)
    if chart_type not in ["Candlestick", "Pie"]:
        if not swap_axes:
            if y2_set:
                fig.update_layout(
                    yaxis=dict(
                        title=dict(
                            text=y_title,
                            font=dict(color=font_color),
                            standoff=y_title_standoff,
                        ),
                        type="log" if y_log else "linear",
                        tickformat=y_tickformat if y_tickformat else None,
                        tickprefix=y_tickprefix or None,
                        ticksuffix=y_ticksuffix or None,
                        tickangle=int(y_tickangle),
                        showgrid=show_y_grid,
                        gridcolor=y_grid_color,
                        gridwidth=y_grid_width,
                        tickfont=dict(color=font_color),
                        showline=show_y_line,
                        linecolor=y_line_color,
                        linewidth=y_line_width,
                    ),
                    yaxis2=dict(
                        title=dict(
                            text=y2_title,
                            font=dict(color=font_color),
                            standoff=y2_title_standoff,
                        ),
                        overlaying="y",
                        side="right",
                        type="log" if y2_log else "linear",
                        tickformat=y2_tickformat if y2_tickformat else None,
                        tickprefix=y2_tickprefix or None,
                        ticksuffix=y2_ticksuffix or None,
                        tickangle=int(y2_tickangle),
                        showgrid=show_y_grid,
                        gridcolor=y_grid_color,
                        gridwidth=y_grid_width,
                        tickfont=dict(color=font_color),
                        showline=show_y_line,
                        linecolor=y_line_color,
                        linewidth=y_line_width,
                    ),
                )
            else:
                fig.update_layout(
                    yaxis=dict(
                        title=dict(
                            text=y_title,
                            font=dict(color=font_color),
                            standoff=y_title_standoff,
                        ),
                        type="log" if y_log else "linear",
                        tickformat=y_tickformat if y_tickformat else None,
                        tickprefix=y_tickprefix or None,
                        ticksuffix=y_ticksuffix or None,
                        tickangle=int(y_tickangle),
                        showgrid=show_y_grid,
                        gridcolor=y_grid_color,
                        gridwidth=y_grid_width,
                        tickfont=dict(color=font_color),
                        showline=show_y_line,
                        linecolor=y_line_color,
                        linewidth=y_line_width,
                    )
                )
        else:
            # Swapped: Y shows original X; X shows original Y
            yaxis_type = "date" if parse_dates else "linear"
            fig.update_layout(
                yaxis=dict(
                    title=dict(
                        text=x_title,
                        font=dict(color=font_color),
                        standoff=y_title_standoff,
                    ),
                    type=yaxis_type,
                    showgrid=show_y_grid,
                    gridcolor=y_grid_color,
                    gridwidth=y_grid_width,
                    tickfont=dict(color=font_color),
                    showline=show_y_line,
                    linecolor=y_line_color,
                    linewidth=y_line_width,
                    tickformat=x_tickformat if x_tickformat else None,
                    tickprefix=x_tickprefix or None,
                    ticksuffix=x_ticksuffix or None,
                    tickangle=int(x_tickangle),
                ),
                xaxis=dict(
                    title=dict(
                        text=y_title,
                        font=dict(color=font_color),
                        standoff=x_title_standoff,
                    ),
                    type="log" if y_log else "linear",
                    showgrid=show_x_grid,
                    gridcolor=x_grid_color,
                    gridwidth=x_grid_width,
                    tickfont=dict(color=font_color),
                    showline=show_x_line,
                    linecolor=x_line_color,
                    linewidth=x_line_width,
                    tickformat=y_tickformat if y_tickformat else None,
                    tickprefix=y_tickprefix or None,
                    ticksuffix=y_ticksuffix or None,
                    tickangle=int(y_tickangle),
                ),
            )
            if y2_set:
                fig.update_layout(
                    xaxis2=dict(
                        title=dict(
                            text=y2_title,
                            font=dict(color=font_color),
                            standoff=y2_title_standoff,
                        ),
                        overlaying="x",
                        side="top",
                        type="log" if y2_log else "linear",
                        showgrid=show_x_grid,
                        gridcolor=x_grid_color,
                        gridwidth=x_grid_width,
                        tickfont=dict(color=font_color),
                        showline=show_x_line,
                        linecolor=x_line_color,
                        linewidth=x_line_width,
                        tickformat=y2_tickformat if y2_tickformat else None,
                        tickprefix=y2_tickprefix or None,
                        ticksuffix=y2_ticksuffix or None,
                        tickangle=int(y2_tickangle),  # <-- fixed here
                    )
                )

    # Background colors
    if bg_transparent:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig.update_layout(
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
        )
    # Add watermark
    if watermark_img is not None:
        try:
            import base64

            img = Image.open(watermark_img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_str}",
                    xref="paper",
                    yref="paper",
                    x=watermark_x,
                    y=watermark_y,
                    sizex=watermark_size,
                    sizey=watermark_size,
                    xanchor="center",
                    yanchor="middle",
                    opacity=watermark_opacity,
                    layer="below",
                )
            )
        except Exception as e:
            st.warning(f"Could not add watermark: {e}")

    # Add logo
    if logo_img is not None:
        try:
            import base64

            img = Image.open(logo_img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_str}",
                    xref="paper",
                    yref="paper",
                    x=logo_x,
                    y=logo_y,
                    sizex=logo_size,
                    sizey=logo_size,
                    xanchor="center",
                    yanchor="middle",
                    opacity=1.0,
                    layer="above",
                )
            )
        except Exception as e:
            st.warning(f"Could not add logo: {e}")

    return fig


def export_image(
    fig: go.Figure,
    fmt: str,
    width: int,
    height: int,
    scale: float,
    caption_text: Optional[str] = None,
    base_font_size: Optional[int] = None,
    font_color: Optional[str] = None,
    extra_l_margin: int = 10,
    caption_yshift: int = -12,
) -> bytes:
    """
    Export the figure to an image. If caption_text is provided, it will be added
    to the exported image as a footer (annotation) at the lower-left corner.
    The on-screen figure is not mutated.

    extra_l_margin: additional left margin (px) to avoid clipping the caption.
    caption_yshift: y-shift in px for the caption annotation (negative pushes
                    the caption down into the bottom margin).
    """
    try:
        fig2 = go.Figure(fig)

        if caption_text and caption_text.strip():
            safe_text = html.escape(caption_text.strip()).replace("\n", "<br>")
            n_lines = caption_text.count("\n") + 1
            cap_font_size = max(8, (base_font_size or 12) - 1)

            # Ensure enough bottom margin for caption and the downward shift
            extra_b_margin = max(
                40, int(cap_font_size * (n_lines + 1.2)) + max(0, -caption_yshift)
            )

            m = fig2.layout.margin or go.layout.Margin()
            m_l = m.l if m.l is not None else 60
            m_r = m.r if m.r is not None else 40
            m_t = m_t if (m_t := m.t) is not None else 60
            m_b = m_b if (m_b := m.b) is not None else 60
            fig2.update_layout(
                margin=dict(
                    l=m_l + int(extra_l_margin),
                    r=m_r,
                    t=m_t,
                    b=m_b + int(extra_b_margin),
                )
            )

            # Bottom-left caption
            fig2.add_annotation(
                xref="paper",
                yref="paper",
                x=0,
                y=0,
                xanchor="left",
                yanchor="top",
                text=safe_text,
                showarrow=False,
                align="left",
                font=dict(size=cap_font_size, color=font_color or "#2c3e50"),
                xshift=max(6, int(cap_font_size * 0.4)),
                yshift=int(caption_yshift),
            )

        img = pio.to_image(
            fig2,
            format=fmt,
            width=width,
            height=height,
            scale=scale,
            #engine="kaleido",
        )
        return img
    except Exception as e:
        st.error(
            "Image export failed with Kaleido 1.x.\n"
            "Tips:\n"
            "- Using new Plotly defaults API (pio.defaults.*)\n"
            "- Chromium flags set (--no-sandbox, --disable-dev-shm-usage, --headless=new)\n"
            "- Install basic fonts (DejaVu/Liberation) if on Linux/containers\n"
            "- Lower export scale/size; increase pio.defaults.image_export_timeout\n\n"
            f"Details: {e}"
        )
        return b""


# ----------------------- Layout profile helpers ---------------------------- #
def make_layout_profile(fig: go.Figure) -> dict:
    """
    Build a pure-layout profile (no data). This captures:
    - fonts, colors, colorway
    - sizes (width/height), margins
    - axes settings, grids, legends
    - title positions, annotations, shapes
    - layout images (watermarks/logos)
    """
    layout = fig.layout.to_plotly_json()
    return {
        "profile_version": 1,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "plotly_layout": layout,
    }


# ----------------------------- UI ------------------------------------------ #

st.title("📊 Graph Studio")
st.caption(
    "Create beautiful, modern charts with full customization. "
    "Export high-res images with transparent or colored backgrounds."
)

with st.sidebar:
    st.header("1) Data")
    upload = st.file_uploader("Upload data (CSV / Excel / Parquet / JSON)", type=None)
    excel_sheet = None
    # If Excel, allow worksheet selection
    if upload is not None and upload.name.lower().endswith((".xlsx", ".xls")):
        sheets = get_excel_sheets(upload)
        if sheets:
            excel_sheet = st.selectbox("Worksheet", sheets, index=0)

    sample_choice = st.selectbox(
        "Or pick a sample", ["—", "Stocks timeseries (synthetic)", "Iris"], index=0
    )

    df = load_data(upload, excel_sheet=excel_sheet)
    if df.empty and sample_choice != "—":
        df = make_sample_data(sample_choice)

    if df.empty:
        st.info("Upload a dataset or pick a sample to get started.")
    else:
        st.success(f"Loaded data with shape: {df.shape[0]} rows × {df.shape[1]} cols")

    st.header("2) Chart")
    chart_type = st.selectbox(
        "Chart type", ["Line", "Area", "Bar", "Scatter", "Pie", "Candlestick"]
    )

    st.header("3) Images")
    st.subheader("Watermark")
    watermark_img = st.file_uploader(
        "Upload watermark image", type=["png", "jpg", "jpeg"], key="watermark"
    )
    if watermark_img:
        watermark_opacity = st.slider("Watermark opacity", 0.0, 1.0, 0.3, 0.05)
        watermark_size = st.slider("Watermark size", 0.1, 1.0, 0.3, 0.05)
        watermark_x = st.slider("Watermark X position", 0.0, 1.0, 0.5, 0.05)
        watermark_y = st.slider("Watermark Y position", 0.0, 1.0, 0.5, 0.05)
    else:
        watermark_opacity = watermark_size = watermark_x = watermark_y = 0.0

    st.subheader("Logo")
    logo_img = st.file_uploader(
        "Upload logo image", type=["png", "jpg", "jpeg"], key="logo"
    )
    if logo_img:
        logo_size = st.slider("Logo size", 0.05, 0.5, 0.15, 0.01)
        logo_x = st.slider("Logo X position", 0.0, 1.0, 0.95, 0.05)
        logo_y = st.slider("Logo Y position", 0.0, 1.0, 0.95, 0.05)
    else:
        logo_size = logo_x = logo_y = 0.0

# Guard clause
if df.empty:
    st.stop()

# Column mapping controls
st.subheader("Mapping")
cols = list(df.columns)

# Guess a time-like column
date_like = next(
    (c for c in cols if "date" in c.lower() or "time" in c.lower()), cols[0]
)

if chart_type == "Candlestick":
    left, right = st.columns([1, 1])
    with left:
        x_col = st.selectbox("X (date/time)", cols, index=cols.index(date_like))
        parse_dates = st.toggle("Parse X as datetime", value=True)
        sort_x = st.toggle("Sort by X ascending", value=True)
    with right:
        c_open = st.selectbox("Open", cols)
        c_high = st.selectbox("High", cols)
        c_low = st.selectbox("Low", cols)
        c_close = st.selectbox("Close", cols)
        y_log = st.toggle("Log scale (Y)", value=False)

    y_cols = []
    color_col = None
    rolling_window = 1
    candlestick_cols = {"open": c_open, "high": c_high, "low": c_low, "close": c_close}
    series_avg_settings = {}
    series_rolling_settings = {}
    series_glow_settings = {}
    avg_line_color = rolling_line_color = "#000000"
    y_col_names = {}
elif chart_type == "Pie":
    left, right = st.columns([1, 1])
    with left:
        pie_labels_col = st.selectbox("Labels (categories)", cols, index=0)
        value_candidates = [c for c in cols if c != pie_labels_col]
        value_candidates = value_candidates or cols
        pie_values_col = st.selectbox("Values (numeric)", value_candidates, index=0)
        parse_dates = False
        sort_x = False
        y_log = False
    with right:
        st.caption("Tip: Fine-tune label content/position in Chart Options below.")

    # Placeholders for unused variables in Pie flow
    x_col = cols[0]
    y_cols = []
    color_col = None
    rolling_window = 1
    candlestick_cols = {}
    series_avg_settings = {}
    series_rolling_settings = {}
    series_glow_settings = {}
    avg_line_color = rolling_line_color = "#000000"
    y_col_names = {}
else:
    left, right = st.columns([1, 1])
    with left:
        x_col = st.selectbox("X axis", cols, index=cols.index(date_like))
        parse_dates = st.toggle("Parse X as datetime", value=("date" in x_col.lower()))
        sort_x = st.toggle("Sort by X ascending", value=True)
        y_options = [c for c in cols if c != x_col]
        y_default = y_options[: min(3, len(y_options))]
        y_cols = st.multiselect("Y columns (one or more)", y_options, default=y_default)
        rolling_window = st.number_input(
            "Data smoothing window",
            min_value=1,
            value=1,
            step=1,
            help="Apply rolling average to smooth data points",
        )
    with right:
        color_options = ["— (none)"] + [c for c in cols if c != x_col]
        color_choice = st.selectbox("Color/group by column", color_options, index=0)
        color_col = None if color_choice.startswith("—") else color_choice
        y_log = st.toggle("Log scale (Y)", value=False)

    candlestick_cols = {}

    if not y_cols:
        st.warning("Select at least one Y column.")
        st.stop()

# Style and Layout
st.subheader("Style and Layout")
c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.2, 1.2])

with c1:
    ENHANCED_PALETTES = {
        "Ceres Wealth": {
            "colors": [
                "#013220",
                "#8c6239",
                "#57575a",
                "#b08568",
                "#09202e",
                "#582308",
                "#7a6200",
            ],
            "paper_bg": "#333652",
            "plot_bg": "#818183",
            "font_color": "#ffffff",
            "grid_color": "rgba(0,0,0,0.8)",
        },
        "Custom Light": {
            "colors": px.colors.qualitative.Set2,
            "paper_bg": "#ffffff",
            "plot_bg": "#ffffff",
            "font_color": "#2c3e50",
            "grid_color": "rgba(0,0,0,0.08)",
        },
        "Custom Dark": {
            "colors": px.colors.qualitative.D3,
            "paper_bg": "#1a1a1a",
            "plot_bg": "#2d2d2d",
            "font_color": "#eaeaea",
            "grid_color": "rgba(255,255,255,0.1)",
        },
        "Dracula": {
            "colors": [
                "#bd93f9",
                "#ff79c6",
                "#50fa7b",
                "#8be9fd",
                "#ffb86c",
                "#f1fa8c",
                "#ff5555",
            ],
            "paper_bg": "#282a36",
            "plot_bg": "#44475a",
            "font_color": "#f8f8f2",
            "grid_color": "rgba(248,248,242,0.1)",
        },
        "Monokai": {
            "colors": [
                "#a6e22e",
                "#f92672",
                "#66d9ef",
                "#fd971f",
                "#ae81ff",
                "#e6db74",
                "#f8f8f2",
            ],
            "paper_bg": "#272822",
            "plot_bg": "#3e3d32",
            "font_color": "#f8f8f2",
            "grid_color": "rgba(248,248,242,0.08)",
        },
        "Catppuccin Mocha": {
            "colors": [
                "#f38ba8",
                "#a6e3a1",
                "#89b4fa",
                "#fab387",
                "#cba6f7",
                "#94e2d5",
                "#f2cdcd",
                "#b4befe",
            ],
            "paper_bg": "#1e1e2e",
            "plot_bg": "#313244",
            "font_color": "#cdd6f4",
            "grid_color": "rgba(205,214,244,0.1)",
        },
        "Catppuccin Latte": {
            "colors": [
                "#d20f39",
                "#40a02b",
                "#1e66f5",
                "#fe640b",
                "#8839ef",
                "#179299",
                "#df8e1d",
                "#ea76cb",
            ],
            "paper_bg": "#eff1f5",
            "plot_bg": "#ffffff",
            "font_color": "#4c4f69",
            "grid_color": "rgba(76,79,105,0.1)",
        },
        "Gruvbox Dark": {
            "colors": [
                "#fb4934",
                "#fabd2f",
                "#b8bb26",
                "#83a598",
                "#d3869b",
                "#8ec07c",
                "#fe8019",
            ],
            "paper_bg": "#282828",
            "plot_bg": "#3c3836",
            "font_color": "#ebdbb2",
            "grid_color": "rgba(235,219,178,0.1)",
        },
        "Gruvbox Light": {
            "colors": [
                "#9d0006",
                "#af3a03",
                "#b57614",
                "#79740e",
                "#076678",
                "#8f3f71",
                "#427b58",
            ],
            "paper_bg": "#fbf1c7",
            "plot_bg": "#ffffff",
            "font_color": "#3c3836",
            "grid_color": "rgba(60,56,54,0.1)",
        },
        "Nord": {
            "colors": [
                "#88c0d0",
                "#a3be8c",
                "#ebcb8b",
                "#bf616a",
                "#b48ead",
                "#5e81ac",
                "#d08770",
            ],
            "paper_bg": "#2e3440",
            "plot_bg": "#3b4252",
            "font_color": "#eceff4",
            "grid_color": "rgba(236,239,244,0.1)",
        },
        "Tokyo Night": {
            "colors": [
                "#7aa2f7",
                "#c0caf5",
                "#9ece6a",
                "#f7768e",
                "#bb9af7",
                "#e0af68",
                "#2ac3de",
            ],
            "paper_bg": "#1a1b26",
            "plot_bg": "#24283b",
            "font_color": "#c0caf5",
            "grid_color": "rgba(192,202,245,0.1)",
        },
        "Material Light": {
            "colors": [
                "#1e88e5",
                "#43a047",
                "#fb8c00",
                "#e53935",
                "#8e24aa",
                "#00acc1",
                "#3949ab",
            ],
            "paper_bg": "#fafafa",
            "plot_bg": "#ffffff",
            "font_color": "#212121",
            "grid_color": "rgba(33,33,33,0.08)",
        },
        "IBM Carbon": {
            "colors": [
                "#0f62fe",
                "#24a148",
                "#ff832b",
                "#da1e28",
                "#a56eff",
                "#1192e8",
                "#007d79",
            ],
            "paper_bg": "#161616",
            "plot_bg": "#262626",
            "font_color": "#f4f4f4",
            "grid_color": "rgba(244,244,244,0.1)",
        },
    }

    palette_name = st.selectbox("Palette", list(ENHANCED_PALETTES.keys()), index=0)
    use_palette_theme = st.toggle(
        "Use palette theme colors",
        value=True,
        help="Apply background, grid, and font colors from the palette",
    )

    custom_colors = st.text_input(
        "Custom colors (comma/space sep HEX/names)", value=""
    )
    colorway = parse_color_list(custom_colors) or ENHANCED_PALETTES[palette_name][
        "colors"
    ]
    reverse_palette = st.toggle("Reverse palette", value=False)
    if reverse_palette:
        colorway = list(reversed(colorway))

    palette_theme = (
        ENHANCED_PALETTES[palette_name]
        if isinstance(ENHANCED_PALETTES[palette_name], dict)
        else {}
    )

    # Inline palette preview (right below selector)
    st.write("Palette preview:")
    swatches = "".join(
        f"<span style='display:inline-block;width:20px;height:20px;"
        f"border-radius:4px;margin:2px;background:{c}' title='{c}'></span>"
        for c in colorway[:40]
    )
    st.markdown(swatches, unsafe_allow_html=True)
    if use_palette_theme and palette_theme:
        st.write("Theme colors:")
        theme_swatch = f"""
        <div style="display:flex;gap:10px;align-items:center;padding:10px;">
            <div style="width:30px;height:30px;background:{palette_theme.get('paper_bg', '#fff')};
                        border:1px solid #ddd;border-radius:4px;" title="Paper BG"></div>
            <div style="width:30px;height:30px;background:{palette_theme.get('plot_bg', '#fff')};
                        border:1px solid #ddd;border-radius:4px;" title="Plot BG"></div>
            <div style="width:30px;height:30px;background:{palette_theme.get('grid_color', '#eee')};
                        border:1px solid #ddd;border-radius:4px;" title="Grid"></div>
            <div style="width:30px;height:30px;background:{palette_theme.get('font_color', '#000')};
                        border:1px solid #ddd;border-radius:4px;" title="Font"></div>
        </div>
        """
        st.markdown(theme_swatch, unsafe_allow_html=True)

with c2:
    font_family = st.selectbox(
        "Font",
        [
            "Inter, Arial, sans-serif",
            "Arial, sans-serif",
            "Helvetica, sans-serif",
            "Roboto",
            "Lato",
        ],
    )
    base_font_size = st.slider("Base font size", 8, 24, 13, 1)
    font_color = st.color_picker(
        "Font color",
        palette_theme.get("font_color", "#2c3e50")
        if use_palette_theme
        else "#2c3e50",
    )

    # Title settings
    title = st.text_input("Title", value="")
    if title:
        title_x = st.slider("Title X position", 0.0, 1.0, 0.5, 0.01)
        title_y = st.slider("Title Y position", 0.0, 1.0, 0.95, 0.01)
    else:
        title_x = title_y = 0.5

    # Optional caption + export caption controls
    caption_text = st.text_area(
        "Figure caption (optional)",
        value="",
        help=(
            "Markdown supported. Appears below the chart in the app and will be "
            "included in the exported image at the lower-left."
        ),
    )
    cap_c1, cap_c2 = st.columns([1, 1])
    with cap_c1:
        extra_l_margin = st.number_input(
            "Caption extra left margin (px)",
            min_value=0,
            max_value=200,
            value=10,
            step=1,
        )
    with cap_c2:
        caption_yshift = st.number_input(
            "Caption y-shift (px; negative = into margin)",
            min_value=-200,
            max_value=200,
            value=-12,
            step=1,
        )

with c3:
    width_px = st.number_input("Figure width (px)", 400, 4000, 1100, 50)
    height_px = st.number_input("Figure height (px)", 300, 3000, 600, 50)

    # Background settings
    bg_transparent = st.toggle("Transparent background (export-ready)", value=True)
    paper_bg = st.color_picker(
        "Page/background color",
        palette_theme.get("paper_bg", "#ffffff")
        if use_palette_theme and not bg_transparent
        else "#ffffff",
        disabled=bg_transparent,
    )
    plot_bg = st.color_picker(
        "Plot area color",
        palette_theme.get("plot_bg", "#ffffff")
        if use_palette_theme and not bg_transparent
        else "#ffffff",
        disabled=bg_transparent,
    )

with c4:
    # Grid settings (independent X/Y)
    st.markdown("Grid lines")
    show_x_grid = st.toggle("Vertical grid (X grid)", value=False)
    show_y_grid = st.toggle("Horizontal grid (Y grid)", value=False)

    # Derive default picker color and opacity from palette (convert rgba to hex+alpha)
    default_grid_hex, default_grid_alpha = rgba_to_hex_and_alpha(
        palette_theme.get("grid_color", "#e1e5ea")
    )

    # X grid style
    if show_x_grid:
        x_grid_color_hex = st.color_picker("X grid color", default_grid_hex)
        x_grid_opacity = st.slider(
            "X grid opacity", 0.0, 1.0, float(default_grid_alpha), 0.05
        )
        x_grid_width = st.number_input("X grid width", 0.0, 5.0, 1.0, 0.1)
    else:
        x_grid_color_hex = default_grid_hex
        x_grid_opacity = float(default_grid_alpha)
        x_grid_width = 1.0

    # Y grid style
    if show_y_grid:
        y_grid_color_hex = st.color_picker("Y grid color", default_grid_hex)
        y_grid_opacity = st.slider(
            "Y grid opacity", 0.0, 1.0, float(default_grid_alpha), 0.05
        )
        y_grid_width = st.number_input("Y grid width", 0.0, 5.0, 1.0, 0.1)
    else:
        y_grid_color_hex = default_grid_hex
        y_grid_opacity = float(default_grid_alpha)
        y_grid_width = 1.0

# Axis titles positioning
st.subheader("Axis Settings")
ax1, ax2 = st.columns([1, 1])
with ax1:
    x_title = st.text_input(
        "X axis title",
        value=date_like if "date" in date_like.lower() else "Date",
    )
    x_title_standoff = st.slider("X title distance from axis", 0, 100, 20, 5)
    # Axis line (X) controls
    show_x_line = st.toggle("Show X axis line", value=False)
    if show_x_line:
        x_line_color = st.color_picker("X axis line color", value=font_color)
        x_line_width = st.number_input(
            "X axis line width", min_value=0.5, max_value=8.0, value=1.5, step=0.5
        )
    else:
        x_line_color = font_color
        x_line_width = 1.0
with ax2:
    y_title = st.text_input("Y axis title", value="Value")
    y_title_standoff = st.slider("Y title distance from axis", 0, 100, 20, 5)
    # Axis line (Y) controls
    show_y_line = st.toggle("Show Y axis line", value=False)
    if show_y_line:
        y_line_color = st.color_picker("Y axis line color", value=font_color)
        y_line_width = st.number_input(
            "Y axis line width", min_value=0.5, max_value=8.0, value=1.5, step=0.5
        )
    else:
        y_line_color = font_color
        y_line_width = 1.0

# X-axis tick formatting (date or numeric)
st.subheader("X axis tick labels")
if "parse_dates" in locals() and parse_dates:
    st.caption("D3 time-format string (applies to datetime X).")
    f1, f2 = st.columns([1, 1])
    with f1:
        x_tickformat_preset = st.selectbox(
            "Date format preset",
            [
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%b %d, %Y",
                "%d %b %Y",
                "%Y-%m",
                "%b %Y",
                "%Y",
                "%H:%M",
                "%H:%M:%S",
                "%Y-%m-%d %H:%M",
            ],
            index=0,
        )
    with f2:
        x_tickformat_custom = st.text_input(
            "Custom date format (optional)",
            value="",
            placeholder="e.g. %d/%m/%Y %H:%M",
        )
    x_tickformat = x_tickformat_custom or x_tickformat_preset

    p1, p2 = st.columns([1, 1])
    with p1:
        x_tickprefix = st.text_input("X tick prefix", value="", placeholder="e.g. ")
    with p2:
        x_ticksuffix = st.text_input("X tick suffix", value="", placeholder="e.g. UTC")
    x_tickangle = st.slider("X tick label angle", -90, 90, 0, 5)
else:
    st.caption("D3 number-format string (applies to numeric X).")
    n1, n2 = st.columns([1, 1])
    with n1:
        x_tickformat_preset_label = st.selectbox(
            "Numeric format preset",
            [
                "Auto (default)",
                "1,234",
                "1,234.56",
                "1.23%",
                "1.23e+3",
                "1.23k (SI)",
                "1.23M (SI)",
            ],
            index=0,
        )
    with n2:
        x_tickformat_custom = st.text_input(
            "Custom numeric format (optional)",
            value="",
            placeholder="e.g. ,.2f, .2%, .3e, ~s",
        )

    preset_map = {
        "Auto (default)": "",
        "1,234": ",.0f",
        "1,234.56": ",.2f",
        "1.23%": ".2%",
        "1.23e+3": ".2e",
        "1.23k (SI)": "~s",
        "1.23M (SI)": "~s",
    }
    x_tickformat = x_tickformat_custom or preset_map.get(x_tickformat_preset_label, "")

    p1, p2 = st.columns([1, 1])
    with p1:
        x_tickprefix = st.text_input("X tick prefix", value="", placeholder="e.g. $")
    with p2:
        x_ticksuffix = st.text_input("X tick suffix", value="", placeholder="e.g. %")
    x_tickangle = st.slider("X tick label angle", -90, 90, 0, 5)

# Y-axis tick formatting (numeric)
st.subheader("Y axis tick labels")
st.caption("D3 number-format string (applies to Y and works with log scale).")
y1, y2 = st.columns([1, 1])
with y1:
    y_tickformat_preset_label = st.selectbox(
        "Y numeric format preset",
        [
            "Auto (default)",
            "1,234",
            "1,234.56",
            "1.23%",
            "1.23e+3",
            "1.23k (SI)",
            "1.23M (SI)",
        ],
        index=0,
    )
with y2:
    y_tickformat_custom = st.text_input(
        "Custom Y numeric format (optional)",
        value="",
        placeholder="e.g. ,.2f, .2%, .3e, ~s",
    )

numeric_preset_map = {
    "Auto (default)": "",
    "1,234": ",.0f",
    "1,234.56": ",.2f",
    "1.23%": ".2%",
    "1.23e+3": ".2e",
    "1.23k (SI)": "~s",
    "1.23M (SI)": "~s",
}
y_tickformat = y_tickformat_custom or numeric_preset_map.get(
    y_tickformat_preset_label, ""
)

yp1, yp2 = st.columns([1, 1])
with yp1:
    y_tickprefix = st.text_input("Y tick prefix", value="", placeholder="e.g. $")
with yp2:
    y_ticksuffix = st.text_input("Y tick suffix", value="", placeholder="e.g. %")
y_tickangle = st.slider("Y tick label angle", -90, 90, 0, 5)

# Legend settings
st.subheader("Legend Settings")
leg1, leg2 = st.columns([1, 1])
with leg1:
    show_legend = st.toggle("Show legend", value=True)
    legend_title = st.text_input("Legend title", value="Legenda")
    legend_orientation = st.selectbox("Legend orientation", ["h", "v"], index=0)

with leg2:
    legend_anchor = st.selectbox(
        "Legend anchor",
        ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        index=1,
    )
    legend_x = st.slider("Legend X position", -0.2, 1.2, 1.0, 0.01)
    legend_y = st.slider("Legend Y position", -0.2, 1.2, 1.02, 0.01)

# Series-specific average, rolling, glow, and Y2 assignment
if chart_type not in ["Candlestick", "Pie"]:

    st.subheader("Series Analysis")
    series_avg_settings = {}
    series_rolling_settings = {}
    series_glow_settings = {}
    y_col_names = {}
    y2_series_set = set()

    for i, y_col in enumerate(y_cols):
        with st.expander(f"📊 {y_col} Settings"):
            col1, col2 = st.columns([1, 1])

            with col1:
                # Rename here
                new_name = st.text_input(
                    f"Display name for '{y_col}'",
                    value=y_col,
                    key=f"name_{y_col}",
                )
                y_col_names[y_col] = new_name

                series_avg_settings[y_col] = st.toggle(
                    "Show average line",
                    value=False,
                    key=f"avg_{y_col}",
                )
                glow_enabled = st.toggle(
                    "Glow (line/scatter)",
                    value=False,
                    key=f"glow_{y_col}",
                )
                on_y2 = st.toggle(
                    "Plot on secondary axis (Y2)",
                    value=False,
                    key=f"y2_{y_col}",
                )
                if on_y2:
                    y2_series_set.add(y_col)

            with col2:
                rolling_enabled = st.toggle(
                    "Show rolling average", value=False, key=f"rolling_{y_col}"
                )
                if rolling_enabled:
                    rolling_window_custom = st.number_input(
                        "Rolling window",
                        min_value=2,
                        value=20,
                        step=1,
                        key=f"rolling_window_{y_col}",
                    )
                    series_rolling_settings[y_col] = {
                        "enabled": True,
                        "window": rolling_window_custom,
                    }
                else:
                    series_rolling_settings[y_col] = {
                        "enabled": False,
                        "window": 10,
                    }

                if glow_enabled:
                    glow_width_boost = st.slider(
                        "Glow width boost (px)",
                        2,
                        30,
                        8,
                        1,
                        key=f"glow_width_{y_col}",
                    )
                    glow_opacity = st.slider(
                        "Glow opacity",
                        0.1,
                        0.9,
                        0.6,
                        0.05,
                        key=f"glow_alpha_{y_col}",
                    )
                    series_glow_settings[y_col] = {
                        "enabled": True,
                        "width_boost": glow_width_boost,
                        "opacity": glow_opacity,
                    }
                else:
                    series_glow_settings[y_col] = {
                        "enabled": False,
                        "width_boost": 8,
                        "opacity": 0.6,
                    }

    # Average and rolling colors
    if any(series_avg_settings.values()) or any(
        s.get("enabled", False) for s in series_rolling_settings.values()
    ):
        avg_colors1, avg_colors2 = st.columns([1, 1])
        with avg_colors1:
            avg_line_color = st.color_picker("Average line color", "#ff6b6b")
        with avg_colors2:
            rolling_line_color = st.color_picker(
                "Rolling average line color", "#4ecdc4"
            )
    else:
        avg_line_color = rolling_line_color = "#000000"

    # Secondary axis settings UI
    st.subheader("Secondary Axis (Y2)")
    if len(y2_series_set) == 0:
        st.caption(
            "No series selected for Y2. Use the toggle inside each series settings "
            "to assign series to the secondary axis."
        )
        y2_title = "Secondary"
        y2_title_standoff = 20
        y2_log = False
        # Defaults for Y2 tick labels (unused)
        y2_tickformat = ""
        y2_tickprefix = ""
        y2_ticksuffix = ""
        y2_tickangle = 0
    else:
        sy1, sy2, sy3 = st.columns([1, 1, 1])
        with sy1:
            y2_title = st.text_input("Y2 axis title", value="Secondary")
        with sy2:
            y2_title_standoff = st.slider("Y2 title distance", 0, 100, 20, 5)
        with sy3:
            y2_log = st.toggle("Log scale (Y2)", value=False)

        st.caption("Y2 tick labels (numeric formatting)")
        z1, z2 = st.columns([1, 1])
        with z1:
            y2_preset_label = st.selectbox(
                "Y2 numeric format preset",
                [
                    "Auto (default)",
                    "1,234",
                    "1,234.56",
                    "1.23%",
                    "1.23e+3",
                    "1.23k (SI)",
                    "1.23M (SI)",
                ],
                index=0,
            )
        with z2:
            y2_tickformat_custom = st.text_input(
                "Custom Y2 numeric format (optional)",
                value="",
                placeholder="e.g. ,.2f, .2%, .3e, ~s",
            )
        y2_tickformat = y2_tickformat_custom or numeric_preset_map.get(
            y2_preset_label, ""
        )

        z3, z4 = st.columns([1, 1])
        with z3:
            y2_tickprefix = st.text_input("Y2 tick prefix", value="", placeholder="$")
        with z4:
            y2_ticksuffix = st.text_input("Y2 tick suffix", value="", placeholder="%")
        y2_tickangle = st.slider("Y2 tick label angle", -90, 90, 0, 5)

else:
    # Defaults for candlestick workflow or Pie (no series analysis)
    y_col_names = {}
    y2_series_set = set()
    y2_title = "Secondary"
    y2_title_standoff = 20
    y2_log = False
    # Defaults for Y2 tick labels (unused)
    y2_tickformat = ""
    y2_tickprefix = ""
    y2_ticksuffix = ""
    y2_tickangle = 0

# Chart-specific options
st.subheader("Chart Options")
cc1, cc2, cc3 = st.columns([1.1, 1.1, 1.1])

if chart_type == "Pie":
    with cc1:
        pie_hole = st.slider("Donut hole size", 0.0, 0.7, 0.3, 0.05)
        pie_text_position = st.selectbox(
            "Slice label position", ["inside", "outside"], index=0
        )
        pie_sort = st.toggle("Sort slices", value=True)
        pie_direction = st.selectbox(
            "Direction", ["counterclockwise", "clockwise"], index=0
        )
    with cc2:
        pie_text_mode = st.selectbox(
            "Slice label content",
            [
                "%",
                "Value",
                "% + Value",
                "Label",
                "Label + %",
                "Label + Value",
                "Label + % + Value",
            ],
            index=0,
        )
        pie_start_angle = st.slider("Start angle (degrees)", 0, 360, 0, 5)
        _mode_map = {
            "%": "%{percent:.0%}",
            "Value": "%{value}",
            "% + Value": "%{percent:.0%}<br>%{value}",
            "Label": "%{label}",
            "Label + %": "%{label}<br>%{percent:.0%}",
            "Label + Value": "%{label}<br>%{value}",
            "Label + % + Value": "%{label}<br>%{percent:.0%}<br>%{value}",
        }
        pie_text_template = _mode_map.get(pie_text_mode, "%{percent:.0%}")
    # Defaults for non-applicable options
    markers = False
    marker_size = 6
    line_width = 3
    line_dash = "solid"
    barmode = "group"
else:
    with cc1:
        # Markers OFF by default
        markers = st.toggle("Markers (line/scatter)", value=False)
        marker_size = st.slider("Marker size", 1, 24, 6, 1)
        line_width = st.slider("Line width", 1, 12, 3, 1)
        line_dash = st.selectbox(
            "Line dash",
            ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"],
            index=0,
        )
    with cc2:
        if chart_type == "Bar":
            barmode = st.selectbox("Bar mode", ["group", "stack"], index=0)
        else:
            barmode = "group"

with cc3:
    export_fmt = st.selectbox("Export format", ["png", "svg", "pdf"], index=0)
    export_scale = st.slider("Export scale (resolution)", 1.0, 5.0, 2.0, 0.5)

# Category grouping (Pie and Bar)
if chart_type in ["Pie", "Bar"]:
    st.subheader('Category Grouping ("Others")')

    if chart_type == "Pie":
        pie_group_small = st.toggle(
            'Group small categories into "Others"', value=False
        )
        col_pg1, col_pg2 = st.columns([1, 1])
        with col_pg1:
            pie_group_threshold = st.number_input(
                "Min value to keep category",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )
        with col_pg2:
            pie_group_others_label = st.text_input("Others label", value="Others")
        # Bar defaults
        bar_group_small = False
        bar_group_threshold = 0.0
        bar_group_by = "x"
        bar_group_others_label = "Others"

    elif chart_type == "Bar":
        # Detect if color dimension exists
        bar_has_color_dimension = (color_col is not None) or (len(y_cols) > 1)
        bar_group_small = st.toggle(
            'Group small categories into "Others"', value=False
        )
        col_bg1, col_bg2, col_bg3 = st.columns([1, 1, 1])
        with col_bg1:
            grp_opts = ["X axis"] + (["Color groups"] if bar_has_color_dimension else [])
            grp_choice = st.selectbox("Group by", grp_opts, index=0)
            bar_group_by = "color" if grp_choice == "Color groups" else "x"
        with col_bg2:
            bar_group_threshold = st.number_input(
                "Min value to keep category",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )
        with col_bg3:
            bar_group_others_label = st.text_input("Others label", value="Others")
        # Pie defaults
        pie_group_small = False
        pie_group_threshold = 0.0
        pie_group_others_label = "Others"
else:
    # Defaults when not applicable
    pie_group_small = False
    pie_group_threshold = 0.0
    pie_group_others_label = "Others"
    bar_group_small = False
    bar_group_threshold = 0.0
    bar_group_by = "x"
    bar_group_others_label = "Others"

# Compose final grid colors for Plotly (rgba from hex + opacity)
x_grid_color_final = hex_to_rgba(x_grid_color_hex, x_grid_opacity)
y_grid_color_final = hex_to_rgba(y_grid_color_hex, y_grid_opacity)

# Optional transform before preview/export
st.subheader("Transform")
swap_axes = st.toggle(
    "Swap X and Y axes for preview/export",
    value=False,
    help="Transpose the plot without losing settings. Candlestick swap is not supported.",
)

# Build figure with all parameters
fig = build_figure(
    chart_type=chart_type,
    df=df,
    x_col=x_col,
    y_cols=y_cols,
    y_col_names=y_col_names,
    color_col=color_col,
    parse_dates=parse_dates,
    sort_x=sort_x,
    rolling_window=int(rolling_window),
    colorway=colorway,
    palette_theme=palette_theme,
    use_palette_theme=use_palette_theme,
    width_px=int(width_px),
    height_px=int(height_px),
    bg_transparent=bg_transparent,
    paper_bg=paper_bg,
    plot_bg=plot_bg,
    font_color=font_color,
    # Per-axis grid configs
    show_x_grid=bool(show_x_grid),
    x_grid_color=x_grid_color_final,
    x_grid_width=float(x_grid_width),
    show_y_grid=bool(show_y_grid),
    y_grid_color=y_grid_color_final,
    y_grid_width=float(y_grid_width),
    # Per-axis line configs
    show_x_line=bool(show_x_line),
    x_line_color=x_line_color,
    x_line_width=float(x_line_width),
    show_y_line=bool(show_y_line),
    y_line_color=y_line_color,
    y_line_width=float(y_line_width),
    show_legend=show_legend,
    legend_orientation=legend_orientation,
    legend_x=float(legend_x),
    legend_y=float(legend_y),
    legend_anchor=legend_anchor,
    legend_title=legend_title,
    font_family=font_family,
    base_font_size=int(base_font_size),
    title=title,
    title_x=float(title_x),
    title_y=float(title_y),
    x_title=x_title,
    x_title_standoff=int(x_title_standoff),
    x_tickformat=x_tickformat or None,
    x_tickprefix=x_tickprefix,
    x_ticksuffix=x_ticksuffix,
    x_tickangle=int(x_tickangle),
    y_title=y_title,
    y_title_standoff=int(y_title_standoff),
    y_tickformat=y_tickformat or None,
    y_tickprefix=y_tickprefix,
    y_ticksuffix=y_ticksuffix,
    y_tickangle=int(y_tickangle),
    y_log=y_log,
    y2_series=list(y2_series_set) if "y2_series_set" in locals() else [],
    y2_title=y2_title,
    y2_title_standoff=int(y2_title_standoff),
    y2_tickformat=y2_tickformat or None,
    y2_tickprefix=y2_tickprefix,
    y2_ticksuffix=y2_ticksuffix,
    y2_tickangle=int(y2_tickangle),
    y2_log=bool(y2_log),
    line_width=float(line_width),
    line_dash=line_dash,
    markers=markers,
    marker_size=int(marker_size),
    barmode=barmode,
    candlestick_cols=candlestick_cols,
    series_avg_settings=series_avg_settings,
    series_rolling_settings=series_rolling_settings,
    series_glow_settings=series_glow_settings,
    avg_line_color=avg_line_color,
    rolling_line_color=rolling_line_color,
    watermark_img=watermark_img,
    watermark_opacity=float(watermark_opacity),
    watermark_size=float(watermark_size),
    watermark_x=float(watermark_x),
    watermark_y=float(watermark_y),
    logo_img=logo_img,
    logo_size=float(logo_size),
    logo_x=float(logo_x),
    logo_y=float(logo_y),
    swap_axes=bool(swap_axes),
    # Pie params (ignored for non-Pie)
    pie_labels_col=pie_labels_col if chart_type == "Pie" else None,
    pie_values_col=pie_values_col if chart_type == "Pie" else None,
    pie_hole=float(pie_hole) if chart_type == "Pie" else 0.0,
    pie_text_template=pie_text_template if chart_type == "Pie" else None,
    pie_text_position=pie_text_position if chart_type == "Pie" else "inside",
    pie_sort=bool(pie_sort) if chart_type == "Pie" else True,
    pie_direction=pie_direction if chart_type == "Pie" else "counterclockwise",
    pie_start_angle=int(pie_start_angle) if chart_type == "Pie" else 0,
    # Others grouping
    pie_group_small=bool(pie_group_small),
    pie_group_threshold=float(pie_group_threshold),
    pie_group_others_label=pie_group_others_label,
    bar_group_small=bool(bar_group_small),
    bar_group_threshold=float(bar_group_threshold),
    bar_group_by=bar_group_by,
    bar_group_others_label=bar_group_others_label,
)

# Optionally apply imported layout profile (if any)
imported_size = False
if "layout_profile" in st.session_state and st.session_state["layout_profile"]:
    try:
        pl = st.session_state["layout_profile"].get("plotly_layout", {}) or {}
        fig.update_layout(pl)
        # If the imported layout defines width/height, respect it
        if pl.get("width") or pl.get("height"):
            fig.update_layout(autosize=False)
            imported_size = True
    except Exception as e:
        st.warning(f"Failed to apply imported layout: {e}")

# Preview
st.subheader("Preview")

# If the imported layout defines width/height, do not stretch to container
use_container = not imported_size

st.plotly_chart(
    fig,
    use_container_width=use_container,
    theme=None,
    config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": "chart"},
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "toggleSpikelines",
        ],
    },
)
if caption_text.strip():
    st.caption(caption_text)

# Export section
st.subheader("Export")

# Optional: use the figure/layout (including imported JSON) size for export
use_layout_size_for_export = st.checkbox(
    "Use layout size for export (from figure/imported JSON)",
    value=False,
    help="If checked, the exported image will use fig.layout.width/height "
    "when present. Otherwise the controls below are used.",
)

exp_c1, exp_c2, exp_c3 = st.columns([1.1, 1.1, 1.1])

with exp_c1:
    exp_width = st.number_input("Export width (px)", 400, 8000, int(width_px), 50)
    exp_height = st.number_input("Export height (px)", 300, 8000, int(height_px), 50)

# with exp_c2:
#     if st.button("Download image"):
#         # Decide which width/height to pass to the exporter
#         width_to_use = int(exp_width)
#         height_to_use = int(exp_height)
#         if use_layout_size_for_export:
#             if fig.layout.width:
#                 width_to_use = int(fig.layout.width)
#             if fig.layout.height:
#                 height_to_use = int(fig.layout.height)
#
#         img_bytes = export_image(
#             fig=fig,
#             fmt=export_fmt,
#             width=width_to_use,
#             height=height_to_use,
#             scale=float(export_scale),
#             caption_text=caption_text,
#             base_font_size=int(base_font_size),
#             font_color=font_color,
#             extra_l_margin=int(extra_l_margin),
#             caption_yshift=int(caption_yshift),
#         )
#         if img_bytes:
#             mime = (
#                 "image/png"
#                 if export_fmt == "png"
#                 else "image/svg+xml"
#                 if export_fmt == "svg"
#                 else "application/pdf"
#             )
#             st.download_button(
#                 "Save file",
#                 data=img_bytes,
#                 file_name=f"graph.{export_fmt}",
#                 mime=mime,
#            )

with exp_c3:
    if chart_type == "Pie":
        out = io.StringIO()
        # Export the aggregated/grouped data used for the pie
        df_pie = df[[pie_labels_col, pie_values_col]].copy()
        if pie_group_small and pie_group_threshold > 0:
            totals = (
                df_pie.groupby(pie_labels_col, dropna=False)[pie_values_col]
                .sum()
                .sort_values(ascending=False)
            )
            small = set(totals[totals < float(pie_group_threshold)].index)
            if small:
                df_pie["_labels"] = df_pie[pie_labels_col].apply(
                    lambda v: pie_group_others_label if v in small else v
                )
                df_pie = (
                    df_pie.groupby("_labels", dropna=False)[pie_values_col]
                    .sum()
                    .reset_index()
                    .rename(columns={"_labels": pie_labels_col})
                )
        df_pie.to_csv(out, index=False)
        st.download_button(
            "Download plotted data (CSV)",
            data=out.getvalue(),
            file_name="plotted_data.csv",
            mime="text/csv",
        )
    elif chart_type != "Candlestick":
        # Export long data in the same shape as plotted, including grouping
        long_df = to_long_df(df, x_col=x_col, y_cols=y_cols, color_col=color_col)
        if y_col_names:
            long_df["SeriesDisplay"] = long_df["Series"].map(y_col_names).fillna(
                long_df["Series"]
            )
        # Determine color column used (for grouping, if any)
        color_col_to_use_export = None
        if color_col and len(y_cols) > 1:
            color_col_to_use_export = "ColorGroup"
        elif color_col and len(y_cols) == 1:
            color_col_to_use_export = color_col
        elif len(y_cols) > 1:
            color_col_to_use_export = "SeriesDisplay"

        # Apply Bar grouping to export data before rolling
        if chart_type == "Bar" and bar_group_small and bar_group_threshold > 0:
            if bar_group_by == "x":
                totals = long_df.groupby(x_col, dropna=False)["Value"].sum()
                small = set(totals[totals < float(bar_group_threshold)].index)
                if small:
                    long_df[x_col] = long_df[x_col].apply(
                        lambda v: bar_group_others_label if v in small else v
                    )
            elif (
                bar_group_by == "color" and color_col_to_use_export is not None
            ):
                totals = long_df.groupby(color_col_to_use_export, dropna=False)[
                    "Value"
                ].sum()
                small = set(totals[totals < float(bar_group_threshold)].index)
                if small:
                    long_df[color_col_to_use_export] = long_df[
                        color_col_to_use_export
                    ].apply(lambda v: bar_group_others_label if v in small else v)

        # Rolling after grouping
        long_df = apply_rolling(long_df, x_col, "ColorGroup", int(rolling_window))
        out = io.StringIO()
        long_df.to_csv(out, index=False)
        st.download_button(
            "Download plotted data (CSV)",
            data=out.getvalue(),
            file_name="plotted_data.csv",
            mime="text/csv",
        )

# Import/Export Layout (no data)
st.subheader("Layout Profiles (Import/Export)")
lp1, lp2, lp3 = st.columns([1.1, 1.1, 1.1])

with lp1:
    profile = make_layout_profile(fig)
    profile_json = json.dumps(profile, indent=2)
    st.download_button(
        "Download layout (JSON)",
        data=profile_json,
        file_name="graph_layout.json",
        mime="application/json",
        help=(
            "Exports Plotly layout only (no data). Includes fonts, colors, sizes, "
            "legend/title positions, grids, images, annotations, and shapes."
        ),
    )

with lp2:
    layout_file = st.file_uploader(
        "Upload layout JSON", type=["json"], key="layout_json_upload"
    )
    if layout_file is not None:
        # Parse JSON only
        try:
            loaded = json.load(layout_file)
        except Exception as e:
            st.error(f"Could not parse layout JSON: {e}")
        else:
            if isinstance(loaded, dict) and "plotly_layout" in loaded:
                st.session_state["layout_profile"] = loaded
                st.success("Layout loaded. It will be applied to the preview/export.")
                if st.button("Apply imported layout now"):
                    # Force a refresh so the profile is applied to the figure build
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for very old Streamlit versions
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.info(
                                "Couldn't auto-refresh. Change any control to refresh."
                            )
            else:
                st.error("Invalid layout file (missing 'plotly_layout').")

with lp3:
    if st.button("Clear imported layout"):
        st.session_state.pop("layout_profile", None)
        # Force refresh to remove the profile
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except Exception:
                st.info("Change any control to refresh.")

st.caption(
    "Tip: For truly transparent PNGs, keep the 'Transparent background' toggle "
    "enabled before export. Upload images for watermarks (background) and logos (foreground)."
)