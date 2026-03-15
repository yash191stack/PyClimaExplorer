"""
modules/visualizations.py
==========================
PyClimaExplorer – Climate Data Visualization Dashboard

All Plotly-based chart generators:
  • create_spatial_heatmap  – global map for one time step
  • create_time_series      – point time-series line chart
  • create_animated_map     – frame-by-frame animated global map (BONUS)
  • create_anomaly_chart    – anomaly bar/line chart  (BONUS)
  • create_comparison_chart – side-by-side two-period comparison (BONUS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
from typing import Optional

from modules.utils import (
    get_colorscale_for_variable,
    get_variable_units,
    rolling_average,
    compute_monthly_anomaly,
)


# ---------------------------------------------------------------------------
# Spatial Heatmap
# ---------------------------------------------------------------------------

def create_spatial_heatmap(
    da_2d: xr.DataArray,
    variable: str,
    time_label: str = "",
    colorscale: Optional[str] = None,
    projection: str = "natural earth",
) -> go.Figure:
    """
    Create an interactive global heatmap (Plotly Choropleth / Densitymapbox).

    Parameters
    ----------
    da_2d       : 2-D xr.DataArray  (lat × lon)
    variable    : str – variable name for title / colour-bar
    time_label  : str – human-readable time string shown in title
    colorscale  : str | None – Plotly colorscale; auto-detected if None
    projection  : str – Plotly geo projection type

    Returns
    -------
    go.Figure
    """
    cscale = colorscale or get_colorscale_for_variable(variable)
    units = get_variable_units(da_2d)
    label = f"{variable} [{units}]" if units else variable

    # Resolve lat/lon dimension names
    lat_dim = _find_dim(da_2d, ("lat", "latitude", "y"))
    lon_dim = _find_dim(da_2d, ("lon", "longitude", "x"))

    if lat_dim is None or lon_dim is None:
        return _error_figure("Cannot find lat/lon dimensions in DataArray.")

    lats = da_2d[lat_dim].values.astype(float)
    lons = da_2d[lon_dim].values.astype(float)
    data = da_2d.values.astype(float)

    # Convert 0-360 longitudes to -180…180
    lons = np.where(lons > 180, lons - 360, lons)

    # Build meshgrid → flatten for scatter-based map
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    z_flat = data.flatten()
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    # Remove NaNs
    mask = ~np.isnan(z_flat)
    z_clean = z_flat[mask]
    lat_clean = lat_flat[mask]
    lon_clean = lon_flat[mask]

    if len(z_clean) == 0:
        return _error_figure("No valid data points to display.")

    # Compute robust colour limits
    vmin = float(np.nanpercentile(z_clean, 2))
    vmax = float(np.nanpercentile(z_clean, 98))

    # Cell size (half the grid spacing, for marker sizing)
    cell_size = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 2.5

    title = f"<b>{variable}</b>"
    if time_label:
        title += f"  ·  {time_label}"

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=lat_clean,
            lon=lon_clean,
            mode="markers",
            marker=dict(
                color=z_clean,
                colorscale=cscale,
                cmin=vmin,
                cmax=vmax,
                size=max(3, int(cell_size * 1.2)),
                opacity=0.85,
                colorbar=dict(
                    title=dict(text=label, side="right"),
                    thickness=16,
                    outlinewidth=0,
                ),
                symbol="square",
            ),
            text=[
                f"Lat: {la:.1f}°<br>Lon: {lo:.1f}°<br>{variable}: {v:.3g} {units}"
                for la, lo, v in zip(lat_clean, lon_clean, z_clean)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color="#0f172a")),
        geo=dict(
            projection_type=projection,
            showland=True,
            landcolor="#f8fafc",
            showocean=True,
            oceancolor="#e0f2fe",
            showcoastlines=True,
            coastlinecolor="#4a90d9",
            coastlinewidth=0.8,
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
            showlakes=True,
            lakecolor="#e0f2fe",
            showcountries=True,
            countrycolor="#e2e8f0",
        ),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=480,
    )

    return fig


# ---------------------------------------------------------------------------
# Time-Series Chart
# ---------------------------------------------------------------------------

def create_time_series(
    series: pd.Series,
    variable: str,
    lat: float,
    lon: float,
    units: str = "",
    show_rolling: bool = True,
    rolling_window: int = 3,
) -> go.Figure:
    """
    Create an interactive time-series line chart.

    Parameters
    ----------
    series         : pd.Series – time-indexed climate values
    variable       : str
    lat, lon       : float – location used for subtitle
    units          : str – unit label
    show_rolling   : bool – overlay a rolling mean
    rolling_window : int  – window size for rolling mean

    Returns
    -------
    go.Figure
    """
    label = f"{variable} [{units}]" if units else variable
    lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"

    fig = go.Figure()

    # Raw values
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines+markers",
            name=variable,
            line=dict(color="#4a90d9", width=1.5),
            marker=dict(size=4, color="#4a90d9"),
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{variable}: %{{y:.3g}} {units}<extra></extra>",
        )
    )

    # Rolling mean overlay
    if show_rolling and len(series) >= rolling_window:
        smoothed = rolling_average(series, rolling_window)
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                name=f"{rolling_window}-step rolling mean",
                line=dict(color="#f5a623", width=2.5, dash="dot"),
                hovertemplate=f"Rolling mean: %{{y:.3g}} {units}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{label}</b>  ·  {lat_str}, {lon_str}",
            x=0.5,
            font=dict(size=15, color="#0f172a"),
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
            tickfont=dict(color="#334155"),
            title_font=dict(color="#334155"),
        ),
        yaxis=dict(
            title=label,
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
            tickfont=dict(color="#334155"),
            title_font=dict(color="#334155"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#e2e8f0",
            font=dict(color="#334155"),
        ),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155"),
        margin=dict(l=60, r=20, t=60, b=60),
        height=380,
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# BONUS: Animated Climate Map
# ---------------------------------------------------------------------------

def create_animated_map(
    da: xr.DataArray,
    variable: str,
    colorscale: Optional[str] = None,
    max_frames: int = 36,
) -> go.Figure:
    """
    Build a Plotly animated global map stepping through time.

    Parameters
    ----------
    da         : xr.DataArray (time × lat × lon)
    variable   : str
    colorscale : str | None
    max_frames : int – cap number of frames to keep file size reasonable

    Returns
    -------
    go.Figure with animation slider
    """
    cscale = colorscale or get_colorscale_for_variable(variable)
    units = get_variable_units(da)

    lat_dim = _find_dim(da, ("lat", "latitude", "y"))
    lon_dim = _find_dim(da, ("lon", "longitude", "x"))
    time_dim = _find_dim(da, ("time", "t", "datetime"))

    if lat_dim is None or lon_dim is None:
        return _error_figure("Cannot find lat/lon dimensions.")
    if time_dim is None:
        return _error_figure("Dataset has no time dimension for animation.")

    lats = da[lat_dim].values.astype(float)
    lons = da[lon_dim].values.astype(float)
    lons = np.where(lons > 180, lons - 360, lons)

    time_vals = da[time_dim].values
    n_time = len(time_vals)
    step = max(1, n_time // max_frames)
    indices = list(range(0, n_time, step))[:max_frames]

    # Global colour limits across all frames
    arr_all = da.isel({time_dim: indices}).values.astype(float)
    vmin = float(np.nanpercentile(arr_all, 2))
    vmax = float(np.nanpercentile(arr_all, 98))

    frames = []
    for i in indices:
        data_2d = da.isel({time_dim: i}).values.astype(float)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        z_flat = data_2d.flatten()
        mask = ~np.isnan(z_flat)

        try:
            t_label = str(pd.Timestamp(time_vals[i]))[:10]
        except Exception:
            t_label = str(i)

        frames.append(
            go.Frame(
                data=[
                    go.Scattergeo(
                        lat=lat_grid.flatten()[mask],
                        lon=lon_grid.flatten()[mask],
                        mode="markers",
                        marker=dict(
                            color=z_flat[mask],
                            colorscale=cscale,
                            cmin=vmin,
                            cmax=vmax,
                            size=4,
                            opacity=0.85,
                            colorbar=dict(title=dict(text=f"{variable} [{units}]")),
                        ),
                    )
                ],
                name=t_label,
                layout=go.Layout(title_text=f"<b>{variable}</b>  ·  {t_label}"),
            )
        )

    # Initial frame
    first_data = da.isel({time_dim: indices[0]}).values.astype(float)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    z_flat = first_data.flatten()
    mask = ~np.isnan(z_flat)

    fig = go.Figure(
        data=[
            go.Scattergeo(
                lat=lat_grid.flatten()[mask],
                lon=lon_grid.flatten()[mask],
                mode="markers",
                marker=dict(
                    color=z_flat[mask],
                    colorscale=cscale,
                    cmin=vmin,
                    cmax=vmax,
                    size=4,
                    opacity=0.85,
                    colorbar=dict(title=dict(text=f"{variable} [{units}]", side="right")),
                ),
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        title=dict(text=f"<b>Animated: {variable}</b>", x=0.5, font=dict(size=15, color="#0f172a")),
        geo=dict(
            projection_type="natural earth",
            showland=True, landcolor="#f8fafc",
            showocean=True, oceancolor="#e0f2fe",
            showcoastlines=True, coastlinecolor="#4a90d9",
            bgcolor="rgba(0,0,0,0)",
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.02, x=0.1,
            xanchor="right",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 300, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                         "mode": "immediate"}],
                        label=f.name, method="animate")
                   for f in frames],
            transition={"duration": 0},
            x=0.1, y=0, len=0.9,
            currentvalue=dict(prefix="Date: ", visible=True, xanchor="center"),
        )],
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155"),
        height=520,
        margin=dict(l=0, r=0, t=60, b=80),
    )

    return fig


# ---------------------------------------------------------------------------
# BONUS: Anomaly Chart
# ---------------------------------------------------------------------------

def create_anomaly_chart(
    series: pd.Series,
    variable: str,
    units: str = "",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> go.Figure:
    """
    Display a bar chart of monthly anomalies with a zero reference line.

    Parameters
    ----------
    series   : pd.Series – time-indexed values
    variable : str
    units    : str
    lat, lon : float | None – for subtitle

    Returns
    -------
    go.Figure
    """
    anomaly = compute_monthly_anomaly(series)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in anomaly.values]

    label = f"{variable} anomaly [{units}]" if units else f"{variable} anomaly"
    subtitle = ""
    if lat is not None and lon is not None:
        lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
        subtitle = f"  ·  {lat_str}, {lon_str}"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=anomaly.index,
            y=anomaly.values,
            marker_color=colors,
            name="Monthly anomaly",
            hovertemplate="<b>%{x|%Y-%m}</b><br>Anomaly: %{y:.3g} " + units + "<extra></extra>",
        )
    )

    # Rolling mean of anomaly
    if len(anomaly) >= 3:
        smooth = rolling_average(anomaly, 3)
        fig.add_trace(
            go.Scatter(
                x=smooth.index,
                y=smooth.values,
                mode="lines",
                name="3-step rolling mean",
                line=dict(color="#f5a623", width=2),
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#666")

    fig.update_layout(
        title=dict(
            text=f"<b>{label}</b>{subtitle}",
            x=0.5,
            font=dict(size=15, color="#0f172a"),
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#334155")),
        yaxis=dict(title=label, showgrid=True, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#334155")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#334155")),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155"),
        margin=dict(l=60, r=20, t=60, b=60),
        height=360,
        bargap=0.1,
    )

    return fig


# ---------------------------------------------------------------------------
# BONUS: Two-Period Comparison
# ---------------------------------------------------------------------------

def create_comparison_chart(
    series1: pd.Series,
    series2: pd.Series,
    label1: str,
    label2: str,
    variable: str,
    units: str = "",
) -> go.Figure:
    """
    Overlay two time series for period comparison.

    Returns
    -------
    go.Figure
    """
    y_label = f"{variable} [{units}]" if units else variable

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(series1))),
        y=series1.values,
        mode="lines+markers",
        name=label1,
        line=dict(color="#4a90d9", width=2),
        marker=dict(size=5),
        hovertemplate=f"{label1}<br>{variable}: %{{y:.3g}} {units}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(series2))),
        y=series2.values,
        mode="lines+markers",
        name=label2,
        line=dict(color="#e74c3c", width=2, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate=f"{label2}<br>{variable}: %{{y:.3g}} {units}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{variable}</b> – Period Comparison",
            x=0.5,
            font=dict(size=15, color="#0f172a"),
        ),
        xaxis=dict(title="Time step", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title=y_label, showgrid=True, gridcolor="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#334155")),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155"),
        margin=dict(l=60, r=20, t=60, b=60),
        height=360,
    )

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_dim(da: xr.DataArray, candidates: tuple) -> Optional[str]:
    for d in da.dims:
        if d.lower() in candidates:
            return d
    return None


def _error_figure(message: str) -> go.Figure:
    """Return a blank figure with an error annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#e74c3c"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        height=400,
    )
    return fig

def create_difference_map(
    da1: xr.DataArray,
    da2: xr.DataArray,
    variable: str,
    label1: str,
    label2: str,
    colorscale: str = "RdBu_r",
    projection: str = "natural earth",
) -> go.Figure:
    """Create a spatial difference heatmap (Period 2 - Period 1)."""
    # Align and diff
    diff_da = da2 - da1
    units = get_variable_units(da1)
    
    lat_dim = _find_dim(diff_da, ("lat", "latitude", "y"))
    lon_dim = _find_dim(diff_da, ("lon", "longitude", "x"))
    if not lat_dim or not lon_dim:
        return _error_figure("Cannot find lat/lon for difference map.")

    lats = diff_da[lat_dim].values.astype(float)
    lons = diff_da[lon_dim].values.astype(float)
    lons = np.where(lons > 180, lons - 360, lons)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    z_flat = diff_da.values.astype(float).flatten()
    mask = ~np.isnan(z_flat)

    z_clean = z_flat[mask]
    lat_clean = lat_grid.flatten()[mask]
    lon_clean = lon_grid.flatten()[mask]
    
    if len(z_clean) == 0:
        return _error_figure("No overlapping valid data to compare.")

    vmax = float(np.nanpercentile(np.abs(z_clean), 98))
    cell_size = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 2.5

    fig = go.Figure(go.Scattergeo(
        lat=lat_clean,
        lon=lon_clean,
        mode="markers",
        marker=dict(
            color=z_clean,
            colorscale=colorscale,
            cmin=-vmax, cmax=vmax,
            size=max(3, int(cell_size * 1.2)),
            opacity=0.85,
            symbol="square",
            colorbar=dict(title=dict(text=f"Diff [{units}]", side="right")),
        ),
        text=[f"Lat: {la:.1f}°<br>Lon: {lo:.1f}°<br>Diff: {v:+.3g} {units}"
              for la, lo, v in zip(lat_clean, lon_clean, z_clean)],
        hovertemplate="%{text}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>Difference: {label2} minus {label1}</b>", x=0.5, font=dict(color="#0f172a")),
        geo=dict(
            projection_type=projection,
            showland=True, landcolor="#f8fafc", showocean=True, oceancolor="#e0f2fe",
            showcoastlines=True, coastlinecolor="#4a90d9", bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)", font=dict(color="#334155"),
        margin=dict(l=0, r=0, t=50, b=0), height=480,
    )
    return fig
