"""
modules/utils.py
================
PyClimaExplorer – Climate Data Visualization Dashboard

Utility / helper functions for coordinate handling, unit conversion,
colour-scale selection, and other miscellaneous tasks.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def find_nearest_latlon(
    ds: xr.Dataset,
    target_lat: float,
    target_lon: float,
) -> tuple[float, float]:
    """
    Return the nearest (lat, lon) grid-point values found in *ds*.

    Parameters
    ----------
    ds         : xr.Dataset
    target_lat : float – desired latitude  (-90 … 90)
    target_lon : float – desired longitude (-180 … 180)

    Returns
    -------
    (actual_lat, actual_lon) as floats
    """
    lat_dim = _find_coord(ds, ("lat", "latitude", "y"))
    lon_dim = _find_coord(ds, ("lon", "longitude", "x"))

    actual_lat = target_lat
    actual_lon = target_lon

    if lat_dim:
        lat_vals = ds[lat_dim].values.astype(float)
        idx = int(np.argmin(np.abs(lat_vals - target_lat)))
        actual_lat = float(lat_vals[idx])

    if lon_dim:
        lon_vals = ds[lon_dim].values.astype(float)
        # Handle 0-360 datasets transparently
        if lon_vals.max() > 180:
            target_lon_adj = target_lon % 360
        else:
            target_lon_adj = target_lon
        idx = int(np.argmin(np.abs(lon_vals - target_lon_adj)))
        actual_lon = float(lon_vals[idx])

    return actual_lat, actual_lon


def extract_point_timeseries(
    da: xr.DataArray,
    lat: float,
    lon: float,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
) -> pd.Series:
    """
    Extract a 1-D time series from *da* at the grid-point nearest
    to (*lat*, *lon*).

    Returns
    -------
    pd.Series  with a DatetimeIndex and float values
    """
    # Resolve actual dimension names
    lat_dim = _find_da_dim(da, ("lat", "latitude", "y")) or lat_dim
    lon_dim = _find_da_dim(da, ("lon", "longitude", "x")) or lon_dim
    time_dim = _find_da_dim(da, ("time", "t", "datetime")) or time_dim

    # Select nearest lat/lon
    sel_kwargs = {}
    if lat_dim in da.dims:
        sel_kwargs[lat_dim] = lat
    if lon_dim in da.dims:
        sel_kwargs[lon_dim] = lon

    point = da.sel(sel_kwargs, method="nearest")

    # Squeeze out any remaining non-time dims
    for dim in list(point.dims):
        if dim != time_dim:
            point = point.isel({dim: 0})

    values = point.values.astype(float)
    try:
        index = pd.DatetimeIndex(point[time_dim].values)
    except Exception:
        try:
            import cftime
            raw = point[time_dim].values
            index = pd.DatetimeIndex(
                [pd.Timestamp(t.strftime("%Y-%m-%d %H:%M:%S")) for t in raw]
            )
        except Exception:
            index = pd.RangeIndex(len(values))

    return pd.Series(values, index=index, name=str(da.name or "value"))


def get_spatial_slice(
    da: xr.DataArray,
    time_value: Optional[str] = None,
    time_index: int = 0,
) -> xr.DataArray:
    """
    Return a 2-D (lat × lon) slice of *da* for a single time step.

    Parameters
    ----------
    da         : xr.DataArray – must have a time dimension
    time_value : str | None   – ISO date string; uses index if None
    time_index : int          – fallback integer index

    Returns
    -------
    2-D xr.DataArray
    """
    time_dim = _find_da_dim(da, ("time", "t", "datetime"))
    if time_dim is None:
        # Already 2-D or no time → return as-is (squeeze extras)
        return da.squeeze()

    if time_value is not None:
        try:
            slice_2d = da.sel({time_dim: time_value}, method="nearest")
        except Exception:
            slice_2d = da.isel({time_dim: time_index})
    else:
        time_index = min(time_index, da.sizes[time_dim] - 1)
        slice_2d = da.isel({time_dim: time_index})

    # Squeeze any leftover length-1 dims
    return slice_2d.squeeze()


def get_time_index_from_str(da: xr.DataArray, time_str: str) -> int:
    """
    Map an ISO date string to the nearest time-axis integer index.
    """
    time_dim = _find_da_dim(da, ("time", "t", "datetime"))
    if time_dim is None:
        return 0
    try:
        times = pd.DatetimeIndex(da[time_dim].values)
        target = pd.Timestamp(time_str)
        return int(np.argmin(np.abs(times - target)))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Unit & label helpers
# ---------------------------------------------------------------------------

def format_coord_label(lat: float, lon: float) -> str:
    """Return a human-readable coordinate string like '23.5°N, 45.0°E'."""
    lat_label = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"
    lon_label = f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
    return f"{lat_label}, {lon_label}"


def get_colorscale_for_variable(variable_name: str) -> str:
    """
    Return a sensible Plotly colorscale name for common climate variables.
    """
    v = variable_name.lower()
    if any(k in v for k in ("temp", "t2m", "sst", "ts", "tmax", "tmin")):
        return "RdBu_r"
    if any(k in v for k in ("precip", "rain", "pr", "prect", "p")):
        return "Blues"
    if any(k in v for k in ("wind", "spd", "u10", "v10", "ws")):
        return "Viridis"
    if any(k in v for k in ("humidity", "hum", "rh", "q")):
        return "YlGnBu"
    if any(k in v for k in ("pressure", "msl", "slp", "psl")):
        return "PuBuGn"
    if any(k in v for k in ("cloud", "cld", "cc")):
        return "Greys"
    if any(k in v for k in ("snow", "ice", "sic", "siconc")):
        return "ice"
    return "Plasma"


def get_variable_units(da: xr.DataArray) -> str:
    """Extract unit string from DataArray attributes."""
    for key in ("units", "unit"):
        if key in da.attrs:
            return da.attrs[key]
    return ""


def rolling_average(series: pd.Series, window: int = 3) -> pd.Series:
    """Return a centred rolling mean of *series*."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_anomaly(series: pd.Series) -> pd.Series:
    """
    Compute anomaly relative to the long-term mean.

    Returns
    -------
    pd.Series  – values minus their overall mean
    """
    return series - series.mean()


def compute_monthly_anomaly(series: pd.Series) -> pd.Series:
    """
    Compute anomaly relative to the long-term *monthly* climatology.
    Works only when *series* has a DatetimeIndex.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return compute_anomaly(series)
    climatology = series.groupby(series.index.month).transform("mean")
    return series - climatology


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_coord(ds: xr.Dataset, candidates: tuple) -> Optional[str]:
    for c in ds.coords:
        if c.lower() in candidates:
            return c
    for d in ds.dims:
        if d.lower() in candidates:
            return d
    return None


def _find_da_dim(da: xr.DataArray, candidates: tuple) -> Optional[str]:
    for d in da.dims:
        if d.lower() in candidates:
            return d
    # Also check coords
    for c in da.coords:
        if c.lower() in candidates and c in da.dims:
            return c
    return None

# ---------------------------------------------------------------------------
# Hackathon Features
# ---------------------------------------------------------------------------

def generate_climate_insights(series: pd.Series, variable: str) -> str:
    """Auto-generate text insights based on time-series trends."""
    if len(series) < 12:
        return "Insufficient data points for robust trend analysis."

    start_mean = series.iloc[:len(series)//3].mean()
    end_mean = series.iloc[-len(series)//3:].mean()
    diff = end_mean - start_mean
    pct_change = (diff / abs(start_mean)) * 100 if start_mean != 0 else 0
    std_dev = series.std()

    trend = "stable"
    if pct_change > 5:
        trend = f"increasing (+{pct_change:.1f}%)"
    elif pct_change < -5:
        trend = f"decreasing ({pct_change:.1f}%)"

    insight = f"📈 **Trend Analysis**: Over the selected period, {variable} is showing a **{trend}** trend. "

    if std_dev > series.mean() * 0.2:
         insight += f"There is high variability (std dev: {std_dev:.2g}), suggesting increased extreme events."
    else:
         insight += "The metric remains relatively consistent with low variability."

    if 'temp' in variable.lower() and diff > 0.5:
        insight += " ⚠️ **Warning**: Steady warming detected in this region."
    elif 'precip' in variable.lower() and pct_change < -10:
        insight += " ⚠️ **Warning**: Significant drying trend detected."

    return insight
