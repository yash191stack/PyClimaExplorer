"""
modules/data_loader.py
======================
PyClimaExplorer – Climate Data Visualization Dashboard

Handles loading NetCDF climate datasets via Xarray, extracting
available variables, and performing time-range slicing.
"""

import io
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_netcdf(source: Union[str, io.BytesIO]) -> xr.Dataset:
    """
    Load a NetCDF dataset from a file path or an in-memory bytes buffer
    (e.g. a Streamlit UploadedFile).

    Parameters
    ----------
    source : str | BytesIO
        Path to the .nc file, OR a file-like object returned by
        ``st.file_uploader``.

    Returns
    -------
    xr.Dataset
        The decoded dataset with CF-conventions applied where possible.

    Raises
    ------
    ValueError
        If the source cannot be opened as a valid NetCDF dataset.
    """
    try:
        if isinstance(source, (str,)):
            ds = xr.open_dataset(source, use_cftime=True)
        else:
            # BytesIO / UploadedFile path
            ds = xr.open_dataset(io.BytesIO(source.read()), engine="scipy", use_cftime=True)
    except Exception:
        try:
            # Fallback: try scipy engine for unusual formats
            if isinstance(source, (str,)):
                ds = xr.open_dataset(source, engine="scipy")
            else:
                source.seek(0)
                ds = xr.open_dataset(io.BytesIO(source.read()), engine="scipy")
        except Exception as exc:
            raise ValueError(f"Could not open NetCDF dataset: {exc}") from exc

    # Decode times safely
    ds = _normalize_coords(ds)
    return ds


def get_variables(ds: xr.Dataset) -> list[str]:
    """
    Return a sorted list of data variable names in the dataset,
    excluding pure coordinate / auxiliary variables.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    list[str]
    """
    # Filter out variables that have no lat/lon/spatial dimension
    spatial_vars = []
    for vname, var in ds.data_vars.items():
        dims = [str(d).lower() for d in var.dims]
        has_spatial = any(
            kw in d
            for d in dims
            for kw in ("lat", "lon", "x", "y", "ncols", "nrows")
        )
        if has_spatial:
            spatial_vars.append(vname)
    return sorted(spatial_vars) if spatial_vars else sorted(ds.data_vars.keys())


def slice_time(
    ds: xr.Dataset,
    variable: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> xr.DataArray:
    """
    Extract a DataArray for *variable* optionally sliced to [start, end].

    Parameters
    ----------
    ds       : xr.Dataset
    variable : str – variable name
    start    : str | None – ISO date string, e.g. "2020-01-01"
    end      : str | None – ISO date string

    Returns
    -------
    xr.DataArray
    """
    da = ds[variable]

    # Identify the time dimension
    time_dim = _find_dim(da, ("time", "t", "datetime"))
    if time_dim is None or start is None or end is None:
        return da

    try:
        da = da.sel({time_dim: slice(start, end)})
    except Exception:
        # If selection fails, return the full array
        pass

    return da


def get_time_range(ds: xr.Dataset) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Return (min_time, max_time) as pandas Timestamps, or (None, None).
    """
    time_dim = _find_dim_in_ds(ds, ("time", "t", "datetime"))
    if time_dim is None:
        return None, None

    raw = ds[time_dim].values
    try:
        times = pd.DatetimeIndex(raw)
    except Exception:
        try:
            import cftime
            times = pd.DatetimeIndex(
                [pd.Timestamp(t.strftime("%Y-%m-%d %H:%M:%S")) for t in raw]
            )
        except Exception:
            return None, None

    return times.min(), times.max()


def get_latlon_range(ds: xr.Dataset) -> dict:
    """
    Return a dict with lat_min, lat_max, lon_min, lon_max.
    Falls back to (-90, 90, -180, 180) if coordinates not found.
    """
    lat_dim = _find_dim_in_ds(ds, ("lat", "latitude", "y"))
    lon_dim = _find_dim_in_ds(ds, ("lon", "longitude", "x"))

    result = {"lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0}

    if lat_dim:
        lat_vals = ds[lat_dim].values.astype(float)
        result["lat_min"] = float(np.nanmin(lat_vals))
        result["lat_max"] = float(np.nanmax(lat_vals))

    if lon_dim:
        lon_vals = ds[lon_dim].values.astype(float)
        result["lon_min"] = float(np.nanmin(lon_vals))
        result["lon_max"] = float(np.nanmax(lon_vals))

    return result


def generate_synthetic_dataset() -> xr.Dataset:
    """
    Generate a small synthetic climate-like NetCDF dataset for demo purposes.

    Returns
    -------
    xr.Dataset
        Dataset with variables: temperature, precipitation, wind_speed
    """
    rng = np.random.default_rng(42)

    lats = np.arange(-90, 91, 5.0)          # 37 points
    lons = np.arange(-180, 181, 5.0)         # 73 points
    times = pd.date_range("2020-01-01", periods=36, freq="MS")  # 3 years monthly

    nlat, nlon, nt = len(lats), len(lons, ), len(times)

    # Temperature: base latitudinal gradient + seasonal + noise
    lat_grid = lats[:, None, None]           # (lat, 1, 1)
    month_grid = np.arange(nt)[None, None, :]  # (1, 1, time) → broadcast
    temperature = (
        30 * np.cos(np.deg2rad(lat_grid))                        # pole-equator
        + 10 * np.sin(2 * np.pi * month_grid / 12)               # seasonal
        + rng.standard_normal((nlat, nlon, nt)) * 2              # noise
    ).transpose(2, 0, 1)  # → (time, lat, lon)

    # Precipitation: tropical peak + noise
    precip_base = np.exp(-((lats[:, None]) ** 2) / 800)          # (lat, 1)
    precipitation = (
        precip_base[None, :, :]
        * (1 + 0.5 * np.sin(2 * np.pi * np.arange(nt)[:, None, None] / 12))
        * rng.exponential(3, (nt, nlat, nlon))
    )

    # Wind speed: random but smooth-ish
    wind_speed = np.abs(
        5 + rng.standard_normal((nt, nlat, nlon)) * 3
    )

    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temperature.astype(np.float32),
                            {"units": "°C", "long_name": "2m Air Temperature"}),
            "precipitation": (["time", "lat", "lon"], precipitation.astype(np.float32),
                              {"units": "mm/day", "long_name": "Daily Precipitation"}),
            "wind_speed": (["time", "lat", "lon"], wind_speed.astype(np.float32),
                           {"units": "m/s", "long_name": "10m Wind Speed"}),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "PyClimaExplorer Synthetic Demo Dataset",
            "source": "Generated by PyClimaExplorer",
            "history": "Synthetic data for demonstration purposes only",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename non-standard coordinate names to canonical forms."""
    rename_map = {}
    for dim in ds.dims:
        dl = dim.lower()
        if dl in ("latitude",) and "lat" not in ds.dims:
            rename_map[dim] = "lat"
        elif dl in ("longitude",) and "lon" not in ds.dims:
            rename_map[dim] = "lon"
        elif dl in ("valid_time",) and "time" not in ds.dims:
            rename_map[dim] = "time"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _find_dim(da: xr.DataArray, candidates: tuple) -> Optional[str]:
    """Find the first matching dimension name (case-insensitive)."""
    for dim in da.dims:
        if dim.lower() in candidates:
            return dim
    return None


def _find_dim_in_ds(ds: xr.Dataset, candidates: tuple) -> Optional[str]:
    """Find the first matching coordinate in a Dataset."""
    for coord in ds.coords:
        if coord.lower() in candidates:
            return coord
    # Also check dims
    for dim in ds.dims:
        if dim.lower() in candidates:
            return dim
    return None
