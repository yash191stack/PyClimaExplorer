# modules/__init__.py
# PyClimaExplorer – Climate Data Visualization Dashboard
# This package contains the core modules for data loading,
# visualization generation, and utility helpers.

from modules.data_loader import load_netcdf, get_variables, slice_time
from modules.utils import find_nearest_latlon, format_coord_label
from modules.visualizations import (
    create_spatial_heatmap,
    create_time_series,
    create_animated_map,
    create_anomaly_chart,
)

__all__ = [
    "load_netcdf",
    "get_variables",
    "slice_time",
    "find_nearest_latlon",
    "format_coord_label",
    "create_spatial_heatmap",
    "create_time_series",
    "create_animated_map",
    "create_anomaly_chart",
]
