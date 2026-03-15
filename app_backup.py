"""
app.py
======
PyClimaExplorer – Climate Data Visualization Dashboard
======================================================

Main Streamlit application.  Run with:
    streamlit run app.py

Controls (sidebar):
  • Upload a NetCDF file  OR  use the built-in synthetic demo dataset
  • Select climate variable
  • Select time range (date slider)
  • Select lat / lon for point extraction

Main panel:
  • 📍 Dataset overview (variables, dimensions, attributes)
  • 🗺  Spatial heatmap for the selected time step
  • 📈 Time-series at the selected location
  • 💡 BONUS – Anomaly chart, animated map, two-period comparison
"""

import io
import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# ── Module imports ──────────────────────────────────────────────────────────
from modules.data_loader import (
    generate_synthetic_dataset,
    get_latlon_range,
    get_time_range,
    get_variables,
    load_netcdf,
    slice_time,
)
from modules.utils import (
    extract_point_timeseries,
    find_nearest_latlon,
    format_coord_label,
    get_colorscale_for_variable,
    get_variable_units,
)
from modules.visualizations import (
    create_animated_map,
    create_anomaly_chart,
    create_comparison_chart,
    create_spatial_heatmap,
    create_time_series,
)

warnings.filterwarnings("ignore")

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PyClimaExplorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
def _inject_css():
    try:
        with open("assets/styles.css") as f:
            css = f.read()
    except FileNotFoundError:
        css = ""
    # Inline fallback dark-theme tweaks
    st.markdown(
        f"""
        <style>
        {css}

        /* Base dark theme overrides */
        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }}

        /* Sidebar gradient */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d1b2a 0%, #1a1a2e 100%);
            border-right: 1px solid #2a3a4a;
        }}

        /* Metric cards */
        [data-testid="metric-container"] {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #2a3a4a;
            border-radius: 12px;
            padding: 1rem;
        }}

        /* Expander polish */
        details summary {{
            font-weight: 600;
            color: #4a90d9;
        }}

        /* Tab strip */
        [data-baseweb="tab-list"] {{
            background: #141824;
            border-radius: 8px;
            gap: 4px;
        }}

        /* Divider */
        hr {{
            border-color: #2a3a4a;
        }}

        /* Badge-style info boxes */
        .info-badge {{
            display: inline-block;
            padding: 0.2rem 0.75rem;
            border-radius: 20px;
            background: #1a2a3a;
            border: 1px solid #4a90d9;
            color: #4a90d9;
            font-size: 0.8rem;
            margin: 2px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()


# ── Session state helpers ─────────────────────────────────────────────────────
def _get_state(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 PyClimaExplorer")
    st.markdown("*Climate Data Visualization Dashboard*")
    st.divider()

    # ── Data source ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    use_demo = st.toggle("Use synthetic demo dataset", value=True, key="use_demo")

    uploaded_file = None
    if not use_demo:
        uploaded_file = st.file_uploader(
            "Upload a NetCDF (.nc) file",
            type=["nc", "nc4"],
            help="Drag and drop or click to browse. Large files may take a moment to load.",
        )
        if uploaded_file is None:
            st.info("👆 Upload a file, or enable the demo toggle above.")

    st.divider()

    # ── Placeholder for dynamic controls (populated after dataset loads) ──────
    sidebar_controls = st.container()

# ── Dataset loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_demo():
    return generate_synthetic_dataset()


@st.cache_data(show_spinner=False)
def _load_uploaded(file_bytes: bytes):
    buf = io.BytesIO(file_bytes)
    return load_netcdf(buf)


ds = None  # type: Optional[xr.Dataset]

with st.spinner("Loading dataset…"):
    if use_demo:
        ds = _load_demo()
        data_label = "🧪 Synthetic Demo Dataset"
    elif uploaded_file is not None:
        try:
            ds = _load_uploaded(uploaded_file.getvalue())
            data_label = f"📄 {uploaded_file.name}"
        except Exception as exc:
            st.error(f"❌ Failed to load dataset: {exc}")
            st.stop()
    else:
        ds = None


# ── Header banner ─────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("# 🌍 PyClimaExplorer")
    st.markdown("**Climate Data Visualization Dashboard** – explore global climate variables interactively.")
with col_badge:
    if ds is not None:
        st.success(data_label, icon="✅")


if ds is None:
    st.info("⬅️ Upload a NetCDF dataset or enable the demo toggle in the sidebar to get started.")
    st.stop()

# ── Extract dataset metadata ──────────────────────────────────────────────────
variables = get_variables(ds)
t_min, t_max = get_time_range(ds)
latlon_range = get_latlon_range(ds)

if not variables:
    st.error("No spatial variables found in this dataset.")
    st.stop()

# ── Sidebar dynamic controls ───────────────────────────────────────────────────
with sidebar_controls:

    # Variable selector
    st.markdown("### 🌡 Variable")
    selected_var = st.selectbox(
        "Climate variable",
        options=variables,
        index=0,
        help="Choose the climate variable you want to explore.",
    )

    st.divider()

    # Time range
    st.markdown("### 🕐 Time Range")
    has_time = t_min is not None and t_max is not None

    if has_time:
        date_min = t_min.date() if isinstance(t_min, pd.Timestamp) else date(2000, 1, 1)
        date_max = t_max.date() if isinstance(t_max, pd.Timestamp) else date(2022, 12, 31)

        sel_start, sel_end = st.slider(
            "Select date range",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_max),
            format="YYYY-MM",
            help="Drag the handles to narrow the time window.",
        )
        start_str = str(sel_start)
        end_str = str(sel_end)
    else:
        st.info("No time axis detected.")
        start_str, end_str = None, None

    st.divider()

    # Spatial selection
    st.markdown("### 📍 Location")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        sel_lat = st.number_input(
            "Latitude",
            min_value=float(latlon_range["lat_min"]),
            max_value=float(latlon_range["lat_max"]),
            value=0.0,
            step=2.5,
            format="%.1f",
        )
    with col_lon:
        sel_lon = st.number_input(
            "Longitude",
            min_value=float(latlon_range["lon_min"]),
            max_value=float(latlon_range["lon_max"]),
            value=0.0,
            step=2.5,
            format="%.1f",
        )

    actual_lat, actual_lon = find_nearest_latlon(ds, sel_lat, sel_lon)
    st.caption(f"📌 Snapped to: {format_coord_label(actual_lat, actual_lon)}")

    st.divider()

    # Bonus feature toggles
    st.markdown("### ✨ Bonus Features")
    show_anomaly  = st.checkbox("📊 Climate anomaly chart", value=True)
    show_animated = st.checkbox("🎬 Animated climate map", value=False)
    show_compare  = st.checkbox("🔀 Two-period comparison", value=False)

    if show_compare and has_time:
        st.markdown("**Period A**")
        ca_start, ca_end = st.slider(
            "Period A",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_min + timedelta(days=365)),
            format="YYYY-MM",
            key="cmp_a",
        )
        st.markdown("**Period B**")
        cb_start, cb_end = st.slider(
            "Period B",
            min_value=date_min,
            max_value=date_max,
            value=(date_max - timedelta(days=365), date_max),
            format="YYYY-MM",
            key="cmp_b",
        )

# ── Prepare data arrays ────────────────────────────────────────────────────────
da_full  = ds[selected_var]
da_slice = slice_time(ds, selected_var, start_str, end_str)

# Time series at selected location
ts = extract_point_timeseries(da_slice, actual_lat, actual_lon)
units = get_variable_units(da_full)

# Spatial 2-D slice – pick midpoint of selected window
time_dim = next(
    (d for d in da_slice.dims if d.lower() in ("time", "t", "datetime")), None
)
mid_idx = len(da_slice[time_dim]) // 2 if time_dim else 0

if time_dim and len(da_slice[time_dim]) > 0:
    mid_time_val = da_slice[time_dim].values[mid_idx]
    try:
        mid_time_label = str(pd.Timestamp(mid_time_val))[:10]
    except Exception:
        mid_time_label = str(mid_idx)
    da_2d = da_slice.isel({time_dim: mid_idx}).squeeze()
else:
    mid_time_label = ""
    da_2d = da_slice.squeeze()

# ── Dataset overview ───────────────────────────────────────────────────────────
with st.expander("📋 Dataset Overview", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables", len(variables))
    col2.metric("Time steps", int(ds.dims.get("time", ds.dims.get("t", 0))))
    col3.metric("Lat points", int(ds.dims.get("lat", ds.dims.get("latitude", 0))))
    col4.metric("Lon points", int(ds.dims.get("lon", ds.dims.get("longitude", 0))))

    st.markdown("**Available Variables**")
    badge_html = " ".join(f'<span class="info-badge">{v}</span>' for v in variables)
    st.markdown(badge_html, unsafe_allow_html=True)

    if ds.attrs:
        st.markdown("**Global Attributes**")
        attrs_df = pd.DataFrame(
            [(k, str(v)) for k, v in ds.attrs.items()],
            columns=["Attribute", "Value"],
        )
        st.dataframe(attrs_df, width="stretch", hide_index=True)

    st.markdown("**Selected Variable Info**")
    var_info_cols = st.columns(3)
    var_info_cols[0].markdown(f"**Name**: `{selected_var}`")
    var_info_cols[1].markdown(f"**Units**: `{units or 'N/A'}`")
    var_info_cols[2].markdown(f"**Dims**: `{', '.join(da_full.dims)}`")
    if "long_name" in da_full.attrs:
        st.caption(da_full.attrs["long_name"])

st.divider()

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_map, tab_ts, tab_bonus = st.tabs(
    ["🗺 Spatial Map", "📈 Time Series", "✨ Bonus Features"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Spatial heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tab_map:
    st.markdown(f"### 🗺 Spatial Distribution — `{selected_var}`")

    map_col, ctrl_col = st.columns([4, 1])

    with ctrl_col:
        projection = st.selectbox(
            "Map projection",
            ["natural earth", "orthographic", "mercator", "equirectangular", "robinson"],
            key="proj",
        )
        colorscale_opt = st.selectbox(
            "Colour scale",
            ["Auto-detect", "RdBu_r", "Viridis", "Plasma", "Blues", "Reds", "YlGnBu", "RdYlGn", "Jet"],
            key="cscale",
        )
        chosen_cs = (
            None if colorscale_opt == "Auto-detect" else colorscale_opt
        )

        if time_dim:
            time_idx_override = st.slider(
                "Time step",
                0,
                max(0, len(da_slice[time_dim]) - 1),
                mid_idx,
                key="time_idx",
            )
            da_2d_disp = da_slice.isel({time_dim: time_idx_override}).squeeze()
            try:
                tval = da_slice[time_dim].values[time_idx_override]
                disp_time_lbl = str(pd.Timestamp(tval))[:10]
            except Exception:
                disp_time_lbl = str(time_idx_override)
        else:
            da_2d_disp = da_2d
            disp_time_lbl = mid_time_label

    with map_col:
        with st.spinner("Rendering map…"):
            fig_map = create_spatial_heatmap(
                da_2d_disp,
                selected_var,
                time_label=disp_time_lbl,
                colorscale=chosen_cs,
                projection=projection,
            )
        st.plotly_chart(fig_map, width="stretch")

    st.caption(
        f"Showing **{selected_var}** at time step **{disp_time_lbl}**  "
        f"| Colour scale: **{colorscale_opt}**  "
        f"| Projection: **{projection}**"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Time-series
# ─────────────────────────────────────────────────────────────────────────────
with tab_ts:
    st.markdown(f"### 📈 Time Series — `{selected_var}` at {format_coord_label(actual_lat, actual_lon)}")

    ts_col_l, ts_col_r = st.columns([5, 1])

    with ts_col_r:
        rolling_w = st.slider("Rolling window", 1, 12, 3, key="roll_w")
        show_roll  = st.checkbox("Show rolling mean", value=True, key="show_roll")

    with ts_col_l:
        if len(ts) == 0:
            st.warning("No data points in the selected time range for this location.")
        else:
            fig_ts = create_time_series(
                ts, selected_var, actual_lat, actual_lon,
                units=units,
                show_rolling=show_roll,
                rolling_window=rolling_w,
            )
            st.plotly_chart(fig_ts, width="stretch")

    # Summary statistics
    if len(ts) > 0:
        st.markdown("#### 📊 Summary Statistics")
        s_cols = st.columns(5)
        s_cols[0].metric("Mean",  f"{ts.mean():.3g} {units}")
        s_cols[1].metric("Std",   f"{ts.std():.3g} {units}")
        s_cols[2].metric("Min",   f"{ts.min():.3g} {units}")
        s_cols[3].metric("Max",   f"{ts.max():.3g} {units}")
        s_cols[4].metric("Range", f"{(ts.max()-ts.min()):.3g} {units}")

        with st.expander("📥 Download time-series data"):
            csv_df = pd.DataFrame({"date": ts.index, selected_var: ts.values})
            csv_bytes = csv_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"{selected_var}_{format_coord_label(actual_lat, actual_lon).replace(',','').replace(' ','_')}.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Bonus Features
# ─────────────────────────────────────────────────────────────────────────────
with tab_bonus:
    st.markdown("### ✨ Bonus Features")

    if not any([show_anomaly, show_animated, show_compare]):
        st.info("Enable one or more bonus features in the sidebar to display them here.")

    # ── Anomaly chart ─────────────────────────────────────────────────────────
    if show_anomaly:
        st.markdown("---")
        st.markdown("#### 📊 Climate Anomaly Analysis")
        st.markdown(
            "Monthly anomalies computed relative to the long-term climatological mean. "
            "Red bars = above average · Blue bars = below average."
        )
        if len(ts) < 2:
            st.warning("Not enough data points to compute anomalies.")
        else:
            fig_anom = create_anomaly_chart(ts, selected_var, units, actual_lat, actual_lon)
            st.plotly_chart(fig_anom, width="stretch")

    # ── Animated map ──────────────────────────────────────────────────────────
    if show_animated:
        st.markdown("---")
        st.markdown("#### 🎬 Animated Climate Map")
        max_fr = st.slider("Max animation frames", 6, 60, 24, key="anim_frames")
        st.info("⚠️ Large datasets may take a moment to render all frames.")
        with st.spinner("Building animation…"):
            fig_anim = create_animated_map(da_slice, selected_var, max_frames=max_fr)
        st.plotly_chart(fig_anim, width="stretch")
        st.caption("Use the ▶ Play button or drag the slider to step through time.")

    # ── Two-period comparison ─────────────────────────────────────────────────
    if show_compare:
        st.markdown("---")
        st.markdown("#### 🔀 Two-Period Comparison")
        if not has_time:
            st.warning("Dataset has no time axis; period comparison is unavailable.")
        else:
            ts_a = extract_point_timeseries(
                slice_time(ds, selected_var, str(ca_start), str(ca_end)),
                actual_lat, actual_lon,
            )
            ts_b = extract_point_timeseries(
                slice_time(ds, selected_var, str(cb_start), str(cb_end)),
                actual_lat, actual_lon,
            )

            if len(ts_a) == 0 or len(ts_b) == 0:
                st.warning("One or both periods returned no data. Adjust the sliders.")
            else:
                fig_cmp = create_comparison_chart(
                    ts_a, ts_b,
                    label1=f"Period A  ({ca_start} – {ca_end})",
                    label2=f"Period B  ({cb_start} – {cb_end})",
                    variable=selected_var,
                    units=units,
                )
                st.plotly_chart(fig_cmp, width="stretch")

                c1, c2 = st.columns(2)
                c1.metric("Period A mean", f"{ts_a.mean():.3g} {units}", delta=None)
                c2.metric(
                    "Period B mean",
                    f"{ts_b.mean():.3g} {units}",
                    delta=f"{ts_b.mean() - ts_a.mean():.3g}",
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#4a6a7a; font-size:0.8rem;'>"
    "🌍 PyClimaExplorer &nbsp;·&nbsp; Built with Streamlit & Plotly &nbsp;·&nbsp; "
    "Data: ERA5 / CESM / Synthetic Demo"
    "</center>",
    unsafe_allow_html=True,
)
