"""
app.py
======
PyClimaExplorer – Climate Data Visualization Dashboard
======================================================
"""

import io
import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import requests
from streamlit_lottie import st_lottie
from streamlit_plotly_events import plotly_events

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
    generate_climate_insights,
)
from modules.visualizations import (
    create_animated_map,
    create_anomaly_chart,
    create_comparison_chart,
    create_spatial_heatmap,
    create_time_series,
    create_difference_map
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
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_inject_css()

# ── Session state helpers ─────────────────────────────────────────────────────
if 'click_lat' not in st.session_state:
    st.session_state['click_lat'] = 0.0
if 'click_lon' not in st.session_state:
    st.session_state['click_lon'] = 0.0
if 'story_step' not in st.session_state:
    st.session_state['story_step'] = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 PyClimaExplorer")
    st.markdown("*Advanced Analytics Edition*")
    st.divider()

    st.markdown("### 📂 Dataset")
    use_demo = st.toggle("Use synthetic demo dataset", value=True, key="use_demo")

    uploaded_file = None
    if not use_demo:
        uploaded_file = st.file_uploader(
            "Upload a NetCDF (.nc) file",
            type=["nc", "nc4"],
        )

    st.divider()
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
        if "first_load" not in st.session_state:
            st.balloons()
            st.toast('Synthetic Dataset Loaded Successfully!', icon='🚀')
            st.session_state["first_load"] = True
    elif uploaded_file is not None:
        try:
            ds = _load_uploaded(uploaded_file.getvalue())
            data_label = f"📄 {uploaded_file.name}"
            if "first_load" not in st.session_state:
                st.snow()
                st.toast('NetCDF File Loaded Successfully!', icon='✅')
                st.session_state["first_load"] = True
        except Exception as exc:
            st.error(f"❌ Failed to load dataset: {exc}")
            st.stop()
    else:
        ds = None

# ── Sticky Top Navbar ──────────────────────────────────────────────────────────
st.markdown("""
<div class="top-navbar">
    <div class="nav-logo">🌍 PyClimaExplorer</div>
    <div class="nav-links">
        <a href="#" class="nav-link active">Dashboard</a>
        <a href="#" class="nav-link">Analytics</a>
        <a href="#" class="nav-link">Documentation</a>
        <a href="https://github.com" target="_blank" class="nav-link button-link">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────────────
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_globe = load_lottieurl("https://lottie.host/7e0767fb-231a-429a-9e1e-2cb6851eb3fa/H7xW2kGkP1.json")

col_badge, col_title, col_lottie = st.columns([1, 4, 1.5])
with col_badge:
    if ds is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(data_label, icon="✅")
with col_title:
    st.markdown('''
    <div class="hero-section">
        <h1>Advanced Climate Analytics</h1>
        <p>Explore, analyze, and gain actionable insights from global climate datasets in real-time.</p>
    </div>
    ''', unsafe_allow_html=True)
with col_lottie:
    if lottie_globe:
        st_lottie(lottie_globe, height=180, key="hero_lottie")

if ds is None:
    st.info("⬅️ Upload a NetCDF dataset or enable the demo toggle to get started.")
    st.stop()

# Meta
variables = get_variables(ds)
t_min, t_max = get_time_range(ds)
latlon_range = get_latlon_range(ds)

if not variables:
    st.error("No spatial variables found in this dataset.")
    st.stop()

# ── Sidebar dynamic controls ───────────────────────────────────────────────────
with sidebar_controls:
    st.markdown("### 🌡 Variable")
    selected_var = st.selectbox("Climate variable", options=variables, index=0)

    st.divider()
    has_time = t_min is not None and t_max is not None

    if has_time:
        date_min = t_min.date() if isinstance(t_min, pd.Timestamp) else date(2000, 1, 1)
        date_max = t_max.date() if isinstance(t_max, pd.Timestamp) else date(2022, 12, 31)

        st.markdown("### 🕐 Time Range")
        sel_start, sel_end = st.slider(
            "Select date range",
            min_value=date_min, max_value=date_max,
            value=(date_min, date_max), format="YYYY-MM"
        )
        start_str, end_str = str(sel_start), str(sel_end)
    else:
        start_str, end_str = None, None

    st.divider()
    st.markdown("### ✨ Mode Selection")
    mode = st.radio("Select View Mode", ["Interactive Explorer", "📖 Story Mode", "🔀 Comparison Mode"])

# ── Prepare data arrays ────────────────────────────────────────────────────────
da_full  = ds[selected_var]
da_slice = slice_time(ds, selected_var, start_str, end_str)

time_dim = next((d for d in da_slice.dims if d.lower() in ("time", "t", "datetime")), None)
mid_idx = len(da_slice[time_dim]) // 2 if time_dim else 0

if time_dim and len(da_slice[time_dim]) > 0:
    da_2d = da_slice.isel({time_dim: mid_idx}).squeeze()
    try:
        disp_time_lbl = str(pd.Timestamp(da_slice[time_dim].values[mid_idx]))[:10]
    except:
        disp_time_lbl = "Time"
else:
    da_2d = da_slice.squeeze()
    disp_time_lbl = ""

units = get_variable_units(da_full)

# ===================================================================================
# MODE 1: Interactive Explorer (Default Maps + Click to analyze)
# ===================================================================================
if mode == "Interactive Explorer":
    tab_map, tab_anim = st.tabs(["📍 Click-to-Analyze Map", "🎬 Animated Map"])

    with tab_map:
        st.markdown("### 📈 Global Overview")
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.metric(f"Global Max ({units})", f"{float(da_2d.max()):.2f}")
        with kpi_col2:
            st.metric(f"Global Avg ({units})", f"{float(da_2d.mean()):.2f}")
        with kpi_col3:
            st.metric(f"Global Min ({units})", f"{float(da_2d.min()):.2f}")
            
        st.markdown("<div class='insight-box'><b>💡 Pro Tip:</b> Click anywhere on the map below to extract instant climate insights and time-series anomalies for that exact location!</div>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns([3, 1])
        with col_m2:
            st.markdown("#### Controls")
            colorscale_opt = st.selectbox("Colour scale", ["Auto", "RdBu_r", "Viridis", "Plasma", "Blues", "Reds", "YlGnBu", "Jet"])
            cs = None if colorscale_opt == "Auto" else colorscale_opt
            projection = st.selectbox("Map projection", ["orthographic", "natural earth", "mercator"])
            
            if time_dim:
                time_idx = st.slider("Time step index", 0, max(0, len(da_slice[time_dim])-1), mid_idx)
                da_2d_disp = da_slice.isel({time_dim: time_idx}).squeeze()
                try:
                    time_lbl = str(pd.Timestamp(da_slice[time_dim].values[time_idx]))[:10]
                except:
                    time_lbl = ""
            else:
                da_2d_disp = da_2d
                time_lbl = disp_time_lbl

        with col_m1:
            fig_map = create_spatial_heatmap(da_2d_disp, selected_var, time_label=time_lbl, colorscale=cs, projection=projection)
            # Plotly events for Click-to-analyze
            clicked_points = plotly_events(fig_map, click_event=True, hover_event=False, override_height=480)
            
            if clicked_points:
                st.session_state['click_lat'] = clicked_points[0]['lat']
                st.session_state['click_lon'] = clicked_points[0]['lon']

        st.markdown("---")
        
        # Click Analysis Result Panel
        act_lat, act_lon = find_nearest_latlon(ds, st.session_state['click_lat'], st.session_state['click_lon'])
        ts = extract_point_timeseries(da_slice, act_lat, act_lon)
        
        st.markdown(f"### 📊 Deep-Dive Analysis for: `{format_coord_label(act_lat, act_lon)}`")
        
        if len(ts) > 0:
            insight_text = generate_climate_insights(ts, selected_var)
            st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)
            
            # 1-Click Export CSV
            csv_data = ts.to_csv().encode('utf-8')
            
            def export_triggered():
                st.toast('Report successfully downloaded!', icon='📥')
                
            st.download_button(
                label="📥 Download Time-Series Data (CSV)",
                data=csv_data,
                file_name=f"pyclima_{selected_var}_{act_lat:.2f}_{act_lon:.2f}.csv",
                mime="text/csv",
                on_click=export_triggered
            )
            
            c_ts, c_an = st.columns(2)
            with c_ts:
                fig_ts = create_time_series(ts, selected_var, act_lat, act_lon, units=units)
                st.plotly_chart(fig_ts, use_container_width=True)
            with c_an:
                fig_an = create_anomaly_chart(ts, selected_var, units, act_lat, act_lon)
                st.plotly_chart(fig_an, use_container_width=True)
        else:
            st.warning("No time-series data available for the clicked location.")

    with tab_anim:
        st.markdown("### 🎬 Temporal Evolution")
        st.markdown("Watch how the climate shifts over the selected period.")
        if st.button("Generate Animation"):
            with st.spinner("Rendering animation frames (this may take a few seconds)..."):
                fig_anim = create_animated_map(da_slice, selected_var, max_frames=36)
                st.plotly_chart(fig_anim, use_container_width=True)

# ===================================================================================
# MODE 2: Story Mode
# ===================================================================================
elif mode == "📖 Story Mode":
    st.markdown("<div class='story-card'><h2>Climate Stories</h2><p>A guided exploration of the dataset highlighting significant patterns.</p></div>", unsafe_allow_html=True)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1,2,1])
    with col_nav1:
        if st.button("⬅️ Previous") and st.session_state['story_step'] > 0:
            st.session_state['story_step'] -= 1
    with col_nav3:
        if st.button("Next ➡️") and st.session_state['story_step'] < 2:
            st.session_state['story_step'] += 1

    step = st.session_state['story_step']
    
    if step == 0:
        st.markdown("### Step 1: Global Extremes")
        st.write(f"Looking at the overall distribution of **{selected_var}**. Which regions experience the highest and lowest values?")
        fig_map = create_spatial_heatmap(da_2d, selected_var, time_label=disp_time_lbl, projection="orthographic")
        st.plotly_chart(fig_map, use_container_width=True)
        
    elif step == 1:
        st.markdown("### Step 2: Temporal Variability")
        st.write("We extract the signal at a key mid-latitude point to observe seasonal and long-term cyclic behaviour. The rolling mean reveals the underlying trend.")
        ts = extract_point_timeseries(da_slice, 45.0, 0.0)
        if len(ts)>0:
            st.plotly_chart(create_time_series(ts, selected_var, 45.0, 0.0, units=units, rolling_window=5), use_container_width=True)
        
    elif step == 2:
        st.markdown("### Step 3: Climate Anomalies")
        st.write("Anomalies show us deviations from the norm. Red spikes indicate higher-than-average events (like heatwaves), while blue spikes indicate lower-than-average events.")
        ts = extract_point_timeseries(da_slice, 45.0, 0.0)
        if len(ts)>0:
            st.plotly_chart(create_anomaly_chart(ts, selected_var, units, 45.0, 0.0), use_container_width=True)

# ===================================================================================
# MODE 3: Comparison Mode
# ===================================================================================
elif mode == "🔀 Comparison Mode":
    if not has_time:
        st.warning("Comparison mode requires a dataset with a time dimension.")
    else:
        st.markdown("### 🔀 Two-Period Spatial & Temporal Comparison")
        st.markdown("Compare two distinct time periods to detect structural climate shifts.")
        
        ca_c, cb_c = st.columns(2)
        with ca_c:
            st.markdown("#### 🔵 Period 1")
            ca_start, ca_end = st.slider("Period 1 Dates", min_value=date_min, max_value=date_max, value=(date_min, date_min + timedelta(days=365)), format="YYYY-MM", key="cmp_a")
        with cb_c:
            st.markdown("#### 🔴 Period 2")
            cb_start, cb_end = st.slider("Period 2 Dates", min_value=date_min, max_value=date_max, value=(date_max - timedelta(days=365), date_max), format="YYYY-MM", key="cmp_b")

        st.markdown("---")
        
        # Spatial Diff
        da1 = slice_time(ds, selected_var, str(ca_start), str(ca_end)).mean(dim=time_dim, skipna=True)
        da2 = slice_time(ds, selected_var, str(cb_start), str(cb_end)).mean(dim=time_dim, skipna=True)
        
        st.markdown("#### Spatial Difference Map (Period 2 minus Period 1")
        st.markdown("Regions in red indicate an increase over time, while blue indicates a decrease.")
        with st.spinner("Calculating spatial differences..."):
            fig_diff = create_difference_map(da1, da2, selected_var, f"P1({ca_start.year})", f"P2({cb_start.year})")
            st.plotly_chart(fig_diff, use_container_width=True)
            
        # Temporal Overlay
        st.markdown("#### Exact Point Temporal Shift")
        act_lat, act_lon = find_nearest_latlon(ds, 0.0, 0.0) # default equator for demo
        ts_a = extract_point_timeseries(slice_time(ds, selected_var, str(ca_start), str(ca_end)), act_lat, act_lon)
        ts_b = extract_point_timeseries(slice_time(ds, selected_var, str(cb_start), str(cb_end)), act_lat, act_lon)
        
        if len(ts_a)>0 and len(ts_b)>0:
            fig_cmp = create_comparison_chart(ts_a, ts_b, f"Period 1 ({ca_start.year})", f"Period 2 ({cb_start.year})", selected_var, units)
            st.plotly_chart(fig_cmp, use_container_width=True)

