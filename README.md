# 🌍 PyClimaExplorer

**Climate Data Visualization Dashboard** — an interactive, web-based Python dashboard for exploring global climate NetCDF datasets.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 NetCDF Upload | Upload any `.nc` file or use the built-in synthetic demo |
| 🌡 Variable Selector | Auto-detects all spatial climate variables |
| 🕐 Time Slider | Filter to any date range present in the dataset |
| 📍 Location Picker | Select lat/lon; snaps to nearest grid point |
| 🗺 Spatial Heatmap | Global map with multiple projections & colour scales |
| 📈 Time Series | Point time-series with rolling mean overlay |
| 📊 Anomaly Chart | Monthly anomaly bars (vs. climatological mean) |
| 🎬 Animated Map | Frame-by-frame animation through time |
| 🔀 Period Comparison | Side-by-side overlay of two custom date ranges |

---

## 🗂 Project Structure

```
PyClimaExplorer/
│
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   └── sample_dataset_info.txt # How to download sample climate datasets
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # NetCDF loading & synthetic dataset generator
│   ├── visualizations.py       # All Plotly chart builders
│   └── utils.py                # Coordinate helpers, unit tools, anomaly math
│
├── assets/
│   └── styles.css              # Premium dark-theme CSS
│
└── notebooks/
    └── dataset_exploration.ipynb  # Jupyter EDA notebook
```

---

## 🚀 Quick Start

### 1 — Clone / download

```bash
git clone https://github.com/yourname/PyClimaExplorer.git
cd PyClimaExplorer
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the app

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## 📦 Required Libraries

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `xarray` | NetCDF / multi-dimensional array handling |
| `netcdf4` | NetCDF4 backend for xarray |
| `pandas` | Tabular data & time-series |
| `numpy` | Numerical operations |
| `plotly` | Interactive visualisations |
| `matplotlib` | Fallback static plots |
| `scipy` | Fallback NetCDF engine (scipy backend) |
| `cftime` | CF-calendar time decoding |

---

## 🌐 Downloading Sample NetCDF Datasets

### Option A — Instant Demo (no download needed)
Enable **"Use synthetic demo dataset"** in the sidebar toggle.  
The app generates a 3-year (2020–2022) monthly global dataset with three variables:  
`temperature`, `precipitation`, `wind_speed`.

### Option B — ERA5 Reanalysis (Recommended)
1. Create a free account at [Copernicus CDS](https://cds.climate.copernicus.eu/)
2. Install the API client: `pip install cdsapi`
3. Configure `~/.cdsapirc` with your UID & API key (see CDS documentation)
4. Run:

```python
import cdsapi
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': ['2020', '2021', '2022'],
        'month': [str(m).zfill(2) for m in range(1, 13)],
        'time': '00:00',
        'format': 'netcdf',
    },
    'data/era5_t2m_2020_2022.nc'
)
```

### Option C — NOAA ERSST (no login required)

```python
import xarray as xr
ds = xr.open_dataset(
    "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"
)
ds.to_netcdf("data/sst_ersst5.nc")
```

### Option D — CESM2 (Community Earth System Model)
Register and download at [NCAR Earth System Grid](https://www.earthsystemgrid.org/).

> **Tip:** Place downloaded `.nc` files in the `data/` folder for easy access.  
> See `data/sample_dataset_info.txt` for more sources.

---

## 🖥 Screenshots

> Launch the app and explore — the dashboard renders beautifully in any modern browser.

| View | Description |
|---|---|
| 🗺 Spatial Map tab | Global heatmap with projection & colour-scale selector |
| 📈 Time Series tab | Line chart + rolling mean + summary statistics |
| ✨ Bonus Features tab | Anomaly bars · Animated map · Two-period comparison |

---

## 🔬 Jupyter Notebook

`notebooks/dataset_exploration.ipynb` contains:
- Loading & inspecting a NetCDF file
- Computing descriptive statistics
- Generating quick matplotlib previews
- Identifying the spatial and temporal extents

Run it with:
```bash
pip install jupyter
jupyter notebook notebooks/dataset_exploration.ipynb
```

---

## 🏗 Architecture Overview

```
app.py
 ├─ loads dataset via modules/data_loader.py
 ├─ extracts metadata (variables, time range, lat/lon bounds)
 ├─ renders sidebar controls
 └─ for each tab:
     ├─ calls modules/utils.py  (coordinate snapping, anomaly math)
     └─ calls modules/visualizations.py  (Plotly figure builders)
```

---

## 🚧 Future Improvements

- [ ] **Multi-level datasets** — pressure-level slicing (e.g., 500 hPa)
- [ ] **Difference maps** — pixel-wise subtraction between two time steps
- [ ] **Trend analysis** — linear regression overlay on time series
- [ ] **Ensemble spread** — visualise model ensemble variance
- [ ] **CSV/GeoTIFF export** — spatial slice export to raster formats
- [ ] **Caching layer** — Redis / Streamlit cache for large remote datasets
- [ ] **Multi-file upload** — concatenate multiple `.nc` files along time axis
- [ ] **Dark/light theme toggle** — user-selectable UI theme

---

## 📄 Licence

MIT © 2024 PyClimaExplorer Contributors

---

<div align="center">Built with ❤️ using Python · Streamlit · Xarray · Plotly</div>
