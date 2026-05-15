#!/usr/bin/env python3
"""
04_cross-sections.py - COMPREHENSIVE HTML REPORT GENERATOR WITH PDF EXPORT
Generates an HTML report AND saves all plots as PDF files
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from pathlib import Path
import pandas as pd
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
data_dir = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/")
plots_dir = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/Cross-sections/Plots")
stats_dir = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/Cross-sections/Statistics")
plots_dir.mkdir(parents=True, exist_ok=True)
stats_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING COMPREHENSIVE HTML REPORT AND PDF EXPORTS")
print("="*80)

# Counter for PDF exports
pdf_counter = {"cross_sections": 0, "difference_maps": 0, "boxplots": 0}

def save_figure_as_pdf(fig, filename, subfolder=""):
    """Save figure as PDF in the plots directory"""
    pdf_path = plots_dir / filename
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"   📄 PDF saved: {pdf_path.name}")
    return pdf_path

# ============================================
# PART 1: LOAD POLARSTERN MEDIAN TRANSECT COORDINATES
# ============================================
print("\n📂 Loading Polarstern median transect coordinates...")

transect_csv = plots_dir / "Polarstern_median_transect_100points.csv"
if not transect_csv.exists():
    print(f"   ❌ Error: {transect_csv} not found!")
    exit(1)

transect_df = pd.read_csv(transect_csv)
transect_df = transect_df.sort_values('latitude', ascending=False).reset_index(drop=True)

transect_lats = transect_df['latitude'].values
transect_lons = transect_df['longitude'].values
transect_name = "POLARSTERN MEDIAN TRANSECT (Bremerhaven to Namibia)"

print(f"   Loaded {len(transect_lats)} points ({transect_lats.min():.1f}°S to {transect_lats.max():.1f}°N)")

# ============================================
# PART 2: PERIOD DETECTION FUNCTION (FIXED FOR ORAS5)
# ============================================
def get_period_from_filename(filepath):
    """FIXED: Maps ORAS5 1981-2000 and 1993-2000 to baseline period 1993-2002"""
    fname = str(filepath.name)
    
    # Period 2: 2021-2025
    if '2021-2025' in fname or '2021_2025' in fname:
        return '2021-2025'
    
    # Period 1: 2003-2020
    elif '2003-2020' in fname or '2003_2020' in fname or '2001-2020' in fname or '2001_2020' in fname:
        return '2003-2020'
    
    # Baseline: 1993-2002 (including ORAS5 1981-2000 and 1993-2000)
    elif '1993-2002' in fname or '1993_2002' in fname or '1993-2000' in fname or '1993_2000' in fname:
        return '1993-2002'
    elif '1981-2000' in fname or '1981_2000' in fname:
        return '1993-2002'  # Map ORAS5 1981-2000 to baseline
    
    else:
        return 'unknown'

# ============================================
# PART 3: EXTRACTION FUNCTIONS
# ============================================

def extract_from_curvilinear_grid(ds, lats, lons, var_name=None):
    if var_name is None:
        for candidate in ['thetao', 'temperature', 'temp', 'votemper']:
            if candidate in ds:
                var_name = candidate
                break
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    
    if 'longitude' in ds.coords and 'latitude' in ds.coords:
        lon_grid = ds.longitude.values
        lat_grid = ds.latitude.values
    elif 'nav_lon' in ds.coords and 'nav_lat' in ds.coords:
        lon_grid = ds.nav_lon.values
        lat_grid = ds.nav_lat.values
    else:
        lon_grid = ds.longitude.values
        lat_grid = ds.latitude.values
    
    n_points = len(lats)
    n_depth = len(ds.lev) if 'lev' in ds.dims else 1
    extracted = np.full((n_depth, n_points), np.nan)
    valid_mask = np.zeros(n_points, dtype=bool)
    ny, nx = lat_grid.shape
    
    for i, (target_lat, target_lon) in enumerate(zip(lats, lons)):
        lon_diff = np.minimum(np.abs(lon_grid - target_lon), 360 - np.abs(lon_grid - target_lon))
        dist = np.sqrt((lat_grid - target_lat)**2 + lon_diff**2)
        i_idx, j_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i_idx = min(i_idx, ny - 1)
        j_idx = min(j_idx, nx - 1)
        
        try:
            if 'i' in ds[var_name].dims and 'j' in ds[var_name].dims:
                data_point = ds[var_name].isel(time=0, i=i_idx, j=j_idx)
            else:
                data_point = ds[var_name].isel(time=0, i=j_idx, j=i_idx)
            if not np.isnan(data_point.values).all():
                extracted[:, i] = data_point.values
                valid_mask[i] = True
        except:
            try:
                data_point = ds[var_name].isel(time=0, i=j_idx, j=i_idx)
                if not np.isnan(data_point.values).all():
                    extracted[:, i] = data_point.values
                    valid_mask[i] = True
            except:
                pass
    return extracted, valid_mask

def extract_from_oras5(ds, lats, lons):
    var_name = None
    for candidate in ['votemper', 'thetao', 'temperature', 'temp']:
        if candidate in ds:
            var_name = candidate
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    
    if 'nav_lat' in ds and 'nav_lon' in ds:
        lat_grid = ds.nav_lat.values
        lon_grid = ds.nav_lon.values
    elif 'latitude' in ds and 'longitude' in ds:
        lat_grid = ds.latitude.values
        lon_grid = ds.longitude.values
    else:
        lat_grid = ds.y.values if 'y' in ds else ds.lat.values
        lon_grid = ds.x.values if 'x' in ds else ds.lon.values
    
    n_points = len(lats)
    n_depth = len(ds.deptht) if 'deptht' in ds.dims else 1
    extracted = np.full((n_depth, n_points), np.nan)
    valid_mask = np.zeros(n_points, dtype=bool)
    
    for i, (target_lat, target_lon) in enumerate(zip(lats, lons)):
        if lon_grid.ndim == 2:
            lon_diff = np.minimum(np.abs(lon_grid - target_lon), 360 - np.abs(lon_grid - target_lon))
            dist = np.sqrt((lat_grid - target_lat)**2 + lon_diff**2)
            y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        else:
            lon_diff = np.minimum(np.abs(lon_grid - target_lon), 360 - np.abs(lon_grid - target_lon))
            x_idx = np.argmin(lon_diff)
            y_idx = np.argmin(np.abs(lat_grid - target_lat))
        
        try:
            if n_depth > 1:
                data_point = ds[var_name].isel(time_counter=0, deptht=slice(None), y=y_idx, x=x_idx)
            else:
                data_point = ds[var_name].isel(time_counter=0, y=y_idx, x=x_idx)
            if not np.isnan(data_point.values).all():
                extracted[:, i] = data_point.values
                valid_mask[i] = True
        except:
            pass
    return extracted, valid_mask

def extract_from_regular_grid(ds, lats, lons):
    var_name = None
    for candidate in ['thetao', 'temperature', 'temp', 'votemper']:
        if candidate in ds:
            var_name = candidate
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    
    lat_name = None
    lon_name = None
    depth_name = None
    time_name = None
    
    for dim in ds.dims:
        if dim in ['latitude', 'lat', 'nav_lat', 'y']:
            lat_name = dim
        if dim in ['longitude', 'lon', 'nav_lon', 'x']:
            lon_name = dim
        if dim in ['depth', 'deptht', 'lev']:
            depth_name = dim
        if dim in ['time', 'time_counter']:
            time_name = dim
    
    try:
        if time_name:
            ds = ds.isel({time_name: 0})
        extracted = ds[var_name].interp(
            {lat_name: xr.DataArray(lats, dims='transect'),
             lon_name: xr.DataArray(lons, dims='transect')},
            method='linear'
        )
        if depth_name and depth_name in extracted.dims:
            result = extracted.values
        else:
            result = extracted.values.reshape(1, -1)
        return result, np.ones(len(lats), dtype=bool)
    except:
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
        n_points = len(lats)
        n_depth = len(ds[depth_name]) if depth_name else 1
        extracted = np.full((n_depth, n_points), np.nan)
        valid_mask = np.zeros(n_points, dtype=bool)
        for i, (target_lat, target_lon) in enumerate(zip(lats, lons)):
            i_lat = np.argmin(np.abs(lat_vals - target_lat))
            i_lon = np.argmin(np.abs(lon_vals - target_lon))
            try:
                if time_name:
                    data_point = ds[var_name].isel({time_name: 0, depth_name: slice(None), 
                                                    lat_name: i_lat, lon_name: i_lon})
                else:
                    data_point = ds[var_name].isel({depth_name: slice(None), 
                                                    lat_name: i_lat, lon_name: i_lon})
                if not np.isnan(data_point.values).all():
                    extracted[:, i] = data_point.values
                    valid_mask[i] = True
            except:
                pass
        return extracted, valid_mask

# ============================================
# PART 4: LOAD DATASETS
# ============================================

def load_dataset_files(pattern, dataset_name, loader_func):
    files = sorted(data_dir.glob(pattern))
    data_dict = {}
    for file_path in files:
        period = get_period_from_filename(file_path)
        if period == 'unknown':
            # Skip unknown periods but don't print warning for every file
            continue
        try:
            ds = xr.open_dataset(file_path)
            temp_data, valid_mask = loader_func(ds, transect_lats, transect_lons)
            
            # Get depth coordinates
            if 'depth' in ds.coords:
                depth = ds.depth.values
            elif 'deptht' in ds.coords:
                depth = ds.deptht.values
            elif 'lev' in ds.coords:
                depth = ds.lev.values
            else:
                depth = np.array([0])
            
            data_dict[period] = {'temperature': temp_data, 'depth': depth, 'valid_mask': valid_mask}
            ds.close()
        except Exception as e:
            print(f"   ❌ Error loading {file_path.name}: {e}")
    return data_dict

print("\n📂 Loading model and reanalysis data...")
print("   Using periods: Baseline (1993-2002), Period 1 (2003-2020), Period 2 (2021-2025)")

# Load all datasets (OND = October-November-December, or Q4)
cmcc_data = load_dataset_files("**/CMCC-ESM2/*OND*.nc", "CMCC-ESM2", extract_from_curvilinear_grid)
ec_data = load_dataset_files("**/EC-Earth3/*OND*.nc", "EC-Earth3", extract_from_curvilinear_grid)
oras5_data = load_dataset_files("**/ORAS5/*OND*.nc", "ORAS5", extract_from_oras5)
glorys_data = load_dataset_files("**/GLORYS/*OND*.nc", "GLORYS", extract_from_regular_grid)

# Print summary of loaded data
print("\n📊 Data loaded summary:")
for name, data_dict in [('CMCC-ESM2', cmcc_data), ('EC-Earth3', ec_data), ('ORAS5', oras5_data), ('GLORYS', glorys_data)]:
    if data_dict:
        print(f"   {name}: {list(data_dict.keys())}")
    else:
        print(f"   {name}: No data loaded!")

# ============================================
# PART 5: PLOTTING FUNCTIONS
# ============================================
print("\n🎨 Creating plots and exporting PDFs...")

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

def plot_cross_section(data, depth, latitude, title, dataset_name, period, save_pdf=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    isotherm_levels = np.arange(2, 30, 2)
    
    # Upper 1000m
    depth_mask = depth <= 1000
    depth_upper = depth[depth_mask]
    if data.ndim == 2:
        data_upper = data[depth_mask, :] if len(depth_mask) == data.shape[0] else data
    else:
        data_upper = data[0, depth_mask, :] if data.shape[0] == 1 else data
    if data_upper.ndim == 1:
        data_upper = data_upper.reshape(1, -1)
    data_upper = np.ma.masked_where(np.isnan(data_upper), data_upper)
    
    cf1 = ax1.contourf(latitude, depth_upper[:data_upper.shape[0]], data_upper, 
                       levels=20, cmap=cmocean.cm.thermal, vmin=0, vmax=30, extend='both')
    if data_upper.shape[0] > 1:
        cs1 = ax1.contour(latitude, depth_upper[:data_upper.shape[0]], data_upper, 
                          levels=isotherm_levels, colors='black', linewidths=0.8, alpha=0.5)
        ax1.clabel(cs1, inline=True, fontsize=8, fmt='%d°C')
    ax1.invert_yaxis()
    ax1.set_xlabel('Latitude (°N)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('0-1000m')
    ax1.set_xlim(latitude.max(), latitude.min())
    
    # Full depth
    if data.ndim == 2:
        plot_data = data
        plot_depth = depth[:data.shape[0]]
    else:
        plot_data = data[0, :, :] if data.shape[0] == 1 else data
        plot_depth = depth[:plot_data.shape[0]]
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
    
    cf2 = ax2.contourf(latitude, plot_depth, plot_data, 
                       levels=20, cmap=cmocean.cm.thermal, vmin=0, vmax=30, extend='both')
    if plot_data.shape[0] > 1:
        cs2 = ax2.contour(latitude, plot_depth, plot_data, 
                          levels=isotherm_levels, colors='black', linewidths=0.8, alpha=0.5)
        ax2.clabel(cs2, inline=True, fontsize=8, fmt='%d°C')
    ax2.invert_yaxis()
    ax2.set_xlabel('Latitude (°N)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('0-6000m')
    ax2.set_xlim(latitude.max(), latitude.min())
    
    plt.suptitle(f'{dataset_name} - {period}', y=1.02)
    plt.tight_layout()
    
    if save_pdf:
        pdf_filename = f"{dataset_name}_{period}_cross_section.pdf"
        save_figure_as_pdf(fig, pdf_filename)
        global pdf_counter
        pdf_counter["cross_sections"] += 1
    
    return fig

def plot_difference_map(data_dict, dataset_name, depth, latitude, baseline_period, period, save_pdf=True, vmin=-2, vmax=2):
    """Create difference plot showing warming (red) and cooling (blue)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    diff_levels = np.linspace(vmin, vmax, 21)
    
    baseline_data = data_dict[baseline_period]['temperature']
    period_data = data_dict[period]['temperature']
    
    # Upper 1000m difference
    depth_mask = depth <= 1000
    depth_upper = depth[depth_mask]
    if baseline_data.ndim == 2:
        baseline_upper = baseline_data[depth_mask, :] if len(depth_mask) == baseline_data.shape[0] else baseline_data
        period_upper = period_data[depth_mask, :] if len(depth_mask) == period_data.shape[0] else period_data
    else:
        baseline_upper = baseline_data[0, depth_mask, :] if baseline_data.shape[0] == 1 else baseline_data
        period_upper = period_data[0, depth_mask, :] if period_data.shape[0] == 1 else period_data
    
    if baseline_upper.ndim == 1:
        baseline_upper = baseline_upper.reshape(1, -1)
        period_upper = period_upper.reshape(1, -1)
    
    difference = period_upper - baseline_upper
    difference = np.ma.masked_where(np.isnan(baseline_upper), difference)
    
    cf1 = ax1.contourf(latitude, depth_upper[:difference.shape[0]], difference, 
                       levels=diff_levels, cmap='RdBu_r', vmin=vmin, vmax=vmax, extend='both')
    ax1.contour(latitude, depth_upper[:difference.shape[0]], difference, 
                levels=[0], colors='black', linewidths=1.0, linestyles='--', alpha=0.7)
    ax1.invert_yaxis()
    ax1.set_xlabel('Latitude (°N)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('0-1000m Difference')
    ax1.set_xlim(latitude.max(), latitude.min())
    
    # Full depth difference
    if baseline_data.ndim == 2 and period_data.ndim == 2:
        difference_full = period_data - baseline_data
        plot_depth = depth[:period_data.shape[0]]
    else:
        difference_full = period_data[0, :, :] - baseline_data[0, :, :]
        plot_depth = depth[:period_data.shape[1]]
    
    difference_full = np.ma.masked_where(np.isnan(baseline_data), difference_full)
    
    cf2 = ax2.contourf(latitude, plot_depth, difference_full, 
                       levels=diff_levels, cmap='RdBu_r', vmin=vmin, vmax=vmax, extend='both')
    ax2.contour(latitude, plot_depth, difference_full, 
                levels=[0], colors='black', linewidths=1.0, linestyles='--', alpha=0.7)
    ax2.invert_yaxis()
    ax2.set_xlabel('Latitude (°N)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('0-6000m Difference')
    ax2.set_xlim(latitude.max(), latitude.min())
    
    plt.suptitle(f'{dataset_name}: {period} minus {baseline_period}\nRed = Warming, Blue = Cooling', y=1.02)
    plt.tight_layout()
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf2, cax=cbar_ax, label='Temperature Difference (°C)')
    cbar.set_ticks(np.arange(vmin, vmax+0.5, 0.5))
    
    if save_pdf:
        pdf_filename = f"{dataset_name}_{period}_difference_map.pdf"
        save_figure_as_pdf(fig, pdf_filename)
        global pdf_counter
        pdf_counter["difference_maps"] += 1
    
    return fig

def create_boxplot_figure(data_dict, dataset_name, depth, baseline_period, period, save_pdf=True, y_limit=None):
    """Create boxplot with consistent y-axis for comparison"""
    layers = [
        (0, 100, "Surface\n(0-100m)"),
        (100, 200, "Upper thermocline\n(100-200m)"),
        (200, 500, "Thermocline\n(200-500m)"),
        (500, 1000, "Upper intermediate\n(500-1000m)"),
        (1000, 2000, "Intermediate\n(1000-2000m)"),
        (2000, 3000, "Deep\n(2000-3000m)"),
        (3000, 4000, "Abyssal\n(3000-4000m)"),
        (4000, 6000, "Deep abyssal\n(4000-6000m)")
    ]
    baseline_data = data_dict[baseline_period]['temperature']
    period_data = data_dict[period]['temperature']
    
    if period_data.ndim == 2 and baseline_data.ndim == 2:
        diff = period_data - baseline_data
    else:
        diff = period_data[0, :, :] - baseline_data[0, :, :]
    valid_mask = ~np.isnan(baseline_data)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0, :, :]
    
    boxplot_data = []
    layer_names = []
    for layer_min, layer_max, layer_name in layers:
        layer_indices = np.where((depth >= layer_min) & (depth < layer_max))[0]
        if len(layer_indices) == 0:
            continue
        if diff.ndim == 2:
            layer_diff = diff[layer_indices, :]
            layer_valid = valid_mask[layer_indices, :]
        else:
            layer_diff = diff[:, layer_indices, :]
            layer_valid = valid_mask[layer_indices, :]
        values = layer_diff[layer_valid].flatten()
        # Remove outliers beyond 3 sigma for cleaner plot
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            values = values[np.abs(values - mean_val) <= 3 * std_val]
        if len(values) > 0:
            boxplot_data.append(values)
            layer_names.append(layer_name)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(boxplot_data, patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='gray', linewidth=1.5),
                   capprops=dict(color='gray', linewidth=1.5))
    for i, box in enumerate(bp['boxes']):
        median = np.median(boxplot_data[i])
        if median > 0:
            box.set_facecolor('lightcoral')
        else:
            box.set_facecolor('lightblue')
        box.set_alpha(0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Temperature Difference (°C)')
    ax.set_title(f'{dataset_name}: {period} minus {baseline_period}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set consistent y-axis limits if provided
    if y_limit is not None:
        ax.set_ylim(y_limit)
    
    plt.tight_layout()
    
    if save_pdf:
        pdf_filename = f"{dataset_name}_{period}_warming_boxplot.pdf"
        save_figure_as_pdf(fig, pdf_filename)
        global pdf_counter
        pdf_counter["boxplots"] += 1
    
    return fig

# ============================================
# Generate all plots and save PDFs
# ============================================

# Cross-section plots for all datasets
print("\n📊 Generating cross-section plots and PDFs...")
cross_section_html = {}
for name, data_dict in [('CMCC-ESM2', cmcc_data), ('EC-Earth3', ec_data), 
                         ('ORAS5', oras5_data), ('GLORYS', glorys_data)]:
    if data_dict:
        cross_section_html[name] = {}
        for period, data in data_dict.items():
            fig = plot_cross_section(data['temperature'], data['depth'], transect_lats, period, name, period)
            cross_section_html[name][period] = fig_to_base64(fig)
            plt.close(fig)

# Difference maps for ALL datasets with baseline 1993-2002
print("\n📊 Generating difference maps and PDFs...")
difference_html = {}
datasets_to_compare = [
    ('CMCC-ESM2', cmcc_data), 
    ('EC-Earth3', ec_data),
    ('ORAS5', oras5_data), 
    ('GLORYS', glorys_data)
]

for name, data_dict in datasets_to_compare:
    if data_dict and '1993-2002' in data_dict:
        difference_html[name] = {}
        depth = data_dict[list(data_dict.keys())[0]]['depth']
        for period in data_dict.keys():
            if period != '1993-2002':
                fig = plot_difference_map(data_dict, name, depth, transect_lats, '1993-2002', period)
                difference_html[name][period] = fig_to_base64(fig)
                plt.close(fig)

# Boxplots for ALL datasets with consistent y-axis
print("\n📊 Generating boxplots and PDFs...")

# First, find the global y-axis limits across all datasets and periods
all_warming_values = []
for name, data_dict in datasets_to_compare:
    if data_dict and '1993-2002' in data_dict:
        depth = data_dict[list(data_dict.keys())[0]]['depth']
        baseline_data = data_dict['1993-2002']['temperature']
        for period in data_dict.keys():
            if period != '1993-2002':
                period_data = data_dict[period]['temperature']
                if period_data.ndim == 2 and baseline_data.ndim == 2:
                    diff = period_data - baseline_data
                else:
                    diff = period_data[0, :, :] - baseline_data[0, :, :]
                valid_mask = ~np.isnan(baseline_data)
                if valid_mask.ndim == 3:
                    valid_mask = valid_mask[0, :, :]
                values = diff[valid_mask].flatten()
                if len(values) > 0:
                    values = values[np.abs(values) < 5]  # Remove extreme outliers
                    all_warming_values.extend(values)

if all_warming_values:
    global_y_limit = (min(-0.5, np.percentile(all_warming_values, 1)), 
                      max(2.5, np.percentile(all_warming_values, 99)))
    print(f"   Global y-axis limits for boxplots: {global_y_limit}")
else:
    global_y_limit = (-1, 3)

boxplot_html = {}
for name, data_dict in datasets_to_compare:
    if data_dict and '1993-2002' in data_dict:
        boxplot_html[name] = {}
        depth = data_dict[list(data_dict.keys())[0]]['depth']
        for period in data_dict.keys():
            if period != '1993-2002':
                fig = create_boxplot_figure(data_dict, name, depth, '1993-2002', period, y_limit=global_y_limit)
                boxplot_html[name][period] = fig_to_base64(fig)
                plt.close(fig)

# ============================================
# PART 6: COMPUTE WARMING STATISTICS
# ============================================
print("\n📊 Computing warming statistics...")

def compute_layer_statistics(data_dict, dataset_name, depth, baseline_period='1993-2002'):
    if baseline_period not in data_dict:
        print(f"   ⚠️ {dataset_name}: Baseline period '{baseline_period}' not found")
        return None
    layers = [
        (0, 100, "Surface (0-100m)"),
        (100, 200, "Upper thermocline (100-200m)"),
        (200, 500, "Thermocline (200-500m)"),
        (500, 1000, "Upper intermediate (500-1000m)"),
        (1000, 2000, "Intermediate (1000-2000m)"),
        (2000, 3000, "Deep (2000-3000m)"),
        (3000, 4000, "Abyssal (3000-4000m)"),
        (4000, 6000, "Deep abyssal (4000-6000m)")
    ]
    baseline_data = data_dict[baseline_period]['temperature']
    other_periods = [p for p in data_dict.keys() if p != baseline_period]
    other_periods = sorted(other_periods)
    results = []
    for period in other_periods:
        period_data = data_dict[period]['temperature']
        if period_data.shape != baseline_data.shape:
            continue
        if period_data.ndim == 2:
            diff = period_data - baseline_data
        else:
            diff = period_data[0, :, :] - baseline_data[0, :, :]
        valid_mask = ~np.isnan(baseline_data)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[0, :, :]
        for layer_min, layer_max, layer_name in layers:
            layer_indices = np.where((depth >= layer_min) & (depth < layer_max))[0]
            if len(layer_indices) == 0:
                continue
            if diff.ndim == 2:
                layer_diff = diff[layer_indices, :]
                layer_valid = valid_mask[layer_indices, :]
            else:
                layer_diff = diff[:, layer_indices, :]
                layer_valid = valid_mask[layer_indices, :]
            valid_values = layer_diff[layer_valid]
            if len(valid_values) > 0:
                results.append({
                    'Dataset': dataset_name,
                    'Period': period,
                    'Baseline': baseline_period,
                    'Depth Layer': layer_name,
                    'Mean Warming (°C)': np.mean(valid_values),
                    'Std Dev (°C)': np.std(valid_values),
                    '25th Percentile': np.percentile(valid_values, 25),
                    '75th Percentile': np.percentile(valid_values, 75)
                })
    return pd.DataFrame(results)

stats_data = {}
for name, data_dict in datasets_to_compare:
    if data_dict and '1993-2002' in data_dict:
        first_period = list(data_dict.keys())[0]
        depth = data_dict[first_period]['depth']
        stats_data[name] = compute_layer_statistics(data_dict, name, depth, '1993-2002')

# Save statistics to CSV
if stats_data:
    all_stats = []
    for df in stats_data.values():
        if df is not None:
            all_stats.append(df)
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        stats_file = stats_dir / "warming_statistics_by_depth.csv"
        combined_stats.to_csv(stats_file, index=False)
        print(f"✅ Statistics saved to: {stats_file}")

# ============================================
# PART 7: GENERATE HTML REPORT
# ============================================
print("\n📄 Generating HTML report...")

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Atlantic Ocean Warming Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #8B0000; }}
        h2 {{ color: #333; border-bottom: 1px solid #ccc; }}
        .section {{ margin-bottom: 30px; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .plot {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
        img {{ width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Atlantic Ocean Warming Analysis</h1>
    <p>Transect: {transect_name}</p>
    <p>Number of sampling points: {len(transect_lats)}</p>
    <p>Latitude range: {transect_lats.min():.1f}°S to {transect_lats.max():.1f}°N</p>
    
    <div class="warning">
        <strong>Note:</strong> EC-Earth3 shows physically implausible patterns (warming increasing with depth) 
        and should be excluded from the analysis. Focus on GLORYS (most reliable), ORAS5, and CMCC-ESM2.
    </div>
    
    <div class="section">
        <h2>Cross-Sections (Absolute Temperatures)</h2>
        <div class="plot-grid">
"""

for name, periods in cross_section_html.items():
    for period, img_base64 in periods.items():
        html_content += f"""
            <div class="plot">
                <h3>{name} - {period}</h3>
                <img src="data:image/png;base64,{img_base64}" alt="{name} {period}">
            </div>
        """

html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Difference Maps (Warming/Cooling)</h2>
        <p>Baseline: 1993-2002 | Red = Warming, Blue = Cooling</p>
        <div class="plot-grid">
"""

for name, periods in difference_html.items():
    for period, img_base64 in periods.items():
        html_content += f"""
            <div class="plot">
                <h3>{name} - {period} minus 1993-2002</h3>
                <img src="data:image/png;base64,{img_base64}" alt="{name} {period} diff">
            </div>
        """

html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Boxplots (Warming by Depth Layer)</h2>
        <p>Baseline: 1993-2002 | Consistent y-axis across all models for comparison</p>
        <div class="plot-grid">
"""

for name, periods in boxplot_html.items():
    for period, img_base64 in periods.items():
        html_content += f"""
            <div class="plot">
                <h3>{name} - {period} minus 1993-2002</h3>
                <img src="data:image/png;base64,{img_base64}" alt="{name} {period} boxplot">
            </div>
        """

if all_stats:
    html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Warming Statistics by Depth Layer</h2>
        <p>Only showing datasets with physically plausible patterns (GLORYS, ORAS5, CMCC-ESM2)</p>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Period</th>
                <th>Depth Layer</th>
                <th>Mean Warming (°C)</th>
                <th>Std Dev (°C)</th>
                <th>25th Percentile</th>
                <th>75th Percentile</th>
            </tr>
    """
    for df in all_stats:
        for _, row in df.iterrows():
            # Skip EC-Earth3 in the table
            if 'EC-Earth3' in row['Dataset']:
                continue
            html_content += f"""
            <tr>
                <td>{row['Dataset']}</td>
                <td>{row['Period']}</td>
                <td>{row['Depth Layer']}</td>
                <td>{row['Mean Warming (°C)']:.3f}</td>
                <td>{row['Std Dev (°C)']:.3f}</td>
                <td>{row['25th Percentile']:.3f}</td>
                <td>{row['75th Percentile']:.3f}</td>
            </tr>
            """
    html_content += """
        </table>
    """

html_content += """
    </div>
</body>
</html>
"""

html_file = plots_dir / "Atlantic_Warming_Analysis_Report.html"
with open(html_file, 'w') as f:
    f.write(html_content)
print(f"✅ HTML report saved to: {html_file}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("✅ COMPLETE!")
print("="*80)
print(f"\n📁 Output files saved to: {plots_dir}")
print(f"\n📊 PDF Export Summary:")
print(f"   • Cross-section plots: {pdf_counter['cross_sections']} PDF files")
print(f"   • Difference maps: {pdf_counter['difference_maps']} PDF files")
print(f"   • Boxplots: {pdf_counter['boxplots']} PDF files")
print(f"   • Total PDFs: {sum(pdf_counter.values())} files")
print(f"\n📄 HTML Report: {html_file}")
print(f"📊 Statistics CSV: {stats_dir}/warming_statistics_by_depth.csv")
print("\n⚠️  Note: EC-Earth3 shows physically implausible patterns and should be excluded from analysis")
print("\n" + "="*80)
