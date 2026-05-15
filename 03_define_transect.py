#!/usr/bin/env python3
"""
04_define_transect.py - Extract Polarstern median transect from Germany to Namibia
Plots points on map, median transect, and 100 random sampling points as 'x'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
data_dir = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/")
plots_dir = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/Cross-sections/Plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Define cherry red color (darker, richer red)
CHERRY_RED = '#8B0000'  # Dark cherry red
CHERRY_RED_LIGHT = '#CD5C5C'  # Lighter cherry for contrast

print("="*70)
print("📍 EXTRACTING POLARSTERN MEDIAN TRANSECT (Germany to Namibia via Western Africa)")
print("="*70)

# ============================================
# PART 1: LOAD THERMO DATA WITH YEARS
# ============================================
print("\n📊 Loading thermosalinograph data...")

def load_thermo_data_with_years():
    """Load thermosalinograph data with year extraction"""
    thermo_files = list(data_dir.glob("**/THERMO_SALINOGRAPH/*.txt.dat"))
    
    data_by_year = {}
    
    for thermo_file in thermo_files:
        print(f"\n📄 Reading: {thermo_file.name}")
        
        try:
            df = pd.read_csv(thermo_file, sep=';', encoding='latin1', 
                           skiprows=3, header=None,
                           on_bad_lines='skip', low_memory=False)
            
            timestamps = pd.to_datetime(df[0], errors='coerce')
            lats = pd.to_numeric(df[1], errors='coerce').values
            lons = pd.to_numeric(df[2], errors='coerce').values
            temps = pd.to_numeric(df[4], errors='coerce').values
            years = timestamps.dt.year.values
            
            # Filter for Western Africa route
            valid = (~np.isnan(lats)) & (~np.isnan(lons)) & (~np.isnan(temps)) & (~np.isnan(years))
            valid &= (lats >= -35) & (lats <= 55)
            valid &= (temps > -5) & (temps < 40)
            
            # CHANGE 1: Fixed northern hemisphere longitude filter to include Bremerhaven (8.5°E)
            # Old code: north_mask = (lats >= 0) & (lons >= -35) & (lons <= -5)
            north_mask_europe = (lats >= 45) & (lons >= -10) & (lons <= 15)  # European waters including Bremerhaven
            north_mask_africa = (lats >= 0) & (lats < 45) & (lons >= -35) & (lons <= -5)  # West African coast
            north_mask = north_mask_europe | north_mask_africa
            south_mask = (lats < 0) & (lons >= -20) & (lons <= 20)
            valid &= (north_mask | south_mask)
            
            if not valid.any():
                continue
            
            # Group by year
            for year in np.unique(years[valid]):
                year = int(year)
                if year not in data_by_year:
                    data_by_year[year] = {'lats': [], 'lons': [], 'temps': []}
                
                year_mask = (years == year) & valid
                data_by_year[year]['lats'].extend(lats[year_mask])
                data_by_year[year]['lons'].extend(lons[year_mask])
                data_by_year[year]['temps'].extend(temps[year_mask])
            
            print(f"   ✅ Added {valid.sum()} points")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Convert to arrays
    for year in data_by_year:
        data_by_year[year]['lats'] = np.array(data_by_year[year]['lats'])
        data_by_year[year]['lons'] = np.array(data_by_year[year]['lons'])
        data_by_year[year]['temps'] = np.array(data_by_year[year]['temps'])
    
    return data_by_year

data_by_year = load_thermo_data_with_years()

print(f"\n📊 Initial data summary:")
print(f"   Years with data: {sorted(data_by_year.keys())}")

# ============================================
# PART 2: IDENTIFY FULL NORTH-SOUTH TRANSECTS
# ============================================
print("\n🔍 Identifying full Germany-Namibia transects...")

full_transects = {}

for year, data in data_by_year.items():
    lats = data['lats']
    temps = data['temps']
    
    if len(lats) < 500:
        continue
    
    lat_min = lats.min()
    lat_max = lats.max()
    
    # CHANGE 2: Increased northern latitude threshold to reach Bremerhaven (52°N instead of 45°N)
    if lat_max >= 52 and lat_min <= -23:
        # Sort by latitude for the transect plot
        sort_idx = np.argsort(lats)
        lats_sorted = lats[sort_idx]
        temps_sorted = temps[sort_idx]
        
        # Apply smoothing
        temps_smoothed = gaussian_filter1d(temps_sorted, sigma=5)
        
        full_transects[year] = {
            'lats': lats_sorted,
            'lons': data['lons'][sort_idx],
            'temps_raw': temps_sorted,
            'temps_smoothed': temps_smoothed,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'n_points': len(lats)
        }
        print(f"   ✅ Year {year}: {len(lats)} points, {lat_min:.1f}° to {lat_max:.1f}°")

print(f"\n📊 Found {len(full_transects)} full transects")

if len(full_transects) == 0:
    print("\n⚠️ No full transects found. Using all years with broad coverage...")
    for year, data in data_by_year.items():
        lats = data['lats']
        if len(lats) >= 500 and (lats.max() - lats.min()) >= 60:
            sort_idx = np.argsort(lats)
            lats_sorted = lats[sort_idx]
            temps_sorted = data['temps'][sort_idx]
            temps_smoothed = gaussian_filter1d(temps_sorted, sigma=5)
            
            full_transects[year] = {
                'lats': lats_sorted,
                'lons': data['lons'][sort_idx],
                'temps_raw': temps_sorted,
                'temps_smoothed': temps_smoothed,
                'lat_min': lats.min(),
                'lat_max': lats.max(),
                'n_points': len(lats)
            }
            print(f"   ✅ Year {year}: {len(lats)} points, {lats.min():.1f}° to {lats.max():.1f}°")

print(f"\n📊 Final: Found {len(full_transects)} transects")

# ============================================
# PART 3: COMPUTE MEDIAN TRANSECT
# ============================================
print("\n📊 Computing median transect...")

# CHANGE 3: Extended latitude range to include Bremerhaven (53.5°N instead of 52°N)
common_lats = np.linspace(-30, 53.5, 500)  # High resolution for sampling
all_interpolated = []
all_longitudes_interp = []

for year, transect in full_transects.items():
    if len(transect['lats']) > 10:
        f_temp = interp1d(transect['lats'], transect['temps_smoothed'], 
                         bounds_error=False, fill_value=np.nan)
        interp_temps = f_temp(common_lats)
        all_interpolated.append(interp_temps)
        
        f_lon = interp1d(transect['lats'], transect['lons'], 
                        bounds_error=False, fill_value=np.nan)
        interp_lons = f_lon(common_lats)
        all_longitudes_interp.append(interp_lons)

if len(all_interpolated) > 0:
    median_temps = np.nanmedian(all_interpolated, axis=0)
    median_lons = np.nanmedian(all_longitudes_interp, axis=0)
    print(f"   Median transect computed from {len(all_interpolated)} years")
else:
    median_lons = None
    print("   ⚠️ Could not compute median transect")

# ============================================
# PART 4: SELECT 100 RANDOM POINTS FROM MEDIAN TRANSECT
# ============================================
print("\n🎲 Selecting 100 random points from median transect...")

if median_lons is not None:
    # Create a mask for valid points (not NaN)
    valid_mask = ~np.isnan(median_lons)
    valid_lats = common_lats[valid_mask]
    valid_lons = median_lons[valid_mask]
    
    # Randomly select 100 points
    np.random.seed(42)  # For reproducibility
    n_points = min(100, len(valid_lats))
    random_indices = np.random.choice(len(valid_lats), n_points, replace=False)
    
    random_points = pd.DataFrame({
        'latitude': valid_lats[random_indices],
        'longitude': valid_lons[random_indices],
        'point_id': range(1, n_points + 1)
    })
    
    # Sort by latitude for easier use
    random_points = random_points.sort_values('latitude').reset_index(drop=True)
    
    print(f"   Selected {n_points} random points along the median transect")
    print(f"   Latitude range: {random_points['latitude'].min():.2f}° to {random_points['latitude'].max():.2f}°")
    print(f"   Longitude range: {random_points['longitude'].min():.2f}° to {random_points['longitude'].max():.2f}°")
    
    # Save to CSV
    csv_points_file = plots_dir / "Polarstern_median_transect_100points.csv"
    random_points.to_csv(csv_points_file, index=False)
    print(f"✅ Saved 100 random points to: {csv_points_file}")

# ============================================
# PART 5: CREATE MAP WITH POINTS + MEDIAN + 100 RANDOM X's
# ============================================
print("\n🎨 Creating map plot...")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    fig_map = plt.figure(figsize=(14, 10))
    ax = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    
    # Plot each year as POINTS (very faint, in background)
    colors = plt.cm.viridis(np.linspace(0, 1, len(full_transects)))
    for idx, (year, transect) in enumerate(full_transects.items()):
        step = max(1, len(transect['lons']) // 1000)
        ax.scatter(transect['lons'][::step], transect['lats'][::step], 
                  s=1, alpha=0.15, color=colors[idx], 
                  label=f'{year}', transform=ccrs.PlateCarree())
    
    # Plot median transect in cherry red with 85% transparency (alpha=0.15 = 85% transparent)
    if median_lons is not None:
        ax.plot(median_lons, common_lats, 
               color=CHERRY_RED, linewidth=2.5, alpha=0.15,
               label='Median Transect', transform=ccrs.PlateCarree())
    
    # Plot 100 random points as 'x' markers in cherry red, 100% opaque, smaller size
    if median_lons is not None:
        ax.scatter(random_points['longitude'], random_points['latitude'], 
                  marker='x', color=CHERRY_RED, s=30, linewidths=1.0, alpha=1.0,
                  label=f'{n_points} Sampling Points', 
                  transform=ccrs.PlateCarree(), zorder=10)
    
    ax.set_xlim(-35, 25)
    ax.set_ylim(-35, 60)
    ax.legend(loc='lower left', fontsize=9, ncol=2, markerscale=1.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(f'Polarstern Germany-Namibia Median Transect\n'
                f'{len(full_transects)} years (colored points) + Median (cherry red, 75% transparent) + {n_points} sampling points (cherry red ×)', 
                fontsize=12, color=CHERRY_RED)
    
    plt.tight_layout()
    
    # Save map
    map_file = plots_dir / "Polarstern_median_transect_with_samples.pdf"
    plt.savefig(map_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved map: {map_file}")
    plt.close()
    
except ImportError:
    print("   Cartopy not available, creating simple map...")
    fig_map, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(full_transects)))
    for idx, (year, transect) in enumerate(full_transects.items()):
        step = max(1, len(transect['lons']) // 1000)
        ax.scatter(transect['lons'][::step], transect['lats'][::step], 
                  s=1, alpha=0.15, color=colors[idx], label=f'{year}')
    
    if median_lons is not None:
        ax.plot(median_lons, common_lats, 
               color=CHERRY_RED, linewidth=2.5, alpha=0.15, label='Median Transect')
        ax.scatter(random_points['longitude'], random_points['latitude'], 
                  marker='x', color=CHERRY_RED, s=30, linewidths=1.0, alpha=1.0,
                  label=f'{n_points} Sampling Points')
    
    ax.set_xlim(-35, 25)
    ax.set_ylim(-35, 60)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='lower left', fontsize=9, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Polarstern Median Transect\n{len(full_transects)} years + Median (cherry red, 75% transparent) + {n_points} sampling points (cherry red ×)',
                color=CHERRY_RED)
    
    plt.tight_layout()
    map_file = plots_dir / "Polarstern_median_transect_with_samples.pdf"
    plt.savefig(map_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved map: {map_file}")
    plt.close()

# ============================================
# PART 6: SAVE FULL MEDIAN TRANSECT DATA
# ============================================
print("\n📊 Saving full median transect data...")

if median_lons is not None:
    full_transect_df = pd.DataFrame({
        'latitude': common_lats,
        'longitude': median_lons,
        'temperature': median_temps if 'median_temps' in locals() else np.nan
    })
    
    csv_full_file = plots_dir / "Polarstern_median_transect_full.csv"
    full_transect_df.to_csv(csv_full_file, index=False)
    print(f"✅ Saved full median transect: {csv_full_file}")

# ============================================
# PRINT SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ DONE!")
print("="*70)
print(f"\n📁 Files saved to: {plots_dir}")
print(f"   • Polarstern_median_transect_with_samples.pdf - Map with median line + sampling points")
print(f"   • Polarstern_median_transect_100points.csv - 100 random sampling points (SHARE THIS!)")
print(f"   • Polarstern_median_transect_full.csv - Complete median transect (all points)")
print(f"\n📊 Summary:")
print(f"   • Total years used: {len(full_transects)}")
print(f"   • Years: {list(full_transects.keys())}")
print(f"   • Random sampling points: {n_points}")
print(f"\n🎨 Styling:")
print(f"   • Median line: Cherry red (#8B0000) with 75% transparency")
print(f"   • Sampling points: Cherry red × symbols, 100% opaque, size 30")
print(f"\n💡 The file 'Polarstern_median_transect_100points.csv' contains the coordinates")
print(f"   you can share with your colleagues for consistent analysis across models!")
print("="*70)

# Print first 10 points as preview
if median_lons is not None:
    print("\n📋 Preview of random sampling points (first 10):")
    print(random_points.head(10).to_string(index=False))
