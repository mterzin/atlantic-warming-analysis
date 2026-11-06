#!/usr/bin/env python3
"""
05_composite_figure.py - COMPOSITE FIGURE: MAP, ANOMALY SECTIONS, BOXPLOTS, BAND PROFILES

Reads the transect extractions written by 04_cross-sections_defined_time-periods.py
(Statistics_OND_1993-2002_baseline/transect_extraction_<product>.nc), so it runs in
seconds and can be re-run freely while trying out layouts.

Layout (panel letters are assigned automatically in this order):
  +-----------+----------------------------------------+-----------+
  | (a) map   | sections 2003-2014: GLORYS ORAS5 CMCC  | boxplots  |
  |  whole    | sections 2015-2025: GLORYS ORAS5 CMCC  | boxplots  |
  | transect  | latitude-band strip                    |           |
  +-----------+----------------------------------------+-----------+
  | legend    | band profiles: one panel per latitude band          |
  +-----------+-----------------------------------------------------+
Sections, boxplots and profiles share one square-root depth axis, and dashed
lines mark the boxplot depth layers, so a box sits level with the depths it summarises.

Everything you may want to change is in the CONFIGURATION block.
Run 04 first whenever the input data change.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
analysis_dir = Path.home() / "Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis"
stats_dir = analysis_dir / "Cross-sections/Statistics_OND_1993-2002_baseline"
out_dir = analysis_dir / "Cross-sections/Composite_figures"
out_dir.mkdir(parents=True, exist_ok=True)
FULL_TRANSECT_CSV = analysis_dir / "Cross-sections/Plots/Polarstern_median_transect_full.csv"

PRODUCTS = ['GLORYS', 'ORAS5', 'CMCC-ESM2']
# Okabe-Ito colours (colour-blind safe), used for products in boxplots and profiles
PRODUCT_COLORS = {'GLORYS': '#0072B2', 'ORAS5': '#E69F00', 'CMCC-ESM2': '#009E73'}

BASELINE = '1993-2002'
PERIODS = ['2003-2014', '2015-2025']   # one row of sections + boxplots per period
PROFILE_MAIN_PERIOD = '2015-2025'     # solid line + IQR shading in the band profiles
PROFILE_SECOND_PERIOD = '2003-2014'   # dotted line (None to hide)

# Depth layers for the boxplots (m); their boundaries are drawn as dashed lines
DEPTH_LAYERS = [(0, 100), (100, 200), (200, 500), (500, 1000),
                (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000)]

# Latitude bands: (key, label, south limit, north limit), ordered NORTH to SOUTH
LAT_BANDS = [
    ('NE', 'NE Atlantic 25–54°N',      25.0, 54.0),
    ('TN', 'Tropical North 10–25°N',   10.0, 25.0),
    ('EQ', 'Equatorial 10°S–10°N',    -10.0, 10.0),
    ('TS', 'Tropical South 20–10°S',  -20.0, -10.0),
    ('SO', 'South 31–20°S',           -31.0, -20.0),
]
BAND_COLORS = ['#cc79a7', '#d55e00', '#999999', '#56b4e9', '#332288']

# Depth axes
SECTION_DEPTH_MAX = 6000   # sections and boxplots
PROFILE_DEPTH_MAX = 2000   # band profiles (deeper changes are small)
DEPTH_TICKS = [0, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 6000]

# Sections: shared diverging colour scale (degC)
SECTION_VMAX = 2.0
SECTION_LEVELS = 21
SHOW_ZERO_CONTOUR = True

# Interpolate every profile to an evenly spaced depth grid before computing
# boxplots/profiles, so products with more native levels do not get more weight
COMMON_DEPTH_GRID = True
GRID_DZ = 10.0             # m
MIN_POINTS_PROFILE = 3     # minimum points with data for a band-profile value

# Boxplots: whiskers at 1.5 x IQR (clipped to the data range), no outliers
WHIS = 1.5
BOX_XLIM = (-2.0, 2.0)     # degC; None = automatic, shared by both rows
PROFILE_XLIM = 'box'       # 'box' = same as BOX_XLIM; or e.g. (-1, 1.5); None = automatic

USE_CARTOPY = True         # falls back to a plain map if cartopy is unavailable
MAP_EXTENT = [-28, 18, -33, 56]   # lon min, lon max, lat min, lat max
MAP_SHOW_BANDS = True      # shade the latitude bands on the map and colour the points

MAKE_BAND_HEATMAP = False  # separate heatmap figure, for comparison
FIG_SIZE = (19, 14)
FIG_FORMATS = ['pdf', 'png']
FIG_NAME = "Composite_transect_sections_boxplots_profiles"

# ============================================
# PART 1: LOAD EXTRACTED TRANSECT VALUES
# ============================================
print("=" * 70)
print("COMPOSITE FIGURE")
print("=" * 70)

data = {}
for prod in PRODUCTS:
    f = stats_dir / f"transect_extraction_{prod}.nc"
    if not f.exists():
        raise SystemExit(f"❌ {f} not found. Run 04_cross-sections_defined_time-periods.py first.")
    data[prod] = xr.load_dataset(f)
    print(f"   Loaded {f.name}: {data[prod].sizes['depth']} levels, {data[prod].sizes['point']} points")

lats = data[PRODUCTS[0]].latitude.values
lons = data[PRODUCTS[0]].longitude.values
for prod in PRODUCTS[1:]:
    if not np.allclose(data[prod].latitude.values, lats):
        raise SystemExit("❌ Products were extracted at different transect points.")

band_of_point = np.full(len(lats), None, dtype=object)
for key, _, south, north in LAT_BANDS:
    band_of_point[(lats >= south) & (lats < north)] = key
for key, label, _, _ in LAT_BANDS:
    print(f"   Band {key:3s} ({label}): {np.sum(band_of_point == key)} points")
outside = int(np.sum(band_of_point == None))  # noqa: E711
if outside:
    print(f"   ⚠️ {outside} point(s) outside LAT_BANDS (used only in whole-transect panels)")

# ============================================
# PART 2: DIFFERENCES (optionally on a common depth grid)
# ============================================
common_grid = np.arange(GRID_DZ / 2, 6000, GRID_DZ)


def to_common_grid(values, depth, grid):
    """Linear interpolation of each column (point) onto grid, without extrapolation."""
    out = np.full((len(grid), values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        ok = ~np.isnan(values[:, j])
        if ok.sum() < 2:
            continue
        d, v = depth[ok], values[ok, j]
        inside = (grid >= d.min()) & (grid <= d.max())
        out[inside, j] = np.interp(grid[inside], d, v)
    return out


diffs = {}   # diffs[prod][period] = (depth, diff[depth, point])
for prod in PRODUCTS:
    ds = data[prod]
    native = ds.depth.values
    base = ds.thetao.sel(period=BASELINE).values
    diffs[prod] = {}
    for period in PERIODS:
        diff = ds.thetao.sel(period=period).values - base
        diffs[prod][period] = (common_grid, to_common_grid(diff, native, common_grid)) \
            if COMMON_DEPTH_GRID else (native, diff)

# ============================================
# PART 3: STATISTICS
# ============================================
ALL_POINTS = np.ones(len(lats), bool)


def layer_values(prod, period, layer, point_mask=ALL_POINTS):
    depth, diff = diffs[prod][period]
    rows = (depth >= layer[0]) & (depth < layer[1])
    v = diff[np.ix_(rows, point_mask)].ravel()
    return v[~np.isnan(v)]


def box_stats(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - WHIS * iqr].min()   # whiskers end at the most extreme
    hi = v[v <= q3 + WHIS * iqr].max()   # data point within 1.5 x IQR
    return lo, q1, med, q3, hi


def band_profile(prod, period, point_mask):
    depth, diff = diffs[prod][period]
    sub = diff[:, point_mask]
    n = np.sum(~np.isnan(sub), axis=1)
    with np.errstate(all='ignore'):
        med = np.nanmedian(sub, axis=1)
        q25 = np.nanpercentile(sub, 25, axis=1)
        q75 = np.nanpercentile(sub, 75, axis=1)
    bad = n < MIN_POINTS_PROFILE
    med[bad] = q25[bad] = q75[bad] = np.nan
    return depth, med, q25, q75


# Export the numbers behind the figure (for writing the Results)
rows = []
for prod in PRODUCTS:
    for period in PERIODS:
        for layer in DEPTH_LAYERS:
            v = layer_values(prod, period, layer)
            if len(v):
                lo, q1, med, q3, hi = box_stats(v)
                rows.append({'Dataset': prod, 'Period': period, 'Baseline': BASELINE,
                             'Depth layer (m)': f"{layer[0]}-{layer[1]}", 'Median': med,
                             'Q25': q1, 'Q75': q3, 'Whisker low': lo, 'Whisker high': hi,
                             'Fraction positive': np.mean(v > 0), 'N values': len(v)})
pd.DataFrame(rows).to_csv(out_dir / f"{FIG_NAME}_boxplot_values.csv", index=False, float_format='%.4f')

prof_rows = []
for key, *_ in LAT_BANDS:
    for prod in PRODUCTS:
        for period in PERIODS:
            depth, med, q25, q75 = band_profile(prod, period, band_of_point == key)
            for d, m, a, b in zip(depth, med, q25, q75):
                if not np.isnan(m) and d <= PROFILE_DEPTH_MAX:
                    prof_rows.append({'Band': key, 'Dataset': prod, 'Period': period,
                                      'Depth (m)': d, 'Median': m, 'Q25': a, 'Q75': b})
pd.DataFrame(prof_rows).to_csv(out_dir / f"{FIG_NAME}_profile_values.csv", index=False, float_format='%.4f')

# ============================================
# PART 4: PLOTTING HELPERS
# ============================================
sqrt_fwd = lambda z: np.sqrt(np.clip(z, 0, None))   # noqa: E731
sqrt_inv = lambda t: np.asarray(t) ** 2              # noqa: E731
letters = iter('abcdefghijklmnopqrstuvwxyz')


def panel_label(ax, inside=True):
    """Add the next panel letter, e.g. '(b)', in the top-left corner."""
    letter = f"({next(letters)})"
    if inside:
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha='left', va='top', fontsize=10,
                fontweight='bold', zorder=20,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    else:
        ax.set_title(letter, loc='left', fontsize=10, fontweight='bold')
    return letter


def depth_axis(ax, zmax, label=True, ticks_right=False):
    ax.set_yscale('function', functions=(sqrt_fwd, sqrt_inv))
    ax.set_ylim(zmax, 0)
    ax.set_yticks([t for t in DEPTH_TICKS if t <= zmax])
    ax.tick_params(axis='y', labelsize=7)
    if ticks_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    if label:
        ax.set_ylabel('Depth (m)', fontsize=8)


def layer_lines(ax, zmax):
    for a, _ in DEPTH_LAYERS[1:]:
        if a < zmax:
            ax.axhline(a, color='k', lw=0.6, ls='--', alpha=0.55, zorder=5)


def draw_map(ax, use_cartopy):
    kw = {}
    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 7}
        kw = {'transform': ccrs.PlateCarree()}
    else:
        ax.set_xlim(MAP_EXTENT[:2])
        ax.set_ylim(MAP_EXTENT[2:])
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude (°E)', fontsize=8)
        ax.set_ylabel('Latitude (°N)', fontsize=8)
        ax.tick_params(labelsize=7)
    if FULL_TRANSECT_CSV.exists():
        full = pd.read_csv(FULL_TRANSECT_CSV).dropna(subset=['longitude'])
        ax.plot(full['longitude'], full['latitude'], color='#8B0000', lw=1.2, alpha=0.5, zorder=2, **kw)
    if MAP_SHOW_BANDS:
        for (key, _, south, north), col in zip(LAT_BANDS, BAND_COLORS):
            lo, hi = max(south, MAP_EXTENT[2]), min(north, MAP_EXTENT[3])
            ax.fill_between(MAP_EXTENT[:2], lo, hi, color=col, alpha=0.15, lw=0, zorder=0.5, **kw)
            ax.text(MAP_EXTENT[0] + 1, (lo + hi) / 2, key, ha='left', va='center', fontsize=9,
                    color=col, fontweight='bold', zorder=4, **kw)
            m = band_of_point == key
            ax.scatter(lons[m], lats[m], s=16, color=col, edgecolor='k', lw=0.3, zorder=3, **kw)
    else:
        ax.scatter(lons, lats, marker='x', s=16, color='#8B0000', lw=0.9, zorder=3, **kw)
    ax.set_anchor('N')
    ax.set_title('Polarstern median transect\n'
                 f'{len(lats)} sampling points', fontsize=9)
    panel_label(ax)


def draw_section(ax, prod, period, norm, levels):
    depth, diff = diffs[prod][period]
    order = np.argsort(lats)[::-1]                  # north on the left
    x, z = lats[order], depth
    field = np.ma.masked_invalid(diff[:, order])
    cf = ax.contourf(x, z, field, levels=levels, cmap='RdBu_r', norm=norm, extend='both')
    if SHOW_ZERO_CONTOUR:
        ax.contour(x, z, field, levels=[0], colors='k', linewidths=0.4, alpha=0.6)
    depth_axis(ax, SECTION_DEPTH_MAX, label=False)
    layer_lines(ax, SECTION_DEPTH_MAX)
    ax.set_xlim(lats.max(), lats.min())
    ax.set_facecolor('#d9d9d9')                     # grey = no data (land/seafloor)
    panel_label(ax)
    return cf


def draw_boxplot(ax, period, xlim, legend):
    """Horizontal boxplots positioned at the depths of each layer (sqrt axis)."""
    n = len(PRODUCTS)
    for layer in DEPTH_LAYERS:
        if layer[0] >= SECTION_DEPTH_MAX:
            continue
        t0, t1 = sqrt_fwd(layer[0]), sqrt_fwd(min(layer[1], SECTION_DEPTH_MAX))
        slot = (t1 - t0) / n
        for k, prod in enumerate(PRODUCTS):
            v = layer_values(prod, period, layer)
            if len(v) == 0:
                continue
            lo, q1, med, q3, hi = box_stats(v)
            a, b = t0 + slot * (k + 0.15), t0 + slot * (k + 0.85)   # box extent (sqrt units)
            za, zb, zm = sqrt_inv(a), sqrt_inv(b), sqrt_inv((a + b) / 2)
            col = PRODUCT_COLORS[prod]
            ax.add_patch(Rectangle((q1, za), q3 - q1, zb - za, facecolor=col, alpha=0.75,
                                   edgecolor='k', lw=0.5, zorder=3))
            ax.plot([med, med], [za, zb], color='k', lw=1.4, zorder=4)
            ax.plot([lo, q1], [zm, zm], color=col, lw=0.9, zorder=2)
            ax.plot([q3, hi], [zm, zm], color=col, lw=0.9, zorder=2)
            ax.plot([lo, lo], [za, zb], color=col, lw=0.9, zorder=2)
            ax.plot([hi, hi], [za, zb], color=col, lw=0.9, zorder=2)
    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
    depth_axis(ax, SECTION_DEPTH_MAX, label=True, ticks_right=True)
    layer_lines(ax, SECTION_DEPTH_MAX)
    ax.set_xlim(xlim)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_title(f'Whole transect, {period}', fontsize=9)
    if legend:
        ax.legend(handles=[Patch(facecolor=PRODUCT_COLORS[p], alpha=0.75, edgecolor='k', label=p)
                           for p in PRODUCTS], fontsize=7, loc='lower right', frameon=True,
                  framealpha=0.9)
    panel_label(ax)


def draw_band_strip(ax):
    ax.set_xlim(lats.max(), lats.min())
    ax.set_ylim(0, 1)
    for (key, _, south, north), col in zip(LAT_BANDS, BAND_COLORS):
        lo, hi = max(south, lats.min()), min(north, lats.max())
        ax.axvspan(hi, lo, color=col, alpha=0.85, lw=0)
        ax.text((lo + hi) / 2, 0.5, key, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')
    ax.scatter(lats, np.full(len(lats), 0.08), marker='|', s=12, color='k', lw=0.6)  # sampling points
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('Latitude (°N)', fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_profile(ax, key, label, color, xlim, first):
    mask = band_of_point == key
    for prod in PRODUCTS:
        col = PRODUCT_COLORS[prod]
        depth, med, q25, q75 = band_profile(prod, PROFILE_MAIN_PERIOD, mask)
        keep = depth <= PROFILE_DEPTH_MAX
        ax.fill_betweenx(depth[keep], q25[keep], q75[keep], color=col, alpha=0.15, lw=0)
        ax.plot(med[keep], depth[keep], color=col, lw=1.6)
        if PROFILE_SECOND_PERIOD:
            d2, m2, _, _ = band_profile(prod, PROFILE_SECOND_PERIOD, mask)
            k2 = d2 <= PROFILE_DEPTH_MAX
            ax.plot(m2[k2], d2[k2], color=col, lw=1.1, ls=':')
    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
    depth_axis(ax, PROFILE_DEPTH_MAX, label=first)
    layer_lines(ax, PROFILE_DEPTH_MAX)
    if not first:
        ax.tick_params(labelleft=False)
    ax.set_xlim(xlim)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('ΔT (°C)', fontsize=8)
    ax.set_title(f"{key}: {label}\n{int(mask.sum())} points", fontsize=8, color=color)
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(1.6)
    panel_label(ax)


def automatic_limits():
    lo_all, hi_all = [], []
    for prod in PRODUCTS:
        for period in PERIODS:
            for layer in DEPTH_LAYERS:
                v = layer_values(prod, period, layer)
                if len(v):
                    lo, _, _, _, hi = box_stats(v)
                    lo_all.append(lo)
                    hi_all.append(hi)
    pad = 0.05 * (max(hi_all) - min(lo_all))
    box = (min(lo_all) - pad, max(hi_all) + pad)
    xs = []
    for key, *_ in LAT_BANDS:
        for prod in PRODUCTS:
            for period in [PROFILE_MAIN_PERIOD, PROFILE_SECOND_PERIOD]:
                if period:
                    depth, _, q25, q75 = band_profile(prod, period, band_of_point == key)
                    keep = depth <= PROFILE_DEPTH_MAX
                    xs.extend(np.abs(q25[keep][~np.isnan(q25[keep])]))
                    xs.extend(np.abs(q75[keep][~np.isnan(q75[keep])]))
    lim = np.percentile(xs, 99) * 1.1 if xs else 1.0
    return box, (-lim, lim)

# ============================================
# PART 5: COMPOSITE FIGURE
# ============================================
print("\n🎨 Drawing composite figure...")
auto_box, auto_prof = automatic_limits()
box_xlim = BOX_XLIM or auto_box
if PROFILE_XLIM == 'box':
    prof_xlim = box_xlim
else:
    prof_xlim = PROFILE_XLIM or auto_prof
levels = np.linspace(-SECTION_VMAX, SECTION_VMAX, SECTION_LEVELS)
norm = BoundaryNorm(levels, ncolors=plt.get_cmap('RdBu_r').N, extend='both')

fig = plt.figure(figsize=FIG_SIZE)
outer = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 6.2], height_ratios=[2.5, 1.0],
                 wspace=0.13, hspace=0.22)
nprod = len(PRODUCTS)
# top right: section columns | colour bar | spacer | boxplots
top = GridSpecFromSubplotSpec(len(PERIODS) + 1, nprod + 3, subplot_spec=outer[0, 1],
                              width_ratios=[1] * nprod + [0.06, 0.32, 1.25],
                              height_ratios=[1] * len(PERIODS) + [0.13], hspace=0.12, wspace=0.08)

# (a) map
use_cartopy = USE_CARTOPY
if use_cartopy:
    try:
        import cartopy.crs as ccrs
        ax_map = fig.add_subplot(outer[0, 0], projection=ccrs.PlateCarree())
    except ImportError:
        print("   cartopy not available, drawing a plain map")
        use_cartopy = False
if not use_cartopy:
    ax_map = fig.add_subplot(outer[0, 0])
draw_map(ax_map, use_cartopy)

# sections first (letters b-g), then boxplots (letters h-i)
section_axes = []
box_axes = []
cf = None
for r, period in enumerate(PERIODS):
    row_axes = []
    for c, prod in enumerate(PRODUCTS):
        ax = fig.add_subplot(top[r, c], sharex=section_axes[0][0] if section_axes else None)
        cf = draw_section(ax, prod, period, norm, levels)
        if r == 0:
            ax.set_title(prod, fontsize=10, fontweight='bold')
        if c == 0:
            ax.set_ylabel(f'{period} minus {BASELINE}\nDepth (m)', fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)
        row_axes.append(ax)
    section_axes.append(row_axes)

for r, period in enumerate(PERIODS):
    ax_box = fig.add_subplot(top[r, nprod + 2], sharex=box_axes[0] if box_axes else None)
    draw_boxplot(ax_box, period, box_xlim, legend=(r == 0))
    if r == len(PERIODS) - 1:
        ax_box.set_xlabel('ΔT (°C)', fontsize=8)
    else:
        ax_box.tick_params(labelbottom=False)
    box_axes.append(ax_box)

# latitude-band strips under the sections (share latitude axis)
for c in range(nprod):
    ax_strip = fig.add_subplot(top[len(PERIODS), c], sharex=section_axes[0][c])
    draw_band_strip(ax_strip)

# colour bar for the sections
cax = fig.add_subplot(top[0:len(PERIODS), nprod])
cb = fig.colorbar(cf, cax=cax, ticks=np.arange(-SECTION_VMAX, SECTION_VMAX + 0.01, 0.5))
cb.set_label(f'ΔT relative to {BASELINE} (°C)', fontsize=8)
cb.ax.tick_params(labelsize=7)

# band profiles under the sections
nb = len(LAT_BANDS)
bottom = GridSpecFromSubplotSpec(1, nb, subplot_spec=outer[1, 1], wspace=0.08)
for c, ((key, label, _, _), col) in enumerate(zip(LAT_BANDS, BAND_COLORS)):
    ax = fig.add_subplot(bottom[0, c])
    draw_profile(ax, key, label, col, prof_xlim, first=(c == 0))

# legend for the profile panels, bottom left
ax_leg = fig.add_subplot(outer[1, 0])
ax_leg.axis('off')
handles = [Line2D([], [], color=PRODUCT_COLORS[p], lw=1.6, label=p) for p in PRODUCTS]
handles.append(Line2D([], [], color='k', lw=1.6, label=f'{PROFILE_MAIN_PERIOD}: median, IQR shaded'))
if PROFILE_SECOND_PERIOD:
    handles.append(Line2D([], [], color='k', lw=1.1, ls=':', label=f'{PROFILE_SECOND_PERIOD}: median'))
handles.append(Line2D([], [], color='k', lw=0.6, ls='--', alpha=0.55, label='Boxplot depth layers'))
ax_leg.legend(handles=handles, loc='center left', fontsize=8, frameon=False,
              title=f'Band profiles\n(ΔT relative to {BASELINE}, OND)', title_fontsize=8)

fig.suptitle(f"October–December temperature change along the Polarstern transect relative to {BASELINE}",
             fontsize=12, y=0.93)

for fmt in FIG_FORMATS:
    out = out_dir / f"{FIG_NAME}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   📄 Saved {out}")
plt.close(fig)

# ============================================
# PART 6 (OPTIONAL): LATITUDE-BAND HEATMAP, SAME BANDS AND LAYERS
# ============================================
if MAKE_BAND_HEATMAP:
    print("\n🎨 Drawing band heatmap for comparison...")
    layers = [layer for layer in DEPTH_LAYERS if layer[1] <= PROFILE_DEPTH_MAX]
    fig, axs = plt.subplots(len(PERIODS), len(PRODUCTS), figsize=(4.2 * len(PRODUCTS), 3.2 * len(PERIODS)),
                            sharex=True, sharey=True, squeeze=False)
    vmax = prof_xlim[1]
    hnorm = TwoSlopeNorm(0, -vmax, vmax)
    for r, period in enumerate(PERIODS):
        for c, prod in enumerate(PRODUCTS):
            ax = axs[r, c]
            M = np.full((len(layers), len(LAT_BANDS)), np.nan)
            F = M.copy()
            for i, layer in enumerate(layers):
                for j, (key, *_) in enumerate(LAT_BANDS):
                    v = layer_values(prod, period, layer, band_of_point == key)
                    if len(v):
                        M[i, j], F[i, j] = np.median(v), np.mean(v > 0)
            im = ax.imshow(M, cmap='RdBu_r', norm=hnorm, aspect='auto')
            for i in range(len(layers)):
                for j in range(len(LAT_BANDS)):
                    if not np.isnan(M[i, j]):
                        mixed = 0.25 < F[i, j] < 0.75
                        ax.text(j, i, f"{M[i, j]:+.2f}" + ("*" if mixed else ""), ha='center', va='center',
                                fontsize=7, color='w' if abs(M[i, j]) > 0.6 * vmax else 'k')
            ax.set_xticks(range(len(LAT_BANDS)))
            ax.set_xticklabels([b[0] for b in LAT_BANDS], fontsize=8)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"{a}–{b} m" for a, b in layers], fontsize=8)
            ax.set_title(f"{prod}: {period} minus {BASELINE}", fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8, label='Median ΔT (°C); * = 25–75% of values positive')
    for fmt in FIG_FORMATS:
        out = out_dir / f"{FIG_NAME}_band_heatmap.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"   📄 Saved {out}")
    plt.close(fig)

print(f"\n✅ Done. Figures and value tables in {out_dir}")
