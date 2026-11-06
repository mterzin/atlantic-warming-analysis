#!/usr/bin/env python3
"""
00_check_input_files.py

Inspect the new October-December (OND) climatology files before re-running
the cross-section and box-plot analysis (manuscript Figs 2-3).

For every NetCDF file found (recursively) under the data folder it reports:
  - the temperature variable: name, units, standard_name (potential vs in-situ)
  - depth levels: number, shallowest, deepest
  - horizontal grid: regular (1D) or curvilinear (2D), and the longitude
    convention (-180..180 or 0..360), which matters for extracting points at
    negative (western) longitudes
  - time value(s) and time bounds, decoded with the file's own calendar
  - provenance attributes (history, experiment_id, ...) and the years and
    scenario names mentioned in them
  - surface temperature range, as a quick units sanity check

A one-line-per-file summary is printed at the end.

Usage:
    python 00_check_input_files.py                         # default folder
    python 00_check_input_files.py /path/to/data           # another folder
    python 00_check_input_files.py > input_check_log.txt   # save the report
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import cftime
except ImportError:  # cftime is normally installed together with netCDF4
    cftime = None

warnings.filterwarnings("ignore")

DEFAULT_DATA_DIR = (
    Path.home()
    / "Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/DATA_OND_1993-2025"
)

TEMP_NAMES = ["thetao", "votemper", "thetao_oras", "temperature", "temp", "to", "potemp"]
TEMP_STD_NAMES = [
    "sea_water_potential_temperature",
    "sea_water_conservative_temperature",
    "sea_water_temperature",
]
DEPTH_NAMES = ["depth", "deptht", "lev", "olevel", "nav_lev", "z", "depth_below_sea"]
TIME_NAMES = ["time", "time_counter", "t"]
LAT_NAMES = ["latitude", "lat", "nav_lat", "y_lat"]
LON_NAMES = ["longitude", "lon", "nav_lon", "x_lon"]
PROVENANCE_ATTRS = [
    "title", "source", "source_id", "experiment_id", "parent_experiment_id",
    "variant_label", "institution", "product_id", "product", "references",
    "CDO", "CDI", "history", "history_of_appended_files",
]
SCENARIO_PATTERN = re.compile(r"(historical|ssp\d{3}|rcp\d{2})", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")


def find_name(ds, candidates):
    """Return the first candidate that exists as a variable or dimension."""
    for name in candidates:
        if name in ds.variables or name in ds.dims:
            return name
    return None


def find_temperature(ds):
    for name in TEMP_NAMES:
        if name in ds.data_vars:
            return name
    for name, var in ds.data_vars.items():
        if var.attrs.get("standard_name", "") in TEMP_STD_NAMES:
            return name
    return None


def find_depth(ds, temp_name):
    name = find_name(ds, DEPTH_NAMES)
    if name:
        return name
    # Fall back on CF metadata of the temperature variable's dimensions
    if temp_name:
        for dim in ds[temp_name].dims:
            if dim in ds.variables:
                attrs = ds[dim].attrs
                if attrs.get("axis") == "Z" or attrs.get("positive") in ("down", "up"):
                    return dim
    return None


def decode(values, units, calendar):
    """Decode numeric times with the file's own calendar; fall back to raw values."""
    if cftime is None or not units or "since" not in units:
        return [str(v) for v in np.atleast_1d(values)]
    try:
        dates = cftime.num2date(np.atleast_1d(values), units, calendar=calendar)
        return [d.strftime("%Y-%m-%d %H:%M") for d in np.atleast_1d(dates)]
    except Exception as err:  # noqa: BLE001
        return [f"{v} (could not decode: {err})" for v in np.atleast_1d(values)]


def describe_time(ds):
    lines = []
    tname = find_name(ds, TIME_NAMES)
    if tname is None or tname not in ds.variables:
        return ["  time: no time variable found"], None, None
    tvar = ds[tname]
    units = tvar.attrs.get("units", "")
    calendar = tvar.attrs.get("calendar", "standard")
    values = tvar.values
    decoded = decode(values, units, calendar)
    lines.append(f"  time variable : {tname}  (n={np.size(values)}, units='{units}', calendar='{calendar}')")
    shown = decoded if len(decoded) <= 12 else decoded[:6] + ["..."] + decoded[-6:]
    lines.append(f"  time value(s) : {', '.join(shown)}")

    # Time bounds: CF 'bounds' or 'climatology' attribute, or common names
    bname = tvar.attrs.get("bounds") or tvar.attrs.get("climatology")
    if not bname:
        for cand in [f"{tname}_bnds", f"{tname}_bounds", "time_bnds", "time_bounds",
                     "climatology_bounds", "time_counter_bounds"]:
            if cand in ds.variables:
                bname = cand
                break
    bounds_text = None
    if bname and bname in ds.variables:
        b = np.asarray(ds[bname].values).reshape(-1, 2)
        b_dec = [decode(row, units, calendar) for row in b]
        bounds_text = "; ".join(f"{row[0]} to {row[1]}" for row in b_dec[:6])
        if len(b_dec) > 6:
            bounds_text += f"; ... ({len(b_dec)} rows)"
        lines.append(f"  time bounds   : {bname}: {bounds_text}")
    else:
        lines.append("  time bounds   : none found (check 'history' below for the averaging period)")
    return lines, decoded, bounds_text


def describe_grid(ds):
    lines = []
    lat = find_name(ds, LAT_NAMES)
    lon = find_name(ds, LON_NAMES)
    lon_conv = "unknown"
    if lat and lon and lat in ds.variables and lon in ds.variables:
        lat_v = ds[lat].values
        lon_v = ds[lon].values
        kind = "regular (1D lat/lon)" if ds[lat].ndim == 1 else "curvilinear (2D lat/lon)"
        lon_min, lon_max = float(np.nanmin(lon_v)), float(np.nanmax(lon_v))
        lon_conv = "0..360" if lon_max > 180.5 else "-180..180"
        lines.append(f"  grid          : {kind}, {lat}{ds[lat].shape} / {lon}{ds[lon].shape}")
        lines.append(f"  lat range     : {np.nanmin(lat_v):.2f} to {np.nanmax(lat_v):.2f}")
        lines.append(f"  lon range     : {lon_min:.2f} to {lon_max:.2f}  -> convention {lon_conv}")
    else:
        lines.append("  grid          : latitude/longitude variables not found")
    return lines, lon_conv


def describe_depth(ds, dname):
    if dname is None or dname not in ds.variables:
        return ["  depth         : no depth coordinate found"], None
    d = ds[dname]
    vals = np.asarray(d.values, dtype=float)
    units = d.attrs.get("units", "?")
    lines = [
        f"  depth coord   : {dname}  (n={vals.size}, units='{units}', positive='{d.attrs.get('positive', '?')}')",
        f"  depth range   : {np.nanmin(vals):.2f} to {np.nanmax(vals):.2f} {units}",
        f"  first levels  : {', '.join(f'{v:.2f}' for v in vals[:5])}",
    ]
    if units.lower() in ("cm", "centimeters", "centimetres"):
        lines.append("  WARNING       : depth is in centimetres; convert to metres before comparing products")
    return lines, vals.size


def surface_range(ds, temp_name, dname):
    """Min/max of the first time step at the shallowest level."""
    var = ds[temp_name]
    sel = {}
    for dim in var.dims:
        if dim in TIME_NAMES or dim == dname:
            sel[dim] = 0
    try:
        surf = var.isel(sel).values.astype(float)
        if np.all(np.isnan(surf)):
            return "  surface range : all NaN at first level"
        lo, hi = np.nanmin(surf), np.nanmax(surf)
        note = ""
        if hi > 100:
            note = "  -> looks like KELVIN, convert before differencing"
        elif hi < 5:
            note = "  -> suspiciously cold for a global/Atlantic surface field"
        return f"  surface range : {lo:.2f} to {hi:.2f} {var.attrs.get('units', '?')}{note}"
    except Exception as err:  # noqa: BLE001
        return f"  surface range : could not compute ({err})"


def describe_provenance(ds, path):
    lines = []
    text_for_search = path.name + " "
    for key in PROVENANCE_ATTRS:
        if key in ds.attrs:
            value = str(ds.attrs[key])
            text_for_search += value + " "
            if key == "history" and len(value) > 1500:
                value = value[:1500] + " [...truncated]"
            lines.append(f"  {key:<14}: {value}")
    if not lines:
        lines.append("  (no provenance attributes found)")
    years = sorted(set(YEAR_PATTERN.findall(text_for_search)))
    scenarios = sorted(set(s.lower() for s in SCENARIO_PATTERN.findall(text_for_search)))
    lines.append(f"  years mentioned in name/attributes: {', '.join(years) if years else 'none'}")
    lines.append(f"  scenarios mentioned               : {', '.join(scenarios) if scenarios else 'none'}")
    return lines


def check_file(path):
    print("=" * 100)
    print(f"FILE: {path}")
    print(f"  size          : {path.stat().st_size / 1e6:.1f} MB")
    summary = {"file": path.name}
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as err:  # noqa: BLE001
        print(f"  ERROR: could not open file ({err})")
        summary["status"] = "ERROR opening"
        return summary

    with ds:
        print(f"  dimensions    : {dict(ds.sizes)}")
        print(f"  data vars     : {', '.join(ds.data_vars)}")

        temp = find_temperature(ds)
        summary["var"] = temp or "NOT FOUND"
        if temp:
            a = ds[temp].attrs
            std = a.get("standard_name", "?")
            print(f"  temperature   : {temp}  dims={ds[temp].dims}")
            print(f"                  standard_name='{std}', long_name='{a.get('long_name', '?')}', units='{a.get('units', '?')}'")
            if std not in ("sea_water_potential_temperature", "?"):
                print("  NOTE          : not potential temperature; check before comparing products")
        else:
            print("  temperature   : NOT FOUND, check variable names above")

        dname = find_depth(ds, temp)
        dlines, nlev = describe_depth(ds, dname)
        print("\n".join(dlines))
        summary["levels"] = nlev

        glines, lon_conv = describe_grid(ds)
        print("\n".join(glines))
        summary["lon"] = lon_conv

        tlines, decoded, bounds = describe_time(ds)
        print("\n".join(tlines))
        summary["time"] = decoded[0] if decoded and len(decoded) == 1 else (f"{len(decoded)} steps" if decoded else "none")
        summary["bounds"] = bounds or "none"

        if temp:
            print(surface_range(ds, temp, dname))

        print("  provenance:")
        print("\n".join(describe_provenance(ds, path)))
    summary["status"] = "ok"
    return summary


def main():
    data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        sys.exit(f"Folder not found: {data_dir}")
    files = sorted(data_dir.rglob("*.nc"))
    if not files:
        sys.exit(f"No .nc files found under {data_dir}")

    print(f"Checking {len(files)} NetCDF file(s) under {data_dir}\n")
    summaries = [check_file(f) for f in files]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for s in summaries:
        rel = s["file"]
        print(f"{rel}\n    var={s.get('var', '-')}, levels={s.get('levels', '-')}, "
              f"lon={s.get('lon', '-')}, time={s.get('time', '-')}, bounds={s.get('bounds', '-')}, "
              f"status={s.get('status', '-')}")


if __name__ == "__main__":
    main()
