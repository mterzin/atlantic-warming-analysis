import os
from pathlib import Path
import xarray as xr
import numpy as np

# Base directory
base_dir = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/")

# Look at the extracted folders
extracted_folders = sorted(base_dir.glob("DATA-*"))
print("Found extracted data folders:")
for folder in extracted_folders:
    size_mb = sum(f.stat().st_size for f in folder.glob('**/*') if f.is_file()) / (1024*1024)
    print(f"  📁 {folder.name} - {size_mb:.1f} MB")

# Let's peek inside each folder
print("\n" + "="*60)
print("EXPLORING DATA STRUCTURE")
print("="*60)

for folder in extracted_folders[:1]:  # Just look at first folder for now
    print(f"\n📂 Contents of {folder.name}:")
    
    # List first level contents
    for item in sorted(folder.glob("*")):
        if item.is_dir():
            print(f"    📁 {item.name}/")
            # Show one level deeper for context
            subitems = list(item.glob("*"))[:3]  # First 3 items
            for sub in subitems:
                if sub.is_dir():
                    print(f"        📁 {sub.name}/")
                else:
                    size_kb = sub.stat().st_size / 1024
                    print(f"        📄 {sub.name} ({size_kb:.1f} KB)")
        else:
            size_mb = item.stat().st_size / (1024*1024)
            print(f"    📄 {item.name} ({size_mb:.2f} MB)")

# Check for NetCDF files specifically
print("\n" + "="*60)
print("LOOKING FOR NETCDF FILES (*.nc)")
print("="*60)

nc_files = list(base_dir.glob("**/*.nc"))
print(f"Found {len(nc_files)} NetCDF files:")

for nc_file in nc_files[:10]:  # Show first 10
    rel_path = nc_file.relative_to(base_dir)
    size_mb = nc_file.stat().st_size / (1024*1024)
    print(f"  📄 {rel_path} ({size_mb:.1f} MB)")

if len(nc_files) > 10:
    print(f"  ... and {len(nc_files)-10} more")

# Try to open one NetCDF file to see its contents
if nc_files:
    print("\n" + "="*60)
    print("PREVIEWING FIRST NETCDF FILE")
    print("="*60)
    
    test_file = nc_files[0]
    print(f"Opening: {test_file.relative_to(base_dir)}")
    
    try:
        ds = xr.open_dataset(test_file)
        print("\nDataset dimensions:", dict(ds.dims))
        print("\nData variables:")
        for var in ds.data_vars:
            print(f"  - {var}: {ds[var].dims} {ds[var].shape}")
        print("\nCoordinates:", list(ds.coords))
        
        # Close the dataset
        ds.close()
    except Exception as e:
        print(f"Could not open file: {e}")