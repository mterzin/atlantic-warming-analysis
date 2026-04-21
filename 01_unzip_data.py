import zipfile
import os
from pathlib import Path

# Specify your exact data path
data_path = Path("/home/marko-terzin/Documents/co-authored_papers/NoSoAt_modelling_paper/Marko_analysis/")

# Find all zip files in that directory
zip_files = sorted(data_path.glob("*.zip"))

print(f"Found {len(zip_files)} zip files in: {data_path}")
print("-" * 60)

for zip_path in zip_files:
    # Create extraction directory name (remove .zip extension)
    extract_dir = data_path / zip_path.stem
    
    print(f"\n📦 Processing: {zip_path.name}")
    print(f"📂 Extracting to: {extract_dir}")
    
    # Create directory if it doesn't exist
    extract_dir.mkdir(exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Count files extracted
    file_count = len(list(extract_dir.glob("**/*")))
    print(f"✅ Extracted {file_count} files/folders")

print("\n" + "=" * 60)
print("🎉 All files extracted successfully!")
print("=" * 60)

# Optional: Show the extracted folders
print("\nExtracted folders now available:")
for folder in sorted(data_path.glob("DATA-*")):
    if folder.is_dir():
        size_mb = sum(f.stat().st_size for f in folder.glob('**/*') if f.is_file()) / (1024*1024)
        print(f"  📁 {folder.name} - {size_mb:.1f} MB")