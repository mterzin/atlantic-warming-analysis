import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"xarray version: {xr.__version__}")
print("All packages loaded successfully!")

# Simple plot test
plt.figure()
plt.title("Your Python environment is working!")
plt.show()