# Atlantic Ocean Warming Analysis

Analysis of Atlantic Ocean warming trends using Polarstern observations, CMIP6 models, and ocean reanalyses.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `01_unzip_data.py` | Extract compressed data files |
| `02_explore_data.py` | Explore data structure and contents |
| `03_define_transect.py` | Define Polarstern median transect and sampling points |
| `04_cross-sections.py` | Generate cross-section plots, difference maps, and HTML report |

## Requirements

See `requirements.txt` for Python dependencies.

## Usage

1. Create virtual environment: `python3 -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run analysis: `python 04_cross-sections.py`

## Data Location

To check with Tido how we can descript where the data can be downloaded from.

## Outputs

- PDF plots (cross-sections, difference maps, boxplots)
- HTML interactive report
- CSV statistics files
