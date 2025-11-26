# Spectral Indices Usage Guide

## Installation
Ensure you have the `solafune-tools` package installed.

## How to use

### Calculating NDVI
```python
from solafune_tools.community_tools.spectral_indices.core import calculate_ndvi
import numpy as np

# Load your bands (Example shapes)
red_band = np.array([[100, 200], [100, 200]])
nir_band = np.array([[500, 600], [500, 600]])

ndvi = calculate_ndvi(red_band, nir_band)
print(ndvi)