import numpy as np

def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Vegetation Index (NDVI)."""
    red = red_band.astype(float)
    nir = nir_band.astype(float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.true_divide(nir - red, nir + red)
        ndvi[np.isnan(ndvi)] = 0  # Handle division by zero
    return ndvi

def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Water Index (NDWI)."""
    green = green_band.astype(float)
    nir = nir_band.astype(float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.true_divide(green - nir, green + nir)
        ndwi[np.isnan(ndwi)] = 0 # Handle division by zero
    return ndwi