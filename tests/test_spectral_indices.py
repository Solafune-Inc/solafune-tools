import unittest
import numpy as np
from solafune_tools.community_tools.spectral_indices.core import calculate_ndvi, calculate_ndwi

class TestSpectralIndices(unittest.TestCase):
    
    def test_ndvi_calculation(self):
        # Create dummy data: Red=10, NIR=50. NDVI should be (50-10)/(50+10) = 0.66
        red = np.array([[10]])
        nir = np.array([[50]])
        result = calculate_ndvi(red, nir)
        self.assertAlmostEqual(result[0,0], 0.6666666, places=5)

    def test_division_by_zero(self):
        # Create data that sums to zero (0,0) to ensure code doesn't crash
        red = np.array([[0]])
        nir = np.array([[0]])
        result = calculate_ndvi(red, nir)
        self.assertEqual(result[0,0], 0)

    def test_ndwi(self):
        green = np.array([[20]])
        nir = np.array([[10]])
        result = calculate_ndwi(green, nir)
        self.assertAlmostEqual(result[0, 0], (20 - 10) / (20 + 10), places=5)

    def test_ndwi_zero_division(self):
        green = np.array([[0]])
        nir = np.array([[0]])
        result = calculate_ndwi(green, nir)
        self.assertEqual(result[0, 0], 0)

if __name__ == '__main__':
    unittest.main()