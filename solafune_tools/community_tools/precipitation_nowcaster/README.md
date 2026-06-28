# Precipitation Nowcaster

A robust Object-Oriented module for training and inferring precipitation models based on sequential satellite imagery. Built natively for the `solafune_tools.community_tools` ecosystem.

> **Note on Origin:** This tool was developed with assistance from Generative AI (Antigravity/Gemma 31B) to ensure rapid prototyping and clean structural compliance with Solafune guidelines.

## Features
- **Seamless PyTorch Integration:** Encapsulates the dataset loaders, PyTorch CNN definitions, and training loop into a simple `PrecipitationNowcaster` class.
- **Anomaly Filtering:** Built-in safeguards to filter out anomalous samples (e.g. cloud outages resulting in fewer than 3 time-step inputs).
- **Device Agnostic:** Automatically utilizes Apple Silicon (`mps`), NVIDIA (`cuda`), or fallback (`cpu`).

## Dependencies
- `torch`
- `pandas`
- `numpy`
- `rasterio`

## Usage Example

```python
from solafune_tools.community_tools.precipitation_nowcaster import PrecipitationNowcaster

# 1. Initialize the Nowcaster
nowcaster = PrecipitationNowcaster()

# 2. Train the model on the Solafune dataset
# (Set limit=500 for a quick prototype run)
nowcaster.train(
    csv_file='data/train_dataset/train_dataset.csv',
    root_dir='data/train_dataset',
    epochs=5,
    batch_size=8,
    limit=500
)

# 3. Predict precipitation on new satellite sequences
# dummy_input shape: (Batch, Channels, Height, Width) -> (1, 48, 81, 81)
import torch
dummy_input = torch.randn(1, 48, 81, 81) 
prediction = nowcaster.predict(dummy_input)

print(f"Predicted precipitation map shape: {prediction.shape}")
```
