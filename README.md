# Nepal TimesFM Model & Benchmark Dataset

This repository contains the **pre-trained Google TimesFM model checkpoint** and **historical multi-location training dataset** for Nepal's topographic weather and microclimates.

## 📦 What's Included

1. **`data/nepal_timesfm_train.csv`**: Standardized continuous multi-location hourly time-series dataset across high-altitude and valley stations in Nepal.
2. **`models/model_config.json`**: Pre-trained TimesFM configuration and adaptation weights.
3. **`project2_timesfm_model/`**: Inference pipeline adapter to load and run predictions with the model.

## 🚀 Usage

Install the model package:

```powershell
pip install -e .
```

Load and run predictions in Python:

```python
from project2_timesfm_model.timesfm_pipeline import NepalTimesFMPipeline

# Initialize model with trained checkpoint
model = NepalTimesFMPipeline(horizon_len=24)

# Past 7 days of temperature (°C)
past_context = [14.2, 13.8, 11.5, 9.8, ...] 

# Forecast next 24 hours
forecast = model.forecast(past_context)
print("Projected 24h trajectory:", forecast)
```

## License
MIT. See [LICENSE](LICENSE).

