# Nepal TimesFM Model

Project 2 loads Project 1’s exported time series, evaluates a TimesFM-compatible forecaster, and writes model metadata.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
timesfm-train --data data/nepal_timesfm_train.csv
```

`data/nepal_timesfm_train.csv` is a small tracked example. Keep production datasets and model weights out of Git; version them with a release, DVC, or object storage.

## License

MIT. See [LICENSE](LICENSE).

