"""Google TimesFM Pipeline & Adaptation Layer for Nepal Weather.
Supports:
1. Official Google TimesFM (timesfm.TimesFm) if installed (GPU/CPU).
2. Clean fallback forecaster for testing & environments before timesfm pip package is installed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dataset_loader import TimeSeriesSample


class NepalTimesFMPipeline:
    def __init__(
        self,
        model_name: str = "google/timesfm-1.0-200m-pytorch",
        context_len: int = 168,
        horizon_len: int = 24,
        backend: str = "auto",
    ):
        self.model_name = model_name
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.backend = backend
        self.tfm = None
        self._init_model()

    def _init_model(self):
        """Attempt to load official google-research timesfm, else fallback gracefully."""
        if self.backend != "fallback":
            try:
                import timesfm
                print(f"[TimesFM] Initializing Google TimesFM ({self.model_name})...")
                self.tfm = timesfm.TimesFm(
                    context_len=self.context_len,
                    horizon_len=self.horizon_len,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend="cpu",
                )
                self.tfm.load_from_checkpoint(repo_id=self.model_name)
                print("[TimesFM] Successfully loaded TimesFM weights.")
                self.backend = "timesfm"
                return
            except Exception as exc:
                print(f"[TimesFM Notice] Could not load official timesfm library ({exc}). Using Nepal adaptive forecaster.")
        
        self.backend = "nepal_adaptive"

    def forecast(self, context_values: list[float], horizon_len: int | None = None) -> list[float]:
        """Forecast future time steps given past context values."""
        h_len = horizon_len or self.horizon_len
        if not context_values:
            return [0.0] * h_len

        # If official TimesFM is active
        if self.backend == "timesfm" and self.tfm is not None:
            import numpy as np
            inputs = np.array([context_values], dtype=np.float32)
            preds = self.tfm.forecast(inputs, freq=[0])  # 0 indicates high-frequency / hourly
            return preds[0][:h_len].tolist()

        # Adaptive Himalayan forecaster: Diurnal 24h harmonic + auto-regressive drift
        # Captures day-night heating cycle and recent trend in mountain terrain
        n = len(context_values)
        recent = context_values[-24:] if n >= 24 else context_values
        recent_mean = sum(recent) / len(recent)
        
        # Calculate 24h diurnal amplitude from last 24h
        max_val = max(recent)
        min_val = min(recent)
        amplitude = (max_val - min_val) / 2.0
        
        # Recent slope/trend
        slope = (context_values[-1] - context_values[-min(n, 12)]) / max(1, min(n, 12))

        predictions = []
        for step in range(1, h_len + 1):
            # 24-hour sinusoidal diurnal pattern
            diurnal = amplitude * math.sin((2 * math.pi * step) / 24.0)
            # Damped trend drift
            trend = slope * step * (0.95 ** step)
            pred = recent_mean + diurnal + trend
            predictions.append(round(pred, 2))

        return predictions

    def evaluate(self, eval_samples: list[TimeSeriesSample]) -> dict[str, float]:
        """Evaluate model against holdout evaluation windows."""
        total_mae = 0.0
        total_mse = 0.0
        total_count = 0

        for sample in eval_samples:
            preds = self.forecast(sample.context_values, horizon_len=len(sample.target_horizon_values))
            for actual, pred in zip(sample.target_horizon_values, preds):
                err = abs(actual - pred)
                total_mae += err
                total_mse += (actual - pred) ** 2
                total_count += 1

        if total_count == 0:
            return {"mae": 0.0, "rmse": 0.0, "evaluated_steps": 0, "evaluated_windows": 0}

        mae = total_mae / total_count
        rmse = math.sqrt(total_mse / total_count)
        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "evaluated_steps": total_count,
            "evaluated_windows": len(eval_samples),
        }

    def save_checkpoint_info(self, output_dir: str):
        """Export model configuration and metadata for Project 3 live inference."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "backend": self.backend,
            "context_len": self.context_len,
            "horizon_len": self.horizon_len,
            "trained_for": "Nepal Topographic Weather & Climate Shifts",
        }
        with open(p / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved model configuration to: {p / 'model_config.json'}")
