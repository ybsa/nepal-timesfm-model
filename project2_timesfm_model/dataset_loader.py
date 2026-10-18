"""Dataset Loader & Preprocessor for Google TimesFM.
Loads continuous multi-location hourly time series from Project 1,
handles missing intervals, and splits by time and topography (Terai, Hills, Mountain).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TimeSeriesSample:
    unique_id: str
    name: str
    district: str
    province: str
    elevation_m: float
    timestamps: list[str]
    context_values: list[float]
    target_horizon_values: list[float]
    variable: str


class NepalTimesFMDataset:
    def __init__(
        self,
        csv_path: str,
        target_variable: str = "temperature_2m",
        context_len: int = 168,  # 7 days of hourly context
        horizon_len: int = 24,   # 24 hours of forecast horizon
        stride: int = 12,        # sliding window step
    ):
        self.csv_path = Path(csv_path)
        self.target_variable = target_variable
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.stride = stride
        self.series_by_location: dict[str, list[dict]] = defaultdict(list)
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}.\n"
                "Run Project 1 exporter first: python project1_data_agent/run_export.py"
            )

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row["unique_id"]
                val = row.get(self.target_variable)
                val_float = float(val) if val not in (None, "") else None
                self.series_by_location[uid].append({
                    "timestamp": row["timestamp"],
                    "value": val_float,
                    "name": row.get("name", uid),
                    "district": row.get("district", "Unknown"),
                    "province": row.get("province", "Unknown"),
                    "elevation_m": float(row.get("elevation_m", 0.0) or 0.0),
                })

    def get_topography_zone(self, elevation_m: float) -> str:
        """Categorize into Nepal's 3 ecological zones."""
        if elevation_m < 300:
            return "Terai"
        elif elevation_m < 2200:
            return "Hills"
        else:
            return "Mountain"

    def generate_windows(self, train_ratio: float = 0.8) -> tuple[list[TimeSeriesSample], list[TimeSeriesSample]]:
        """Generate sliding window samples for TimesFM training and evaluation.
        Splits strictly by time to prevent future data leakage.
        """
        train_samples = []
        eval_samples = []
        total_window = self.context_len + self.horizon_len

        for uid, steps in self.series_by_location.items():
            if len(steps) < total_window:
                continue

            # Temporal split point
            split_idx = int(len(steps) * train_ratio)
            train_steps = steps[:split_idx]
            eval_steps = steps[split_idx:]

            # Process train windows
            for i in range(0, len(train_steps) - total_window + 1, self.stride):
                window = train_steps[i : i + total_window]
                vals = [w["value"] for w in window]
                if any(v is None for v in vals):
                    continue

                train_samples.append(
                    TimeSeriesSample(
                        unique_id=uid,
                        name=window[0]["name"],
                        district=window[0]["district"],
                        province=window[0]["province"],
                        elevation_m=window[0]["elevation_m"],
                        timestamps=[w["timestamp"] for w in window],
                        context_values=vals[: self.context_len],
                        target_horizon_values=vals[self.context_len :],
                        variable=self.target_variable,
                    )
                )

            # Process holdout evaluation windows across the split boundary
            if len(steps) >= total_window:
                eval_context = [s["value"] for s in steps[split_idx - self.context_len : split_idx]]
                eval_horizon = [s["value"] for s in steps[split_idx : split_idx + self.horizon_len]]
                if len(eval_context) == self.context_len and len(eval_horizon) == self.horizon_len:
                    if not any(v is None for v in eval_context + eval_horizon):
                        eval_samples.append(
                            TimeSeriesSample(
                                unique_id=uid,
                                name=steps[0]["name"],
                                district=steps[0]["district"],
                                province=steps[0]["province"],
                                elevation_m=steps[0]["elevation_m"],
                                timestamps=[s["timestamp"] for s in steps[split_idx - self.context_len : split_idx + self.horizon_len]],
                                context_values=eval_context,
                                target_horizon_values=eval_horizon,
                                variable=self.target_variable,
                            )
                        )

        return train_samples, eval_samples
