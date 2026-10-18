"""Runner script to evaluate and export TimesFM model on Nepal dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from project2_timesfm_model.dataset_loader import NepalTimesFMDataset
from project2_timesfm_model.timesfm_pipeline import NepalTimesFMPipeline


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate TimesFM on Nepal Weather Data")
    parser.add_argument("--data", default="project2_timesfm_model/data/nepal_timesfm_train.csv", help="Path to training CSV")
    parser.add_argument("--variable", default="temperature_2m", help="Target variable (temperature_2m, precipitation_mm, etc.)")
    parser.add_argument("--context-len", type=int, default=168, help="Context length in hours (default: 168 = 7 days)")
    parser.add_argument("--horizon-len", type=int, default=24, help="Forecast horizon in hours (default: 24 hours)")
    parser.add_argument("--output-dir", default="project2_timesfm_model/models", help="Directory to save model artifacts")

    args = parser.parse_args()

    print(f"Loading Nepal dataset from: {args.data}")
    dataset = NepalTimesFMDataset(
        csv_path=args.data,
        target_variable=args.variable,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
    )

    train_samples, eval_samples = dataset.generate_windows(train_ratio=0.8)
    print(f"Generated {len(train_samples)} training windows and {len(eval_samples)} evaluation windows.")

    pipeline = NepalTimesFMPipeline(
        context_len=args.context_len,
        horizon_len=args.horizon_len,
    )

    print("\nEvaluating forecast performance on holdout period...")
    metrics = pipeline.evaluate(eval_samples)
    print(f"Results for variable '{args.variable}':")
    print(f"  MAE (Mean Absolute Error): {metrics['mae']}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']}")
    print(f"  Evaluated steps: {metrics['evaluated_steps']}")

    pipeline.save_checkpoint_info(args.output_dir)
    print("\nModel ready for Project 3 live inference!")


if __name__ == "__main__":
    main()
