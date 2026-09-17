"""Unified Command-Line Interface for the Anacity Facility Usage Prediction System."""

import argparse
import json
import os
import sys
import time

import pandas as pd

from src.data.generator import CommunityDatasetGenerator, generate_and_save_dataset
from src.features.pipeline import FeatureExtractor
from src.features.temporal_splitter import split_dataset_chronologically
from src.models.cascaded_pipeline import CascadedPredictionPipeline
from src.models.baselines import HabitualBaseline, GlobalPopularityBaseline
from src.evaluation.metrics import PredictionEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer
from src.presentation.formatter import PredictionFormatter


def run_pipeline(
    data_path: str = "data/facility_bookings.csv",
    num_residents: int = 350,
    seed: int = 42,
    mode: str = "all",
) -> None:
    """Executes the complete end-to-end workflow."""
    start_time = time.time()
    print("================================================================================")
    print("🏢 ANACITY FACILITY USAGE PREDICTION SYSTEM — END-TO-END PIPELINE")
    print(f"Mode: {mode.upper()} | Python: {sys.version.split()[0]} | Seed: {seed}")
    print("================================================================================")

    # -------------------------------------------------------------
    # 1. Dataset Generation
    # -------------------------------------------------------------
    if mode in ["all", "generate"]:
        print("\n[PHASE 1] Generating Realistic Synthetic Dataset...")
        df = generate_and_save_dataset(
            output_path=data_path, num_residents=num_residents, seed=seed
        )
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Run with --mode generate first.")
        df = pd.read_csv(data_path)

    print(f"Dataset loaded: {len(df)} total bookings across {df['resident_id'].nunique()} residents.")

    # -------------------------------------------------------------
    # 2. Chronological Splitting & Leakage-Safe Feature Extraction
    # -------------------------------------------------------------
    print("\n[PHASE 2] Chronological Temporal Splitting & Leakage-Safe Feature Extraction...")
    train_df, val_df, test_df = split_dataset_chronologically(df)
    print(f"Split sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Unseen Test: {len(test_df)}")

    extractor = FeatureExtractor()
    extractor.fit_global_priors(train_df)

    print("Extracting training features...")
    X_train, y_train, meta_train, train_hist = extractor.extract_features(train_df)
    print("Extracting validation features (chaining history)...")
    X_val, y_val, meta_val, val_hist = extractor.extract_features(val_df, initial_history=train_hist)
    print("Extracting unseen holdout test features (strictly prior history)...")
    X_test, y_test, meta_test, _ = extractor.extract_features(test_df, initial_history=val_hist)

    # -------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------
    pipeline = CascadedPredictionPipeline(random_state=seed)
    hab_base = HabitualBaseline()
    glob_base = GlobalPopularityBaseline()

    if mode in ["all", "train"]:
        print("\n[PHASE 3] Training Cascaded ML Pipeline & Baselines...")
        pipeline.fit(X_train, y_train)
        pipeline.save("models/artifacts")

        hab_base.fit(train_df)
        glob_base.fit(train_df)
    else:
        pipeline.load("models/artifacts")
        hab_base.fit(train_df)
        glob_base.fit(train_df)

    # -------------------------------------------------------------
    # 4. Evaluation on Unseen Test Holdout
    # -------------------------------------------------------------
    if mode in ["all", "evaluate", "export"]:
        print("\n[PHASE 4] Evaluating on Chronological Unseen Test Set (Month 6)...")
        test_preds = pipeline.predict(X_test, meta_test)
        hab_preds = hab_base.predict(meta_test)
        glob_preds = glob_base.predict(meta_test)

        evaluator = PredictionEvaluator(nudge_tolerance_minutes=60)
        ml_eval = evaluator.evaluate(y_test, test_preds, meta_test)
        hab_eval = evaluator.evaluate(y_test, hab_preds, meta_test)
        glob_eval = evaluator.evaluate(y_test, glob_preds, meta_test)

        summary = ml_eval["summary"]
        details = ml_eval["details"]

        # Persist metrics summary
        os.makedirs("output", exist_ok=True)
        with open("output/metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("               HOLD-OUT EVALUATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"Evaluated Test Samples:        {summary['n_samples']}")
        print(f"Output 1: Facility Accuracy:   {summary['facility_accuracy']*100:.2f}% (Macro F1: {summary['facility_macro_f1']:.3f})")
        print(f"Output 2: Usage Day Accuracy:  {summary['usage_day_accuracy']*100:.2f}% (Macro F1: {summary['usage_day_macro_f1']:.3f})")
        print(f"Output 3: Hour ±1hr Accuracy:  {summary['usage_hour_within_1hr_accuracy']*100:.2f}% (MAE: {summary['usage_hour_mae']}h, Circular MAE: {summary['usage_hour_circular_mae']}h)")
        print(f"Output 4: Nudge Window Match:  {summary['nudge_window_accuracy']*100:.2f}% (Proactive Rate: {summary['nudge_proactive_actionability_rate']*100:.2f}%)")
        print("-" * 70)
        print(f"OVERALL: Exact 4 of 4 Match:   {summary['exact_4_of_4_match_rate']*100:.2f}%")
        print(f"OVERALL: Partial 3+ of 4 Match:{summary['partial_3_plus_match_rate']*100:.2f}%")
        print(f"OVERALL: Avg Outputs Matched:  {summary['average_outputs_matched']} / 4.0")
        print("=" * 70)

        # Baseline Benchmark Comparison
        print("\n--- BENCHMARK COMPARISON AGAINST BASELINES ---")
        comparison_table = pd.DataFrame(
            [
                {
                    "Model": "Cascaded GBDT Pipeline (Ours)",
                    "Facility Acc": f"{summary['facility_accuracy']*100:.1f}%",
                    "Day Acc": f"{summary['usage_day_accuracy']*100:.1f}%",
                    "Hour ±1h Acc": f"{summary['usage_hour_within_1hr_accuracy']*100:.1f}%",
                    "Exact 4 of 4": f"{summary['exact_4_of_4_match_rate']*100:.1f}%",
                    "Avg Matched": f"{summary['average_outputs_matched']:.2f}",
                },
                {
                    "Model": "Habitual Historical Mode Baseline",
                    "Facility Acc": f"{hab_eval['summary']['facility_accuracy']*100:.1f}%",
                    "Day Acc": f"{hab_eval['summary']['usage_day_accuracy']*100:.1f}%",
                    "Hour ±1h Acc": f"{hab_eval['summary']['usage_hour_within_1hr_accuracy']*100:.1f}%",
                    "Exact 4 of 4": f"{hab_eval['summary']['exact_4_of_4_match_rate']*100:.1f}%",
                    "Avg Matched": f"{hab_eval['summary']['average_outputs_matched']:.2f}",
                },
                {
                    "Model": "Global Community Mode Baseline",
                    "Facility Acc": f"{glob_eval['summary']['facility_accuracy']*100:.1f}%",
                    "Day Acc": f"{glob_eval['summary']['usage_day_accuracy']*100:.1f}%",
                    "Hour ±1h Acc": f"{glob_eval['summary']['usage_hour_within_1hr_accuracy']*100:.1f}%",
                    "Exact 4 of 4": f"{glob_eval['summary']['exact_4_of_4_match_rate']*100:.1f}%",
                    "Avg Matched": f"{glob_eval['summary']['average_outputs_matched']:.2f}",
                },
            ]
        )
        print(comparison_table.to_string(index=False))

        # Stratified Error Analysis
        print("\n--- COHORT ERROR ANALYSIS (BY USER ENGAGEMENT TIER) ---")
        cohort_analysis = ErrorAnalyzer.analyze_by_engagement_tier(X_test, details)
        print(cohort_analysis.to_string(index=False))

        # -------------------------------------------------------------
        # 5. Prediction Review Table Deliverable Formatting
        # -------------------------------------------------------------
        print("\n[PHASE 5] Formatting Prediction Review Output (Assignment §3.4 & §3.5)...")
        review_table = PredictionFormatter.build_comparison_table(
            meta_df=meta_test,
            y_pred_df=test_preds,
            y_true_df=y_test,
            eval_details=details,
        )
        PredictionFormatter.export_all_formats(review_table, summary, output_dir="output")

    elapsed = time.time() - start_time
    print(f"\n Pipeline executed successfully in {elapsed:.2f} seconds!")
    print("Deliverables available in output/:")
    print("  1. CSV Table:     output/prediction_review_table.csv")
    print("  2. Excel Table:   output/prediction_review_table.xlsx")
    print("  3. HTML Viewer:   output/prediction_review.html")
    print("  4. Streamlit UI:  streamlit run src/ui/app.py")


def main():
    parser = argparse.ArgumentParser(description="Anacity Facility Usage Prediction System")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "generate", "train", "evaluate", "export"],
        help="Pipeline execution mode",
    )
    parser.add_argument("--residents", type=int, default=350, help="Number of simulated residents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-path", type=str, default="data/facility_bookings.csv", help="Dataset path")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        num_residents=args.residents,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
