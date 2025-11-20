"""
SaaS Idea Validator - Main Entry Point

Complete pipeline with:
- SL Training with Precision, Recall, Confusion Matrix
- Model evaluation using ModelEvaluator
- Multiple model comparison using evaluate_multiple_models
- Visualization generation
- Results reporting

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path

# Import modules
from data_loader import load_training_data
from ml_system import MLSystem
from evaluation import ModelEvaluator, evaluate_multiple_models


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Execute complete ML pipeline"""

    try:
        print("\n" + "="*80)
        print(" "*20 + "SaaS Idea Validator - ML Training Pipeline")
        print(" "*15 + "With Precision, Recall, Confusion Matrix")
        print("="*80)

        # ============================================================
        # STEP 1: Initialize ML System
        # ============================================================
        logger.info("\nSTEP 1: Initializing ML System...")
        ml_system = MLSystem(config_path='config.yaml')

        # ============================================================
        # STEP 2: Load Data
        # ============================================================
        logger.info("\nSTEP 2: Loading Data...")
        X, y = load_training_data()
        logger.info(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

        # ============================================================
        # STEP 3: Build Models
        # ============================================================
        logger.info("\nSTEP 3: Building Models...")
        models = ml_system.build_models()
        logger.info(f"✓ Models built: {list(models.keys())}")

        # ============================================================
        # STEP 4: Train with Cross-Validation
        # ============================================================
        logger.info("\nSTEP 4: Training Models with Cross-Validation...")
        logger.info("   (Includes Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)")
        cv_results, batch_reports = ml_system.train_with_cv(X, y)

        # ============================================================
        # STEP 5: Generate Summary Reports
        # ============================================================
        logger.info("\nSTEP 5: Generating Summary Reports...")
        summary_reports = ml_system.generate_summary_reports()

        # ============================================================
        # STEP 6: Model Comparison
        # ============================================================
        logger.info("\nSTEP 6: Creating Model Comparison...")
        comparison_df = ml_system.generate_model_comparison(summary_reports)

        # ============================================================
        # STEP 7: Save All Models
        # ============================================================
        logger.info("\nSTEP 7: Saving All Models to JSON...")
        ml_system.save_all_models_to_json()

        # ============================================================
        # STEP 8: Evaluate with ModelEvaluator (NEW!)
        # ============================================================
        logger.info("\nSTEP 8: Detailed Evaluation with ModelEvaluator...")
        from sklearn.preprocessing import StandardScaler

        # Prepare data for evaluation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data (using same split as training for consistency)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Evaluate best model (LightGBM)
        best_model = models[ml_system.best_model_name]
        evaluator = ModelEvaluator(best_model, scaler, X_test, y_test, ml_system.best_model_name)

        logger.info(f"\n✓ Evaluating Best Model: {ml_system.best_model_name}")
        best_model_metrics = evaluator.evaluate()

        # Print detailed metrics
        evaluator.print_metrics()

        # Get confusion matrix analysis
        cm_analysis = evaluator.get_confusion_matrix_analysis()
        logger.info("\nConfusion Matrix Analysis:")
        for class_name, analysis in cm_analysis.items():
            logger.info(f"  {class_name}:")
            logger.info(f"    TP: {analysis['true_positives']}, FP: {analysis['false_positives']}, "
                       f"FN: {analysis['false_negatives']}, TN: {analysis['true_negatives']}")

        # ✅ NEW: Use evaluate_multiple_models for all models comparison
        logger.info("\nSTEP 8B: Comparing All Models with evaluate_multiple_models...")
        comparison_results = evaluate_multiple_models(models, scaler, X_test, y_test)
        logger.info(f"\n✓ Evaluated {len(comparison_results)} models")
        logger.info("\nMultiple Models Comparison:")
        logger.info(comparison_results.to_string(index=False))

        # Save best model evaluation to JSON
        # ✅ NEW: Use Path for file operations
        reports_dir = Path(ml_system.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        eval_output_path = reports_dir / f"{ml_system.best_model_name}_evaluation.json"
        evaluator.save_metrics_to_json(str(eval_output_path))

        # Save multiple models comparison to CSV
        comparison_csv_path = reports_dir / "multiple_models_evaluation.csv"
        comparison_results.to_csv(comparison_csv_path, index=False)
        logger.info(f"✓ Saved multiple models comparison to: {comparison_csv_path}")

        # ============================================================
        # PRINT RESULTS
        # ============================================================
        print("\n" + "="*80)
        print("✓ TRAINING AND EVALUATION COMPLETE!")
        print("="*80)

        print("\n📊 Model Comparison Results (from ml_system):")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)

        print("\n📊 Multiple Models Comparison (from evaluate_multiple_models):")
        print("="*80)
        print(comparison_results.to_string(index=False))
        print("="*80)

        # ============================================================
        # PRINT BEST MODEL INFO
        # ============================================================
        best_model_name = ml_system.best_model_name
        best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]

        print(f"\n★ BEST MODEL: {best_model_name}")
        print(f"{'='*80}")
        print(f"  Accuracy:  {best_row['Accuracy_Mean']:.4f} ± {best_row['Accuracy_Std']:.4f}")
        print(f"  Precision: {best_row['Precision_Mean']:.4f} ± {best_row['Precision_Std']:.4f}")
        print(f"  Recall:    {best_row['Recall_Mean']:.4f} ± {best_row['Recall_Std']:.4f}")
        print(f"  F1-Score:  {best_row['F1_Mean']:.4f} ± {best_row['F1_Std']:.4f}")
        print(f"  ROC-AUC:   {best_row['ROC_AUC_Mean']:.4f}")
        print(f"{'='*80}")

        # ============================================================
        # PRINT DETAILED EVALUATION RESULTS
        # ============================================================
        print(f"\n📋 Detailed {best_model_name} Evaluation (Test Set):")
        print(f"{'='*80}")
        print(f"  Accuracy:  {best_model_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_model_metrics['precision']:.4f}")
        print(f"  Recall:    {best_model_metrics['recall']:.4f}")
        print(f"  F1-Score:  {best_model_metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {best_model_metrics['roc_auc']:.4f}")
        print(f"\n  Confusion Matrix Shape: {best_model_metrics['cm_shape']}")

        import numpy as np
        cm_array = np.array(best_model_metrics['confusion_matrix'])
        print(f"  Confusion Matrix:")
        for row in cm_array:
            print(f"    {row}")
        print(f"{'='*80}")

        # ============================================================
        # OUTPUT LOCATIONS
        # ============================================================
        print("\n📁 Output Files:")
        print(f"{'='*80}")
        print(f"  Reports:        {reports_dir}/")
        print(f"  Models:         {Path(ml_system.models_dir)}/")
        print(f"  Visualizations: {Path(ml_system.output_dir) / 'visualizations'}/")
        print(f"\n  Key Files:")
        print(f"    - model_comparison.csv (from ml_system)")
        print(f"    - multiple_models_evaluation.csv (from evaluate_multiple_models) ← NEW!")
        print(f"    - LightGBM_batch_reports.json (all 10 folds with confusion matrices)")
        print(f"    - LightGBM_evaluation.json (detailed evaluation from ModelEvaluator)")
        print(f"    - sl_LightGBM_confusion_matrix.png (visualization)")
        print(f"{'='*80}")

        # ============================================================
        # NEXT STEPS
        # ============================================================
        print("\n📋 Next Steps:")
        print(f"{'='*80}")
        print("  1. Generate visualizations:")
        print("     python visualize_sl_models.py")
        print()
        print("  2. View model comparison (from ml_system):")
        print("     cat ml_outputs/reports/model_comparison.csv")
        print()
        print("  3. View multiple models evaluation (from evaluate_multiple_models):")
        print("     cat ml_outputs/reports/multiple_models_evaluation.csv")
        print()
        print("  4. View detailed evaluation (from ModelEvaluator):")
        print("     cat ml_outputs/reports/LightGBM_evaluation.json")
        print()
        print("  5. View all 10 fold confusion matrices:")
        print("     cat ml_outputs/reports/LightGBM_batch_reports.json | grep -A 5 'confusion_matrix'")
        print()
        print("  6. Train RL system with this baseline:")
        print("     python rl_train_main.py")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
