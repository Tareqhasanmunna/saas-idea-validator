"""
Complete Integration Example - SaaS Idea Validator ML Pipeline
=============================================================

This guide shows how all components work together with a complete example.

Run this as a Jupyter notebook or Python script to execute the full pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: IMPORT COMPONENTS
# ============================================================================

from ml_system import MLSystem
from evaluation import generate_final_evaluation_report, EvaluationReporter
from visualization import VisualizationGenerator

logger.info("All components imported successfully!")

# ============================================================================
# STEP 2: INITIALIZE AND CONFIGURE SYSTEM
# ============================================================================

def initialize_ml_system(config_path='config.yaml'):
    """Initialize the ML system with configuration"""
    ml_system = MLSystem(config_path=config_path)
    logger.info(f"ML System initialized")
    logger.info(f"Output directory: {ml_system.output_dir}")
    return ml_system

# ============================================================================
# STEP 3: LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare_data(ml_system, data_path, target_column='label_numeric'):
    """Load and prepare data for training"""

    # Load data
    X, y = ml_system.load_data(data_path, target_column=target_column)

    # Display data info
    logger.info(f"\nData Summary:")
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Target shape: {y.shape}")
    logger.info(f"  Feature types: {X.dtypes.value_counts().to_dict()}")
    logger.info(f"\n  Class Distribution:")
    logger.info(y.value_counts())

    # Preprocess (scale features)
    X_scaled = ml_system.preprocess_data(X, fit_scaler=True)

    return X, y, X_scaled

# ============================================================================
# STEP 4: BUILD AND TRAIN MODELS WITH 10-FOLD CV
# ============================================================================

def train_models(ml_system, X_scaled, y):
    """Train models using 10-fold cross-validation"""

    # Build models
    models = ml_system.build_models()
    logger.info(f"\nBuilt models: {list(models.keys())}")

    # Train with cross-validation
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODELS WITH 10-FOLD CROSS-VALIDATION")
    logger.info("="*70)

    cv_results, batch_reports = ml_system.train_with_cv(X_scaled, y)

    logger.info("\n✓ Training completed!")
    return cv_results, batch_reports

# ============================================================================
# STEP 5: GENERATE COMPREHENSIVE REPORTS
# ============================================================================

def generate_reports(ml_system, y):
    """Generate all reports and comparisons"""

    logger.info("\n" + "="*70)
    logger.info("GENERATING COMPREHENSIVE REPORTS")
    logger.info("="*70)

    # Summary reports for each model
    summary_reports = ml_system.generate_summary_reports(y)
    logger.info(f"\n✓ Generated {len(summary_reports)} summary reports")

    # Model comparison
    comparison_df = ml_system.generate_model_comparison(summary_reports)
    logger.info("\n✓ Model Comparison Table:")
    logger.info(comparison_df.to_string())

    # Final evaluation reports
    logger.info("\n" + "-"*70)
    logger.info("Generating detailed evaluation reports...")
    logger.info("-"*70)

    evaluation_summaries, detailed_comparison = generate_final_evaluation_report(
        ml_system, y
    )

    logger.info(f"✓ Generated {len(evaluation_summaries)} evaluation summaries")

    return summary_reports, comparison_df, evaluation_summaries

# ============================================================================
# STEP 6: GENERATE VISUALIZATIONS
# ============================================================================

def generate_visualizations(ml_system, y):
    """Generate all visualizations for thesis documentation"""

    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)

    viz_gen = VisualizationGenerator(ml_system.visualizations_dir)

    # 1. Class distribution
    logger.info("\n1. Generating class distribution plot...")
    y_array = y.values if hasattr(y, 'values') else y
    viz_gen.plot_class_distribution(y_array)

    # 2. Per-model visualizations
    for model_name, result in ml_system.cv_results.items():
        logger.info(f"\n2. Generating visualizations for {model_name}...")

        y_pred = result['predictions']
        y_pred_proba = result['probabilities']
        cv_scores = result['cv_scores']

        # Confusion matrix
        viz_gen.plot_confusion_matrix(y, y_pred, model_name)
        logger.info(f"   ✓ Confusion matrix")

        # ROC curve
        viz_gen.plot_roc_curve(y, y_pred_proba, model_name)
        logger.info(f"   ✓ ROC curve")

        # Precision-Recall curve
        viz_gen.plot_precision_recall_curve(y, y_pred_proba, model_name)
        logger.info(f"   ✓ Precision-Recall curve")

        # CV scores distribution
        viz_gen.plot_cv_scores_distribution(cv_scores, model_name)
        logger.info(f"   ✓ CV scores distribution")

        # Fold metrics
        viz_gen.plot_fold_metrics_heatmap(
            ml_system.batch_reports, model_name
        )
        logger.info(f"   ✓ Fold metrics heatmap")

    # 3. Model comparison
    logger.info(f"\n3. Generating model comparison plot...")
    comparison_df = pd.read_csv(ml_system.reports_dir / 'model_comparison.csv')
    viz_gen.plot_model_comparison(comparison_df)
    logger.info(f"   ✓ Model comparison")

    logger.info("\n✓ All visualizations generated!")

# ============================================================================
# STEP 7: SAVE MODELS FOR FUTURE USE
# ============================================================================

def save_trained_models(ml_system):
    """Save trained models for deployment/future use"""

    logger.info("\n" + "="*70)
    logger.info("SAVING TRAINED MODELS")
    logger.info("="*70)

    ml_system.save_models()
    logger.info("✓ All models saved to models/")

# ============================================================================
# STEP 8: GENERATE SUMMARY FOR THESIS
# ============================================================================

def print_thesis_summary(ml_system, comparison_df):
    """Print summary suitable for thesis documentation"""

    thesis_summary = f"""

{'='*70}
THESIS DOCUMENTATION SUMMARY
{'='*70}

SYSTEM CONFIGURATION:
  - Cross-Validation: 10-Fold Stratified
  - Models Evaluated: {len(ml_system.models)}
  - Total Cross-Validations: {len(ml_system.models) * ml_system.config.get('n_splits', 10)}

MODELS COMPARED:
{comparison_df.to_string()}

OUTPUT STRUCTURE:
  Reports Directory: {ml_system.reports_dir}/
    - model_comparison.csv (Overall model performance)
    - [model_name]_batch_reports.json (Per-fold metrics)
    - [model_name]_evaluation_summary.json (Comprehensive evaluation)
    - [model_name]_detailed_report.txt (Text format for thesis)

  Models Directory: {ml_system.models_dir}/
    - [model_name]_model.pkl (Trained models)
    - scaler.pkl (Feature scaler)

  Visualizations Directory: {ml_system.visualizations_dir}/
    - [model_name]_confusion_matrix.png
    - [model_name]_roc_curve.png
    - [model_name]_pr_curve.png
    - [model_name]_cv_scores.png
    - [model_name]_fold_metrics.png
    - model_comparison.png
    - class_distribution.png

FOR YOUR THESIS:
  1. Methods Section: Include SYSTEM_GUIDE.md content
  2. Results Section: Include comparison_df and visualizations
  3. Appendix: Include detailed reports and batch_reports.json
  4. Code Reference: Include config.yaml and ml_system.py snippets

{'='*70}
"""

    logger.info(thesis_summary)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(data_path, config_path='config.yaml'):
    """Execute complete pipeline"""

    try:
        logger.info("\n" + "="*70)
        logger.info("SAAS IDEA VALIDATOR - COMPLETE ML PIPELINE")
        logger.info("="*70)

        # Initialize
        ml_system = initialize_ml_system(config_path)

        # Load and prepare data
        X, y, X_scaled = load_and_prepare_data(ml_system, data_path)

        # Train models
        cv_results, batch_reports = train_models(ml_system, X_scaled, y)

        # Generate reports
        summary_reports, comparison_df, eval_summaries = generate_reports(
            ml_system, y
        )

        # Generate visualizations
        generate_visualizations(ml_system, y)

        # Save models
        save_trained_models(ml_system)

        # Print thesis summary
        print_thesis_summary(ml_system, comparison_df)

        logger.info("\n" + "="*70)
        logger.info("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\nAll outputs saved to: {ml_system.output_dir}")
        logger.info("Ready for thesis documentation!")

        return ml_system, summary_reports, comparison_df

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


# ============================================================================
# USAGE
# ============================================================================

if __name__ == '__main__':
    # Example usage:
    # ml_system, reports, comparison = main('your_data.csv', 'config.yaml')

    # For interactive notebook:
    # 1. Replace 'your_data.csv' with your actual data path
    # 2. Run: python integration_guide.py
    # 3. Check ml_outputs/ for all results

    print("Integration guide loaded. Call main() with your data path.")
    print("Example: main('preprocessed_data.csv', 'config.yaml')")
