"""
Enhanced main_sl.py with Thesis Evaluation System Integration

This upgraded version includes:
1. Your existing SL training pipeline
2. Integrated SLEvaluator for automatic visualization
3. Export of all thesis-ready artifacts
4. Automatic best model identification and evaluation
"""

import os
import warnings
import joblib
from pathlib import Path
from datetime import datetime

# Your existing imports
from src.training.sl_training.data_loader import DataLoader
from src.training.sl_training.train_pipeline import TrainingPipeline
from src.training.sl_training.model_definitions import ModelFactory
from src.training.sl_training.best_model_finder import find_and_copy_best_model

# New: Thesis Evaluation System
from src.training.sl_training.sl_evaluation import SLEvaluator

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': r'E:\saas-idea-validator\data\processed\vectorized_dataset.csv',
    'vector_col': 'vector',
    'target_col': 'label_numeric',
    'model_save_dir': r'E:\saas-idea-validator\models',
    'report_dir': r'E:\saas-idea-validator\reports',
    'thesis_eval_dir': r'E:\saas-idea-validator\thesis_eval\sl_results',
    'test_size': 0.15,
    'validation_size': 0.15,
    'cv_folds': 10,
}


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_and_prepare_data():
    """Load data and split into train/val/test"""
    print("\n" + "="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80 + "\n")
    
    data_loader = DataLoader(
        CONFIG['data_path'],
        vector_col=CONFIG['vector_col'],
        target_col=CONFIG['target_col']
    )
    
    X, y = data_loader.load_data()
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split into train/validation/test
    from sklearn.model_selection import train_test_split
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Split temp into val and test (50-50 of 30% = 15% each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    return data_loader, X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# STEP 2: TRAIN ALL MODELS
# ============================================================================

def train_all_models(data_loader):
    """Train all SL models using existing pipeline"""
    print("\n" + "="*80)
    print("STEP 2: TRAINING ALL MODELS")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        data_loader,
        report_dir=CONFIG['report_dir'],
        model_save_dir=CONFIG['model_save_dir']
    )
    
    # Get all models and hyperparameter grids
    models_dict, param_grids = ModelFactory.get_models_and_grids()
    
    # Run training with cross-validation
    results = pipeline.run(
        models_dict=models_dict,
        param_grids=param_grids,
        cv_folds=CONFIG['cv_folds']
    )
    
    print(f"\n✓ Training complete! {len(models_dict)} models trained.")
    
    return models_dict, results


# ============================================================================
# STEP 3: FIND BEST MODEL
# ============================================================================

def find_best_model():
    """Find and copy best model using existing best_model_finder"""
    print("\n" + "="*80)
    print("STEP 3: FINDING BEST MODEL")
    print("="*80 + "\n")
    
    result = find_and_copy_best_model(
        report_csv=os.path.join(CONFIG['report_dir'], 'model_comparison.csv'),
        models_folder=CONFIG['model_save_dir'],
        best_model_dest=os.path.join(CONFIG['model_save_dir'], 'best_sl_model'),
        metric='accuracy'
    )
    
    if result['success']:
        print(f"\n✓ Best model found: {result['best_model_name']}")
        print(f"✓ Accuracy: {result['metric_value']:.4f}")
        print(f"✓ Location: {result['best_model_path']}")
        return result['best_model_path']
    else:
        print(f"✗ Error finding best model: {result['errors']}")
        return None


# ============================================================================
# STEP 4: EVALUATE ALL MODELS (THESIS)
# ============================================================================

def evaluate_models_for_thesis(models_dict, X_test, y_test):
    """
    Evaluate all trained models and generate thesis visualizations
    This is the key integration point with SLEvaluator
    """
    print("\n" + "="*80)
    print("STEP 4: THESIS EVALUATION - GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Create evaluator
    evaluator = SLEvaluator(output_dir=CONFIG['thesis_eval_dir'])
    
    # Load trained models from disk
    print("Loading trained models...")
    loaded_models = {}
    for model_name in models_dict.keys():
        model_path = os.path.join(
            CONFIG['model_save_dir'],
            f"{model_name}_model.joblib"
        )
        if os.path.exists(model_path):
            loaded_models[model_name] = joblib.load(model_path)
            print(f"  ✓ Loaded {model_name}")
        else:
            print(f"  ✗ Model not found: {model_path}")
    
    if not loaded_models:
        print("✗ No models to evaluate!")
        return None
    
    # Compare all models
    print("\nEvaluating models on test set...")
    comparison_df = evaluator.compare_models(loaded_models, X_test, y_test)
    
    print("\n" + "-"*80)
    print("MODEL COMPARISON RESULTS")
    print("-"*80)
    print(comparison_df.to_string())
    print("-"*80 + "\n")
    
    # Generate all visualizations and exports
    print("Generating thesis artifacts...")
    
    evaluator.save_comparison_csv()
    print("  ✓ Saved comparison CSV")
    
    evaluator.save_detailed_metrics_json()
    print("  ✓ Saved detailed metrics JSON")
    
    evaluator.plot_model_comparison()
    print("  ✓ Generated model comparison chart")
    
    evaluator.plot_confusion_matrices()
    print("  ✓ Generated confusion matrices")
    
    evaluator.plot_roc_curves()
    print("  ✓ Generated ROC curves")
    
    evaluator.generate_summary_report()
    print("  ✓ Generated summary report")
    
    print(f"\n✓ All thesis artifacts saved to: {CONFIG['thesis_eval_dir']}")
    
    return evaluator, comparison_df


# ============================================================================
# STEP 5: SUMMARY REPORT
# ============================================================================

def print_summary_report(best_model_path, comparison_df):
    """Print final summary"""
    print("\n" + "="*80)
    print("FINAL SUMMARY - SUPERVISED LEARNING TRAINING")
    print("="*80 + "\n")
    
    if best_model_path:
        best_model = joblib.load(best_model_path)
        print(f"✓ Best model: {os.path.basename(best_model_path)}")
    
    print(f"\n✓ All models trained: {len(comparison_df)}")
    print(f"✓ Best accuracy: {comparison_df['accuracy'].max():.4f}")
    print(f"✓ Worst accuracy: {comparison_df['accuracy'].min():.4f}")
    print(f"✓ Average accuracy: {comparison_df['accuracy'].mean():.4f}")
    
    print(f"\n✓ Thesis artifacts generated:")
    print(f"  - Model comparison chart (PNG)")
    print(f"  - Confusion matrices (PNG)")
    print(f"  - ROC curves (PNG)")
    print(f"  - Metrics CSV")
    print(f"  - Detailed JSON metrics")
    print(f"  - Summary report (TXT)")
    
    print(f"\n✓ Location: {CONFIG['thesis_eval_dir']}")
    
    print("\n" + "="*80)
    print("SL TRAINING COMPLETE - Ready for RL integration")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete SL training and thesis evaluation pipeline"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SAAS IDEA VALIDATOR - SL TRAINING" + " "*25 + "║")
    print("║" + " "*16 + "with Thesis Evaluation System Integration" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Step 1: Load data
        data_loader, X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()
        
        # Step 2: Train models
        models_dict, results = train_all_models(data_loader)
        
        # Step 3: Find best model
        best_model_path = find_best_model()
        
        # Step 4: Evaluate for thesis
        evaluator, comparison_df = evaluate_models_for_thesis(models_dict, X_test, y_test)
        
        # Step 5: Summary
        print_summary_report(best_model_path, comparison_df)
        
        print("✓ SUCCESS! SL training and thesis evaluation complete.")
        print(f"✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
