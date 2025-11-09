import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.sl_training.data_loader import DataLoader
from src.training.sl_training.train_pipeline import TrainingPipeline
from src.training.sl_training.model_definitions import ModelFactory
import yaml
import warnings

warnings.filterwarnings("ignore")

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(PROJECT_ROOT, 'config.yaml')

# Load config
if not os.path.exists(config_path):
    raise FileNotFoundError(f"config.yaml not found at {config_path}")

with open(config_path, "r", encoding='utf-8') as f:
    CONFIG_YAML = yaml.safe_load(f)

# Build paths from config
CONFIG = {
    "data_path": os.path.join(PROJECT_ROOT, CONFIG_YAML['paths']['processed_data_dir'], "vectorised_dataset_*.csv"),
    "vector_col": "vector",
    "target_col": "label_numeric",
    "model_save_dir": os.path.join(PROJECT_ROOT, CONFIG_YAML['paths']['models_dir']),
    "report_dir": os.path.join(PROJECT_ROOT, CONFIG_YAML['paths']['reports_dir']),
    "best_model_dir": os.path.join(PROJECT_ROOT, CONFIG_YAML['paths']['best_sl_model_dir'])
}

def train_sl_model(logger_obj=None):
    """Train SL models and return results"""
    import logging
    logger = logger_obj or logging.getLogger(__name__)
    
    try:
        logger.info("[SL TRAINING] Starting SL training...")
        
        # Find latest processed CSV
        import glob
        data_files = sorted(glob.glob(
            os.path.join(PROJECT_ROOT, CONFIG_YAML['paths']['processed_data_dir'], "vectorised_dataset_*.csv")
        ))
        
        if not data_files:
            logger.error("[SL TRAINING] No processed data found")
            return {
                'success': False,
                'message': "No processed data found"
            }
        
        data_path = data_files[-1]
        logger.info(f"[SL TRAINING] Using data: {data_path}")
        
        # Create output directories
        os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
        os.makedirs(CONFIG["report_dir"], exist_ok=True)
        os.makedirs(CONFIG["best_model_dir"], exist_ok=True)
        
        # Load data
        logger.info("[SL TRAINING] Loading data...")
        data_loader = DataLoader(
            data_path,
            vector_col=CONFIG["vector_col"],
            target_col=CONFIG["target_col"]
        )
        
        # Initialize pipeline
        logger.info("[SL TRAINING] Initializing training pipeline...")
        pipeline = TrainingPipeline(
            data_loader,
            report_dir=CONFIG["report_dir"],
            model_save_dir=CONFIG["model_save_dir"],
            logger_obj=logger
        )
        
        # Get models
        logger.info("[SL TRAINING] Getting model definitions...")
        models_dict, param_grids = ModelFactory.get_models_and_grids()
        
        # Run training
        logger.info("[SL TRAINING] Running training, CV, and model selection...")
        results = pipeline.run(models_dict=models_dict, param_grids=param_grids, cv_folds=5)
        
        # Print results
        logger.info("[SL TRAINING] Training complete")
        logger.info("[SL TRAINING] Saved models:")
        for model_name in models_dict.keys():
            model_file = os.path.join(CONFIG["model_save_dir"], f"{model_name}.joblib")
            if os.path.exists(model_file):
                logger.info(f"  - {model_name}: {model_file}")
        
        return {
            'success': True,
            'message': "SL training completed successfully",
            'results': results
        }
    
    except Exception as e:
        logger.error(f"[SL TRAINING] Error: {str(e)}")
        return {
            'success': False,
            'message': f"SL training failed: {str(e)}"
        }

def main():
    """Main entry point for standalone execution"""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run training
    result = train_sl_model(logger_obj=logger)
    
    print("\n" + "="*80)
    print(f"Training Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Message: {result['message']}")
    print("="*80)

import subprocess

if __name__ == "__main__":
    main()
    # ✅ Auto-select the best SL model
    subprocess.run(["python", "src/training/sl_training/best_model_finder.py"])
