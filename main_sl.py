import os
import warnings
import yaml
import joblib
from datetime import datetime
import logging

warnings.filterwarnings("ignore")

from src.training.sl_training.data_loader import DataLoader
from src.training.sl_training.train_pipeline import TrainingPipeline
from src.training.sl_training.model_definitions import ModelFactory
from src.training.sl_training.best_model_finder import find_and_copy_best_model

config_path = os.path.join(os.path.dirname(__file__), '../..', 'config.yaml')
with open(config_path, "r", encoding='utf-8') as f:
    CONFIG_FILE = yaml.safe_load(f)

logger = logging.getLogger(__name__)

CONFIG = {
    "data_path": "data/processed/vectorised_dataset.csv",
    "vector_col": "vector",
    "target_col": "label_numeric",
    "model_save_dir": "models",
    "report_dir": "src/training/reports"
}

def train_sl_model(data_path=None, vector_col=None, target_col=None, 
                   model_save_dir=None, report_dir=None, cv_folds=10, logger_obj=None):
    data_path = data_path or CONFIG["data_path"]
    vector_col = vector_col or CONFIG["vector_col"]
    target_col = target_col or CONFIG["target_col"]
    model_save_dir = model_save_dir or CONFIG["model_save_dir"]
    report_dir = report_dir or CONFIG["report_dir"]
    
    log = logger_obj or logger
    
    log.info(f"[SL TRAIN] Starting SL model training...")
    
    try:
        log.info(f"[SL TRAIN] Loading data...")
        data_loader = DataLoader(data_path, vector_col=vector_col, target_col=target_col)
        
        log.info(f"[SL TRAIN] Initializing training pipeline...")
        pipeline = TrainingPipeline(data_loader, report_dir=report_dir, model_save_dir=model_save_dir)
        
        log.info(f"[SL TRAIN] Loading model definitions...")
        models_dict, param_grids = ModelFactory.get_models_and_grids()
        
        log.info(f"[SL TRAIN] Running training pipeline...")
        results = pipeline.run(models_dict=models_dict, param_grids=param_grids, cv_folds=cv_folds)
        
        comparison_csv = os.path.join(report_dir, "model_comparison.csv")
        best_model_path = find_and_copy_best_model()["best_model_path"] if find_and_copy_best_model() else None
        
        if best_model_path is None:
            raise Exception("No best model found")
        
        best_model_name = os.path.basename(best_model_path)
        
        log.info(f"[SL TRAIN] Best model: {best_model_name}")
        
        return {
            "success": True,
            "models_trained": len(models_dict),
            "best_model_name": best_model_name,
            "best_model_path": best_model_path,
            "comparison_csv": comparison_csv,
            "errors": []
        }
    
    except Exception as e:
        error_msg = f"[SL TRAIN] Error: {str(e)}"
        log.error(error_msg)
        return {
            "success": False,
            "models_trained": 0,
            "best_model_name": None,
            "best_model_path": None,
            "comparison_csv": None,
            "errors": [error_msg]
        }

def find_best_sl_model(comparison_csv, models_folder, metric="accuracy"):
    try:
        import pandas as pd
        df = pd.read_csv(comparison_csv)
        best_idx = df[metric].idxmax()
        best_model_name = df.loc[best_idx, 'model_name']
        if not best_model_name.endswith('.joblib'):
            best_model_name += '.joblib'
        best_model_path = os.path.join(models_folder, best_model_name)
        return best_model_path if os.path.exists(best_model_path) else None
    except:
        return None

def load_best_sl_model(best_model_path):
    try:
        if not os.path.exists(best_model_path):
            logger.error(f"Model not found: {best_model_path}")
            return None
        model = joblib.load(best_model_path)
        logger.info(f"Loaded: {best_model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading: {str(e)}")
        return None

def copy_best_model_to_serving_dir(best_model_path, serving_dir):
    try:
        import shutil
        os.makedirs(serving_dir, exist_ok=True)
        model_name = os.path.basename(best_model_path)
        dest_path = os.path.join(serving_dir, model_name)
        shutil.copy2(best_model_path, dest_path)
        logger.info(f"Copied to serving: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Copy error: {str(e)}")
        return None

def main():
    result = train_sl_model()
    if result["success"]:
        print(f"✓ Training complete: {result['best_model_name']}")
    else:
        print(f"✗ Training failed: {result['errors']}")

if __name__ == "__main__":
    main()