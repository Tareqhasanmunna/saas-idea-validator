import os
import pandas as pd
import shutil
import logging
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

DEFAULT_REPORT_CSV = 'reports/model_comparison.csv'
DEFAULT_MODELS_FOLDER = 'models'
DEFAULT_BEST_MODEL_DIR = 'models/best_sl_model'

def find_and_copy_best_model(report_csv=None, models_folder=None, best_model_dest=None,
                             metric="accuracy", logger_obj=None):
    log = logger_obj or logger
    
    report_csv = report_csv or DEFAULT_REPORT_CSV
    models_folder = models_folder or DEFAULT_MODELS_FOLDER
    best_model_dest = best_model_dest or DEFAULT_BEST_MODEL_DIR
    
    log.info(f"[BEST MODEL] Finding best model...")
    
    try:
        if not os.path.exists(report_csv):
            error_msg = f"Report not found: {report_csv}"
            log.error(f"[BEST MODEL] {error_msg}")
            return {'success': False, 'best_model_name': None, 'best_model_path': None, 'metric_value': None, 'errors': [error_msg]}
        
        log.info(f"[BEST MODEL] Reading comparison CSV...")
        df = pd.read_csv(report_csv)
        
        if metric not in df.columns:
            log.warning(f"[BEST MODEL] Metric '{metric}' not found, using 'accuracy'")
            metric = 'accuracy' if 'accuracy' in df.columns else df.columns[0]
        
        best_idx = df[metric].idxmax()
        best_model_row = df.loc[best_idx]
        best_model_name = best_model_row.get('model_name', df.index[best_idx])
        metric_value = best_model_row[metric]
        
        if not best_model_name.endswith('.joblib'):
            best_model_name = f"{best_model_name}.joblib"
        
        source_path = os.path.join(models_folder, best_model_name)
        
        if not os.path.exists(source_path):
            error_msg = f"Model not found: {source_path}"
            log.error(f"[BEST MODEL] {error_msg}")
            return {'success': False, 'best_model_name': None, 'best_model_path': None, 'metric_value': metric_value, 'errors': [error_msg]}
        
        os.makedirs(best_model_dest, exist_ok=True)
        destination_path = os.path.join(best_model_dest, best_model_name)
        
        try:
            shutil.copy2(source_path, destination_path)
            log.info(f"[BEST MODEL] ✓ Copied: {destination_path}")
            return {'success': True, 'best_model_name': best_model_name, 'best_model_path': destination_path, 'metric_value': metric_value, 'errors': []}
        except Exception as e:
            error_msg = f"Copy failed: {str(e)}"
            log.error(f"[BEST MODEL] {error_msg}")
            return {'success': False, 'best_model_name': None, 'best_model_path': None, 'metric_value': metric_value, 'errors': [error_msg]}
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        log.error(f"[BEST MODEL] {error_msg}")
        return {'success': False, 'best_model_name': None, 'best_model_path': None, 'metric_value': None, 'errors': [error_msg]}

def get_model_metrics(report_csv, metric="accuracy"):
    try:
        if not os.path.exists(report_csv):
            return {}
        df = pd.read_csv(report_csv)
        if metric in df.columns:
            return df.set_index(df.index.map(str))[metric].to_dict()
        return {}
    except:
        return {}

def compare_model_performance(report_csv, top_n=5):
    try:
        if not os.path.exists(report_csv):
            return None
        df = pd.read_csv(report_csv)
        if 'accuracy' in df.columns:
            return df.nlargest(top_n, 'accuracy')
        return df.head(top_n)
    except:
        return None

def main():
    print("\n[BEST MODEL FINDER] Starting...")
    result = find_and_copy_best_model()
    if result['success']:
        print(f"✓ Best model: {result['best_model_name']} (accuracy={result['metric_value']:.4f})")
    else:
        print(f"✗ Failed: {result['errors']}")

if __name__ == "__main__":
    main()