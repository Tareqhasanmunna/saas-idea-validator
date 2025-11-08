import os
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, data_loader, report_dir="src/training/reports", 
                 model_save_dir="models", add_timestamp=False, logger_obj=None):
        self.data_loader = data_loader
        self.report_dir = report_dir
        self.model_save_dir = model_save_dir
        self.add_timestamp = add_timestamp
        self.logger = logger_obj or logger
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
    
    def _get_model_filename(self, model_name):
        if self.add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{model_name}_{timestamp}.joblib"
        else:
            return f"{model_name}.joblib"
    
    def run(self, models_dict, param_grids=None, cv_folds=10, return_trained_models=False):
        self.logger.info(f"[PIPELINE] Starting training for {len(models_dict)} models")
        
        try:
            X, y = self.data_loader.load_data()
            self.logger.info(f"[PIPELINE] Data loaded: X={X.shape}, y={y.shape}")
        except Exception as e:
            self.logger.error(f"[PIPELINE] Data loading error: {str(e)}")
            return {
                'results': {},
                'model_paths': {},
                'trained_models': {},
                'comparison_csv': None,
                'success': False,
                'errors': [str(e)]
            }
        
        results = {}
        model_paths = {}
        trained_models = {}
        errors = []
        
        for model_name, model in models_dict.items():
            try:
                self.logger.info(f"[PIPELINE] Training {model_name}...")
                
                if param_grids and model_name in param_grids:
                    self.logger.info(f"[TUNING] Tuning {model_name}...")
                    grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring="f1_weighted", n_jobs=-1)
                    grid.fit(X, y)
                    best_model = grid.best_estimator_
                    self.logger.info(f"[TUNING] Best params: {grid.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X, y)
                
                from src.training.sl_training.cross_validator_with_logging import CrossValidatorWithLogging
                avg_metrics = CrossValidatorWithLogging.perform_cross_validation(
                    model_name=model_name, model=best_model, X=X, y=y, 
                    cv_folds=cv_folds, log_dir=os.path.join(self.report_dir, model_name)
                )
                
                results[model_name] = avg_metrics
                
                model_filename = self._get_model_filename(model_name)
                model_file = os.path.join(self.model_save_dir, model_filename)
                joblib.dump(best_model, model_file)
                model_paths[model_name] = model_file
                
                self.logger.info(f"[SAVED] {model_name} → {model_file}")
                
                if return_trained_models:
                    trained_models[model_name] = best_model
            
            except Exception as e:
                error_msg = f"Error training {model_name}: {str(e)}"
                self.logger.error(f"[ERROR] {error_msg}")
                errors.append(error_msg)
        
        try:
            df = pd.DataFrame(results).T
            comparison_csv = os.path.join(self.report_dir, "model_comparison.csv")
            df.to_csv(comparison_csv)
            self.logger.info(f"[PIPELINE] Comparison saved: {comparison_csv}")
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            errors.append(error_msg)
            comparison_csv = None
        
        return {
            'results': results,
            'model_paths': model_paths,
            'trained_models': trained_models if return_trained_models else {},
            'comparison_csv': comparison_csv,
            'success': len(errors) == 0,
            'errors': errors
        }