import os
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from src.training.sl_training.cross_validator_with_logging import CrossValidatorWithLogging

class TrainingPipeline:
    def __init__(self, data_loader, report_dir="reports", model_save_dir="E:/saas-idea-validator/models"):
        self.data_loader = data_loader
        self.report_dir = report_dir
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def run(self, models_dict, param_grids=None, cv_folds=10):
        X, y = self.data_loader.load_data()  # X: numeric + vectors, y: label_numeric

        results = {}
        for model_name, model in models_dict.items():
            print(f"\n[MODEL] {model_name} Training Started")

            # Hyperparameter tuning
            if param_grids and model_name in param_grids:
                print(f"[TUNING] {model_name} hyperparameter tuning...")
                grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring="f1_weighted", n_jobs=-1)
                grid.fit(X, y)
                best_model = grid.best_estimator_
                print(f"[SUCCESS] Best params: {grid.best_params_}")
            else:
                best_model = model
                best_model.fit(X, y)

            # 10-fold CV with logging
            avg_metrics = CrossValidatorWithLogging.perform_cross_validation(
                model_name=model_name,
                model=best_model,
                X=X,
                y=y,
                cv_folds=cv_folds,
                log_dir=os.path.join(self.report_dir, model_name)
            )
            results[model_name] = avg_metrics

            # Save trained model
            model_file = os.path.join(self.model_save_dir, f"{model_name}_model.joblib")
            joblib.dump(best_model, model_file)
            print(f"[SAVED] {model_name} → {model_file}")

        # Generate comparison table
        df = pd.DataFrame(results).T
        df.to_csv(os.path.join(self.report_dir, "model_comparison.csv"))
        print("\n[INFO] Comparison table saved at model_comparison.csv")
        return results
