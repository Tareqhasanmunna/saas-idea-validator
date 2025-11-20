import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from data_loader import load_training_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLSystem:
    """ML Pipeline with Proper CV (No Data Leakage) and Precision"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', 'ml_outputs'))
        self.output_dir.mkdir(exist_ok=True)

        self.reports_dir = self.output_dir / 'reports'
        self.models_dir = self.output_dir / 'models'

        for d in [self.reports_dir, self.models_dir]:
            d.mkdir(exist_ok=True)

        self.models: Dict[str, object] = {}
        self.cv_results: Dict = {}
        self.batch_reports: Dict = {}
        self.best_model_name: str = None

        # Initialize last_X and last_y as None; will assign during training
        self.last_X: pd.DataFrame = None
        self.last_y: pd.Series = None

        logger.info(f"ML System initialized. Output directory: {self.output_dir}")

    def _load_config(self, config_path: str = None) -> Dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        return {
            'output_dir': 'ml_outputs',
            'n_splits': 10,
            'random_state': 42,
            'models': {
                'DecisionTree': {'max_depth': 10, 'random_state': 42},
                'LogisticRegression': {'max_iter': 1000, 'solver': 'lbfgs'},
                'RandomForest': {'n_estimators': 100, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1},
                'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
                'LightGBM': {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42, 'verbose': -1}
            }
        }

    def build_models(self) -> Dict:
        model_configs = self.config.get('models', {})

        models = {
            'DecisionTree': DecisionTreeClassifier(**model_configs.get('DecisionTree', {})),
            'LogisticRegression': LogisticRegression(**model_configs.get('LogisticRegression', {})),
            'RandomForest': RandomForestClassifier(**model_configs.get('RandomForest', {})),
            'GradientBoosting': GradientBoostingClassifier(**model_configs.get('GradientBoosting', {})),
        }

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(**model_configs.get('LightGBM', {}))
            logger.info("✓ Using 5 models: DT, LogReg, RF, GB, LightGBM")
        else:
            logger.warning("⚠ LightGBM not available, using 4 models")

        self.models = models
        return models

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        n_splits = self.config.get('n_splits', 10)
        random_state = self.config.get('random_state', 42)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting {n_splits}-Fold Cross-Validation (NO DATA LEAKAGE)")
        logger.info(f"{'='*70}")

        cv_results = {}
        batch_reports = {}

        # Save copy of training data safely
        self.last_X = X.copy()
        self.last_y = y.copy()

        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name}...")
            fold_reports = self._train_model_cv(model_name, skf, X, y, model)
            batch_reports[model_name] = fold_reports

            fold_accs = [f['accuracy'] for f in fold_reports]
            fold_prec = [f['precision'] for f in fold_reports]
            fold_rec = [f['recall'] for f in fold_reports]
            fold_f1s = [f['f1_weighted'] for f in fold_reports]
            fold_roc_aucs = [f['roc_auc_ovr'] for f in fold_reports]

            cv_results[model_name] = {
                'mean_accuracy': np.mean(fold_accs),
                'std_accuracy': np.std(fold_accs),
                'mean_precision': np.mean(fold_prec),
                'std_precision': np.std(fold_prec),
                'mean_recall': np.mean(fold_rec),
                'std_recall': np.std(fold_rec),
                'mean_f1': np.mean(fold_f1s),
                'std_f1': np.std(fold_f1s),
                'mean_roc_auc': np.mean(fold_roc_aucs),
                'std_roc_auc': np.std(fold_roc_aucs),
            }

            logger.info(f"✓ Completed {model_name}:")
            logger.info(f"   Accuracy: {cv_results[model_name]['mean_accuracy']:.4f} ± {cv_results[model_name]['std_accuracy']:.4f}")
            logger.info(f"   Precision: {cv_results[model_name]['mean_precision']:.4f} ± {cv_results[model_name]['std_precision']:.4f}")
            logger.info(f"   Recall: {cv_results[model_name]['mean_recall']:.4f} ± {cv_results[model_name]['std_recall']:.4f}")
            logger.info(f"   F1 Score: {cv_results[model_name]['mean_f1']:.4f} ± {cv_results[model_name]['std_f1']:.4f}")

        self.cv_results = cv_results
        self.batch_reports = batch_reports
        return cv_results, batch_reports

    def _train_model_cv(self, model_name: str, skf, X: pd.DataFrame, y: pd.Series, model) -> List[Dict]:
        fold_reports = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            cm = confusion_matrix(y_test, y_pred)
            y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

            accuracy = accuracy_score(y_test_array, y_pred)
            precision = precision_score(y_test_array, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_array, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_array, y_pred, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(y_test_array, y_pred_proba, multi_class='ovr')

            class_report = classification_report(y_test_array, y_pred, output_dict=True, zero_division=0)
            roc_auc_per_class = {}
            ap_per_class = {}
            for i in range(len(np.unique(y_test_array))):
                y_test_binary = (y_test_array == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                roc_auc_per_class[str(i)] = float(auc(fpr, tpr))
                ap = average_precision_score(y_test_binary, y_pred_proba[:, i])
                ap_per_class[str(i)] = float(ap)

            # Precision-Recall Curve
            precision_recall_curves = {}
            for i in range(len(np.unique(y_test_array))):
                y_test_binary = (y_test_array == i).astype(int)
                precision_vals, recall_vals, _ = precision_recall_curve(
                    y_test_binary, y_pred_proba[:, i]
                )
                precision_recall_curves[str(i)] = {
                    'precision': precision_vals.tolist(),
                    'recall': recall_vals.tolist()
                }

            metrics = {
                'fold': fold_idx + 1,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_weighted': float(f1),
                'roc_auc_ovr': float(roc_auc),
                'roc_auc_per_class': roc_auc_per_class,
                'average_precision_per_class': ap_per_class,
                'classification_report': class_report,
                'train_size': len(y_train),
                'test_size': len(y_test),
                'cm': cm.tolist(),
                'precision_recall_curves': precision_recall_curves,
            }

            fold_reports.append(metrics)
            logger.info(f"  Fold {fold_idx + 1}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

        return fold_reports

    def generate_summary_reports(self) -> Dict:
        logger.info("\nGenerating summary reports...")
        summary_reports = {}
        for model_name, result in self.cv_results.items():
            summary = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'n_splits': self.config.get('n_splits', 10),
                'metrics': {
                    'accuracy_mean': float(result['mean_accuracy']),
                    'accuracy_std': float(result['std_accuracy']),
                    'precision_mean': float(result['mean_precision']),
                    'precision_std': float(result['std_precision']),
                    'recall_mean': float(result['mean_recall']),
                    'recall_std': float(result['std_recall']),
                    'f1_mean': float(result['mean_f1']),
                    'f1_std': float(result['std_f1']),
                    'roc_auc_mean': float(result['mean_roc_auc']),
                    'roc_auc_std': float(result['std_roc_auc']),
                }
            }
            summary_reports[model_name] = summary
            report_path = self.reports_dir / f'{model_name}_summary_report.json'
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary: {report_path}")
        return summary_reports

    def generate_model_comparison(self, summary_reports: Dict) -> pd.DataFrame:
        logger.info("\nGenerating model comparison...")
        comparison_data = []
        for model_name, report in summary_reports.items():
            row = {
                'Model': model_name,
                'Accuracy_Mean': report['metrics']['accuracy_mean'],
                'Accuracy_Std': report['metrics']['accuracy_std'],
                'Precision_Mean': report['metrics']['precision_mean'],
                'Precision_Std': report['metrics']['precision_std'],
                'Recall_Mean': report['metrics']['recall_mean'],
                'Recall_Std': report['metrics']['recall_std'],
                'F1_Mean': report['metrics']['f1_mean'],
                'F1_Std': report['metrics']['f1_std'],
                'ROC_AUC_Mean': report['metrics']['roc_auc_mean'],
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        best_idx = comparison_df['F1_Mean'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']

        logger.info(f"\n★ BEST MODEL: {self.best_model_name}")
        logger.info(f"   Accuracy: {comparison_df.loc[best_idx, 'Accuracy_Mean']:.4f}")
        logger.info(f"   Precision: {comparison_df.loc[best_idx, 'Precision_Mean']:.4f}")
        logger.info(f"   Recall: {comparison_df.loc[best_idx, 'Recall_Mean']:.4f}")
        logger.info(f"   F1: {comparison_df.loc[best_idx, 'F1_Mean']:.4f}")
        return comparison_df

    def save_all_models_to_json(self, X=None):
        data_to_scale = X if X is not None else self.last_X
        if data_to_scale is None:
            raise ValueError("Training data for scaling not provided and self.last_X is None.")

        if isinstance(data_to_scale, float) and np.isnan(data_to_scale):
            raise ValueError("Training data is NaN, expected 2D feature array.")

        if len(getattr(data_to_scale, 'shape', ())) != 2:
            raise ValueError(f"Expected 2D feature array for scaling, got shape {getattr(data_to_scale, 'shape', None)}.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_to_scale)

        self.models_dir.mkdir(exist_ok=True)
        for model_name, model in self.models.items():
            model_json_path = self.models_dir / f"{model_name}_model.json"
            model_data = {
                'model_name': model_name,
                'params': model.get_params(),
                'timestamp': datetime.now().isoformat(),
            }
            with open(model_json_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info(f"Saved {model_name} to JSON: {model_json_path}")

    def run_pipeline(self):
        logger.info(f"{'='*70}")
        logger.info("SaaS Idea Validator - ML Pipeline")
        logger.info("No Data Leakage + Complete Metrics (Including Precision)")
        logger.info(f"{'='*70}")

        logger.info("\nLoading data...")
        X, y = load_training_data()
        self.last_X = X
        self.last_y = y

        self.build_models()
        cv_results, batch_reports = self.train_with_cv(X, y)
        summary_reports = self.generate_summary_reports()
        comparison_df = self.generate_model_comparison(summary_reports)

        self.save_all_models_to_json(X)

        logger.info(f"\n{'='*70}")
        logger.info("✓ Training Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Best Model: {self.best_model_name}")
        logger.info(f"Reports stored in: {self.reports_dir}")
        logger.info(f"Models stored in: {self.models_dir}")

        return cv_results, summary_reports, comparison_df


def main():
    ml_system = MLSystem(config_path='config.yaml')
    ml_system.run_pipeline()


if __name__ == '__main__':
    main()
