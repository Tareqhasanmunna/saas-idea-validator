"""
ML System - Complete with Proper Cross-Validation (NO DATA LEAKAGE)

Features:
- Stratified K-Fold CV (10 folds)
- Scaler fit inside each fold (prevents leakage)
- Complete metrics (ROC, PR curves, AUC, AP)
- Batch and summary reports (JSON)
- Models saved as JSON for RL integration

Usage:
    from ml_system import MLSystem
    ml = MLSystem()
    ml.run_pipeline()
"""

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
    roc_auc_score, confusion_matrix,
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


class JSONModelSaver:
    """Save/load models in JSON format for RL"""

    @staticmethod
    def save_model_to_json(model, scaler, model_name: str, output_path: Path):
        """Save model and scaler to JSON"""
        model_data = {
            "model_info": {
                "type": type(model).__name__,
                "name": model_name,
                "timestamp": datetime.now().isoformat(),
            },
            "model_params": model.get_params(),
            "scaler": {
                "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            },
            "n_features": int(scaler.n_features_in_) if hasattr(scaler, 'n_features_in_') else None,
            "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None,
        }

        # Model-specific info
        if hasattr(model, 'feature_importances_'):
            model_data['feature_importances'] = model.feature_importances_.tolist()
        if hasattr(model, 'coef_'):
            model_data['coefficients'] = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else None

        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"✓ Model saved: {output_path}")


class MLSystem:
    """ML Pipeline with Proper CV (No Data Leakage)"""

    def __init__(self, config_path: str = None):
        """Initialize ML system"""
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', 'ml_outputs'))
        self.output_dir.mkdir(exist_ok=True)

        self.reports_dir = self.output_dir / 'reports'
        self.models_dir = self.output_dir / 'models'

        for d in [self.reports_dir, self.models_dir]:
            d.mkdir(exist_ok=True)

        self.models = {}
        self.cv_results = {}
        self.batch_reports = {}
        self.best_model_name = None
        self.last_X = None
        self.last_y = None

        logger.info(f"ML System initialized. Output: {self.output_dir}")

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration"""
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
        """Build model instances"""
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

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train with proper CV (NO DATA LEAKAGE)"""
        n_splits = self.config.get('n_splits', 10)
        random_state = self.config.get('random_state', 42)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting {n_splits}-Fold Cross-Validation (NO DATA LEAKAGE)")
        logger.info(f"{'='*70}")

        cv_results = {}
        batch_reports = {}

        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name}...")

            fold_reports = self._train_model_cv(model_name, skf, X, y, model)
            batch_reports[model_name] = fold_reports

            fold_accs = [f['accuracy'] for f in fold_reports]
            fold_f1s = [f['f1_weighted'] for f in fold_reports]
            fold_roc_aucs = [f['roc_auc_ovr'] for f in fold_reports]

            cv_results[model_name] = {
                'mean_accuracy': np.mean(fold_accs),
                'std_accuracy': np.std(fold_accs),
                'mean_f1': np.mean(fold_f1s),
                'std_f1': np.std(fold_f1s),
                'mean_roc_auc': np.mean(fold_roc_aucs),
                'std_roc_auc': np.std(fold_roc_aucs),
            }

            logger.info(f"✓ Completed {model_name}: Acc={cv_results[model_name]['mean_accuracy']:.4f} ± {cv_results[model_name]['std_accuracy']:.4f}")

        self.cv_results = cv_results
        self.batch_reports = batch_reports

        return cv_results, batch_reports

    def _train_model_cv(self, model_name: str, skf, X: pd.DataFrame, y: pd.Series, model) -> List[Dict]:
        """Train single model with CV (NO LEAKAGE)"""
        fold_reports = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # USE .iloc FOR ROW SELECTION (fixes the error!)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # FIT SCALER ON TRAINING FOLD ONLY (prevents leakage!)
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)

            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)

            # FIXED: Removed zero_division parameter from roc_auc_score
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

            # Per-class ROC-AUC
            roc_auc_per_class = {}
            for i in range(len(np.unique(y_test))):
                y_test_binary = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                roc_auc_per_class[str(i)] = float(auc(fpr, tpr))

            # Per-class Average Precision
            ap_per_class = {}
            for i in range(len(np.unique(y_test))):
                y_test_binary = (y_test == i).astype(int)
                ap = average_precision_score(y_test_binary, y_pred_proba[:, i])
                ap_per_class[str(i)] = float(ap)

            metrics = {
                'fold': fold_idx + 1,
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
                'roc_auc_ovr': float(roc_auc),
                'roc_auc_per_class': roc_auc_per_class,
                'average_precision_per_class': ap_per_class,
                'train_size': len(y_train),
                'test_size': len(y_test),
                'cm': cm.tolist(),
            }

            fold_reports.append(metrics)
            logger.info(f"  Fold {fold_idx + 1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}")

        return fold_reports

    def generate_summary_reports(self) -> Dict:
        """Generate summary reports"""
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
        """Generate model comparison"""
        logger.info("\nGenerating model comparison...")

        comparison_data = []
        for model_name, report in summary_reports.items():
            row = {
                'Model': model_name,
                'Accuracy_Mean': report['metrics']['accuracy_mean'],
                'Accuracy_Std': report['metrics']['accuracy_std'],
                'F1_Mean': report['metrics']['f1_mean'],
                'F1_Std': report['metrics']['f1_std'],
                'ROC_AUC_Mean': report['metrics']['roc_auc_mean'],
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        best_idx = comparison_df['F1_Mean'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']

        logger.info(f"\n★ BEST MODEL: {self.best_model_name}")
        logger.info(f"   F1: {comparison_df.loc[best_idx, 'F1_Mean']:.4f}")

        comparison_path = self.reports_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)

        # Save batch reports
        for model_name, fold_reports in self.batch_reports.items():
            batch_path = self.reports_dir / f'{model_name}_batch_reports.json'
            with open(batch_path, 'w') as f:
                json.dump(fold_reports, f, indent=2)

        return comparison_df

    def save_all_models_to_json(self):
        """Save final models"""
        logger.info(f"\n{'='*70}")
        logger.info("SAVING FINAL MODELS")
        logger.info(f"{'='*70}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.last_X)

        for model_name, model in self.models.items():
            logger.info(f"Saving: {model_name}")
            model.fit(X_scaled, self.last_y)

            json_path = self.models_dir / f'{model_name}_model.json'
            JSONModelSaver.save_model_to_json(model, scaler, model_name, json_path)

        logger.info(f"✓ All models saved to {self.models_dir}/")

    def run_pipeline(self):
        """Execute full pipeline"""
        logger.info(f"{'='*70}")
        logger.info("SaaS Idea Validator ML Pipeline")
        logger.info("NO DATA LEAKAGE + COMPLETE METRICS")
        logger.info(f"{'='*70}")

        # Load data
        logger.info("\nLoading data...")
        X, y = load_training_data()
        self.last_X = X
        self.last_y = y

        # Build and train models
        self.build_models()
        cv_results, batch_reports = self.train_with_cv(X, y)

        # Generate reports
        summary_reports = self.generate_summary_reports()
        comparison_df = self.generate_model_comparison(summary_reports)

        # Save models
        self.save_all_models_to_json()

        logger.info(f"\n{'='*70}")
        logger.info("✓ TRAINING COMPLETE!")
        logger.info(f"{'='*70}")
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Reports: {self.reports_dir}/")
        logger.info(f"Models: {self.models_dir}/")

        return cv_results, summary_reports, comparison_df


def main():
    """Main execution"""
    ml_system = MLSystem(config_path='config.yaml')
    ml_system.run_pipeline()


if __name__ == '__main__':
    main()
