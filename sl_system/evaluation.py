"""
Evaluation Module - SL Model Evaluation with Precision, Recall, Confusion Matrix

Comprehensive evaluation of trained models including:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix
- Per-class metrics
- Comparison with baseline

Usage:
    from evaluation import ModelEvaluator
    evaluator = ModelEvaluator(model, scaler, X_test, y_test)
    metrics = evaluator.evaluate()
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from typing import Dict, Tuple
import json


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics"""

    def __init__(self, model, scaler, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"):
        """
        Initialize evaluator

        Args:
            model: Trained sklearn model
            scaler: Fitted StandardScaler
            X_test: Test features
            y_test: Test labels
            model_name: Name of model for reporting
        """
        self.model = model
        self.scaler = scaler
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.predictions = None
        self.probabilities = None
        self.metrics = None

    def predict(self):
        """Make predictions on test set"""
        X_test_scaled = self.scaler.transform(self.X_test)
        self.predictions = self.model.predict(X_test_scaled)
        self.probabilities = self.model.predict_proba(X_test_scaled)
        return self.predictions

    def evaluate(self) -> Dict:
        """
        Evaluate model on all metrics

        Returns:
            metrics: Dict with accuracy, precision, recall, f1, roc_auc, confusion matrix
        """
        if self.predictions is None:
            self.predict()

        y_test_array = self.y_test if isinstance(self.y_test, np.ndarray) else self.y_test.values

        # ✓ Calculate all metrics
        accuracy = accuracy_score(y_test_array, self.predictions)
        precision = precision_score(y_test_array, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test_array, self.predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test_array, self.predictions, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test_array, self.probabilities, multi_class='ovr')

        # ✓ Confusion Matrix
        cm = confusion_matrix(y_test_array, self.predictions)

        # ✓ Per-class metrics
        class_report = classification_report(y_test_array, self.predictions, output_dict=True, zero_division=0)
        
        # ✅ USE roc_curve and auc
        roc_curves = {}
        for i in range(len(np.unique(y_test_array))):
            y_test_binary = (y_test_array == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, self.probabilities[:, i])
            roc_auc_score_val = auc(fpr, tpr)
            roc_curves[str(i)] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc_score_val)
            }


        self.metrics = {
            'model_name': self.model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),           # ✓ NEW
            'recall': float(recall),                 # ✓ NEW
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),         # ✓ NEW
            'cm_shape': cm.shape,
            'classification_report': class_report,
            'roc_curves': roc_curves,
   # ✓ NEW
        }

        return self.metrics

    def print_metrics(self):
        """Print metrics in readable format"""
        if self.metrics is None:
            self.evaluate()

        print(f"\n{'='*70}")
        print(f"Model Evaluation: {self.model_name}")
        print(f"{'='*70}")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")  # ✓ NEW
        print(f"Recall:    {self.metrics['recall']:.4f}")     # ✓ NEW
        print(f"F1-Score:  {self.metrics['f1']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")

        print(f"\nConfusion Matrix (3×3):")  # ✓ NEW
        cm = np.array(self.metrics['confusion_matrix'])
        print(cm)

        print(f"\nPer-Class Metrics:")  # ✓ NEW
        for class_id in ['0', '1', '2']:
            class_metrics = self.metrics['classification_report'].get(class_id, {})
            if class_metrics:
                print(f"  Class {class_id}: Precision={class_metrics.get('precision', 0):.4f}, "
                      f"Recall={class_metrics.get('recall', 0):.4f}, "
                      f"F1={class_metrics.get('f1-score', 0):.4f}")
        print(f"{'='*70}\n")

    def get_confusion_matrix_analysis(self) -> Dict:
        """
        Analyze confusion matrix for insights

        Returns:
            analysis: Dict with FP, FN, TP, TN for each class
        """
        if self.metrics is None:
            self.evaluate()

        cm = np.array(self.metrics['confusion_matrix'])
        analysis = {}

        for i in range(3):  # 3 classes: Bad, Neutral, Good
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            class_names = {0: 'Bad', 1: 'Neutral', 2: 'Good'}
            analysis[class_names[i]] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
            }

        return analysis

    def compare_with_baseline(self, baseline_metrics: Dict) -> Dict:
        """
        Compare current metrics with baseline

        Args:
            baseline_metrics: Dict with baseline accuracy, precision, recall, f1

        Returns:
            comparison: Dict with improvements
        """
        if self.metrics is None:
            self.evaluate()

        comparison = {
            'current': {
                'accuracy': self.metrics['accuracy'],
                'precision': self.metrics['precision'],
                'recall': self.metrics['recall'],
                'f1': self.metrics['f1'],
            },
            'baseline': baseline_metrics,
            'improvements': {
                'accuracy': self.metrics['accuracy'] - baseline_metrics.get('accuracy', 0),
                'precision': self.metrics['precision'] - baseline_metrics.get('precision', 0),
                'recall': self.metrics['recall'] - baseline_metrics.get('recall', 0),
                'f1': self.metrics['f1'] - baseline_metrics.get('f1', 0),
            }
        }

        print(f"\nComparison with Baseline:")
        print(f"{'='*70}")
        print(f"{'Metric':<15} {'Current':<15} {'Baseline':<15} {'Improvement':<15}")
        print(f"{'='*70}")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            curr = comparison['current'][metric]
            base = comparison['baseline'].get(metric, 0)
            impr = comparison['improvements'][metric]
            print(f"{metric:<15} {curr:<15.4f} {base:<15.4f} {impr:<15.4f}")
        print(f"{'='*70}\n")

        return comparison

    def save_metrics_to_json(self, output_path: str):
        """Save metrics to JSON file"""
        if self.metrics is None:
            self.evaluate()

        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"✓ Metrics saved to: {output_path}")


def evaluate_multiple_models(models: Dict, scaler, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison

    Args:
        models: Dict of {model_name: trained_model}
        scaler: Fitted StandardScaler
        X_test: Test features
        y_test: Test labels

    Returns:
        comparison_df: DataFrame with all metrics for all models
    """
    results = []

    for model_name, model in models.items():
        evaluator = ModelEvaluator(model, scaler, X_test, y_test, model_name)
        metrics = evaluator.evaluate()

        result = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],          # ✓ NEW
            'Recall': metrics['recall'],                # ✓ NEW
            'F1': metrics['f1'],
            'ROC_AUC': metrics['roc_auc'],
        }
        results.append(result)

    comparison_df = pd.DataFrame(results)
    return comparison_df
