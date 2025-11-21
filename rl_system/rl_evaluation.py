"""
RL Evaluation Module

Evaluates RL agent performance against baseline SL model.
Compares metrics: accuracy, precision, recall, F1, ROC-AUC.

Usage:
    evaluator = RLEvaluator(agent, X_test, y_test)
    metrics = evaluator.evaluate()
    comparison = evaluator.compare_with_baseline(baseline_predictions)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from typing import Dict, Tuple, List
import pandas as pd


class RLEvaluator:
    """Evaluate RL agent performance"""

    def __init__(self, agent, X_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize evaluator

        Args:
            agent: Trained RL agent
            X_test: Test feature matrix
            y_test: Test labels
        """
        self.agent = agent
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = None
        self.probabilities = None

    def evaluate(self) -> Dict:
        """
        Evaluate agent on test set

        Returns:
            metrics: Dict with comprehensive evaluation metrics
        """
        print("Evaluating RL agent on test set...")

        # Get predictions
        predictions = []
        probabilities = []

        for i in range(len(self.X_test)):
            state = self.X_test[i] if isinstance(self.X_test, np.ndarray) else self.X_test.iloc[i].values
            action = self.agent.get_action_deterministic(state)
            predictions.append(action)

        predictions = np.array(predictions)
        y_test_array = self.y_test if isinstance(self.y_test, np.ndarray) else self.y_test.values

        self.predictions = predictions

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test_array, predictions)),
            'precision': float(precision_score(y_test_array, predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test_array, predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test_array, predictions, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test_array, predictions).tolist(),
        }

        # Per-class metrics
        per_class_metrics = classification_report(y_test_array, predictions, output_dict=True, zero_division=0)
        metrics['per_class_metrics'] = per_class_metrics

        # ROC-AUC (if binary or multiclass)
        try:
            roc_auc = roc_auc_score(y_test_array, np.eye(len(np.unique(y_test_array)))[predictions], multi_class='ovr', zero_division=0)
            metrics['roc_auc'] = float(roc_auc)
        except:
            metrics['roc_auc'] = None

        print(f"✓ Evaluation complete")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        return metrics

    def compare_with_baseline(self, baseline_predictions: np.ndarray) -> Dict:
        """
        Compare RL agent with baseline SL model

        Args:
            baseline_predictions: Predictions from baseline SL model

        Returns:
            comparison: Dict with comparison metrics
        """
        y_test_array = self.y_test if isinstance(self.y_test, np.ndarray) else self.y_test.values

        rl_metrics = {
            'accuracy': accuracy_score(y_test_array, self.predictions),
            'f1': f1_score(y_test_array, self.predictions, average='weighted', zero_division=0),
            'precision': precision_score(y_test_array, self.predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test_array, self.predictions, average='weighted', zero_division=0),
        }

        baseline_metrics = {
            'accuracy': accuracy_score(y_test_array, baseline_predictions),
            'f1': f1_score(y_test_array, baseline_predictions, average='weighted', zero_division=0),
            'precision': precision_score(y_test_array, baseline_predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test_array, baseline_predictions, average='weighted', zero_division=0),
        }

        comparison = {
            'rl_metrics': rl_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': {
                'accuracy': rl_metrics['accuracy'] - baseline_metrics['accuracy'],
                'f1': rl_metrics['f1'] - baseline_metrics['f1'],
                'precision': rl_metrics['precision'] - baseline_metrics['precision'],
                'recall': rl_metrics['recall'] - baseline_metrics['recall'],
            }
        }

        print(f"\nComparison with Baseline:")
        print(f"{'='*70}")
        print(f"{'Metric':<15} {'RL Agent':<15} {'Baseline':<15} {'Improvement':<15}")
        print(f"{'='*70}")

        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            rl_val = comparison['rl_metrics'][metric]
            base_val = comparison['baseline_metrics'][metric]
            improvement = comparison['improvements'][metric]
            print(f"{metric:<15} {rl_val:<15.4f} {base_val:<15.4f} {improvement:<15.4f}")

        print(f"{'='*70}")

        return comparison

    def get_improvement_summary(self) -> str:
        """Generate summary of improvements"""
        return f"RL Agent improvements summary"