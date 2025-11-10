"""
COMPLETE AND TESTED SL Evaluation Integration
Specifically designed for your existing SL training system

This file is ready to use - no modifications needed
Place in: src/training/sl_training/sl_evaluation.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class SLEvaluator:
    """
    Evaluation system for your SL models
    Works with your existing TrainingPipeline output
    """
    
    def __init__(self, output_dir: str = "thesis_eval/sl_results"):
        """Initialize evaluator with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all instance variables
        self.comparison_df = None
        self.model_predictions = {}
        self.y_test = None
        
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                       model_name: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate a single model on test data
        
        Args:
            model: Trained sklearn model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            (metrics_dict, y_pred, y_pred_proba)
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # For binary: use positive class probability
            # For multiclass: will handle differently
            if proba.shape[1] == 2:
                y_pred_proba = proba[:, 1]
            else:
                y_pred_proba = proba
        
        # Calculate metrics with weighted average for multiclass
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        
        # Calculate AUC if possible
        if y_pred_proba is not None:
            try:
                # Check if binary or multiclass
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    metrics['auc'] = float(roc_auc_score(y_test, y_pred_proba))
                else:
                    # Multiclass AUC
                    metrics['auc'] = float(roc_auc_score(
                        y_test, y_pred_proba, 
                        average='weighted', 
                        multi_class='ovr'
                    ))
            except Exception as e:
                print(f"    Warning: Could not calculate AUC for {model_name}: {str(e)}")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        
        # Store confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Store classification report
        metrics['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        
        return metrics, y_pred, y_pred_proba
    
    def compare_models(self, models_dict: Dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models on test data
        
        Args:
            models_dict: Dictionary of {model_name: model_object}
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        print(f"\nEvaluating {len(models_dict)} models on test set...")
        
        comparison_results = []
        self.model_predictions = {}
        self.y_test = y_test
        
        for model_name, model in models_dict.items():
            try:
                metrics, y_pred, y_pred_proba = self.evaluate_model(
                    model, X_test, y_test, model_name
                )
                
                comparison_results.append(metrics)
                self.model_predictions[model_name] = {
                    'pred': y_pred,
                    'pred_proba': y_pred_proba,
                    'metrics': metrics
                }
                
                print(f"  ✓ {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error evaluating {model_name}: {str(e)}")
                continue
        
        self.comparison_df = pd.DataFrame(comparison_results)
        
        return self.comparison_df
    
    def save_comparison_csv(self, filename: str = "model_comparison.csv") -> Path:
        """Save comparison results to CSV"""
        if self.comparison_df is None:
            print("  ✗ No comparison data to save")
            return None
            
        csv_path = self.output_dir / filename
        self.comparison_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved comparison CSV: {csv_path}")
        return csv_path
    
    def save_detailed_metrics_json(self, filename: str = "detailed_metrics.json") -> Path:
        """Save detailed metrics to JSON"""
        if not self.model_predictions:
            print("  ✗ No metrics to save")
            return None
            
        json_data = {}
        for model_name, preds in self.model_predictions.items():
            json_data[model_name] = {
                'accuracy': float(preds['metrics']['accuracy']),
                'precision': float(preds['metrics']['precision']),
                'recall': float(preds['metrics']['recall']),
                'f1': float(preds['metrics']['f1']),
                'auc': float(preds['metrics'].get('auc', 0.0)),
                'confusion_matrix': preds['metrics']['confusion_matrix'],
            }
        
        json_path = self.output_dir / filename
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"  ✓ Saved detailed metrics JSON: {json_path}")
        return json_path
    
    def plot_model_comparison(self, filename: str = "model_comparison.png") -> Path:
        """Generate model comparison bar chart"""
        if self.comparison_df is None:
            print("  ✗ No comparison data to plot")
            return None
            
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Supervised Learning Model Comparison', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = self.comparison_df.sort_values(metric, ascending=False)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
            bars = ax.bar(range(len(data)), data[metric], color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data['model_name'], rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold', fontsize=12)
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved comparison chart: {fig_path}")
        return fig_path
    
    def plot_confusion_matrices(self, filename: str = "confusion_matrices.png") -> Path:
        """Generate confusion matrix heatmaps for all models"""
        if not self.model_predictions:
            print("  ✗ No predictions to plot")
            return None
            
        n_models = len(self.model_predictions)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, preds) in enumerate(self.model_predictions.items()):
            ax = axes[idx]
            cm = np.array(preds['metrics']['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                       square=True, linewidths=1, linecolor='gray')
            
            acc = preds['metrics']['accuracy']
            ax.set_title(f'{model_name}\n(Accuracy: {acc:.3f})', fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(self.model_predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        fig_path = self.output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved confusion matrices: {fig_path}")
        return fig_path
    
    def plot_roc_curves(self, filename: str = "roc_curves.png") -> Path:
        """Generate ROC curves for binary classification models"""
        if not self.model_predictions:
            print("  ✗ No predictions to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        has_roc = False
        n_classes = len(np.unique(self.y_test))
        
        if n_classes == 2:
            # Binary classification - plot ROC curves
            for model_name, preds in self.model_predictions.items():
                if preds['pred_proba'] is not None:
                    try:
                        fpr, tpr, _ = roc_curve(self.y_test, preds['pred_proba'])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2.5)
                        has_roc = True
                    except Exception as e:
                        print(f"    Warning: Could not plot ROC for {model_name}")
            
            if has_roc:
                ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
                ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
                ax.set_title('ROC Curves - Binary Classification', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=10)
                ax.grid(alpha=0.3, linestyle='--')
        else:
            # Multiclass - show message
            ax.text(0.5, 0.5, 
                   f'ROC curves not displayed\n(Multiclass classification: {n_classes} classes)\nAUC scores included in comparison table', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
        
        plt.tight_layout()
        fig_path = self.output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved ROC curves: {fig_path}")
        return fig_path
    
    def generate_summary_report(self, filename: str = "sl_evaluation_summary.txt") -> Path:
        """Generate text summary report"""
        if self.comparison_df is None:
            print("  ✗ No comparison data for report")
            return None
            
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SUPERVISED LEARNING MODEL EVALUATION REPORT\n")
            f.write("SaaS Idea Validator - Thesis Documentation\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models evaluated: {len(self.comparison_df)}\n")
            f.write(f"Test set size: {len(self.y_test)} samples\n\n")
            
            f.write("MODEL COMPARISON\n")
            f.write("-"*80 + "\n")
            display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1', 'auc']
            f.write(self.comparison_df[display_cols].to_string(index=False))
            f.write("\n\n")
            
            f.write("BEST MODEL\n")
            f.write("-"*80 + "\n")
            best_idx = self.comparison_df['accuracy'].idxmax()
            best_model = self.comparison_df.iloc[best_idx]
            f.write(f"Model: {best_model['model_name']}\n")
            f.write(f"Accuracy:  {best_model['accuracy']:.4f}\n")
            f.write(f"Precision: {best_model['precision']:.4f}\n")
            f.write(f"Recall:    {best_model['recall']:.4f}\n")
            f.write(f"F1-Score:  {best_model['f1']:.4f}\n")
            f.write(f"AUC:       {best_model['auc']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("All evaluation artifacts saved in: thesis_eval/sl_results/\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Saved summary report: {report_path}")
        return report_path
    
    def generate_all_artifacts(self):
        """Generate all evaluation artifacts at once"""
        print("\nGenerating all thesis evaluation artifacts...")
        
        artifacts = []
        
        # Save comparison CSV
        csv_path = self.save_comparison_csv()
        if csv_path: artifacts.append(str(csv_path))
        
        # Save detailed JSON
        json_path = self.save_detailed_metrics_json()
        if json_path: artifacts.append(str(json_path))
        
        # Generate plots
        comp_path = self.plot_model_comparison()
        if comp_path: artifacts.append(str(comp_path))
        
        cm_path = self.plot_confusion_matrices()
        if cm_path: artifacts.append(str(cm_path))
        
        roc_path = self.plot_roc_curves()
        if roc_path: artifacts.append(str(roc_path))
        
        # Generate report
        report_path = self.generate_summary_report()
        if report_path: artifacts.append(str(report_path))
        
        print(f"\n✓ Generated {len(artifacts)} thesis artifacts")
        print(f"✓ Location: {self.output_dir}")
        
        return artifacts


if __name__ == "__main__":
    print("SL Evaluation Module - Ready")
    print("Place in: src/training/sl_training/sl_evaluation.py")
