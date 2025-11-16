"""
Evaluation and Reporting Module
==============================

Comprehensive evaluation metrics and batch-wise reporting:
- Per-fold evaluation metrics
- Batch-wise performance reports (JSON format)
- Summary statistics for each model
- Detailed classification reports
- Model comparison metrics

Usage:
    from evaluation import generate_batch_reports, generate_summary_evaluation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score
)

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate comprehensive evaluation reports"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path('ml_outputs/reports')
        self.output_dir.mkdir(exist_ok=True)

    def generate_batch_report(self, model_name: str, fold: int, y_true, y_pred, 
                             y_pred_proba, fold_data: Dict = None) -> Dict:
        """Generate detailed report for a single batch/fold"""

        # Calculate comprehensive metrics
        report = {
            'model_name': model_name,
            'fold_number': fold,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
                'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
                'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
                'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
                'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
            },
            'data_info': {
                'total_samples': len(y_true),
                'unique_classes': len(np.unique(y_true)),
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

        # Add optional fold data
        if fold_data:
            report['fold_data'] = fold_data

        return report

    def generate_batch_reports_file(self, model_name: str, batch_reports: List[Dict]):
        """Save batch reports to JSON file"""

        batch_file = {
            'model_name': model_name,
            'total_folds': len(batch_reports),
            'generation_timestamp': datetime.now().isoformat(),
            'folds': batch_reports,
            'aggregate_stats': self._calculate_aggregate_stats(batch_reports)
        }

        file_path = self.output_dir / f'{model_name}_batch_reports.json'
        with open(file_path, 'w') as f:
            json.dump(batch_file, f, indent=2)

        logger.info(f"Batch reports saved: {file_path}")
        return file_path

    def _calculate_aggregate_stats(self, batch_reports: List[Dict]) -> Dict:
        """Calculate aggregate statistics across folds"""
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        aggregate = {}
        for metric in metrics:
            values = [report['metrics'][metric] for report in batch_reports if metric in report['metrics']]
            if values:
                aggregate[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }

        return aggregate

    def generate_model_evaluation_summary(self, model_name: str, y_true, y_pred, 
                                        y_pred_proba, cv_scores: Dict) -> Dict:
        """Generate comprehensive model evaluation summary"""

        # Prepare classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        summary = {
            'model_name': model_name,
            'evaluation_date': datetime.now().isoformat(),
            'overall_metrics': {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
                'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
            },
            'cross_validation_metrics': {},
            'classification_report': class_report,
        }

        # Add cross-validation metrics
        for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            test_scores = cv_scores.get(f'test_{metric}', [])
            if len(test_scores) > 0:
                summary['cross_validation_metrics'][metric] = {
                    'mean': float(np.mean(test_scores)),
                    'std': float(np.std(test_scores)),
                    'min': float(np.min(test_scores)),
                    'max': float(np.max(test_scores)),
                    'all_scores': [float(s) for s in test_scores]
                }

        return summary

    def save_evaluation_summary(self, model_name: str, summary: Dict):
        """Save evaluation summary to file"""
        file_path = self.output_dir / f'{model_name}_evaluation_summary.json'
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Evaluation summary saved: {file_path}")
        return file_path

    def generate_comparison_table(self, summaries: Dict[str, Dict]) -> pd.DataFrame:
        """Generate model comparison table from summaries"""

        comparison_data = []

        for model_name, summary in summaries.items():
            overall = summary['overall_metrics']
            cv_acc = summary['cross_validation_metrics'].get('accuracy', {})
            cv_f1 = summary['cross_validation_metrics'].get('f1_weighted', {})

            row = {
                'Model': model_name,
                'Accuracy': overall['accuracy'],
                'Precision': overall['precision_weighted'],
                'Recall': overall['recall_weighted'],
                'F1-Score': overall['f1_weighted'],
                'Matthews_CC': overall['matthews_corrcoef'],
                'Cohen_Kappa': overall['cohen_kappa'],
                'CV_Accuracy_Mean': cv_acc.get('mean', 0),
                'CV_Accuracy_Std': cv_acc.get('std', 0),
                'CV_F1_Mean': cv_f1.get('mean', 0),
                'CV_F1_Std': cv_f1.get('std', 0),
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Save to CSV and Excel
        csv_path = self.output_dir / 'model_evaluation_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Comparison table (CSV) saved: {csv_path}")

        # Also save as formatted Excel if possible
        try:
            excel_path = self.output_dir / 'model_evaluation_comparison.xlsx'
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
            logger.info(f"Comparison table (Excel) saved: {excel_path}")
        except:
            logger.warning("Excel export not available")

        return comparison_df

    def generate_detailed_report_file(self, model_name: str, summary: Dict) -> Path:
        """Generate detailed text report"""

        report_text = f"""
{"="*70}
DETAILED MODEL EVALUATION REPORT
{"="*70}

Model: {model_name}
Date: {summary['evaluation_date']}

{"-"*70}
OVERALL PERFORMANCE METRICS
{"-"*70}

Accuracy:                  {summary['overall_metrics']['accuracy']:.4f}
Precision (Weighted):      {summary['overall_metrics']['precision_weighted']:.4f}
Recall (Weighted):         {summary['overall_metrics']['recall_weighted']:.4f}
F1-Score (Weighted):       {summary['overall_metrics']['f1_weighted']:.4f}
Matthews Correlation Coef: {summary['overall_metrics']['matthews_corrcoef']:.4f}
Cohen's Kappa:             {summary['overall_metrics']['cohen_kappa']:.4f}

{"-"*70}
CROSS-VALIDATION PERFORMANCE
{"-"*70}

"""

        for metric, values in summary['cross_validation_metrics'].items():
            report_text += f"\n{metric.upper()}:\n"
            report_text += f"  Mean:   {values['mean']:.4f}\n"
            report_text += f"  Std:    {values['std']:.4f}\n"
            report_text += f"  Min:    {values['min']:.4f}\n"
            report_text += f"  Max:    {values['max']:.4f}\n"
            report_text += f"  Scores: {[f'{s:.4f}' for s in values['all_scores']]}\n"

        report_text += f"\n{'="*70}\n"

        # Save to file
        file_path = self.output_dir / f'{model_name}_detailed_report.txt'
        with open(file_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Detailed report saved: {file_path}")
        return file_path


def generate_final_evaluation_report(ml_system, y: np.ndarray):
    """Generate final comprehensive evaluation report for all models"""

    reporter = EvaluationReporter(ml_system.reports_dir)

    logger.info("\nGenerating final evaluation reports...")

    summaries = {}

    for model_name, result in ml_system.cv_results.items():
        y_pred = result['predictions']
        y_pred_proba = result['probabilities']
        cv_scores = result['cv_scores']

        # Generate summary
        summary = reporter.generate_model_evaluation_summary(
            model_name, y, y_pred, y_pred_proba, cv_scores
        )
        summaries[model_name] = summary

        # Save evaluation summary
        reporter.save_evaluation_summary(model_name, summary)

        # Save detailed report
        reporter.generate_detailed_report_file(model_name, summary)

    # Generate comparison table
    comparison_df = reporter.generate_comparison_table(summaries)

    logger.info("Final evaluation reports generated!")

    return summaries, comparison_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module loaded.")
