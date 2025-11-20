import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI runs
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Environment-driven output dirs (can override with ML_OUTPUT_DIR)
OUTPUT_DIR = Path(os.getenv('ML_OUTPUT_DIR', 'ml_outputs/visualizations'))
REPORTS_DIR = Path(os.getenv('ML_REPORTS_DIR', 'ml_outputs/reports'))

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_context("notebook", font_scale=1.25)
sns.set_style("whitegrid")

def plot_model_comparison():
    """Bar plots for mean±std for all metrics by model (CV averaged)"""
    csv_path = REPORTS_DIR / 'model_comparison.csv'
    df = pd.read_csv(csv_path)

    metrics = [
        ('Accuracy_Mean', 'Accuracy_Std', 'Accuracy'),
        ('Precision_Mean', 'Precision_Std', 'Precision'),
        ('Recall_Mean', 'Recall_Std', 'Recall'),
        ('F1_Mean', 'F1_Std', 'F1 Score')
    ]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    for i, (mean_col, std_col, title) in enumerate(metrics):
        axs[i].bar(df['Model'], df[mean_col], yerr=df[std_col], capsize=6, color=sns.color_palette()[i])
        axs[i].set_title(f'Model {title}')
        axs[i].set_ylabel(title)
        axs[i].set_ylim([0, 1])
        axs[i].grid(True, axis='y', alpha=0.25)
        for j, v in enumerate(df[mean_col]):
            axs[i].text(j, v+.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'sl_model_comparison_all_metrics.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Model comparison bar plots: {out_path}")

def plot_per_fold_metrics(model_name):
    """Per-fold metrics trend for a specific model (e.g., LightGBM)"""
    json_path = REPORTS_DIR / f"{model_name}_batch_reports.json"
    if not json_path.exists():
        print(f"Batch report for {model_name} not found.")
        return

    with open(json_path) as f:
        batch_reports = json.load(f)

    folds = [r['fold'] for r in batch_reports]
    metrics = ['accuracy', 'precision', 'recall', 'f1_weighted']

    plt.figure(figsize=(12, 6))
    for m in metrics:
        plt.plot(folds, [r[m] for r in batch_reports], marker='o', label=m.capitalize(), linewidth=3)
    plt.xlabel('Fold')
    plt.ylabel('Metric Score')
    plt.title(f'{model_name} - Per-Fold Metrics')
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'sl_{model_name}_per_fold_metrics.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ {model_name} per-fold metrics: {out_path}")

def plot_precision_focus(model_name):
    """Per-fold precision plot"""
    json_path = REPORTS_DIR / f"{model_name}_batch_reports.json"
    with open(json_path) as f:
        batch_reports = json.load(f)
    folds = [r['fold'] for r in batch_reports]
    precisions = [r['precision'] for r in batch_reports]
    plt.figure(figsize=(10, 6))
    plt.plot(folds, precisions, "o-", linewidth=3, markersize=9, color='darkred')
    plt.fill_between(folds, precisions, alpha=0.2, color="red")
    plt.title(f"{model_name} - Precision per Fold")
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.ylim([min(precisions)-.05, max(precisions)+.05])
    plt.grid(True, alpha=0.25)
    for x, y in zip(folds, precisions):
        plt.text(x, y+.01, f"{y:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'sl_{model_name}_precision_focus.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ {model_name} precision plot: {out_path}")

def plot_recall_focus(model_name):
    """Per-fold recall plot"""
    json_path = REPORTS_DIR / f"{model_name}_batch_reports.json"
    with open(json_path) as f:
        batch_reports = json.load(f)
    folds = [r['fold'] for r in batch_reports]
    recalls = [r['recall'] for r in batch_reports]
    plt.figure(figsize=(10, 6))
    plt.plot(folds, recalls, "s-", linewidth=3, markersize=9, color='darkgreen')
    plt.fill_between(folds, recalls, alpha=0.2, color="green")
    plt.title(f"{model_name} - Recall per Fold")
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.ylim([min(recalls)-.05, max(recalls)+.05])
    plt.grid(True, alpha=0.25)
    for x, y in zip(folds, recalls):
        plt.text(x, y+.01, f"{y:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'sl_{model_name}_recall_focus.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ {model_name} recall plot: {out_path}")

def plot_confusion_matrix(model_name):
    """Plot confusion matrix heatmap for fold 1"""
    json_path = REPORTS_DIR / f"{model_name}_batch_reports.json"
    with open(json_path) as f:
        batch_reports = json.load(f)
    cm = np.array(batch_reports[0]['cm'])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"{model_name} - Confusion Matrix (Fold 1)")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    classes = ['Bad', 'Neutral', 'Good']
    plt.xticks(np.arange(3)+0.5, classes, rotation=0)
    plt.yticks(np.arange(3)+0.5, classes, rotation=0)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'sl_{model_name}_confusion_matrix.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ {model_name} confusion matrix: {out_path}")

def main():
    print("="*60)
    print("SaaS Idea Validator - Visualization Pipeline")
    print("="*60)
    plot_model_comparison()
    plot_per_fold_metrics("LightGBM")   # Change to your best model
    plot_precision_focus("LightGBM")
    plot_recall_focus("LightGBM")
    plot_confusion_matrix("LightGBM")
    print("="*60)
    print("✓ All visualizations complete and saved in", OUTPUT_DIR)

if __name__ == '__main__':
    main()
