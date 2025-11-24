# ============================================================================
# FILE 2: visualization.py
# ============================================================================
"""
Comprehensive Visualization Suite for SL Training
Usage: python visualization.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path('ml_outputs_thesis/visualizations')
REPORTS_DIR = Path('ml_outputs_thesis/model_reports')

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

def plot_all_models_comparison():
    """Compare all models across metrics"""
    
    print("📊 Generating all models comparison...")
    
    # Collect metrics from all model reports
    all_metrics = []
    
    for model_dir in REPORTS_DIR.glob('*/'):
        metrics_file = model_dir / 'metrics_report.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            all_metrics.append({
                'Model': model_dir.name,
                'CV_Accuracy': data['cv_results']['accuracy_mean'],
                'Test_Accuracy': data['test_results']['accuracy'],
                'F1_Score': data['test_results']['f1'],
                'ROC_AUC': data['test_results']['roc_auc'],
            })
    
    df = pd.DataFrame(all_metrics)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].bar(df['Model'], df['CV_Accuracy'], alpha=0.8, color='#2E86AB')
    axes[0, 0].set_title('CV Accuracy')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    
    axes[0, 1].bar(df['Model'], df['Test_Accuracy'], alpha=0.8, color='#A23B72')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    
    axes[1, 0].bar(df['Model'], df['F1_Score'], alpha=0.8, color='#F18F01')
    axes[1, 0].set_title('F1-Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    
    axes[1, 1].bar(df['Model'], df['ROC_AUC'], alpha=0.8, color='#06A77D')
    axes[1, 1].set_title('ROC-AUC')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    
    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '00_all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 00_all_models_comparison.png\n")

def plot_overfitting_analysis():
    """Plot overfitting analysis for all models"""
    
    print("🔍 Generating overfitting analysis...")
    
    overfitting_data = []
    
    for model_dir in REPORTS_DIR.glob('*/'):
        metrics_file = model_dir / 'metrics_report.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            cv_acc = data['cv_results']['accuracy_mean']
            test_acc = data['test_results']['accuracy']
            gap = cv_acc - test_acc
            
            overfitting_data.append({
                'Model': model_dir.name,
                'CV_Accuracy': cv_acc,
                'Test_Accuracy': test_acc,
                'Overfitting_Gap': gap,
            })
    
    df = pd.DataFrame(overfitting_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # CV vs Test
    x = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x - width/2, df['CV_Accuracy'], width, label='CV', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, df['Test_Accuracy'], width, label='Test', color='#A23B72', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('CV vs Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Overfitting gap
    colors = ['#d62728' if gap > 0.15 else '#ff7f0e' if gap > 0.1 else '#2ca02c' 
              for gap in df['Overfitting_Gap']]
    
    ax2.barh(df['Model'], df['Overfitting_Gap'], color=colors, alpha=0.8)
    ax2.set_xlabel('Overfitting Gap')
    ax2.set_title('Overfitting Severity')
    ax2.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='Warning')
    ax2.axvline(0.15, color='red', linestyle='--', linewidth=2, label='Critical')
    ax2.legend()
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 01_overfitting_analysis.png\n")

def main():
    print("\n" + "="*80)
    print("🎨 VISUALIZATION SUITE")
    print("="*80 + "\n")
    
    try:
        plot_all_models_comparison()
        plot_overfitting_analysis()
        
        print("="*80)
        print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())