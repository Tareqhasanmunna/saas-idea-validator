"""
Thesis Detailed Visualization Generator (Final - Headless Fix)
- Sets backend to 'Agg' to prevent Tkinter/Threading crashes.
- Generates Confusion Matrix, ROC Curve, and Feature Importance.
"""

# --- CRITICAL FIX FOR TKINTER CRASHES ---
import matplotlib
matplotlib.use('Agg')  # Must be done before importing pyplot
# ----------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from data_loader import load_training_data

# --- CONFIG ---
MODEL_DIR = Path('models')
VIS_DIR = Path('ml_outputs_thesis/visualizations')
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generates and saves a Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title(f'{model_name}: Confusion Matrix', pad=15, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    save_path = VIS_DIR / f'cm_{model_name}.png'
    plt.savefig(save_path)
    print(f"   [+] Saved Confusion Matrix: {save_path.name}")
    plt.close()

def plot_roc_curve(y_true, y_probs, model_name):
    """Generates and saves an ROC Curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#d62728', lw=2.5, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name}: ROC Curve', pad=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = VIS_DIR / f'roc_{model_name}.png'
    plt.savefig(save_path)
    print(f"   [+] Saved ROC Curve: {save_path.name}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Generates a Bar Chart of the Top 20 Features"""
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    
    if importances is None:
        return

    # Create DataFrame for plotting
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    
    plt.title(f'{model_name}: Top 20 Features', pad=15, fontweight='bold')
    plt.xlabel('Relative Importance')
    plt.ylabel('')
    plt.tight_layout()
    
    save_path = VIS_DIR / f'feat_imp_{model_name}.png'
    plt.savefig(save_path)
    print(f"   [+] Saved Feature Importance: {save_path.name}")
    plt.close()

def get_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1: scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.zeros((len(X), 2))

def main():
    print("="*60)
    print("GENERATING INDIVIDUAL MODEL PLOTS (HEADLESS MODE)")
    print("="*60)

    # 1. Load Data
    try:
        X_full, y_full = load_training_data()
        feature_names = X_full.columns.tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Must match the split used in training
    _, X_test, _, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    # 2. Iterate through Models
    model_names = ['GradientBoosting', 'LightGBM', 'RandomForest', 'LogisticRegression', 'DecisionTree']
    
    for name in model_names:
        print(f"\nProcessing {name}...")
        model_path = MODEL_DIR / name / f'{name}_model.pkl'
        scaler_path = MODEL_DIR / name / f'{name}_scaler.pkl'
        
        if not model_path.exists():
            print(f"   [-] Model file not found: {model_path}")
            continue
            
        # Load Model
        model = joblib.load(model_path)
        
        # Prepare Data
        X_input = X_test.copy()
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_input = scaler.transform(X_input)
        else:
            X_input = X_input.values
            
        # Generate Plots
        preds = model.predict(X_input)
        probs = get_proba_safe(model, X_input)[:, 1]
        
        plot_confusion_matrix(y_test, preds, name)
        plot_roc_curve(y_test, probs, name)
        plot_feature_importance(model, feature_names, name)

    print("\n" + "="*60)
    print(f"✅ DONE. All individual plots saved to: {VIS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()