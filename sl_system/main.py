"""
Comprehensive SL Training System - Fixed Unicode Encoding for Windows
Run this: python sl_training_main.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ========== FIX: Unicode encoding for Windows ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sl_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    'data_path': r"E:\saas-idea-validator\data\processed\balanced\vectorized_features_balanced.csv",
    'output_base': Path('ml_outputs_thesis'),
    'n_splits': 10,
    'test_size': 0.2,
    'random_state': 42,
}

os.makedirs(CONFIG['output_base'], exist_ok=True)

# ========== STEP 1: DATA LOADING ==========
def load_and_prepare_data():
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING & PREPARATION")
    logger.info("="*80)
    
    df = pd.read_csv(CONFIG['data_path'])
    logger.info(f"[OK] Loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    X = df.drop(columns=['label_numeric'])
    y = df['label_numeric']
    
    logger.info(f"\n[INFO] Class Distribution:")
    for class_id in sorted(y.unique()):
        count = (y == class_id).sum()
        pct = count / len(y) * 100
        logger.info(f"   Class {class_id}: {count:5d} ({pct:5.1f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'],
        stratify=y
    )
    
    logger.info(f"\n[OK] Stratified Train-Test Split:")
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Test:  {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

# ========== STEP 2: BUILD MODELS ==========
def build_models():
    logger.info("\n" + "="*80)
    logger.info("STEP 2: BUILDING MODELS")
    logger.info("="*80)
    
    models = {
        'DecisionTree': DecisionTreeClassifier(
            max_depth=15, min_samples_split=10, min_samples_leaf=5,
            random_state=CONFIG['random_state'], class_weight='balanced'
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, solver='lbfgs', class_weight='balanced',
            random_state=CONFIG['random_state']
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=CONFIG['random_state'], n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_samples_split=10, min_samples_leaf=5,
            random_state=CONFIG['random_state'], validation_fraction=0.1,
            n_iter_no_change=20
        ),
    }
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_data_in_leaf=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=CONFIG['random_state'],
            verbose=-1
        )
        logger.info("[OK] Built 5 models")
    else:
        logger.info("[OK] Built 4 models")
    
    return models

# ========== STEP 3: CV TRAINING ==========
def train_with_cv(X_train, y_train, models):
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CROSS-VALIDATION TRAINING (10-Fold Stratified)")
    logger.info("="*80)
    
    skf = StratifiedKFold(
        n_splits=CONFIG['n_splits'],
        shuffle=True,
        random_state=CONFIG['random_state']
    )
    
    cv_results = {}
    fold_reports = {}
    
    for model_name, model in models.items():
        logger.info(f"\n[TRAIN] Training {model_name}...")
        fold_reports[model_name] = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            model.fit(X_fold_train_scaled, y_fold_train)
            
            y_pred = model.predict(X_fold_val_scaled)
            y_proba = model.predict_proba(X_fold_val_scaled)
            
            # Calculate ROC-AUC (handle both binary and multiclass)
            n_classes = len(np.unique(y_fold_val))
            if n_classes == 2:
                # Binary classification: use probabilities of positive class
                roc_auc = roc_auc_score(y_fold_val, y_proba[:, 1])
            else:
                # Multiclass: use ovr (one-vs-rest)
                roc_auc = roc_auc_score(y_fold_val, y_proba, multi_class='ovr')
            
            metrics = {
                'fold': fold_idx + 1,
                'accuracy': accuracy_score(y_fold_val, y_pred),
                'precision': precision_score(y_fold_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_fold_val, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_fold_val, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc,
                'cm': confusion_matrix(y_fold_val, y_pred).tolist(),
            }
            
            fold_reports[model_name].append(metrics)
            logger.info(f"  Fold {fold_idx+1}: Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")
        
        accs = [f['accuracy'] for f in fold_reports[model_name]]
        f1s = [f['f1'] for f in fold_reports[model_name]]
        rocs = [f['roc_auc'] for f in fold_reports[model_name]]
        
        cv_results[model_name] = {
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'roc_auc_mean': np.mean(rocs),
            'roc_auc_std': np.std(rocs),
        }
        
        logger.info(f"[OK] CV Result: {cv_results[model_name]['accuracy_mean']:.4f} +/- {cv_results[model_name]['accuracy_std']:.4f}")
    
    return cv_results, fold_reports

# ========== STEP 4: FINAL TRAINING ==========
def final_training_and_test(X_train, X_test, y_train, y_test, models):
    logger.info("\n" + "="*80)
    logger.info("STEP 4: FINAL TRAINING & TEST EVALUATION")
    logger.info("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    test_results = {}
    trained_models = {}
    
    for model_name, model in models.items():
        logger.info(f"\n[TRAIN] Final training {model_name}...")
        
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = (model, scaler)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate ROC-AUC (handle both binary and multiclass)
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            # Binary classification: use probabilities of positive class
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            # Multiclass: use ovr (one-vs-rest)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        test_results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc,
            'cm': confusion_matrix(y_test, y_pred),
            'class_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        
        logger.info(f"[OK] Test Accuracy: {test_results[model_name]['accuracy']:.4f}")
    
    return test_results, trained_models, scaler

# ========== STEP 5: OVERFITTING ANALYSIS ==========
def analyze_overfitting(cv_results, test_results):
    logger.info("\n" + "="*80)
    logger.info("STEP 5: OVERFITTING/UNDERFITTING ANALYSIS")
    logger.info("="*80)
    
    analysis = {}
    
    for model_name in cv_results.keys():
        cv_acc = cv_results[model_name]['accuracy_mean']
        test_acc = test_results[model_name]['accuracy']
        gap = cv_acc - test_acc
        gap_pct = (gap / cv_acc * 100) if cv_acc > 0 else 0
        
        if gap > 0.15:
            status = "[SEVERE] OVERFITTING DETECTED"
        elif gap > 0.10:
            status = "[MODERATE] OVERFITTING"
        elif gap > 0.05:
            status = "[MILD] OVERFITTING"
        elif gap < -0.05:
            status = "[UNDERFITTING]"
        else:
            status = "[GOOD] GENERALIZATION"
        
        analysis[model_name] = {
            'cv_accuracy': cv_acc,
            'test_accuracy': test_acc,
            'gap': gap,
            'gap_pct': gap_pct,
            'status': status,
        }
        
        logger.info(f"\n{model_name}:")
        logger.info(f"   CV Accuracy:   {cv_acc:.4f}")
        logger.info(f"   Test Accuracy: {test_acc:.4f}")
        logger.info(f"   Gap:           {gap:.4f} ({gap_pct:.1f}%)")
        logger.info(f"   Status:        {status}")
    
    return analysis

# ========== STEP 6: GENERATE REPORTS ==========
def generate_model_reports(model_name, cv_results, test_results, fold_reports, 
                           overfitting_analysis, X_test, y_test, trained_model_info):
    
    model_dir = CONFIG['output_base'] / 'model_reports' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n[REPORT] Generating {model_name} report...")
    
    # 1. METRICS JSON
    metrics_report = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'cv_results': {
            'accuracy_mean': float(cv_results[model_name]['accuracy_mean']),
            'accuracy_std': float(cv_results[model_name]['accuracy_std']),
            'f1_mean': float(cv_results[model_name]['f1_mean']),
            'f1_std': float(cv_results[model_name]['f1_std']),
            'roc_auc_mean': float(cv_results[model_name]['roc_auc_mean']),
            'roc_auc_std': float(cv_results[model_name]['roc_auc_std']),
        },
        'test_results': {
            'accuracy': float(test_results[model_name]['accuracy']),
            'precision': float(test_results[model_name]['precision']),
            'recall': float(test_results[model_name]['recall']),
            'f1': float(test_results[model_name]['f1']),
            'roc_auc': float(test_results[model_name]['roc_auc']),
        },
        'overfitting_analysis': {
            'cv_accuracy': float(overfitting_analysis[model_name]['cv_accuracy']),
            'test_accuracy': float(overfitting_analysis[model_name]['test_accuracy']),
            'gap': float(overfitting_analysis[model_name]['gap']),
            'gap_percentage': float(overfitting_analysis[model_name]['gap_pct']),
            'status': overfitting_analysis[model_name]['status'],
        },
        'confusion_matrix': test_results[model_name]['cm'].tolist(),
        'classification_report': test_results[model_name]['class_report'],
        'fold_reports': fold_reports[model_name],
    }
    
    with open(model_dir / 'metrics_report.json', 'w') as f:
        json.dump(metrics_report, f, indent=2)
    
    # 2. TEXT REPORT
    with open(model_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL EVALUATION REPORT: {model_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CROSS-VALIDATION RESULTS (Training Data):\n")
        f.write(f"  Accuracy:  {cv_results[model_name]['accuracy_mean']:.4f} +/- {cv_results[model_name]['accuracy_std']:.4f}\n")
        f.write(f"  F1-Score:  {cv_results[model_name]['f1_mean']:.4f} +/- {cv_results[model_name]['f1_std']:.4f}\n")
        f.write(f"  ROC-AUC:   {cv_results[model_name]['roc_auc_mean']:.4f} +/- {cv_results[model_name]['roc_auc_std']:.4f}\n\n")
        
        f.write("TEST SET RESULTS (Hold-out Data):\n")
        f.write(f"  Accuracy:  {test_results[model_name]['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_results[model_name]['precision']:.4f}\n")
        f.write(f"  Recall:    {test_results[model_name]['recall']:.4f}\n")
        f.write(f"  F1-Score:  {test_results[model_name]['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {test_results[model_name]['roc_auc']:.4f}\n\n")
        
        f.write("OVERFITTING ANALYSIS:\n")
        f.write(f"  CV Accuracy:   {overfitting_analysis[model_name]['cv_accuracy']:.4f}\n")
        f.write(f"  Test Accuracy: {overfitting_analysis[model_name]['test_accuracy']:.4f}\n")
        f.write(f"  Gap:           {overfitting_analysis[model_name]['gap']:.4f} ({overfitting_analysis[model_name]['gap_pct']:.1f}%)\n")
        f.write(f"  Status:        {overfitting_analysis[model_name]['status']}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"{test_results[model_name]['cm']}\n")
    
    # 3. VISUALIZATIONS
    plot_visualizations(model_name, cv_results, test_results, fold_reports, 
                       y_test, test_results[model_name]['cm'], model_dir)
    
    # 4. SAVE MODELS
    model, scaler = trained_model_info[model_name]
    joblib.dump(model, model_dir / f'{model_name}_model.pkl')
    joblib.dump(scaler, model_dir / f'{model_name}_scaler.pkl')

# ========== VISUALIZATIONS ==========
def plot_visualizations(model_name, cv_results, test_results, fold_reports, 
                       y_test, cm, model_dir):
    
    # 1. CV vs Test
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    cv_vals = [
        cv_results[model_name]['accuracy_mean'],
        np.mean([f['precision'] for f in fold_reports[model_name]]),
        np.mean([f['recall'] for f in fold_reports[model_name]]),
        cv_results[model_name]['f1_mean'],
        cv_results[model_name]['roc_auc_mean'],
    ]
    test_vals = [
        test_results[model_name]['accuracy'],
        test_results[model_name]['precision'],
        test_results[model_name]['recall'],
        test_results[model_name]['f1'],
        test_results[model_name]['roc_auc'],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, cv_vals, width, label='CV', color='#2E86AB', alpha=0.8)
    axes[0].bar(x + width/2, test_vals, width, label='Test', color='#A23B72', alpha=0.8)
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'{model_name}: CV vs Test')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # 2. Fold trend
    fold_accs = [f['accuracy'] for f in fold_reports[model_name]]
    folds = list(range(1, len(fold_accs) + 1))
    
    axes[1].plot(folds, fold_accs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[1].axhline(np.mean(fold_accs), color='green', linestyle='--', linewidth=2)
    axes[1].axhline(test_results[model_name]['accuracy'], color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name}: Per-Fold Trend')
    axes[1].set_xticks(folds)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / '01_cv_vs_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax)
    ax.set_title(f'{model_name}: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(model_dir / '02_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-Class Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = sorted(y_test.unique())
    class_accs = [cm[i, i] / cm[i].sum() for i in range(len(classes))]
    
    colors = ['#2ca02c' if acc >= 0.9 else '#ff7f0e' if acc >= 0.8 else '#d62728' for acc in class_accs]
    ax.bar(classes, class_accs, color=colors, alpha=0.8)
    ax.axhline(0.9, color='green', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Class')
    ax.set_title(f'{model_name}: Per-Class Accuracy')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / '03_per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== MAIN ==========
def main():
    print("\n" + "="*100)
    print(" "*20 + "COMPREHENSIVE SL TRAINING SYSTEM")
    print("="*100)
    
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        models = build_models()
        cv_results, fold_reports = train_with_cv(X_train, y_train, models)
        test_results, trained_models, scaler = final_training_and_test(X_train, X_test, y_train, y_test, models)
        overfitting_analysis = analyze_overfitting(cv_results, test_results)
        
        logger.info("\n" + "="*80)
        logger.info("STEP 6: GENERATING REPORTS & VISUALIZATIONS")
        logger.info("="*80)
        
        for model_name in models.keys():
            generate_model_reports(
                model_name, cv_results, test_results, fold_reports,
                overfitting_analysis, X_test, y_test, trained_models
            )
        
        logger.info("\n" + "="*80)
        logger.info("[COMPLETE] TRAINING FINISHED")
        logger.info("="*80)
        logger.info(f"\n[OUTPUT] {CONFIG['output_base']}/model_reports/")
        
        for model_name in models.keys():
            logger.info(f"\n{model_name}:")
            logger.info(f"  CV:   {cv_results[model_name]['accuracy_mean']:.4f}")
            logger.info(f"  Test: {test_results[model_name]['accuracy']:.4f}")
            logger.info(f"  Status: {overfitting_analysis[model_name]['status']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit(main())