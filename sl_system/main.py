"""
Final SL Training System (10-Fold CV + Full Batch Reporting)
- Uses 10-Fold Stratified Cross-Validation
- SAVES ALL 10 FOLD SCORES (Batch Report) for stability analysis
- Checks for Overfitting
- Saves models for Bandit Stacking
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
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

# --- IMPORTS ---
from data_loader import load_training_data

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sl_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
CONFIG = {
    'output_base': Path('models'),
    'test_size': 0.2,      
    'n_splits': 10,        
    'random_state': 42,
}

os.makedirs(CONFIG['output_base'], exist_ok=True)
Path(CONFIG['output_base'] / 'model_reports').mkdir(parents=True, exist_ok=True)

# --- UTILS ---
def get_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1: scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.zeros((len(X), 2))

# --- MAIN WORKFLOW ---
def main():
    print("\n" + "="*80)
    print(f"SL TRAINING SYSTEM ({CONFIG['n_splits']}-Fold Batch Reporting)")
    print("="*80)

    # 1. Load Data
    try:
        X_full, y_full = load_training_data()
    except Exception as e:
        logger.error(f"Data load failed: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], stratify=y_full
    )
    
    # 2. Build Models
    models = {}
    models['DecisionTree'] = DecisionTreeClassifier(max_depth=15, random_state=CONFIG['random_state'], class_weight='balanced')
    models['LogisticRegression'] = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=CONFIG['random_state'], class_weight='balanced')
    models['RandomForest'] = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=CONFIG['random_state'], class_weight='balanced', n_jobs=-1)
    models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=CONFIG['random_state'])
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=CONFIG['random_state'], class_weight='balanced', verbose=-1)

    # 3. Training & Batch Reporting
    print(f"\n{'Model':<20} | {'CV Mean':<10} | {'Test Acc':<10} | {'Gap':<8}")
    print("-" * 65)
    
    for name, model in models.items():
        # Handle Scaling for LR
        scaler = None
        if name == 'LogisticRegression':
            scaler = StandardScaler()
            X_tr_final = scaler.fit_transform(X_train)
            X_te_final = scaler.transform(X_test)
        else:
            X_tr_final = X_train
            X_te_final = X_test
            
        # --- A. 10-FOLD CV (BATCH RUN) ---
        cv_scores = cross_val_score(
            model, X_tr_final, y_train, 
            cv=CONFIG['n_splits'], 
            scoring='accuracy', 
            n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # --- B. FINAL TRAINING ---
        model.fit(X_tr_final, y_train)
        
        # --- C. TEST EVALUATION ---
        test_preds = model.predict(X_te_final)
        test_acc = accuracy_score(y_test, test_preds)
        test_probs = get_proba_safe(model, X_te_final)
        try:
            test_auc = roc_auc_score(y_test, test_probs[:, 1])
        except:
            test_auc = 0.0
            
        gap = cv_mean - test_acc
        print(f"{name:<20} | {cv_mean:.4f}     | {test_acc:.4f}     | {gap:.4f}")
        
        # --- D. SAVE ARTIFACTS ---
        save_dir = CONFIG['output_base'] / name
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / f'{name}_model.pkl')
        if scaler:
            joblib.dump(scaler, save_dir / f'{name}_scaler.pkl')
            
        # --- E. SAVE DETAILED BATCH REPORT ---
        report = classification_report(y_test, test_preds, output_dict=True)
        report['model_name'] = name
        report['cv_accuracy_mean'] = cv_mean
        report['cv_accuracy_std'] = cv_std
        report['test_accuracy'] = test_acc
        report['roc_auc'] = test_auc
        
        # CRITICAL ADDITION: Save the raw scores of all 10 folds
        report['cv_fold_scores'] = cv_scores.tolist() 
        
        with open(save_dir / 'test_report.json', 'w') as f:
            json.dump(report, f, indent=2)

    print("-" * 65)
    print("DONE. Full batch reports saved to JSON.")

if __name__ == '__main__':
    main()