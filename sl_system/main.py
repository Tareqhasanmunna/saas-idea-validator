"""
Comprehensive SL Training System - Connected with HPO
Imports QuickHPO from advanced_hpo.py

Run: python main.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import QuickHPO (expects QuickHPO(X_train, y_train, model_name).run(iterations) -> best_params dict or {})
from advanced_hpo import AutoTrainer


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
    'output_base': Path('models'),
    'n_splits': 10,
    'test_size': 0.2,
    'random_state': 42,
    'run_hpo': True,
    'hpo_iterations': 50,
}

os.makedirs(CONFIG['output_base'], exist_ok=True)
(os.path.join(CONFIG['output_base'], 'model_reports'))
Path(CONFIG['output_base'] / 'model_reports').mkdir(parents=True, exist_ok=True)


# ========== UTIL: safe proba ==========
def get_proba_safe(model, X):
    """
    Return probability estimates in a robust way.
    Falls back to decision_function if predict_proba missing.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            # binary case -> convert to two-column score
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # If neither available, raise - but try to avoid this upstream
    raise RuntimeError("Model provides neither predict_proba nor decision_function.")


# ========== LOAD DATA ==========
def load_and_prepare_data():
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING & PREPARATION")
    logger.info("="*80)

    df = pd.read_csv(CONFIG['data_path'])
    logger.info(f"[OK] Loaded: {df.shape[0]} samples, {df.shape[1]} columns")

    X = df.drop(columns=['label_numeric'])
    y = df['label_numeric']

    logger.info(f"\n[INFO] Class Distribution:")
    for class_id in sorted(y.unique()):
        count = int((y == class_id).sum())
        pct = count / len(y) * 100
        logger.info(f"   Class {class_id}: {count:5d} ({pct:5.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'],
        stratify=y
    )

    logger.info(f"\n[OK] Stratified Train-Test Split:")
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Test:  {len(X_test)} samples")

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


# ========== BUILD MODELS WITH (OPTIONAL) HPO ==========
def build_models(X_train, y_train):
    logger.info("\n" + "="*80)
    logger.info("STEP 2: BUILDING MODELS")
    logger.info("="*80)

    models = {}
    model_names = ['DecisionTree', 'LogisticRegression', 'RandomForest', 'GradientBoosting']
    if LIGHTGBM_AVAILABLE:
        model_names.append('LightGBM')

    for model_name in model_names:
        logger.info(f"\n[SETUP] Preparing {model_name}")
        best_params = {}

        if CONFIG['run_hpo']:
            try:
                logger.info(f"[HPO] Running QuickHPO for {model_name} ({CONFIG['hpo_iterations']} iters)")
                hpo =AutoTrainer(X_train, y_train, model_name)
                best_params = hpo.run(iterations=CONFIG['hpo_iterations']) or {}
                logger.info(f"[HPO] Best params (sample): {dict(list(best_params.items())[:5])}")
            except Exception as e:
                logger.warning(f"[HPO ERROR] {model_name}: {e}. Using defaults.")
                best_params = {}

        # Build model using best_params if present, else sensible defaults
        if model_name == 'DecisionTree':
            models[model_name] = DecisionTreeClassifier(
                max_depth=int(best_params.get('max_depth', 15)),
                min_samples_split=int(best_params.get('min_samples_split', 10)),
                min_samples_leaf=int(best_params.get('min_samples_leaf', 5)),
                random_state=CONFIG['random_state'],
                class_weight='balanced'
            )
        elif model_name == 'LogisticRegression':
            models[model_name] = LogisticRegression(
                max_iter=int(best_params.get('max_iter', 1000)),
                solver=best_params.get('solver', 'lbfgs'),
                C=float(best_params.get('C', 1.0)),
                random_state=CONFIG['random_state'],
                class_weight='balanced'
            )
        elif model_name == 'RandomForest':
            models[model_name] = RandomForestClassifier(
                n_estimators=int(best_params.get('n_estimators', 200)),
                max_depth=int(best_params.get('max_depth', 15)) if best_params.get('max_depth', None) not in (None, -1) else None,
                min_samples_split=int(best_params.get('min_samples_split', 10)),
                min_samples_leaf=int(best_params.get('min_samples_leaf', 5)),
                max_features=best_params.get('max_features', 'sqrt'),
                random_state=CONFIG['random_state'],
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_name == 'GradientBoosting':
            models[model_name] = GradientBoostingClassifier(
                n_estimators=int(best_params.get('n_estimators', 200)),
                learning_rate=float(best_params.get('learning_rate', 0.05)),
                max_depth=int(best_params.get('max_depth', 5)),
                min_samples_split=int(best_params.get('min_samples_split', 10)),
                min_samples_leaf=int(best_params.get('min_samples_leaf', 5)),
                subsample=float(best_params.get('subsample', 1.0)),
                random_state=CONFIG['random_state'],
                validation_fraction=0.1,
                n_iter_no_change=20
            )
        elif model_name == 'LightGBM':
            models[model_name] = lgb.LGBMClassifier(
                n_estimators=int(best_params.get('n_estimators', 200)),
                learning_rate=float(best_params.get('learning_rate', 0.05)),
                num_leaves=int(best_params.get('num_leaves', 31)),
                max_depth=int(best_params.get('max_depth', -1)),
                min_data_in_leaf=int(best_params.get('min_data_in_leaf', 10)),
                lambda_l1=float(best_params.get('lambda_l1', 0.0)),
                lambda_l2=float(best_params.get('lambda_l2', 0.0)),
                bagging_fraction=float(best_params.get('bagging_fraction', 1.0)),
                random_state=CONFIG['random_state'],
                class_weight='balanced',
                verbose=-1
            )

    status = "with HPO" if CONFIG['run_hpo'] else "with default params"
    logger.info(f"\n[OK] Built {len(models)} models {status}")
    return models


# ========== CV TRAINING ==========
def train_with_cv(X_train, y_train, models):
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CROSS-VALIDATION TRAINING (10-Fold Stratified)")
    logger.info("="*80)

    skf = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['random_state'])

    cv_results = {}
    fold_reports = {}

    for model_name, model in models.items():
        logger.info(f"\n[TRAIN] Training {model_name} (CV)...")
        fold_reports[model_name] = []

        # Only LogisticRegression needs scaling (others are tree-based)
        needs_scaling = isinstance(model, LogisticRegression)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            if needs_scaling:
                scaler = StandardScaler()
                X_fold_train_proc = scaler.fit_transform(X_fold_train)
                X_fold_val_proc = scaler.transform(X_fold_val)
            else:
                X_fold_train_proc = X_fold_train.values
                X_fold_val_proc = X_fold_val.values

            fold_model = clone(model)
            fold_model.fit(X_fold_train_proc, y_fold_train)

            y_pred = fold_model.predict(X_fold_val_proc)
            try:
                y_proba = get_proba_safe(fold_model, X_fold_val_proc)
            except RuntimeError:
                # fallback: create dummy proba from predictions (bad but safe)
                # For binary: second column = predicted label (0/1)
                preds = np.asarray(y_pred)
                if preds.ndim == 1:
                    y_proba = np.vstack([1 - preds, preds]).T
                else:
                    # one-hot fallback
                    y_proba = np.eye(len(np.unique(y_fold_val)))[preds]

            n_classes = len(np.unique(y_fold_val))
            if n_classes == 2:
                roc_auc = roc_auc_score(y_fold_val, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_fold_val, y_proba, multi_class='ovr')

            metrics = {
                'fold': fold_idx + 1,
                'accuracy': float(accuracy_score(y_fold_val, y_pred)),
                'precision': float(precision_score(y_fold_val, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_fold_val, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)),
                'roc_auc': float(roc_auc),
                'cm': confusion_matrix(y_fold_val, y_pred).tolist(),
            }

            fold_reports[model_name].append(metrics)
            logger.info(f"  Fold {fold_idx+1:2d}: Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")

        # aggregate
        accs = [f['accuracy'] for f in fold_reports[model_name]]
        f1s = [f['f1'] for f in fold_reports[model_name]]
        rocs = [f['roc_auc'] for f in fold_reports[model_name]]

        cv_results[model_name] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'roc_auc_mean': float(np.mean(rocs)),
            'roc_auc_std': float(np.std(rocs)),
        }

        logger.info(f"[OK] CV Result: {cv_results[model_name]['accuracy_mean']:.4f} +/- {cv_results[model_name]['accuracy_std']:.4f}")

        # Save fold reports for this model
        out_dir = CONFIG['output_base'] / 'model_reports' / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'cv_fold_reports.json', 'w', encoding='utf-8') as f:
            json.dump(fold_reports[model_name], f, indent=2)

    return cv_results, fold_reports


# ========== FINAL TRAINING & TEST ==========
def final_training_and_test(X_train, X_test, y_train, y_test, models):
    logger.info("\n" + "="*80)
    logger.info("STEP 4: FINAL TRAINING & TEST EVALUATION")
    logger.info("="*80)

    test_results = {}
    trained_models = {}
    # We'll only scale for LogisticRegression at final training
    lr_scaler = None

    for model_name, model in models.items():
        logger.info(f"\n[TRAIN] Final training {model_name}...")

        needs_scaling = isinstance(model, LogisticRegression)
        if needs_scaling:
            lr_scaler = StandardScaler()
            X_train_proc = lr_scaler.fit_transform(X_train)
            X_test_proc = lr_scaler.transform(X_test)
        else:
            X_train_proc = X_train.values
            X_test_proc = X_test.values

        model.fit(X_train_proc, y_train)
        trained_models[model_name] = {'model': model, 'scaler': (lr_scaler if needs_scaling else None)}

        y_pred = model.predict(X_test_proc)
        try:
            y_proba = get_proba_safe(model, X_test_proc)
        except RuntimeError:
            preds = np.asarray(y_pred)
            if preds.ndim == 1:
                y_proba = np.vstack([1 - preds, preds]).T
            else:
                y_proba = np.eye(len(np.unique(y_test)))[preds]

        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        test_results[model_name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'roc_auc': float(roc_auc),
            'cm': confusion_matrix(y_test, y_pred).tolist(),
            'class_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }

        logger.info(f"[OK] Test Accuracy: {test_results[model_name]['accuracy']:.4f}")

        # Save test report
        out_dir = CONFIG['output_base'] / 'model_reports' / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'test_report.json', 'w', encoding='utf-8') as f:
            json.dump(test_results[model_name], f, indent=2)

    return test_results, trained_models


# ========== OVERFITTING ANALYSIS ==========
def analyze_overfitting(cv_results, test_results):
    logger.info("\n" + "="*80)
    logger.info("STEP 5: OVERFITTING/UNDERFITTING ANALYSIS")
    logger.info("="*80)

    analysis = {}

    for model_name in cv_results.keys():
        cv_acc = cv_results[model_name]['accuracy_mean']
        test_acc = test_results[model_name]['accuracy']
        gap = float(cv_acc - test_acc)
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
            'cv_accuracy': float(cv_acc),
            'test_accuracy': float(test_acc),
            'gap': float(gap),
            'gap_pct': float(gap_pct),
            'status': status,
        }

        logger.info(f"\n{model_name}:")
        logger.info(f"   CV Accuracy:   {cv_acc:.4f}")
        logger.info(f"   Test Accuracy: {test_acc:.4f}")
        logger.info(f"   Gap:           {gap:.4f} ({gap_pct:.1f}%)")
        logger.info(f"   Status:        {status}")

    # save analysis
    with open(CONFIG['output_base'] / 'model_reports' / 'overfitting_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    return analysis


# ========== MAIN ==========
def main():
    print("\n" + "="*100)
    print(" " * 20 + "COMPREHENSIVE SL TRAINING SYSTEM")
    if CONFIG['run_hpo']:
        print(" " * 15 + "With Hyperparameter Optimization (HPO)")
    print("=" * 100)

    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        models = build_models(X_train, y_train)
        cv_results, fold_reports = train_with_cv(X_train, y_train, models)
        test_results, trained_models = final_training_and_test(X_train, X_test, y_train, y_test, models)
        overfitting_analysis = analyze_overfitting(cv_results, test_results)

        logger.info("\n" + "=" * 80)
        logger.info("[COMPLETE] TRAINING FINISHED")
        logger.info("=" * 80)
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
