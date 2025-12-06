"""
THESIS FINAL AUDIT: SL vs RL COMPARISON (Fixed)
-----------------------------------------------
1. Loads dataset & creates immutable Test Set.
2. Loads Expert Models (SL).
3. Retrieves Neural Bandit (RL) metrics.
4. Generates Final Evidence Table.
5. PRINTS & SAVES CLASSIFICATION REPORT (Per-Class Metrics).
6. Saves Confusion Matrices.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report  # <--- Actually used now
)

# --- IMPORTS ---
from data_loader import load_training_data
from paths import MODEL_PATH, RL_RESULTS_PATH

# --- CONFIG ---
CONFIG = {
    'model_dir': MODEL_PATH,
    'output_dir': RL_RESULTS_PATH,
    'random_state': 42,
    'test_size': 0.2
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def get_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1: scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.zeros((len(X), 2))

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """Generates and saves a Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_dir / f'audit_cm_{model_name}.png')
    plt.close()

def main():
    print("="*80)
    print("FINAL THESIS AUDIT: FULL METRICS + CLASSIFICATION REPORT")
    print("="*80)

    # 1. LOAD DATA
    print("\n1. Establishing Ground Truth (Test Set)...")
    try:
        X_full, y_full = load_training_data()
    except Exception as e:
        print(f"Error: {e}")
        return

    _, X_test, _, y_test = train_test_split(
        X_full, y_full, test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], stratify=y_full
    )
    print(f"   Test Set Size: {len(X_test)} samples (Fixed Seed: {CONFIG['random_state']})")

    # 2. LOAD SL MODELS
    print("\n2. Loading Supervised Models...")
    models = {}
    sl_names = ['GradientBoosting', 'LightGBM', 'RandomForest', 'LogisticRegression']
    for name in sl_names:
        path = CONFIG['model_dir'] / name / f'{name}_model.pkl'
        if path.exists():
            models[name] = joblib.load(path)
            print(f"   [+] Loaded: {name}")

    # 3. RUN EVALUATION LOOP
    results = []
    print("\n3. Running Independent Audit...")
    
    # Open a text file to save the full classification reports
    report_file = CONFIG['output_dir'] / 'full_classification_reports.txt'
    with open(report_file, 'w') as f:
        f.write("THESIS DETAILED CLASSIFICATION REPORTS\n")
        f.write("======================================\n\n")

    for name, model in models.items():
        # Handle LR scaling
        if name == 'LogisticRegression':
            scaler_path = CONFIG['model_dir'] / name / f'{name}_scaler.pkl'
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_in = scaler.transform(X_test)
            else:
                X_in = X_test.values 
        else:
            X_in = X_test.values

        # Inference
        preds = model.predict(X_in)
        probs = get_proba_safe(model, X_in)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        auc_score = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # --- NEW: GENERATE & PRINT CLASSIFICATION REPORT ---
        report_str = classification_report(y_test, preds, digits=4)
        print(f"\n--- {name} ---")
        print(report_str)
        
        # Save to text file
        with open(report_file, 'a') as f:
            f.write(f"--- {name} ---\n")
            f.write(report_str + "\n\n")
        # ---------------------------------------------------
        
        results.append({
            'Model': name,
            'Type': 'Supervised (Static)',
            'Accuracy': acc,
            'ROC-AUC': auc_score,
            'F1-Score': f1,
            'Precision': prec,
            'Recall': rec
        })
        
        # Save CM Plot
        plot_confusion_matrix(y_test, preds, name, CONFIG['output_dir'])

    # 4. INTEGRATE RL RESULTS
    rl_metrics_file = CONFIG['output_dir'] / 'final_thesis_metrics.csv'
    
    if rl_metrics_file.exists():
        print("\n4. Retrieving Hybrid Bandit Results...")
        try:
            df_existing = pd.read_csv(rl_metrics_file)
            rl_row = df_existing[df_existing['Model'].str.contains("Hybrid")]
            
            if not rl_row.empty:
                cols = rl_row.columns
                rl_acc = rl_row.iloc[0]['Accuracy'] if 'Accuracy' in cols else 0
                rl_auc = rl_row.iloc[0]['ROC-AUC'] if 'ROC-AUC' in cols else 0
                rl_f1 = rl_row.iloc[0]['F1-Score'] if 'F1-Score' in cols else 0
                
                results.append({
                    'Model': 'Hybrid Neural Bandit',
                    'Type': 'Reinforcement Learning',
                    'Accuracy': rl_acc,
                    'ROC-AUC': rl_auc,
                    'F1-Score': rl_f1,
                    'Precision': rl_row.iloc[0]['Precision'] if 'Precision' in cols else 0,
                    'Recall': rl_row.iloc[0]['Recall'] if 'Recall' in cols else 0
                })
                print(f"   [+] Hybrid Agent: {rl_acc:.4f}")
        except Exception as e:
            print(f"   [!] Error reading RL metrics: {e}")

    # 5. GENERATE FINAL OUTPUTS
    df_res = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("FINAL THESIS LEADERBOARD (Full Metrics)")
    print("="*80)
    print(df_res.to_string(index=False))
    
    csv_path = CONFIG['output_dir'] / 'audit_comparison_full.csv'
    df_res.to_csv(csv_path, index=False)
    
    print(f"\n[+] Full Leaderboard saved to: {csv_path}")
    print(f"[+] Full Text Reports saved to: {report_file}")
    print(f"[+] Confusion Matrices saved to: {CONFIG['output_dir']}")

if __name__ == '__main__':
    main()