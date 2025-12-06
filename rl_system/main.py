"""
FINAL THESIS SYSTEM: HYBRID NEURAL BANDIT
-----------------------------------------
1. Loads pre-trained Experts.
2. Stacks predictions to train Neural Bandit.
3. Evaluates Bandit on Test Set.
4. Generates & Saves CLASSIFICATION REPORT (Precision/Recall per class).
5. AUTOMATICALLY evaluates Experts on the same Test Set.
6. Generates the Final "SL vs RL" Leaderboard.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    confusion_matrix, roc_curve, auc, f1_score, 
    precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- IMPORTS ---
from data_loader import load_training_data
from bandit_agent import NeuralBanditAgent
from paths import MODEL_PATH, RL_VIS_PATH, RL_RESULTS_PATH

# --- CONFIG ---
CONFIG = {
    'model_dir': MODEL_PATH,
    'vis_dir': RL_VIS_PATH,
    'comparison_dir': RL_RESULTS_PATH,
    'random_state': 42,
    'test_size': 0.2
}

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

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Hybrid Bandit Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2ca02c', lw=3, label=f'Hybrid Agent (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Hybrid Bandit ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("\n" + "="*80)
    print("HYBRID NEURAL BANDIT SYSTEM (Training & Audit)")
    print("="*80)

    # 1. LOAD DATA
    try:
        X_full, y_full = load_training_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. LOAD EXPERTS
    print("\n1. Loading Expert Models (The Boardroom)...")
    experts = {}
    expert_names = ['GradientBoosting', 'LightGBM', 'RandomForest']
    
    for name in expert_names:
        model_path = CONFIG['model_dir'] / name / f'{name}_model.pkl'
        if model_path.exists():
            experts[name] = joblib.load(model_path)
            print(f"   [+] Expert Joined: {name}")
        else:
            print(f"   [-] Warning: {name} not found. Skipping.")

    if not experts:
        print("Error: No experts found. Run main.py first.")
        return

    # 3. GENERATE META-FEATURES
    print("\n2. Generating Expert Opinions (Meta-State)...")
    meta_features = []
    X_raw_values = X_full.values
    
    for name, model in experts.items():
        probs = get_proba_safe(model, X_raw_values)[:, 1]
        meta_features.append(probs)
    
    meta_matrix = np.column_stack(meta_features)
    X_stacked = np.hstack([X_raw_values, meta_matrix])
    
    print(f"   Original Features: {X_full.shape[1]}")
    print(f"   Expert Inputs:     {len(experts)}")
    print(f"   Final State Size:  {X_stacked.shape[1]}")

    # 4. SPLIT DATA
    X_train_stack, X_test_stack, y_train, y_test = train_test_split(
        X_stacked, y_full, test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], stratify=y_full
    )
    
    _, X_test_raw, _, _ = train_test_split(
        X_raw_values, y_full, test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], stratify=y_full
    )

    # 5. TRAIN NEURAL BANDIT
    print("\n3. Training Neural Bandit Agent (Meta-Learning)...")
    agent = NeuralBanditAgent(state_size=X_train_stack.shape[1], action_size=2)
    agent.train(X_train_stack, y_train, epochs=100) 

    # 6. EVALUATION & COMPARISON
    print("\n4. Running Final Thesis Audit (Head-to-Head)...")
    
    comparison_results = []
    
    # --- A. Evaluate Bandit ---
    bandit_preds = []
    bandit_probs = []
    
    for i in range(len(X_test_stack)):
        state = X_test_stack[i]
        bandit_preds.append(agent.act(state))
        bandit_probs.append(agent.predict_proba(state))
        
    b_acc = accuracy_score(y_test, bandit_preds)
    b_auc = roc_auc_score(y_test, bandit_probs)
    b_f1 = f1_score(y_test, bandit_preds)
    b_prec = precision_score(y_test, bandit_preds)
    b_rec = recall_score(y_test, bandit_preds)
    
    comparison_results.append({
        'Model': 'Hybrid Neural Bandit (RL)',
        'Accuracy': b_acc,
        'ROC-AUC': b_auc,
        'F1-Score': b_f1,
        'Precision': b_prec,
        'Recall': b_rec
    })
    
    # --- NEW: PRINT & SAVE CLASSIFICATION REPORT ---
    report_str = classification_report(y_test, bandit_preds, digits=4)
    print("\n--- HYBRID BANDIT CLASSIFICATION REPORT ---")
    print(report_str)
    
    # Save report to file
    with open(CONFIG['comparison_dir'] / 'bandit_report.txt', 'w') as f:
        f.write("HYBRID NEURAL BANDIT - FINAL THESIS REPORT\n")
        f.write("==========================================\n\n")
        f.write(report_str)
    print(f"   [+] Detailed Report saved to: {CONFIG['comparison_dir'] / 'bandit_report.txt'}")
    # -----------------------------------------------

    print(f"   > Bandit Accuracy: {b_acc:.4f} | Recall: {b_rec:.4f}")
    plot_confusion_matrix(y_test, bandit_preds, CONFIG['vis_dir'] / 'hybrid_confusion_matrix.png')
    plot_roc_curve(y_test, bandit_probs, CONFIG['vis_dir'] / 'hybrid_roc_curve.png')
    agent.save(CONFIG['model_dir'] / 'final_bandit_agent.pkl')

    # --- B. Evaluate Experts ---
    for name, model in experts.items():
        sl_preds = model.predict(X_test_raw)
        sl_probs = get_proba_safe(model, X_test_raw)[:, 1]
        
        sl_acc = accuracy_score(y_test, sl_preds)
        sl_auc = roc_auc_score(y_test, sl_probs)
        sl_f1 = f1_score(y_test, sl_preds)
        sl_prec = precision_score(y_test, sl_preds)
        sl_rec = recall_score(y_test, sl_preds)
        
        comparison_results.append({
            'Model': name,
            'Accuracy': sl_acc,
            'ROC-AUC': sl_auc,
            'F1-Score': sl_f1,
            'Precision': sl_prec,
            'Recall': sl_rec
        })
        print(f"   > {name} Accuracy: {sl_acc:.4f}")

    # 7. GENERATE FINAL LEADERBOARD
    df_results = pd.DataFrame(comparison_results).sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("FINAL THESIS LEADERBOARD")
    print("="*80)
    print(df_results.to_string(index=False))
    
    csv_path = CONFIG['comparison_dir'] / 'final_thesis_metrics.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n[+] Leaderboard saved to: {csv_path}")
    print("[+] Plots saved to visualizations folder.")
    print("System Complete.")

if __name__ == '__main__':
    main()