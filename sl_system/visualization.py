import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for scripts

import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = r'E:\saas-idea-validator\ml_outputs\visualizations'
REPORTS_DIR = r'E:\saas-idea-validator\ml_outputs\reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model comparison CSV
df = pd.read_csv(os.path.join(REPORTS_DIR, 'model_comparison.csv'))

# 1. Model Comparison Plot (Accuracy & F1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].bar(df['Model'], df['Accuracy_Mean'], yerr=df['Accuracy_Std'], capsize=5, color='skyblue')
ax[0].set_title('Model Accuracy Comparison')
ax[0].set_ylim(0, 1)
ax[0].set_ylabel('Accuracy')

ax[1].bar(df['Model'], df['F1_Mean'], yerr=df['F1_Std'], capsize=5, color='lightgreen')
ax[1].set_title('Model F1 Score Comparison')
ax[1].set_ylim(0, 1)
ax[1].set_ylabel('F1 Score')

plt.tight_layout()
model_comp_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
plt.savefig(model_comp_path)
plt.close()
print(f'Saved model comparison plot to {model_comp_path}')

# 2. Per-model fold metrics & confusion matrices
for model_name in df['Model']:
    batch_report_file = os.path.join(REPORTS_DIR, f'{model_name}_batch_reports.json')
    if not os.path.isfile(batch_report_file):
        print(f"Batch reports missing for {model_name}, skipping...")
        continue

    with open(batch_report_file) as f:
        batch_reports = json.load(f)

    folds = [r['fold'] for r in batch_reports]
    accuracies = [r['accuracy'] for r in batch_reports]
    f1_scores = [r['f1_weighted'] for r in batch_reports]

    # Plot fold accuracy & F1
    plt.figure(figsize=(10, 5))
    plt.plot(folds, accuracies, 'o-', label='Accuracy')
    plt.plot(folds, f1_scores, 's-', label='F1 Score')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'{model_name} Per-Fold Accuracy & F1')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    fold_metrics_path = os.path.join(OUTPUT_DIR, f'{model_name}_fold_metrics.png')
    plt.savefig(fold_metrics_path)
    plt.close()
    print(f'Saved {model_name} fold metrics plot to {fold_metrics_path}')

    # Confusion matrix for fold 1
    cm = np.array(batch_reports[0]['cm'])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title(f'{model_name} Confusion Matrix - Fold 1')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    conf_matrix_path = os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix_fold1.png')
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f'Saved {model_name} confusion matrix heatmap to {conf_matrix_path}')
