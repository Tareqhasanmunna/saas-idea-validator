import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Logger:
    @staticmethod
    def write_fold_report(log_dir, fold_num, metrics, model_name, y_true=None, y_pred=None):
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{model_name}_fold_{fold_num}.txt")
        with open(file_path, "w") as f:
            f.write(f"Model: {model_name} | Fold: {fold_num}\n")
            f.write("="*80 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            if y_true is not None and y_pred is not None:
                f.write("\nConfusion Matrix:\n")
                f.write(np.array2string(confusion_matrix(y_true, y_pred)))
        return file_path

    @staticmethod
    def write_summary_report(log_dir, model_name, avg_metrics):
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{model_name}_summary.txt")
        with open(file_path, "w") as f:
            f.write("model_name: {}\n".format(model_name))  # explicitly add model_name label
            f.write(f"{model_name} | 10-Fold CV Summary\n")
            f.write("="*80 + "\n")
            for metric, value in avg_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        return file_path
