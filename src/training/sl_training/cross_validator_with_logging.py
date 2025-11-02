import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

class CrossValidatorWithLogging:
    @staticmethod
    def perform_cross_validation(model_name, model, X, y, cv_folds, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        print(f"[CV] {model_name} - {cv_folds}-Fold Cross-Validation Started")
        
        for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=cv_folds, desc=f"{model_name}", ncols=100)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_val, y_pred)

            # Save fold report
            with open(os.path.join(log_dir, f"fold_{fold+1}_report.txt"), "w") as f:
                f.write(f"Fold {fold+1} Metrics for {model_name}\n")
                f.write("="*40 + "\n")
                f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(cm))
            
            fold_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        # Save summary
        with open(os.path.join(log_dir, "summary.txt"), "w") as f:
            f.write(f"Average Metrics for {model_name}\n")
            f.write("="*40 + "\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        print(f"[CV] {model_name} - Completed")
        return avg_metrics
