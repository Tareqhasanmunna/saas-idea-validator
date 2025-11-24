import os
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_evaluation_report(policy_model, X_test, y_test, running_norm_obj, 
                              reward_history, acc_history, best_val, OUT_DIR, DEVICE):
    """
    Generate comprehensive evaluation report after training.
    """
    # Make predictions
    policy_model.eval()
    with torch.no_grad():
        X_test_np = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test
        y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test
        
        X_test_np = X_test_np.astype(np.float32)
        y_test_np = y_test_np.astype(np.int64)
        
        Xn = running_norm_obj.normalize(X_test_np)
        logits = policy_model(torch.from_numpy(Xn.astype(np.float32)).to(DEVICE))
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    
    # Calculate metrics
    test_acc = accuracy_score(y_test_np, preds)
    class_report = classification_report(y_test_np, preds, output_dict=True)
    conf_matrix = confusion_matrix(y_test_np, preds)
    
    # Training metrics
    final_train_acc = np.mean(acc_history[-50:]) if len(acc_history) > 0 else 0
    final_reward = np.mean(reward_history[-50:]) if len(reward_history) > 0 else 0
    
    # Create report dictionary
    report = {
        "timestamp": datetime.now().isoformat(),
        "training_summary": {
            "total_episodes": len(acc_history),
            "final_training_accuracy": float(final_train_acc),
            "final_training_reward": float(final_reward),
            "best_validation_accuracy": float(best_val)
        },
        "test_results": {
            "test_accuracy": float(test_acc),
            "confusion_matrix": conf_matrix.tolist()
        },
        "per_class_metrics": {}
    }
    
    # Add per-class metrics
    n_classes = len(np.unique(y_test_np))
    for c in range(n_classes):
        report["per_class_metrics"][str(c)] = {
            "precision": float(class_report.get(str(c), {}).get('precision', 0)),
            "recall": float(class_report.get(str(c), {}).get('recall', 0)),
            "f1-score": float(class_report.get(str(c), {}).get('f1-score', 0)),
            "support": int(class_report.get(str(c), {}).get('support', 0))
        }
    
    # Save JSON report
    report_path = os.path.join(OUT_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"✓ JSON report saved to {report_path}")
    
    # Save text report
    text_report_path = os.path.join(OUT_DIR, "evaluation_report.txt")
    with open(text_report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("RL TRAINING EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("TRAINING SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Episodes: {len(acc_history)}\n")
        f.write(f"Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final Training Reward: {final_reward:.4f}\n")
        f.write(f"Best Validation Accuracy: {best_val:.4f}\n\n")
        
        f.write("TEST SET RESULTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        
        f.write("CLASSIFICATION REPORT (Per Class)\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*60 + "\n")
        for c in range(n_classes):
            metrics = report["per_class_metrics"][str(c)]
            f.write(f"{c:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"Report generated at: {report['timestamp']}\n")
    
    print(f"✓ Text report saved to {text_report_path}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val:.4f}")
    print(f"Final Train Accuracy: {final_train_acc:.4f}")
    print("="*60 + "\n")
    
    return report
