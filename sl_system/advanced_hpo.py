"""
advanced_hpo.py
----------------
Auto Hyperparameter Trainer for SL Models (LightGBM, etc.)
Tries random hyperparameters until target accuracy is reached and saves model + metadata.

Usage:
    from advanced_hpo import AutoTrainer
    trainer = AutoTrainer(X_train, y_train, target_accuracy=0.90)
    best_model, best_metadata = trainer.train_model_until_target('LightGBM')
"""

import json
import pickle
import random
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

class AutoTrainer:
    def __init__(self, X_train, y_train, target_accuracy=0.90, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.target_accuracy = target_accuracy
        self.random_state = random_state
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def generate_lgb_params(self):
        """Randomly generate LightGBM hyperparameters"""
        params = {
            'num_leaves': random.choice([20, 30, 40, 50, 70]),
            'learning_rate': random.choice([0.01, 0.05, 0.1, 0.15]),
            'max_depth': random.choice([5, 7, 10, 15, 20]),
            'min_data_in_leaf': random.choice([10, 20, 30, 40]),
            'lambda_l1': random.choice([0, 0.1, 0.5, 1.0]),
            'lambda_l2': random.choice([0, 0.1, 0.5, 1.0]),
            'feature_fraction': random.choice([0.7, 0.8, 0.9, 1.0]),
            'bagging_fraction': random.choice([0.7, 0.8, 0.9, 1.0]),
        }
        return params

    def train_model_until_target(self, model_name='LightGBM', max_iterations=200):
        """
        Train the model with random hyperparameters until target accuracy is reached.
        Saves model + metadata to models/<ModelName>/ directory.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)

        best_accuracy = 0
        best_params = None
        best_model = None
        results = []

        model_dir = Path("models") / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, max_iterations + 1):
            if model_name == 'LightGBM':
                params = self.generate_lgb_params()
                model = lgb.LGBMClassifier(
                    **params,
                    n_estimators=200,
                    random_state=self.random_state,
                    class_weight='balanced',
                    verbose=-1
                )
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")

            try:
                model.fit(X_scaled, self.y_train)
                cv_scores = cross_val_score(model, X_scaled, self.y_train, cv=5, scoring='accuracy')
                cv_accuracy = cv_scores.mean()
            except Exception as e:
                print(f"[WARN] Iteration {iteration} failed: {e}")
                continue

            results.append({'iteration': iteration, 'params': params, 'cv_accuracy': cv_accuracy})

            if cv_accuracy > best_accuracy:
                best_accuracy = cv_accuracy
                best_params = params
                best_model = model

            print(f"Iteration {iteration:3d} | CV Acc: {cv_accuracy:.4f} | Best: {best_accuracy:.4f}")

            if best_accuracy >= self.target_accuracy:
                print(f"\n✓ Target accuracy reached: {best_accuracy:.4f} at iteration {iteration}")
                break

        if best_model is not None:
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)

            # Save scaler
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            # Save metadata
            metadata = {
                'best_accuracy': float(best_accuracy),
                'best_params': best_params,
                'total_iterations': iteration,
                'all_results': results
            }
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"Saved model and metadata to: {model_dir}")

        return best_model, metadata
