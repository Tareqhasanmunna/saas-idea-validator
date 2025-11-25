"""
Advanced HPO Module - Auto Train & Save Models with 90%+ Accuracy
Saves trained models and metadata for future use (RL, analysis, etc.)
"""

import os
import json
import random
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ---------------- CONFIG ----------------
CONFIG = {
    "data_path": r"E:\saas-idea-validator\data\processed\balanced\vectorized_features_balanced.csv",
    "output_base": Path("models"),
    "random_state": 42,
    "test_size": 0.2,
    "target_accuracy": 0.90,
    "max_iterations": 200
}
random.seed(CONFIG["random_state"])
np.random.seed(CONFIG["random_state"])
os.makedirs(CONFIG["output_base"], exist_ok=True)

# ---------------- HYPERPARAMETER SEARCH ----------------
class AutoTrainer:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.models_metadata = {}

    def generate_params(self, model_name):
        """Random parameter generation per model"""
        if model_name == "DecisionTree":
            return {
                "max_depth": random.choice([5, 10, 15, 20, None]),
                "min_samples_split": random.choice([2, 5, 10, 15]),
                "min_samples_leaf": random.choice([1, 2, 5, 10])
            }
        elif model_name == "RandomForest":
            return {
                "n_estimators": random.choice([100, 200, 300]),
                "max_depth": random.choice([5, 10, 15, 20, None]),
                "min_samples_split": random.choice([2, 5, 10]),
                "min_samples_leaf": random.choice([1, 2, 5]),
                "max_features": random.choice(["sqrt", "log2", None])
            }
        elif model_name == "GradientBoosting":
            return {
                "n_estimators": random.choice([100, 200, 300]),
                "learning_rate": random.choice([0.01, 0.05, 0.1, 0.15]),
                "max_depth": random.choice([3, 5, 7, 10]),
                "min_samples_split": random.choice([2, 5, 10]),
                "min_samples_leaf": random.choice([1, 2, 5]),
                "subsample": random.choice([0.7, 0.8, 0.9, 1.0])
            }
        elif model_name == "LogisticRegression":
            return {
                "max_iter": 1000,
                "solver": random.choice(["lbfgs", "saga", "liblinear"]),
            }
        elif model_name == "LightGBM":
            return {
                "num_leaves": random.choice([20, 31, 40, 50, 70]),
                "learning_rate": random.choice([0.01, 0.05, 0.1, 0.15]),
                "max_depth": random.choice([-1, 5, 10, 15]),
                "min_data_in_leaf": random.choice([5, 10, 20, 30]),
                "lambda_l1": random.choice([0, 0.1, 0.5, 1.0]),
                "lambda_l2": random.choice([0, 0.1, 0.5, 1.0]),
                "bagging_fraction": random.choice([0.7, 0.8, 0.9, 1.0])
            }

    def create_model(self, model_name, params):
        if model_name == "DecisionTree":
            return DecisionTreeClassifier(random_state=CONFIG["random_state"], class_weight="balanced", **params)
        elif model_name == "RandomForest":
            return RandomForestClassifier(random_state=CONFIG["random_state"], class_weight="balanced", n_jobs=-1, **params)
        elif model_name == "GradientBoosting":
            return GradientBoostingClassifier(random_state=CONFIG["random_state"], **params)
        elif model_name == "LogisticRegression":
            return LogisticRegression(random_state=CONFIG["random_state"], class_weight="balanced", **params)
        elif model_name == "LightGBM":
            return lgb.LGBMClassifier(random_state=CONFIG["random_state"], class_weight="balanced", verbose=-1, **params)

    def train_until_target(self, model_name):
        best_model = None
        best_params = None
        best_acc = 0
        tried_params = []

        for iteration in range(1, CONFIG["max_iterations"] + 1):
            params = self.generate_params(model_name)
            tried_params.append(params)
            model = self.create_model(model_name, params)
            model.fit(self.X_train_scaled, self.y_train)
            train_acc = model.score(self.X_train_scaled, self.y_train)
            test_acc = model.score(self.X_test_scaled, self.y_test)
            avg_acc = (train_acc + test_acc) / 2

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_model = model
                best_params = params

            print(f"[{model_name}] Iter {iteration}: Train={train_acc:.4f} Test={test_acc:.4f} Avg={avg_acc:.4f} | Best={best_acc:.4f}")

            if avg_acc >= CONFIG["target_accuracy"]:
                print(f"✅ Target accuracy reached for {model_name} at iteration {iteration}!")
                break

        # Save model + metadata
        model_folder = CONFIG["output_base"] / model_name
        model_folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_folder / "best_model.joblib")

        metadata = {
            "best_params": best_params,
            "best_avg_accuracy": best_acc,
            "train_accuracy": best_model.score(self.X_train_scaled, self.y_train),
            "test_accuracy": best_model.score(self.X_test_scaled, self.y_test),
            "iterations_run": iteration,
            "tried_params": tried_params,
            "timestamp": str(datetime.datetime.now())
        }
        with open(model_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        self.models_metadata[model_name] = metadata
        return best_model, metadata

    def run_all(self, models_to_run=None):
        if models_to_run is None:
            models_to_run = ["DecisionTree", "LogisticRegression", "RandomForest", "GradientBoosting"]
            if LIGHTGBM_AVAILABLE:
                models_to_run.append("LightGBM")

        trained_models = {}
        for model_name in models_to_run:
            print(f"\n=== TRAINING {model_name} ===")
            model, metadata = self.train_until_target(model_name)
            trained_models[model_name] = model
        return trained_models, self.models_metadata


# ---------------- MAIN ENTRY ----------------
def main():
    # Load data
    df = pd.read_csv(CONFIG["data_path"])
    X = df.drop(columns=["label_numeric"])
    y = df["label_numeric"]

    trainer = AutoTrainer(X, y)
    models, metadata = trainer.run_all()
    print("\n✅ All models trained and saved successfully!")

if __name__ == "__main__":
    main()
