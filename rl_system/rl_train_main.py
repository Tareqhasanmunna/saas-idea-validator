"""
RL Training Main Script

End-to-end training of RL agent starting from best SL model.

Usage:
    python rl_train_main.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from pathlib import Path
import json

# Import RL modules
from rl_environment import SaaSValidatorEnvironment
from rl_agent import RLAgent
from rl_evaluation import RLEvaluator
from rl_visualization import RLVisualizer
from sl_system.data_loader import load_training_data

def main():
    """Main RL training pipeline"""

    print("\n" + "="*70)
    print("SaaS Idea Validator - RL Training Pipeline")
    print("="*70)

    # Configuration
    config = {
        'best_model_json': 'ml_outputs/models/LightGBM_model.json',
        'episodes': 100,
        'reward_scheme': 'balanced',
        'gamma': 0.99,
        'device': 'cpu',  # Change to 'cuda' if GPU available
        'test_split': 0.2,
    }

    print(f"\nConfiguration:")
    print(f"  Episodes: {config['episodes']}")
    print(f"  Reward scheme: {config['reward_scheme']}")
    print(f"  Device: {config['device']}")

    # 1. Load data
    print(f"\n{'='*70}")
    print("STEP 1: Loading Data")
    print(f"{'='*70}")

    X, y = load_training_data()

    # Train-test split
    n_samples = len(X)
    split_idx = int(n_samples * (1 - config['test_split']))

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"✓ Data loaded and split")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # 2. Initialize environment
    print(f"\n{'='*70}")
    print("STEP 2: Initializing RL Environment")
    print(f"{'='*70}")

    env = SaaSValidatorEnvironment(X_train, y_train, reward_scheme=config['reward_scheme'])
    print(f"✓ Environment initialized")
    print(f"  State space: {env.n_features}")
    print(f"  Action space: {env.n_samples} samples")

    # 3. Initialize RL Agent
    print(f"\n{'='*70}")
    print("STEP 3: Initializing RL Agent")
    print(f"{'='*70}")

    agent = RLAgent(
        best_model_json_path=config['best_model_json'],
        n_actions=3,
        n_features=X_train.shape[1],
        device=config['device']
    )

    # 4. Train agent
    print(f"\n{'='*70}")
    print("STEP 4: Training RL Agent")
    print(f"{'='*70}")

    training_history = agent.train(
        env,
        episodes=config['episodes'],
        gamma=config['gamma'],
        verbose=True
    )

    # Save trained policy
    policy_save_path = 'ml_outputs/rl_models/rl_policy_best.pt'
    Path(policy_save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save_policy(policy_save_path)

    # 5. Evaluate agent
    print(f"\n{'='*70}")
    print("STEP 5: Evaluating RL Agent")
    print(f"{'='*70}")

    evaluator = RLEvaluator(agent, X_test.values, y_test)
    rl_metrics = evaluator.evaluate()

    # Get baseline predictions from best SL model for comparison
    print(f"\nGetting baseline predictions from SL model...")
    try:
        # Load best model predictions or retrain quick model
        # For now, use simple prediction as baseline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        baseline_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        baseline_model.fit(X_train_scaled, y_train)
        baseline_predictions = baseline_model.predict(X_test_scaled)

    except Exception as e:
        print(f"⚠ Could not load baseline: {e}")
        baseline_predictions = np.zeros(len(y_test), dtype=int)

    # 6. Compare with baseline
    print(f"\n{'='*70}")
    print("STEP 6: Comparison Analysis")
    print(f"{'='*70}")

    comparison = evaluator.compare_with_baseline(baseline_predictions)

    # 7. Visualization
    print(f"\n{'='*70}")
    print("STEP 7: Generating Visualizations")
    print(f"{'='*70}")

    viz_dir = 'ml_outputs/rl_visualizations'
    visualizer = RLVisualizer(output_dir=viz_dir)

    visualizer.plot_training_history(training_history)
    visualizer.plot_convergence_analysis(training_history, window_size=5)
    visualizer.plot_comparison(comparison)
    visualizer.plot_confusion_matrix(rl_metrics['confusion_matrix'], 'RL Agent Confusion Matrix')

    if 'per_class_metrics' in rl_metrics and isinstance(rl_metrics['per_class_metrics'], dict):
        visualizer.plot_per_class_metrics(rl_metrics['per_class_metrics'])

    # Generate summary report
    report = visualizer.generate_summary_report(training_history, comparison)

    # 8. Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✓ Results Summary:")
    print(f"  RL Agent Accuracy: {comparison['rl_metrics']['accuracy']:.4f}")
    print(f"  Baseline Accuracy: {comparison['baseline_metrics']['accuracy']:.4f}")
    print(f"  Accuracy Improvement: {comparison['improvements']['accuracy']:.4f}")
    print(f"\n  RL Agent F1 Score: {comparison['rl_metrics']['f1']:.4f}")
    print(f"  Baseline F1 Score: {comparison['baseline_metrics']['f1']:.4f}")
    print(f"  F1 Improvement: {comparison['improvements']['f1']:.4f}")
    print(f"\n✓ Outputs saved to:")
    print(f"  Policy: {policy_save_path}")
    print(f"  Visualizations: {viz_dir}/")
    print(f"  Report: {viz_dir}/rl_summary_report.json")
    print(f"{'='*70}\n")

    return agent, training_history, comparison


if __name__ == '__main__':
    agent, history, comparison = main()