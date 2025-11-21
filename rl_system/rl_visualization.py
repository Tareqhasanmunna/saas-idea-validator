"""
RL Visualization Module

Visualizes RL training progress, convergence, and evaluation results.
Generates plots for thesis documentation.

Usage:
    visualizer = RLVisualizer(output_dir='ml_outputs/rl_visualizations')
    visualizer.plot_training_history(training_history)
    visualizer.plot_comparison(comparison_metrics)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json


class RLVisualizer:
    """Visualize RL training and evaluation"""

    def __init__(self, output_dir: str = 'ml_outputs/rl_visualizations'):
        """Initialize visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)

        print(f"✓ RL Visualizer initialized")
        print(f"  Output directory: {self.output_dir}")

    def plot_training_history(self, history: Dict):
        """
        Plot training history metrics

        Args:
            history: Dict with training metrics
        """
        print(f"\nGenerating training history visualizations...")

        episodes = range(1, len(history['episode_rewards']) + 1)

        # Plot 1: Episode Rewards and Accuracy
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode Rewards
        axes[0, 0].plot(episodes, history['episode_rewards'], label='Episode Reward', linewidth=2)
        axes[0, 0].fill_between(episodes, history['episode_rewards'], alpha=0.3)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('RL Training: Episode Rewards Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Episode Accuracy
        axes[0, 1].plot(episodes, history['episode_accuracies'], label='Episode Accuracy', 
                       color='green', linewidth=2)
        axes[0, 1].fill_between(episodes, history['episode_accuracies'], alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title('RL Training: Episode Accuracy Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Actor Loss
        axes[1, 0].plot(episodes, history['actor_loss'], label='Actor Loss', color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('RL Training: Actor Loss Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Value Loss
        axes[1, 1].plot(episodes, history['value_loss'], label='Value Loss', 
                       color='red', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('RL Training: Critic (Value) Loss Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        save_path = self.output_dir / 'rl_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def plot_convergence_analysis(self, history: Dict, window_size: int = 10):
        """
        Plot convergence analysis with moving averages

        Args:
            history: Training history
            window_size: Window for moving average
        """
        episodes = range(1, len(history['episode_rewards']) + 1)

        # Calculate moving averages
        rewards_ma = self._moving_average(history['episode_rewards'], window_size)
        accuracy_ma = self._moving_average(history['episode_accuracies'], window_size)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Rewards convergence
        axes[0].plot(episodes, history['episode_rewards'], alpha=0.3, label='Raw Episode Reward')
        axes[0].plot(episodes[window_size-1:], rewards_ma, label=f'{window_size}-Episode MA', 
                    linewidth=2, color='red')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title('Reward Convergence (Moving Average)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Accuracy convergence
        axes[1].plot(episodes, history['episode_accuracies'], alpha=0.3, label='Raw Episode Accuracy')
        axes[1].plot(episodes[window_size-1:], accuracy_ma, label=f'{window_size}-Episode MA', 
                    linewidth=2, color='green')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim([0, 1])
        axes[1].set_title('Accuracy Convergence (Moving Average)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        save_path = self.output_dir / 'rl_convergence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def plot_comparison(self, comparison: Dict):
        """
        Plot RL vs Baseline comparison

        Args:
            comparison: Comparison metrics dict
        """
        print(f"\nGenerating comparison visualizations...")

        metrics = ['accuracy', 'f1', 'precision', 'recall']
        rl_values = [comparison['rl_metrics'][m] for m in metrics]
        baseline_values = [comparison['baseline_metrics'][m] for m in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Metrics comparison
        x = np.arange(len(metrics))
        width = 0.35

        axes[0].bar(x - width/2, baseline_values, width, label='Baseline SL Model', alpha=0.8)
        axes[0].bar(x + width/2, rl_values, width, label='RL Agent', alpha=0.8)
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('RL Agent vs Baseline Model Performance')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].set_ylim([0, 1])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Improvements
        improvements = [comparison['improvements'][m] for m in metrics]
        colors = ['green' if imp >= 0 else 'red' for imp in improvements]

        axes[1].bar(metrics, improvements, color=colors, alpha=0.8)
        axes[1].set_ylabel('Improvement')
        axes[1].set_title('RL Agent Improvements Over Baseline')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(improvements):
            axes[1].text(i, v + 0.01 if v >= 0 else v - 0.02, f'{v:.4f}', 
                        ha='center', va='bottom' if v >= 0 else 'top')

        plt.tight_layout()
        save_path = self.output_dir / 'rl_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def plot_confusion_matrix(self, cm: List, title: str = 'Confusion Matrix'):
        """Plot confusion matrix heatmap"""
        cm_array = np.array(cm)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Bad', 'Neutral', 'Good'])
        ax.set_yticklabels(['Bad', 'Neutral', 'Good'])

        plt.tight_layout()
        save_path = self.output_dir / 'rl_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def plot_per_class_metrics(self, per_class_metrics: Dict):
        """Plot per-class evaluation metrics"""
        classes = ['Bad (0)', 'Neutral (1)', 'Good (2)']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_to_plot = ['precision', 'recall', 'f1-score']

        for idx, metric in enumerate(metrics_to_plot):
            values = [per_class_metrics[str(i)].get(metric, 0) for i in range(3)]
            colors = ['#ff9999', '#66b3ff', '#99ff99']

            axes[idx].bar(classes, values, color=colors, alpha=0.8)
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'Per-Class {metric.title()}')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        save_path = self.output_dir / 'rl_per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    @staticmethod
    def _moving_average(values: List, window: int) -> np.ndarray:
        """Calculate moving average"""
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, mode='valid')

    def generate_summary_report(self, training_history: Dict, comparison: Dict, 
                               output_file: str = 'rl_summary_report.json'):
        """Generate JSON summary report"""
        report = {
            'training_summary': {
                'total_episodes': len(training_history['episode_rewards']),
                'final_accuracy': float(training_history['episode_accuracies'][-1]),
                'final_reward': float(training_history['episode_rewards'][-1]),
                'max_accuracy': float(max(training_history['episode_accuracies'])),
                'max_reward': float(max(training_history['episode_rewards'])),
            },
            'rl_performance': comparison['rl_metrics'],
            'baseline_performance': comparison['baseline_metrics'],
            'improvements': comparison['improvements'],
        }

        report_path = self.output_dir / output_file
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Summary report saved: {report_path}")
        return report