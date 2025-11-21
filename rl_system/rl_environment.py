"""
RL Environment for SaaS Idea Validator

Simulates the environment where the RL agent makes decisions (predictions)
and receives rewards based on ground-truth labels from the dataset.

Usage:
    env = SaaSValidatorEnvironment(X, y)
    state = env.reset()
    action, prob = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
"""

import numpy as np
from typing import Tuple, Dict


class SaaSValidatorEnvironment:
    """Environment for RL agent interaction"""

    def __init__(self, X: np.ndarray, y: np.ndarray, reward_scheme: str = 'accuracy'):
        """
        Initialize environment

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            reward_scheme: 'accuracy', 'f1', or 'balanced'
        """
        self.X = X
        self.y = y
        self.reward_scheme = reward_scheme
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.current_idx = 0
        self.episode_steps = 0
        self.max_steps = self.n_samples

        # Track statistics
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_correct_predictions = 0

    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.current_idx = 0
        self.episode_steps = 0
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_correct_predictions = 0

        return self.X.iloc[self.current_idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in environment

        Args:
            action: Predicted class label (0, 1, or 2)

        Returns:
            next_state: Next feature vector
            reward: Reward signal for this action
            done: Whether episode is complete
            info: Additional information
        """
        true_label = self.y.iloc[self.current_idx] if hasattr(self.y, 'iloc') else self.y[self.current_idx]

        # Compute reward based on prediction correctness
        is_correct = (action == true_label)
        reward = self._compute_reward(action, true_label, is_correct)

        if is_correct:
            self.episode_correct_predictions += 1

        self.episode_rewards.append(reward)
        self.episode_steps += 1
        self.current_idx += 1

        done = (self.current_idx >= self.n_samples)

        # Get next state or terminal state
        if done:
            next_state = np.zeros(self.n_features)  # Terminal state
        else:
            next_state = self.X.iloc[self.current_idx] if not isinstance(self.X, np.ndarray) else self.X[self.current_idx]

        info = {
            'true_label': true_label,
            'predicted_label': action,
            'is_correct': is_correct,
            'episode_accuracy': self.episode_correct_predictions / self.episode_steps,
        }

        return next_state, reward, done, info

    def _compute_reward(self, action: int, true_label: int, is_correct: bool) -> float:
        """
        Compute reward based on prediction

        Reward scheme options:
        - accuracy: +1 for correct, -0.5 for incorrect
        - f1: +1 for correct, varying penalty by class
        - balanced: Class-balanced rewards
        """
        if self.reward_scheme == 'accuracy':
            return 1.0 if is_correct else -0.5

        elif self.reward_scheme == 'f1':
            # Higher reward for correct predictions of minority class (0)
            if is_correct:
                if true_label == 0:
                    return 2.0  # Reward for correctly identifying rare class
                else:
                    return 1.0
            else:
                if true_label == 0:
                    return -2.0  # Penalty for missing rare class
                else:
                    return -0.5

        elif self.reward_scheme == 'balanced':
            # Class-balanced rewards
            class_weights = {0: 2.0, 1: 1.0, 2: 1.0}  # Weight minority class higher
            weight = class_weights.get(true_label, 1.0)
            return weight if is_correct else -weight * 0.5

        else:
            return 1.0 if is_correct else -0.5

    def get_episode_summary(self) -> Dict:
        """Get summary statistics for current episode"""
        if len(self.episode_rewards) == 0:
            return {'episode_reward': 0, 'episode_accuracy': 0}

        return {
            'episode_reward': np.mean(self.episode_rewards),
            'total_reward': np.sum(self.episode_rewards),
            'episode_accuracy': self.episode_correct_predictions / self.episode_steps,
            'steps': self.episode_steps,
        }