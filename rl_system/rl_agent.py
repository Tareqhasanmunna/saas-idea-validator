"""
RL Agent for SaaS Idea Validator

Loads best SL model (from JSON) and improves it through policy learning.
Uses Actor-Critic or Policy Gradient methods.

Usage:
    agent = RLAgent(best_model_json_path, n_actions=3, n_features=104)
    agent.train(env, episodes=100, learning_rate=0.001)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, List, Dict


class PolicyNetwork(nn.Module):
    """Neural network for policy (actor)"""

    def __init__(self, n_features: int, n_actions: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)  # For critic/value estimation

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns action probabilities and value estimate"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))

        action_probs = self.softmax(self.policy_head(x))
        value = self.value_head(x)

        return action_probs, value


class RLAgent:
    """Reinforcement Learning Agent with Actor-Critic"""

    def __init__(self, best_model_json_path: str, n_actions: int = 3, n_features: int = 104,
                 hidden_dim: int = 128, device: str = 'cpu'):
        """
        Initialize RL agent

        Args:
            best_model_json_path: Path to best SL model JSON
            n_actions: Number of action classes (e.g., 3 for Good/Neutral/Bad)
            n_features: Input feature dimension
            hidden_dim: Hidden layer size
            device: 'cpu' or 'cuda'
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.device = device
        self.best_model_json_path = best_model_json_path

        # Initialize policy network
        self.policy_net = PolicyNetwork(n_features, n_actions, hidden_dim).to(device)

        # Load SL model info to warm-start policy
        self._load_sl_model_info(best_model_json_path)

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_accuracies': [],
            'actor_loss': [],
            'value_loss': [],
            'total_loss': []
        }

        print(f"✓ RL Agent initialized")
        print(f"  Actions: {n_actions}, Features: {n_features}")
        print(f"  Device: {device}")

    def _load_sl_model_info(self, json_path: str):
        """Load SL model info and warm-start policy network"""
        try:
            with open(json_path, 'r') as f:
                model_data = json.load(f)

            print(f"✓ Loaded SL model: {model_data['model_info']['name']}")
            print(f"  Classes: {model_data['classes']}")

            # Note: Full weight transfer is complex, so we initialize with default weights
            # In production, you'd extract feature importance and initialize accordingly

        except Exception as e:
            print(f"⚠ Could not load model info: {e}")

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Get action from policy network

        Args:
            state: Current state (feature vector)

        Returns:
            action: Selected action (class label)
            action_prob: Probability of selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor)

        # Sample action from probability distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_prob = action_probs[0, action.item()].item()

        return action.item(), action_prob

    def get_action_deterministic(self, state: np.ndarray) -> int:
        """Get deterministic action (greedy policy)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor)

        action = torch.argmax(action_probs, dim=1)
        return action.item()

    def train_step(self, state: np.ndarray, action: int, reward: float, 
                   next_state: np.ndarray, done: bool, gamma: float = 0.99) -> Dict:
        """
        Single training step using Actor-Critic algorithm

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            gamma: Discount factor

        Returns:
            losses: Dict with actor_loss, value_loss, total_loss
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Forward pass
        action_probs, value = self.policy_net(state_tensor)
        _, next_value = self.policy_net(next_state_tensor)

        # Compute TD target and advantage
        if done:
            target = reward
        else:
            target = reward + gamma * next_value.item()

        advantage = target - value.item()

        # Actor loss (policy gradient)
        log_prob = torch.log(action_probs[0, action] + 1e-8)
        actor_loss = -log_prob * advantage

        # Critic loss (value function)
        value_loss = (torch.FloatTensor([target]).to(self.device) - value).pow(2)

        # Total loss
        total_loss = actor_loss + 0.5 * value_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'advantage': advantage
        }

    def train(self, env, episodes: int = 100, gamma: float = 0.99, verbose: bool = True) -> Dict:
        """
        Train RL agent on environment

        Args:
            env: Environment instance
            episodes: Number of training episodes
            gamma: Discount factor
            verbose: Print progress

        Returns:
            training_history: Dict with training metrics
        """
        print(f"\nStarting RL training for {episodes} episodes...")
        print(f"{'='*70}")

        for episode in range(episodes):
            state = env.reset()
            episode_loss = {'actor': [], 'value': [], 'total': []}
            done = False

            while not done:
                # Agent selects action
                action, _ = self.get_action(state)

                # Environment responds
                next_state, reward, done, info = env.step(action)

                # Train agent
                loss_dict = self.train_step(state, action, reward, next_state, done, gamma)
                episode_loss['actor'].append(loss_dict['actor_loss'])
                episode_loss['value'].append(loss_dict['value_loss'])
                episode_loss['total'].append(loss_dict['total_loss'])

                state = next_state

            # Episode summary
            episode_summary = env.get_episode_summary()

            self.training_history['episode_rewards'].append(episode_summary['episode_reward'])
            self.training_history['episode_accuracies'].append(episode_summary['episode_accuracy'])
            self.training_history['actor_loss'].append(np.mean(episode_loss['actor']))
            self.training_history['value_loss'].append(np.mean(episode_loss['value']))
            self.training_history['total_loss'].append(np.mean(episode_loss['total']))

            if verbose and (episode + 1) % max(1, episodes // 10) == 0:
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Episode Accuracy: {episode_summary['episode_accuracy']:.4f}")
                print(f"  Episode Reward: {episode_summary['episode_reward']:.4f}")
                print(f"  Avg Total Loss: {np.mean(episode_loss['total']):.4f}")

        print(f"{'='*70}")
        print(f"✓ Training complete!")

        return self.training_history

    def save_policy(self, save_path: str):
        """Save trained policy network"""
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"✓ Policy saved to {save_path}")

    def load_policy(self, load_path: str):
        """Load trained policy network"""
        self.policy_net.load_state_dict(torch.load(load_path))
        print(f"✓ Policy loaded from {load_path}")