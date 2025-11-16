# rl_system/src/agent/q_learning_agent.py
import random
import pickle
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.action_count = 0

    def choose_action(self, state, available_actions=None):
        if available_actions is None:
            available_actions = list(range(self.num_actions))
        self.action_count += 1
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        qvals = self.q_table[state]
        # pick best among available actions
        best = max(available_actions, key=lambda a: qvals[a])
        return best

    def update(self, state, action, reward, next_state, done):
        cur = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = cur + self.lr * (target - cur)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "action_count": self.action_count
            }, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions), data.get("q_table", {}))
        self.epsilon = data.get("epsilon", self.epsilon)
        self.action_count = data.get("action_count", 0)

    def get_stats(self):
        return {"num_states": len(self.q_table), "epsilon": float(self.epsilon), "action_count": int(self.action_count)}
