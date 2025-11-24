# env_wrapper.py
# Simple environment wrapper over a supervised dataset treated as episodic RL.
# Each episode = one sample (state = feature vector, agent selects class action).
# Returns reward based on prediction correctness + optional shaping.

import numpy as np
from typing import Tuple, Dict, List
import random

class DatasetEnv:
    """
    Episodic env where each episode is one sample from dataset.
    action: integer 0..(n_classes-1)
    observation: 1D numpy array (feature vector)
    reward: shaped per config
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, n_classes:int,
                 class_weights:List[float]=None,
                 success_bonus:float=0.0,
                 correct_reward:float=1.0,
                 wrong_reward:float=-0.5,
                 seed:int=0):
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_classes = n_classes
        self.n = X.shape[0]
        self.indices = list(range(self.n))
        self.rng = random.Random(seed)
        self.class_weights = np.array(class_weights) if class_weights is not None else np.ones(n_classes, dtype=np.float32)
        self.success_bonus = success_bonus
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.curr_idx = 0
        self._permute()

    def _permute(self):
        self.rng.shuffle(self.indices)
        self.curr_idx = 0

    def reset(self) -> np.ndarray:
        if self.curr_idx >= self.n:
            self._permute()
        idx = self.indices[self.curr_idx]
        self.curr_idx += 1
        self._last_idx = idx
        return self.X[idx].copy()

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Action is predicted class index.
        Return next_obs (we'll return zeros since episode ends after one step),
        reward, done (True), info dict with label and correctness.
        """
        true_label = int(self.y[self._last_idx])
        correct = int(action == true_label)
        weight = float(self.class_weights[true_label])
        reward = (self.correct_reward if correct else self.wrong_reward) * weight
        if correct:
            reward += self.success_bonus
        info = {"true_label": true_label, "correct": bool(correct)}
        done = True
        # observation: return zero vector (episode ends) to keep simple; agent should act on reset
        next_obs = np.zeros_like(self.X[self._last_idx], dtype=np.float32)
        return next_obs, float(reward), done, info

    def sample_balanced_batch_indices(self, batch_size:int):
        """
        Returns indices for a balanced batch: attempts to sample equally across classes.
        """
        per_class = max(1, batch_size // self.n_classes)
        sel = []
        for c in range(self.n_classes):
            cls_idx = np.where(self.y == c)[0]
            if len(cls_idx) == 0:
                continue
            chosen = np.random.choice(cls_idx, size=per_class, replace=len(cls_idx) < per_class)
            sel.extend(chosen.tolist())
        if len(sel) < batch_size:
            more = np.random.choice(self.n, size=(batch_size - len(sel)), replace=True)
            sel.extend(more.tolist())
        np.random.shuffle(sel)
        return sel[:batch_size]

    def get_all(self):
        return self.X.copy(), self.y.copy()
