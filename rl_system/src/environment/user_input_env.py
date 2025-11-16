# rl_system/src/environment/user_input_env.py
import numpy as np
import random

class UserInputEnvironment:
    """
    Light-weight environment that simulates asking up to 10 questions (from 104 features).
    Uses provided SL model wrapper to compute reward at episode end.
    """
    def __init__(self, sl_model, num_features=104, max_questions=10, max_steps=20):
        self.sl_model = sl_model
        self.num_features = num_features
        self.max_questions = max_questions
        self.max_steps = max_steps

        self.state = [None] * num_features
        self.asked_questions = set()
        self.step_count = 0
        self.dropped_off = False

    def reset(self):
        self.state = [None] * self.num_features
        self.asked_questions = set()
        self.step_count = 0
        self.dropped_off = False
        return self._get_state()

    def _get_state(self):
        # small representation for Q-table keys (first 16 asked flags)
        return tuple([1 if i in self.asked_questions else 0 for i in range(min(16, self.num_features))])

    def _simulate_user_response(self, idx):
        # simple simulation: more answered -> higher chance to answer next
        progress = len(self.asked_questions) / max(1, self.max_questions)
        prob = 0.6 + 0.3 * progress
        if random.random() < prob:
            # numeric or vector, return float
            if idx == 2:  # upvotes
                return float(np.random.uniform(0, 1000))
            if idx == 3:  # upvote_ratio
                return float(np.random.uniform(0, 1))
            return float(np.random.uniform(-1, 1))
        # sometimes user drops
        if random.random() < 0.15:
            self.dropped_off = True
        return None

    def step(self, action):
        self.step_count += 1
        # duplicate ask penalty
        if action in self.asked_questions:
            return self._get_state(), -0.5, False, {"reason": "duplicate", "completeness": len(self.asked_questions)/self.max_questions}
        self.asked_questions.add(action)
        resp = self._simulate_user_response(action)
        if resp is not None:
            self.state[action] = resp
        if self.dropped_off:
            return self._get_state(), -2.0, True, {"reason": "user_dropout", "completeness": len(self.asked_questions)/self.max_questions}
        done = (len(self.asked_questions) >= self.max_questions) or (self.step_count >= self.max_steps)
        reward = self._calculate_reward(done)
        info = {"reason": "completed" if done else "in_progress",
                "completeness": len(self.asked_questions)/self.max_questions,
                "answered": sum(1 for x in self.state if x is not None)/self.num_features}
        return self._get_state(), reward, done, info

    def _calculate_reward(self, done):
        answered = sum(1 for x in self.state if x is not None)
        if done:
            features = [0 if x is None else x for x in self.state]
            try:
                _, conf = self.sl_model.predict(features)
                base = 5.0 if answered >= 8 else answered * 0.5
                return base + conf * 3.0
            except Exception:
                return answered * 0.5
        return 0.2 * answered
