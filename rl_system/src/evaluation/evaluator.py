# rl_system/src/evaluation/evaluator.py
import numpy as np
import random
from math import erf, sqrt

class RandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
    def choose_action(self, state, available_actions=None):
        if available_actions is None:
            available_actions = list(range(self.num_actions))
        return random.choice(available_actions)

class Evaluator:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def evaluate(self, num_episodes=200, use_best_policy=True):
        results = {
            "total_rewards": [], "completeness_rates": [], "answer_rates": [],
            "dropout_rates": [], "steps": [], "confidences": []
        }
        orig_eps = self.agent.epsilon
        if use_best_policy:
            self.agent.epsilon = 0.0

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            total = 0.0
            while not done:
                avail = [i for i in range(self.env.num_features) if i not in self.env.asked_questions]
                if not avail:
                    break
                action = self.agent.choose_action(state, avail)
                next_state, reward, done, info = self.env.step(action)
                total += reward
                state = next_state
            results["total_rewards"].append(float(total))
            results["completeness_rates"].append(float(info.get("completeness", 0.0)))
            results["answer_rates"].append(float(info.get("answered", 0.0)))
            results["dropout_rates"].append(1.0 if info.get("reason") == "user_dropout" else 0.0)
            results["steps"].append(float(self.env.step_count))
            try:
                _, conf = self.env.sl_model.predict([0 if x is None else x for x in self.env.state])
            except Exception:
                conf = 0.0
            results["confidences"].append(float(conf))

        self.agent.epsilon = orig_eps

        summary = {
            "avg_reward": float(np.mean(results["total_rewards"])),
            "avg_confidence": float(np.mean(results["confidences"])),
            "avg_answer_rate": float(np.mean(results["answer_rates"])),
            "avg_completeness": float(np.mean(results["completeness_rates"])),
            "dropout_rate": float(np.mean(results["dropout_rates"])),
            "success_rate": float(sum(1 for c in results["completeness_rates"] if c >= 0.9) / num_episodes),
            "avg_steps": float(np.mean(results["steps"])),
            # keep raw lists for deeper analysis if needed (convert to python floats)
            "predictions": [int(x) for x in []],  # placeholder if needed
        }
        return results, summary

    def _evaluate_sl_only(self, num_episodes=200):
        rand = RandomAgent(self.env.num_features)
        results = {
            "total_rewards": [], "completeness_rates": [], "answer_rates": [],
            "dropout_rates": [], "steps": [], "confidences": []
        }
        for _ in range(num_episodes):
            self.env.reset()
            done = False
            total = 0.0
            while not done:
                avail = [i for i in range(self.env.num_features) if i not in self.env.asked_questions]
                if not avail:
                    break
                action = rand.choose_action(None, avail)
                next_state, reward, done, info = self.env.step(action)
                total += reward
            results["total_rewards"].append(float(total))
            results["completeness_rates"].append(float(info.get("completeness", 0.0)))
            results["answer_rates"].append(float(info.get("answered", 0.0)))
            results["dropout_rates"].append(1.0 if info.get("reason") == "user_dropout" else 0.0)
            results["steps"].append(float(self.env.step_count))
            try:
                _, conf = self.env.sl_model.predict([0 if x is None else x for x in self.env.state])
            except Exception:
                conf = 0.0
            results["confidences"].append(float(conf))

        summary = {
            "avg_reward": float(np.mean(results["total_rewards"])),
            "avg_confidence": float(np.mean(results["confidences"])),
            "avg_answer_rate": float(np.mean(results["answer_rates"])),
            "avg_completeness": float(np.mean(results["completeness_rates"])),
            "dropout_rate": float(np.mean(results["dropout_rates"])),
            "success_rate": float(sum(1 for c in results["completeness_rates"] if c >= 0.9) / num_episodes),
            "avg_steps": float(np.mean(results["steps"]))
        }
        return results, summary

    def compare_with_sl_only(self, num_episodes=200):
        print("Evaluating RL agent...")
        rl_res, rl_sum = self.evaluate(num_episodes, use_best_policy=True)
        print("Evaluating SL-only (random questions) baseline...")
        sl_res, sl_sum = self._evaluate_sl_only(num_episodes)
        improvement = {}
        for k in ["avg_confidence", "avg_answer_rate", "avg_completeness", "success_rate"]:
            sv = sl_sum.get(k, 0)
            rv = rl_sum.get(k, 0)
            improvement[k] = float(((rv - sv) / abs(sv)) * 100) if sv != 0 else 0.0
        return rl_res, rl_sum, sl_res, sl_sum, improvement
