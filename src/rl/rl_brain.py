# src/rl/rl_brain.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch

def make_agent(env, policy="MlpPolicy", **ppo_kwargs):
    """
    Wrap env in DummyVecEnv and return PPO agent.
    """
    vec_env = DummyVecEnv([lambda: env])
    agent = PPO(policy, vec_env, verbose=1, tensorboard_log="./tb_logs", device="auto", **ppo_kwargs)
    return agent

def save_agent(agent, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    agent.save(path)

def load_agent(path, env):
    # loads into same class
    agent = PPO.load(path, env=env)
    return agent
