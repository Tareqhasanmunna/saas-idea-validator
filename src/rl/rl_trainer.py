# src/rl/rl_trainer.py
import os
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from src.rl.environments.idea_validation_env import IdeaValidationEnv
from src.rl.rl_brain import make_agent, save_agent, load_agent
from src.rl.reward_model import RewardModel
from utils.auto_cleaner import clean_everything

def train_rl(
    processed_data_csv="E:/saas-idea-validator/data/processed/vectorised_dataset.csv",
    vector_col="vector",
    sl_model_path="E:/saas-idea-validator/best_sl_model/best_model.joblib",
    reward_model_path=None,
    total_timesteps=200000,
    eval_episodes=200,
    save_dir="E:/saas-idea-validator/rl_models",
    retrain_reward_model=False
):
    os.makedirs(save_dir, exist_ok=True)
    # cleanup old files before starting
    clean_everything(project_root="E:/saas-idea-validator")

    env = IdeaValidationEnv(
        data_csv=processed_data_csv,
        vector_col=vector_col,
        target_col="label_numeric",
        sl_model_path=sl_model_path,
        reward_model_path=reward_model_path,
        use_reward_model=(reward_model_path is not None)
    )
    agent = make_agent(env, policy="MlpPolicy")
    best_mean_reward = -1e9
    best_path = os.path.join(save_dir, "best_rl_model")

    # Train
    timesteps = 0
    checkpoint_interval = max(int(total_timesteps / 10), 1000)
    while timesteps < total_timesteps:
        to_learn = min(checkpoint_interval, total_timesteps - timesteps)
        agent.learn(total_timesteps=to_learn, reset_num_timesteps=False)
        timesteps += to_learn

        # Evaluate
        mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=eval_episodes, return_episode_rewards=False)
        print(f"[EVAL] t={timesteps} mean_reward={mean_reward:.4f} std={std_reward:.4f}")

        # Save checkpoint
        ckpt = os.path.join(save_dir, f"rl_ckpt_{timesteps}.zip")
        save_agent(agent, ckpt)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            save_agent(agent, best_path)
            print(f"[BEST] New best model saved to {best_path} (mean_reward={mean_reward:.4f})")

    print("[DONE] Training finished. Best mean reward: ", best_mean_reward)
    return best_path
