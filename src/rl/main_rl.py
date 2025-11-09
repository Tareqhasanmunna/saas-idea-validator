"""
Main entry point for Reinforcement Learning (RL) training pipeline
"""

import os
from src.rl.rl_trainer import train_rl
from src.rl.reward_model import RewardModel

# === CONFIGURATION ===
CONFIG = {
    "processed_data_csv": "E:/saas-idea-validator/data/processed/vectorised_dataset.csv",
    "vector_col": "vector",
    "sl_model_path": "E:/saas-idea-validator/best_sl_model/best_model.joblib",
    "reward_model_path": "E:/saas-idea-validator/rl_models/reward_model.pkl",
    "save_dir": "E:/saas-idea-validator/rl_models",
    "total_timesteps": 200_000,
    "eval_episodes": 100
}

def main():
    print("🔁 [RL] Starting Reinforcement Learning Training Pipeline...")

    # 1️⃣ Train reward model if enough feedback available
    feedback_csv = "E:/saas-idea-validator/data/raw/raw_batch/feedback.csv"
    if os.path.exists(feedback_csv):
        try:
            RewardModel.train_from_feedback_csv(
                feedback_csv=feedback_csv,
                vector_len=100,  # update if your vector size differs
                reward_model_path=CONFIG["reward_model_path"]
            )
            print("✅ Reward model trained and saved.")
        except Exception as e:
            print(f"[WARN] Reward model not trained: {e}")
    else:
        print("[INFO] No feedback found, proceeding without reward model.")

    # 2️⃣ Train RL agent using SL model + reward model
    best_model_path = train_rl(
        processed_data_csv=CONFIG["processed_data_csv"],
        vector_col=CONFIG["vector_col"],
        sl_model_path=CONFIG["sl_model_path"],
        reward_model_path=CONFIG["reward_model_path"] if os.path.exists(CONFIG["reward_model_path"]) else None,
        total_timesteps=CONFIG["total_timesteps"],
        eval_episodes=CONFIG["eval_episodes"],
        save_dir=CONFIG["save_dir"]
    )

    print(f"🎯 RL training completed. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
