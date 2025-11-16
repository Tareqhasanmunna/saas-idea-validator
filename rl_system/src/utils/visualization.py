# rl_system/src/utils/visualization.py
import os, json, csv
import matplotlib.pyplot as plt
import numpy as np

def create_output_dir(output_dir="./rl_results"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    return output_dir

def moving_average(data, window=50):
    if len(data) == 0: return []
    return [float(np.mean(data[max(0, i - window): i + 1])) for i in range(len(data))]

def save_training_charts(history, output_dir="./rl_results"):
    create_output_dir(output_dir)
    rewards = history.get("rewards", [])
    completeness = history.get("completeness", [])
    confidence = history.get("confidence", [])

    episodes = list(range(1, len(rewards) + 1))
    if not episodes:
        return

    r_ma = moving_average(rewards)
    c_ma = moving_average(completeness)
    conf_ma = moving_average(confidence)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(episodes, rewards, alpha=0.3); axes[0].plot(episodes, r_ma, linewidth=2)
    axes[0].set_title("Episode Rewards")
    axes[1].plot(episodes, completeness, alpha=0.3); axes[1].plot(episodes, c_ma, linewidth=2)
    axes[1].set_title("Completeness")
    axes[2].plot(episodes, confidence, alpha=0.3); axes[2].plot(episodes, conf_ma, linewidth=2)
    axes[2].set_title("SL Confidence")
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "training_progress.png")
    plt.savefig(path, dpi=200); plt.close()
    print("✓ Saved:", path)

def save_comparison_charts(rl_summary, sl_summary, output_dir="./rl_results"):
    create_output_dir(output_dir)
    metrics = ["avg_reward", "avg_completeness", "avg_answer_rate", "avg_confidence", "success_rate"]
    labels = ["Reward", "Completeness", "Answer rate", "SL confidence", "Success rate"]
    rl_vals = [rl_summary.get(m, 0.0) for m in metrics]
    sl_vals = [sl_summary.get(m, 0.0) for m in metrics]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, rl_vals, width, label="RL")
    ax.bar(x + width/2, sl_vals, width, label="SL")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(rl_vals, sl_vals)):
        ax.text(i - width/2, a + 0.01, f"{a:.3f}", ha="center", va="bottom")
        ax.text(i + width/2, b + 0.01, f"{b:.3f}", ha="center", va="bottom")
    path = os.path.join(output_dir, "plots", "rl_vs_sl.png")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    print("✓ Saved:", path)

def save_evaluation_data(rl_summary, sl_summary, improvement, output_dir):
    import os
    import json

    def safe_value(v):
        if isinstance(v, (int, float)):
            return float(v)
        return v  # list, dict, etc.

    data = {
        "rl": {k: safe_value(v) for k, v in rl_summary.items()},
        "sl": {k: safe_value(v) for k, v in sl_summary.items()},
        "improvement": {k: safe_value(v) for k, v in improvement.items()}
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(data, f, indent=4)

    print("✓ Evaluation data saved (no float/list crash).")

