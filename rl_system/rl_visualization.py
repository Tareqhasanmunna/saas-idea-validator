# visualize_training.py
import matplotlib.pyplot as plt
import numpy as np

def plot_rl_training(reward_history, acc_history, window=20, save_path=None):
    """
    reward_history: list of reward per episode
    acc_history: list of accuracy per episode
    window: moving average window
    """
    rewards = np.array(reward_history)
    accs = np.array(acc_history)

    mov_r = np.convolve(rewards, np.ones(window)/window, mode='same')
    mov_a = np.convolve(accs, np.ones(window)/window, mode='same')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # -------- Episode Rewards --------
    axs[0, 0].plot(rewards, color='blue', linewidth=1)
    axs[0, 0].set_title("Episode Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].fill_between(range(len(rewards)), rewards, alpha=0.15, color='blue')

    # -------- Episode Accuracies --------
    axs[0, 1].plot(accs, color='green', linewidth=1)
    axs[0, 1].set_title("Episode Accuracies")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].fill_between(range(len(accs)), accs, alpha=0.15, color='green')

    # -------- Reward Convergence (Moving Average) --------
    axs[1, 0].plot(rewards, color='lightblue', alpha=0.4, label='Raw')
    axs[1, 0].plot(mov_r, color='red', linewidth=2, label='Moving Average')
    axs[1, 0].set_title("Reward Convergence")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].legend()

    # -------- Accuracy Convergence (Moving Average) --------
    axs[1, 1].plot(accs, color='lightgray', alpha=0.5, label='Raw')
    axs[1, 1].plot(mov_a, color='orange', linewidth=2, label='Moving Average')
    axs[1, 1].set_title("Accuracy Convergence")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
