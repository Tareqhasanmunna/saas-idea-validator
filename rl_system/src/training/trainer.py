# rl_system/src/training/trainer.py
import numpy as np

class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.episode_rewards = []
        self.episode_completeness = []
        self.episode_confidence = []
        self.episode_dropouts = []

    def train(self, num_episodes=2000, verbose=True, log_every=100):
        print(f"Starting training: {num_episodes} episodes")
        for ep in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                avail = [i for i in range(self.env.num_features) if i not in self.env.asked_questions]
                if not avail:
                    break
                action = self.agent.choose_action(state, avail)
                next_state, reward, done, info = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            self.agent.decay_epsilon()

            # logging stats
            self.episode_rewards.append(float(total_reward))
            self.episode_completeness.append(float(info.get("completeness", 0.0)))
            self.episode_dropouts.append(1.0 if info.get("reason") == "user_dropout" else 0.0)

            features = [0 if x is None else x for x in self.env.state]
            try:
                _, conf = self.env.sl_model.predict(features)
            except Exception:
                conf = 0.0
            self.episode_confidence.append(float(conf))

            if verbose and ((ep + 1) % log_every == 0):
                print(f"Episode {ep+1}/{num_episodes} | Reward: {np.mean(self.episode_rewards[-log_every:]):.2f} | "
                      f"Complete: {np.mean(self.episode_completeness[-log_every:]):.2%} | "
                      f"Confidence: {np.mean(self.episode_confidence[-log_every:]):.2%} | "
                      f"Dropout: {np.mean(self.episode_dropouts[-log_every:]):.2%} | "
                      f"Eps: {self.agent.epsilon:.3f}")

        print("Training finished.")
        return self.get_training_history()

    def get_training_history(self):
        return {
            "rewards": self.episode_rewards,
            "completeness": self.episode_completeness,
            "confidence": self.episode_confidence,
            "dropouts": self.episode_dropouts,
            "agent_stats": self.agent.get_stats()
        }
