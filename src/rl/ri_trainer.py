import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import yaml

logger = logging.getLogger(__name__)

class RLEnvironment:
    def __init__(self, X, y, test_size=0.2, logger_obj=None):
        self.X = X
        self.y = y
        self.logger = logger_obj or logger
        
        n_samples = len(X)
        n_train = int(n_samples * (1 - test_size))
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        self.X_train, self.X_test = X[train_idx], X[test_idx]
        self.y_train, self.y_test = y[train_idx], y[test_idx]
        
        self.logger.info(f"[ENV] Train: {len(self.X_train)}, Test: {len(self.X_test)}")
    
    def evaluate_model(self, model):
        try:
            y_pred = model.predict(self.X_test)
            return {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            }
        except:
            return {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}

class QLearningAgent:
    def __init__(self, state_size=10, action_size=5, logger_obj=None):
        self.state_size = state_size
        self.action_size = action_size
        self.logger = logger_obj or logger
        
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        self.episode_count = 0
        self.memory = []
        
        self.logger.info(f"[AGENT] Initialized: states={state_size}, actions={action_size}")
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

class RLTrainer:
    def __init__(self, processed_csv, config_file='config.yaml', logger_obj=None):
        self.processed_csv = processed_csv
        self.logger = logger_obj or logger
        
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.rl_config = self.config.get('rl', {})
        
        self.logger.info(f"[TRAINER] Loading data...")
        df = pd.read_csv(processed_csv)
        
        feature_cols = ['post_sentiment', 'avg_comment_sentiment', 'upvote_ratio', 'post_recency']
        X_numeric = df[feature_cols].fillna(0).values
        
        vectors = []
        for vec_str in df['vector']:
            if isinstance(vec_str, str):
                vec = np.fromstring(vec_str.strip('[]'), sep=' ')
            else:
                vec = np.zeros(200)
            vectors.append(vec)
        
        X_vector = np.array(vectors)
        self.X = np.hstack([X_numeric, X_vector])
        self.y = df['label_numeric'].values
        
        self.logger.info(f"[TRAINER] Data loaded: X={self.X.shape}, y={self.y.shape}")
        
        self.env = RLEnvironment(self.X, self.y, logger_obj=self.logger)
        self.agent = QLearningAgent(state_size=10, action_size=5, logger_obj=self.logger)
    
    def train_episode(self, episode_num, best_model=None):
        self.logger.info(f"[TRAINER] Episode {episode_num}")
        
        state = 0
        total_reward = 0
        episode_metrics = []
        
        try:
            for step in range(self.agent.action_size):
                action = self.agent.get_action(state)
                
                if best_model:
                    metrics = self.env.evaluate_model(best_model)
                else:
                    metrics = {'accuracy': 0.5, 'f1': 0.5, 'precision': 0.5, 'recall': 0.5}
                
                reward = metrics['accuracy'] * 100
                total_reward += reward
                
                episode_metrics.append({'step': step, 'action': action, 'reward': reward})
                
                next_state = min(state + 1, self.agent.state_size - 1)
                self.agent.update_q_value(state, action, reward, next_state)
                self.agent.remember(state, action, reward, next_state, step == self.agent.action_size - 1)
                
                state = next_state
            
            self.agent.decay_epsilon()
            self.agent.episode_count += 1
            avg_reward = total_reward / self.agent.action_size
            
            self.logger.info(f"[TRAINER] Episode {episode_num}: reward={avg_reward:.2f}, eps={self.agent.epsilon:.3f}")
            
            return {
                'episode': episode_num,
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'epsilon': self.agent.epsilon,
                'metrics': episode_metrics
            }
        except Exception as e:
            self.logger.error(f"[TRAINER] Episode {episode_num} error: {str(e)}")
            return None
    
    def train(self, num_episodes=10, best_model=None):
        self.logger.info(f"[TRAINER] Starting RL training: {num_episodes} episodes")
        history = []
        
        for episode in range(num_episodes):
            result = self.train_episode(episode + 1, best_model=best_model)
            if result:
                history.append(result)
        
        return {
            'success': True,
            'episodes_trained': len(history),
            'history': history,
            'final_epsilon': self.agent.epsilon
        }