# src/rl/environments/idea_validation_env.py
import gym
from gym import spaces
import numpy as np
import joblib
import os
from src.training.sl_training.data_loader import DataLoader
from utils.weight_validator import AutomatedWeightValidator
from src.rl.reward_model import RewardModel  # we'll create this
from utils.helpers import replace_nan_with_mean_plus_noise

class IdeaValidationEnv(gym.Env):
    """
    Single-step environment:
      - Observation: numeric features + vector (flattened) for a single idea (1D array)
      - Action: 4 continuous values representing weights for features [post_sentiment, avg_comment_sentiment, upvote_ratio, post_recency].
        We'll convert them to a normalized weight vector internally.
      - Reward: predicted user feedback (via reward model) OR proxy reward from SL label if no feedback exists.
      - Episode: single-step (done after one action). This is simplest to start with; can be extended to multi-step later.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 data_csv,
                 vector_col="vector",
                 target_col="label_numeric",
                 sl_model_path=None,
                 reward_model_path=None,
                 use_reward_model=True):
        super().__init__()
        self.data_csv = data_csv
        self.vector_col = vector_col
        self.target_col = target_col
        self.sl_model_path = sl_model_path
        self.use_reward_model = use_reward_model

        # Load dataset into memory (we'll step through rows)
        self.df = self._load_dataframe(data_csv)
        self.n_samples = len(self.df)
        self.current_index = 0

        # Load SL model if provided (joblib)
        self.sl_model = None
        if sl_model_path and os.path.exists(sl_model_path):
            try:
                self.sl_model = joblib.load(sl_model_path)
            except Exception:
                self.sl_model = None

        # Reward Model (optional)
        self.reward_model = None
        if use_reward_model and reward_model_path and os.path.exists(reward_model_path):
            self.reward_model = RewardModel.load(reward_model_path)

        # derive observation dim by sample 0
        sample_obs = self._get_obs_from_row(self.df.iloc[0])
        obs_dim = sample_obs.shape[0]

        # Observation space: Box with shape (obs_dim,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space: 4 continuous values in [0,1] (we'll softmax/normalize them to get weights)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # helper: weight validator (keeps minimum weights if needed)
        self.weight_validator = AutomatedWeightValidator(min_weight=0.01)

    def _load_dataframe(self, path):
        import pandas as pd, numpy as np
        df = pd.read_csv(path)
        # ensure vector col parsed to numpy arrays
        if self.vector_col in df.columns:
            def parse_vec(s):
                if pd.isna(s):
                    return np.zeros(100)  # fallback length - adjust if necessary
                return np.fromstring(s.strip("[]"), sep=" ")
            vecs = df[self.vector_col].apply(parse_vec)
            maxlen = max(v.size for v in vecs)
            # pad to same length
            vecs_padded = np.vstack([np.pad(v, (0, maxlen - v.size)) for v in vecs])
            df["_vec_arr"] = list(vecs_padded)
        else:
            df["_vec_arr"] = [np.zeros(100) for _ in range(len(df))]
        return df.reset_index(drop=True)

    def _get_obs_from_row(self, row):
        # numeric features + vector flattened
        numeric = np.array([
            row.get("post_sentiment", 0.0),
            row.get("avg_comment_sentiment", 0.0),
            row.get("upvotes", 0.0),
            row.get("upvote_ratio", 0.0),
            row.get("post_recency", 0.0)
        ], dtype=np.float32)
        vec = np.array(row["_vec_arr"], dtype=np.float32)
        obs = np.concatenate([numeric, vec.astype(np.float32)])
        # handle NaNs
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def reset(self):
        # if we reached end, wrap around
        if self.current_index >= self.n_samples:
            self.current_index = 0
        row = self.df.iloc[self.current_index]
        self.current_index += 1
        self.current_row = row
        return self._get_obs_from_row(row)

    def step(self, action):
        """
        action: 4-d raw outputs -> convert to normalized weights
        returns observation, reward, done, info
        """
        # normalize action to weights
        action = np.array(action, dtype=np.float32)
        # avoid all zeros by adding epsilon
        exp = np.exp(action)
        weights = exp / (exp.sum() + 1e-8)
        # enforce min weights and renormalize using your validator
        weights = self.weight_validator.enforce_minimum_weights(weights.tolist())

        # compute validation_score using weight_validator (reuse its compute)
        rec = {
            "post_sentiment": float(self.current_row.get("post_sentiment", 0.5)),
            "avg_comment_sentiment": float(self.current_row.get("avg_comment_sentiment", 0.5)),
            "upvote_ratio": float(self.current_row.get("upvote_ratio", 0.5)),
            "post_recency": float(self.current_row.get("post_recency", 0.5)),
        }
        # calculate score 0-100
        score = self.weight_validator.calculate_validation_score(rec, weights)

        # determine reward:
        reward = 0.0
        info = {"score": score, "weights": weights}

        # if we have a reward model, use it: it expects state+action maybe; we'll pass simple vector
        if self.reward_model is not None:
            # construct features for reward model: concatenation numeric + weights
            obs = self._get_obs_from_row(self.current_row)
            feat = np.concatenate([obs.flatten(), np.array(weights, dtype=np.float32)])
            reward = float(self.reward_model.predict_reward_from_array(feat))
        else:
            # fallback proxy reward:
            # if SL label exists, reward=1 if weight-based label equals SL label, else 0
            if self.target_col in self.current_row and self.sl_model is not None:
                # SL predicted label
                X = self._get_obs_from_row(self.current_row).reshape(1, -1)
                try:
                    sl_pred = self.sl_model.predict(X)[0]
                except Exception:
                    sl_pred = None
                # derive label from validation score
                validator_label = "good" if score >= 70 else ("neutral" if score >= 40 else "bad")
                # convert sl_pred numeric to labels consistent with your system:
                # If sl_model returns 0/1/2 mapping unknown; we'll simple reward partial:
                if sl_pred is not None:
                    # give reward by probability of matching class if predict_proba available
                    try:
                        probs = self.sl_model.predict_proba(X)[0]
                        # assume highest prob class corresponds to sl_pred
                        # reward proportional to probability assigned to class consistent with validator_label when mapping known
                        reward = float(probs.max()) * 0.5
                    except Exception:
                        reward = 0.0
                else:
                    reward = 0.0
            else:
                # no SL model and no reward model -> use score as weak reward scaled to [0,1]
                reward = float(score / 100.0) * 0.2  # small signal

        done = True  # single-step episode
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)  # no next step
        return observation, reward, done, info

    def render(self, mode="human"):
        print(f"Row idx: {self.current_index-1}")
