# src/rl/reward_model.py
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

class RewardModel:
    """
    Trains a regressor that maps (obs_features + action_weights) -> reward in [0,1]
    Save / load utilities included.
    """

    def __init__(self, model=None):
        self.model = model or RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_reward_from_array(self, arr):
        arr = np.array(arr).reshape(1, -1)
        pred = self.model.predict(arr)[0]
        return max(0.0, min(1.0, float(pred)))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        m = joblib.load(path)
        return cls(model=m)

    @classmethod
    def train_from_feedback_csv(cls, feedback_csv, vector_len, reward_model_path):
        """
        Expects CSV with columns: post_id, features... vector (string), weight_0..weight_3, feedback (0 or 1)
        We'll parse vector and build X,y.
        """
        import pandas as pd, numpy as np
        df = pd.read_csv(feedback_csv)
        # must contain 'feedback' column
        if 'feedback' not in df.columns or df.shape[0] < 10:
            raise ValueError("Not enough feedback rows or missing 'feedback' column to train reward model")

        def vec_to_arr(s):
            try:
                return np.fromstring(s.strip("[]"), sep=" ")
            except Exception:
                return np.zeros(vector_len)

        vecs = np.vstack(df['vector'].apply(vec_to_arr).values)
        numerics = df[['post_sentiment', 'avg_comment_sentiment', 'upvotes', 'upvote_ratio', 'post_recency']].fillna(0).values
        weights = df[['weight_0', 'weight_1', 'weight_2', 'weight_3']].values if {'weight_0','weight_1','weight_2','weight_3'}.issubset(df.columns) else np.zeros((len(df),4))
        X = np.hstack([numerics, vecs, weights])
        y = df['feedback'].astype(float).values
        clf = cls()
        clf.fit(X, y)
        clf.save(reward_model_path)
        return clf
