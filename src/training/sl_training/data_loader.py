import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path, vector_col="vector", target_col="label_numeric"):
        self.file_path = file_path
        self.vector_col = vector_col
        self.target_col = target_col

    def load_data(self):
        df = pd.read_csv(self.file_path)
        X_numeric = df[["post_sentiment", "avg_comment_sentiment", "upvotes", "upvote_ratio"]].values
        X_vector = np.vstack(df[self.vector_col].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")))
        X = np.hstack([X_numeric, X_vector])
        y = df[self.target_col].values
        return X, y
