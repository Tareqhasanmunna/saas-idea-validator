import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_CSV = "data/processed/vectorised_dataset.csv"  # Update if needed
OUTPUT_PATH = "src/preprocessing/tfidf_vectorizer.pkl"

def generate_vectorizer():
    if not os.path.isfile(DATA_CSV):
        print("Data CSV not found: {DATA_CSV}")
        return
    df = pd.read_csv(DATA_CSV)
    if 'text' not in df.columns:
        print("Data CSV missing 'text' column")
        return
    texts = df['text'].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True
    )
    vectorizer.fit(texts)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(vectorizer, OUTPUT_PATH)
    print("Vectorizer saved at {OUTPUT_PATH}")

if __name__ == '__main__':
    generate_vectorizer()
