import streamlit as st
import pandas as pd
import pickle
from utils.helpers import handle_nan_values
from utils.weight_validator import AutomatedWeightValidator
import glob
import os

MODELS_DIR = "models"

st.title("SaaS Idea Validator - User Input Test")

# 1. Input
st.subheader("Enter your post data")
post_text = st.text_area("Post content")
post_sentiment = st.number_input("Post sentiment (0-1)", 0.0, 1.0, 0.5)
avg_comment_sentiment = st.number_input("Avg comment sentiment (0-1)", 0.0, 1.0, 0.5)
upvote_ratio = st.number_input("Upvote ratio (0-1)", 0.0, 1.0, 0.5)
post_recency = st.number_input("Recency (0-1)", 0.0, 1.0, 0.5)

if st.button("Predict Label"):

    # 2. Prepare dataframe
    df = pd.DataFrame([{
        'post_sentiment': post_sentiment,
        'avg_comment_sentiment': avg_comment_sentiment,
        'upvote_ratio': upvote_ratio,
        'post_recency': post_recency,
        'token_text': post_text
    }])
    df = handle_nan_values(df)

    # 3. Load latest model
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not model_files:
        st.error("No trained models found!")
    else:
        latest_model_file = max(model_files, key=os.path.getctime)
        with open(latest_model_file, "rb") as f:
            model_data = pickle.load(f)

        # 4. Weight validation
        validator = AutomatedWeightValidator()
        best_weights, _, _ = validator.find_best_weights(df.to_dict('records'))
        labeled = validator.validate_and_label_batch(df.to_dict('records'), best_weights)

        result = labeled[0]
        st.success(f"Predicted label: {result['label']} | Validation score: {result['validation_score']}")
