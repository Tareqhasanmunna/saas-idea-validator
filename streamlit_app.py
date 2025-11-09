# streamlit_app.py
import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from src.training.sl_training.data_loader import DataLoader
from src.rl.rl_brain import load_agent
from src.rl.environments.idea_validation_env import IdeaValidationEnv
from utils.auto_cleaner import delete_old_files

PROJECT_ROOT = "E:/saas-idea-validator"  # adjust if different
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data/processed/vectorised_dataset.csv")
BEST_RL_PATH = os.path.join(PROJECT_ROOT, "rl_models", "best_rl_model.zip")
BEST_SL_PATH = os.path.join(PROJECT_ROOT, "best_sl_model")  # directory (your best model filename may vary)
FEEDBACK_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw", "raw_batch")
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
FEEDBACK_FILE = os.path.join(FEEDBACK_FOLDER, "feedback.csv")

@st.cache_resource(ttl=3600)
def load_sl_model(sl_dir):
    # tries to find a joblib model in sl_dir
    if not os.path.exists(sl_dir):
        return None
    for f in os.listdir(sl_dir):
        if f.endswith(".joblib"):
            try:
                return joblib.load(os.path.join(sl_dir, f))
            except Exception:
                continue
    return None

@st.cache_resource
def get_rl_agent():
    # create small env with a single sample to load agent
    if not os.path.exists(BEST_RL_PATH):
        return None
    # env needs a dummy csv — use the processed csv but train env in deterministic mode
    env = IdeaValidationEnv(data_csv=PROCESSED_CSV, use_reward_model=False)
    from stable_baselines3 import PPO
    agent = PPO.load(BEST_RL_PATH, env=env)
    return agent

def log_feedback(record: dict):
    # append to feedback.csv
    df = pd.DataFrame([record])
    if not os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, index=False, mode='a', header=False)

st.title("SaaS Idea Validator — RL Inference")
st.markdown("Enter an idea or paste a Reddit post content and get the RL evaluation. Give feedback (thumbs up/down) to teach the system.")

# UI inputs
text = st.text_area("Paste idea / post text (optional)", height=180)
# For quick testing, we can sample a row from processed CSV
if st.button("Use random sample from dataset"):
    df = pd.read_csv(PROCESSED_CSV)
    sample = df.sample(1).iloc[0]
    st.write("Using sample id:", sample.get("post_id", "N/A"))
    text = sample.get("post_text", "")

# placeholder features inputs (advanced users)
col1, col2, col3, col4 = st.columns(4)
post_sentiment = col1.number_input("post_sentiment", value=0.5, min_value=-1.0, max_value=1.0, step=0.01)
avg_comment_sentiment = col2.number_input("avg_comment_sentiment", value=0.5, min_value=-1.0, max_value=1.0, step=0.01)
upvotes = col3.number_input("upvotes", value=10, min_value=0)
upvote_ratio = col4.number_input("upvote_ratio", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

# load models
rl_agent = get_rl_agent()
sl_model = load_sl_model(BEST_SL_PATH)

# Build observation row for the single input (must match env obs)
# For simplicity use zeros for vector section if not available
def build_obs_from_inputs():
    vec = np.zeros(100)
    numeric = np.array([post_sentiment, avg_comment_sentiment, upvotes, upvote_ratio, 0.0])
    return np.concatenate([numeric, vec])

obs = build_obs_from_inputs()
st.markdown("### Model prediction")
if rl_agent:
    # RL agent expects vec-env; we'll call env.reset & agent.predict
    env = IdeaValidationEnv(data_csv=PROCESSED_CSV, use_reward_model=False)
    env.current_row = {"post_sentiment": post_sentiment, "avg_comment_sentiment": avg_comment_sentiment,
                       "upvotes": upvotes, "upvote_ratio": upvote_ratio, "post_recency": 0.0, "_vec_arr": np.zeros(100)}
    action, _ = rl_agent.predict(env._get_obs_from_row(env.current_row), deterministic=True)
    # normalize into weights
    exp = np.exp(action)
    weights = exp / (exp.sum() + 1e-8)
    from utils.weight_validator import format_weights_string
    score = env.weight_validator.calculate_validation_score({
        "post_sentiment": post_sentiment,
        "avg_comment_sentiment": avg_comment_sentiment,
        "upvote_ratio": upvote_ratio,
        "post_recency": 0.0
    }, weights.tolist())
    label = env.weight_validator.assign_label(score)
    st.write("RL model weights:", weights)
    st.write(f"Validation score: {score:.2f}  →  Label: **{label}**")
else:
    st.write("No RL model found. Falling back to SL model.")
    if sl_model is not None:
        X = obs.reshape(1, -1)
        try:
            pred = sl_model.predict(X)[0]
            st.write("SL predicted label:", pred)
        except Exception:
            st.write("SL model cannot predict with current inputs.")
    else:
        st.write("No SL model found either. Train models first.")

# Feedback widget
st.markdown("### Give feedback")
col_a, col_b = st.columns(2)
thumbs_up = col_a.button("👍 Helpful")
thumbs_down = col_b.button("👎 Not helpful")

if thumbs_up or thumbs_down:
    feedback_val = 1 if thumbs_up else 0
    # persist a feedback record
    rec = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "post_text": text,
        "post_sentiment": post_sentiment,
        "avg_comment_sentiment": avg_comment_sentiment,
        "upvotes": upvotes,
        "upvote_ratio": upvote_ratio,
        "vector": np.array2string(np.zeros(100), separator=" "),  # placeholder
        "weight_0": float(weights[0]) if rl_agent else 0.0,
        "weight_1": float(weights[1]) if rl_agent else 0.0,
        "weight_2": float(weights[2]) if rl_agent else 0.0,
        "weight_3": float(weights[3]) if rl_agent else 0.0,
        "feedback": int(feedback_val)
    }
    log_feedback(rec)
    st.success("Thanks — feedback logged.")
    # optional: clean old raw batches occasionally
    delete_old_files(FEEDBACK_FOLDER, days=365)
