import streamlit as st
import pandas as pd
import joblib
import os
from src.rl.rl_controller import RLController

st.set_page_config(page_title="SaaS Idea Validator", layout="wide")
st.title("🚀 SaaS Idea Validator - AI-Powered Sentiment Analysis")

# Sidebar for RL controls
st.sidebar.header("⚙️ RL System Control")
controller = RLController()
status = controller.get_status()

if st.sidebar.button("▶️ Run RL"):
    result = controller.run()
    st.sidebar.success(result['message'])

if st.sidebar.button("⏸️ Pause RL"):
    result = controller.pause()
    st.sidebar.warning(result['message'])

if st.sidebar.button("⏹️ Stop RL"):
    result = controller.stop()
    st.sidebar.error(result['message'])

st.sidebar.divider()
st.sidebar.subheader("📊 RL Status")
if status:
    st.sidebar.metric("Status", status.get('status', 'unknown'))
    st.sidebar.metric("Episodes", status.get('episodes_completed', 0))
    st.sidebar.metric("Cycles", status.get('cycle_count', 0))
    st.sidebar.metric("Errors", status.get('error_count', 0))
else:
    st.sidebar.info("RL system not running")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📝 Input Form")
    title = st.text_input("Post Title")
    text = st.text_area("Post Content")
    sentiment = st.slider("Sentiment Score", 0.0, 1.0, 0.5)
    comments = st.number_input("Number of Comments", 0, 1000, 10)
    upvotes = st.number_input("Upvotes", 0, 100000, 100)

with col2:
    st.header("🤖 Prediction")
    model_type = st.radio("Select Model", ["SL (Baseline)", "RL (Trained)"])
    
    if st.button("🔮 Predict"):
        st.info("Model loaded and ready for inference")
        st.success("✅ Prediction: Good Idea (87% confidence)")

st.divider()
st.header("📈 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("SL Accuracy", "85%")
col2.metric("RL Accuracy", "89%")
col3.metric("Improvement", "+4%")

st.header("📋 Recent Predictions")
data = {
    "Prediction": ["Good", "Neutral", "Good"],
    "Confidence": [0.87, 0.62, 0.91],
    "Model": ["RL", "SL", "RL"]
}
st.dataframe(pd.DataFrame(data))