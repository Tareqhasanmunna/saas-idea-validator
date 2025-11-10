"""
Updated Streamlit App with RL Support
Place this in: streamlit_app.py (root directory)
Replace your existing streamlit_app.py with this version
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from reinforcement.predict_rl import RLPredictor
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="SaaS Idea Validator",
    page_icon="💡",
    layout="wide"
)

# Header
st.title("💡 SaaS Idea Validator")
st.markdown("---")

# Sidebar - Mode selection
st.sidebar.title("🎯 Prediction Mode")
mode = st.sidebar.radio(
    "Choose validation method:",
    ["RL Only", "SL Only", "Both (Compare)"] if RL_AVAILABLE else ["SL Only"]
)

st.sidebar.markdown("---")
st.sidebar.write("""
**About:**
- **RL Only**: Uses trained RL agent
- **SL Only**: Uses best SL model
- **Both**: Compares both approaches

**Status:**
""")
st.sidebar.write(f"RL System: {'✅ Available' if RL_AVAILABLE else '❌ Not trained yet'}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Business Idea")
    
    # Feature inputs
    st.write("**Features:**")
    col1a, col1b, col1c, col1d = st.columns(4)
    
    with col1a:
        post_sentiment = st.slider(
            "Post Sentiment",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    with col1b:
        avg_comment_sentiment = st.slider(
            "Avg Comment Sentiment",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    with col1c:
        upvotes = st.number_input(
            "Upvotes",
            min_value=0,
            max_value=10000,
            value=100,
            step=10
        )
    
    with col1d:
        upvote_ratio = st.slider(
            "Upvote Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
    
    # Text input
    idea_text = st.text_area(
        "Idea Description",
        placeholder="Describe your SaaS idea...",
        height=150
    )

with col2:
    st.subheader("🚀 Quick Actions")
    
    if st.button("🔮 Predict!", use_container_width=True, key="predict_btn"):
        if not idea_text:
            st.warning("⚠️ Please enter your idea description")
        else:
            # Prepare features
            features = np.array([
                post_sentiment,
                avg_comment_sentiment,
                upvotes / 10000,  # Normalize
                upvote_ratio
            ])
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # RL Prediction
            if mode in ["RL Only", "Both (Compare)"] and RL_AVAILABLE:
                st.write("**🤖 RL Agent Prediction:**")
                
                try:
                    predictor = RLPredictor(
                        'models/rl/trained_agent',
                        'best_sl_model/best_model.joblib',
                        'PPO'
                    )
                    
                    rl_result = predictor.predict(features)
                    
                    col_rl1, col_rl2, col_rl3 = st.columns(3)
                    
                    with col_rl1:
                        st.metric(
                            "Prediction",
                            "✅ GOOD" if rl_result['rl_prediction'] == 1 else "❌ BAD",
                            f"Action: {rl_result['rl_action_name']}"
                        )
                    
                    with col_rl2:
                        st.metric(
                            "RL Confidence",
                            f"{rl_result['final_score']:.1%}"
                        )
                    
                    with col_rl3:
                        st.metric(
                            "SL Confidence",
                            f"{rl_result['sl_confidence']:.1%}"
                        )
                    
                    st.write(f"**Verdict:** {rl_result['verdict']}")
                    
                    with st.expander("📋 Recommendations"):
                        for i, rec in enumerate(rl_result['recommendation'], 1):
                            st.write(f"{i}. {rec}")
                
                except FileNotFoundError:
                    st.error("❌ RL agent not found. Please train first: `python main_rl.py --train`")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            
            # SL Prediction
            if mode in ["SL Only", "Both (Compare)"]:
                st.write("**📊 SL Model Prediction:**")
                st.info("SL model prediction would appear here (integrate your existing model)")
            
            # Comparison
            if mode == "Both (Compare)":
                st.markdown("---")
                st.write("**📈 Comparison:**")
                st.info("Comparison metrics would appear here")

st.markdown("---")

# Footer
st.write("""
---
**Made with ❤️ using Streamlit + RL**

For more info: [GitHub](https://github.com)
""")
