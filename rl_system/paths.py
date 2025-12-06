"""
CENTRAL PATH CONFIGURATION (Absolute Fix)
-----------------------------------------
Dynamically finds the project root 'saas-idea-validator'
to guarantee data loading works from ANY subfolder.
"""

import os
from pathlib import Path

def find_project_root(current_path, marker="data"):
    """
    Walks up the directory tree until it finds the folder containing the marker.
    """
    root = current_path
    for _ in range(5): # Look up to 5 levels deep
        if (root / marker).exists():
            return root
        root = root.parent
    return None

# 1. Determine Project Root
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = find_project_root(CURRENT_FILE.parent)

if ROOT_DIR is None:
    # Fallback: Just assume we are in the root if detection fails
    ROOT_DIR = CURRENT_FILE.parent
    print(f"[Paths] Warning: Could not auto-detect root. Using: {ROOT_DIR}")
else:
    print(f"[Paths] Project Root Detected: {ROOT_DIR}")

# 2. Define Absolute Paths based on the User's File Structure
# User Path: E:\saas-idea-validator\data\processed\balanced\vectorized_features_balanced.csv
DATA_PATH = ROOT_DIR / "data" / "processed" / "balanced" / "vectorized_features_balanced.csv"

# User Path: E:\saas-idea-validator\models\
MODEL_PATH = ROOT_DIR / "models"

# 3. Output Paths
THESIS_OUTPUT_DIR = ROOT_DIR / "ml_outputs_thesis"
RL_VIS_PATH = THESIS_OUTPUT_DIR / "visualizations"
RL_RESULTS_PATH = THESIS_OUTPUT_DIR / "final_comparison"

# 4. Specific Model Paths (for handy reference)
GBM_MODEL = MODEL_PATH / "GradientBoosting" / "GradientBoosting_model.pkl"
LGBM_MODEL = MODEL_PATH / "LightGBM" / "LightGBM_model.pkl"
RF_MODEL = MODEL_PATH / "RandomForest" / "RandomForest_model.pkl"
RL_AGENT_MODEL = MODEL_PATH / "final_bandit_agent.pkl"

# 5. Create directories if they don't exist
if ROOT_DIR.exists():
    for path in [THESIS_OUTPUT_DIR, RL_VIS_PATH, RL_RESULTS_PATH]:
        path.mkdir(parents=True, exist_ok=True)

# 6. Debug Print
print(f"--- PATH CONFIGURATION ---")
print(f"   Root:   {ROOT_DIR}")
print(f"   Data:   {DATA_PATH}")
print(f"   Models: {MODEL_PATH}")

if not DATA_PATH.exists():
    print(f"   [!] CRITICAL: Data file still not found at calculated path.")
else:
    print(f"   [+] OK: Data file found.")