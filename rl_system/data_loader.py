"""
Data Loader Utility (Fixed Import)
----------------------------------
- Uses the smart DATA_PATH from paths.py
- Removed the broken 'DATA_PATH_ALT' import
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- FIX: Only import DATA_PATH ---
from paths import DATA_PATH

def load_training_data():
    """
    Loads the processed SaaS dataset using the central paths config.
    """
    
    print(f"   [+] Loading data from: {DATA_PATH}")
    
    # 1. Check if file exists
    if not DATA_PATH.exists():
        print(f"\n[!] ERROR: Dataset not found at: {DATA_PATH}")
        print("    Please check your paths.py configuration.")
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH}")

    # 2. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # 3. Validate Columns
    target_col = 'label_numeric'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # 4. Split Features and Target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 5. Ensure numeric only
    X = X.select_dtypes(include=[np.number])

    return X, y

if __name__ == "__main__":
    # Test run
    try:
        X, y = load_training_data()
        print(f"   [OK] Data Loaded Successfully.")
        print(f"   Features: {X.shape}")
        print(f"   Labels:   {y.shape}")
    except Exception as e:
        print(f"   [Error] {e}")