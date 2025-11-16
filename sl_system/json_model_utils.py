"""
JSON Model Utilities - Load and use models for inference

Usage:
    from json_model_utils import load_json_model, predict_json_model
    model, scaler, info = load_json_model('path/to/model.json')
    predictions = predict_json_model(model, scaler, X_new)
"""

import json
import numpy as np
from pathlib import Path


def load_json_model(json_path: str):
    """Load model and scaler from JSON"""
    with open(json_path, 'r') as f:
        model_data = json.load(f)

    return model_data


def predict_json_model(model_data: dict, X_new: np.ndarray) -> np.ndarray:
    """
    Make predictions using loaded JSON model

    Note: For actual predictions in production, you need to:
    1. Load the actual sklearn model object (not just JSON)
    2. Use model.predict(X_scaled)
    3. This is a placeholder for RL integration
    """

    # Get model info
    model_info = model_data['model_info']
    scaler_data = model_data['scaler']

    print(f"Model: {model_info['name']}")
    print(f"Features: {model_data['n_features']}")
    print(f"Classes: {model_data['classes']}")

    # Scale input
    mean = np.array(scaler_data['mean'])
    scale = np.array(scaler_data['scale'])
    X_scaled = (X_new - mean) / scale

    return X_scaled  # For actual predictions, use real model


# Alternative: For full model reconstruction
def reconstruct_model_from_json(json_path: str):
    """
    Reconstruct sklearn model from JSON parameters

    Note: This is limited. For full functionality, save models using:
    - pickle/joblib (full object serialization)
    - Or retrain using these parameters
    """
    with open(json_path, 'r') as f:
        model_data = json.load(f)

    model_type = model_data['model_info']['type']
    params = model_data['model_params']

    print(f"Model type: {model_type}")
    print(f"Parameters: {params}")

    return model_data
