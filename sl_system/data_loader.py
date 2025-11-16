"""
Unified Data Loader for Pre-Vectorized Dataset

Loads 100-dimensional vectors + 4 numeric features + label_numeric
No text processing needed - data is already vectorized!

Usage:
    from data_loader import load_training_data
    X, y = load_training_data()
"""

import os
from pathlib import Path
import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Load pre-vectorized dataset"""

    DATASET_PATHS = [
        r"E:\saas-idea-validator\data\processed\vectorized_features.csv",
        "data/processed/vectorized_features.csv",
        "vectorized_features.csv",
    ]

    TARGET_COLUMN = 'label_numeric'
    EXPECTED_FEATURES = 104  # 100 vectors + 4 numeric features

    def __init__(self, custom_path: str = None):
        """Initialize data loader"""
        self.custom_path = custom_path
        self.dataset_path = None
        self.data = None
        self.X = None
        self.y = None

    def find_dataset(self) -> Path:
        """Find dataset from common locations"""
        if self.custom_path:
            path = Path(self.custom_path)
            if path.exists():
                logger.info(f"✓ Found custom dataset: {path}")
                return path
            else:
                raise FileNotFoundError(f"Custom dataset not found: {self.custom_path}")

        logger.info("Searching for dataset...")
        for path_str in self.DATASET_PATHS:
            path = Path(path_str)
            if path.exists():
                logger.info(f"✓ Found dataset: {path}")
                return path

        raise FileNotFoundError(f"Dataset not found. Searched: {self.DATASET_PATHS}")

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate pre-vectorized dataset"""
        self.dataset_path = self.find_dataset()

        logger.info(f"Loading dataset from: {self.dataset_path}")
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Check target column exists
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Column '{self.TARGET_COLUMN}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        # Extract features and target
        X = df.drop(columns=[self.TARGET_COLUMN])
        y = df[self.TARGET_COLUMN]

        logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"Feature columns: {X.columns.tolist()[:5]}... (showing first 5)")

        # Check for missing values
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()

        if missing_X > 0 or missing_y > 0:
            logger.warning(f"Missing values - X: {missing_X}, y: {missing_y}")
            X = X.fillna(X.mean())
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            logger.info("Filled/removed missing values")

        # Validate data types (should be numeric)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns found: {non_numeric}")

        logger.info("✓ Dataset validated successfully")
        logger.info(f"Class distribution:\n{y.value_counts().to_string()}")

        self.X = X
        self.y = y

        return X, y


def load_training_data(custom_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load pre-vectorized training data"""
    loader = DataLoader(custom_path=custom_path)
    return loader.load()
