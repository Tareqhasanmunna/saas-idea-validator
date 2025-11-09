from datetime import datetime, timezone
import math
import pandas as pd
import numpy as np
import logging
import os
import sys

def setup_rl_logger(log_file, logger_name="RL"):
    """Setup logger for RL system with proper UTF-8 encoding"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with UTF-8
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    if hasattr(ch.stream, 'reconfigure'):
        ch.stream.reconfigure(encoding='utf-8')
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def format_timestamp(utc):
    """UTC timestamp to ISO string"""
    if not utc:
        return ""
    return datetime.fromtimestamp(utc, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def recency_weight(created_utc, decay_days=30):
    """Returns a weight for a post based on its age"""
    if not created_utc:
        return 0
    time_diff = datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)
    days_old = time_diff.total_seconds() / 86400
    return math.exp(-days_old / decay_days)

def age_in_days(created_utc):
    """Returns age in days from UTC timestamp"""
    if not created_utc:
        return None
    return (datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)).days

def handle_nan_values(df, method='mean', min_value=0.01):
    """Handle NaN values in feature columns"""
    feature_columns = ['post_sentiment', 'avg_comment_sentiment', 'upvote_ratio', 'post_recency']
    for col in feature_columns:
        if col in df.columns:
            if method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            df[col] = df[col].apply(lambda x: max(x, min_value) if pd.notna(x) else min_value)
    return df

def get_nan_statistics(df, feature_columns=['post_sentiment', 'avg_comment_sentiment', 'upvote_ratio', 'post_recency']):
    """Get NaN statistics for feature columns"""
    stats = {}
    total_records = len(df)
    for col in feature_columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_percentage = (nan_count / total_records) * 100
            stats[col] = {'nan_count': nan_count, 'nan_percentage': nan_percentage}
    return stats