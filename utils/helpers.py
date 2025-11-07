from datetime import datetime, timezone
import math
import pandas as pd
import numpy as np


# Timestamp formatting
def format_timestamp(utc):
    """UTC timestamp to ISO string"""
    if not utc:
        return ""
    return datetime.fromtimestamp(utc, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# Recency weight
def recency_weight(created_utc, decay_days=30):
    """
    Returns a weight for a post based on its age.
    Newer items have higher weight, decaying exponentially.
    Uses fractional days for precise recency calculation.
    """
    if not created_utc:
        return 0
    
    time_diff = datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)
    days_old = time_diff.total_seconds() / 86400  # Fractional days
    
    return math.exp(-days_old / decay_days)


# Post age in days
def age_in_days(created_utc):
    """
    Returns age in days (integer) from UTC timestamp.
    """
    if not created_utc:
        return None
    return (datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)).days


# NaN handling functions

def replace_nan_with_mean_plus_noise(df, column, noise_std=0.03, min_value=0.01):
    """
    Replace NaN values with mean + random noise, ensuring minimum value.
    
    This approach:
    - Preserves variance by adding small random noise
    - Prevents identical values that collapse variance
    - Ensures no value falls below min_value (0.01 default)
    
    Args:
        df: pandas DataFrame
        column: column name to replace NaN values
        noise_std: standard deviation of random noise (default: 0.03)
        min_value: minimum value to ensure (default: 0.01)
    
    Returns:
        DataFrame with NaN replaced by mean + noise
    """
    if column in df.columns:
        mean_value = df[column].mean()
        if not pd.isna(mean_value):
            nan_mask = df[column].isna()
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                # Generate noise
                noise = np.random.normal(0, noise_std, nan_count)
                # Calculate replacement values
                replacement = mean_value + noise
                # Ensure minimum value
                replacement = np.maximum(replacement, min_value)
                # Replace NaN values
                df.loc[nan_mask, column] = replacement
    
    return df


def replace_nan_with_mean(df, column, min_value=0.01):
    """
    Replace NaN values with mean, ensuring minimum value.
    
    Args:
        df: pandas DataFrame
        column: column name to replace NaN values
        min_value: minimum value to ensure (default: 0.01)
    
    Returns:
        DataFrame with NaN replaced by mean
    """
    if column in df.columns:
        mean_value = df[column].mean()
        if not pd.isna(mean_value):
            mean_value = max(mean_value, min_value)
            df[column].fillna(mean_value, inplace=True)
    return df


def replace_nan_with_median(df, column, min_value=0.01):
    """
    Replace NaN values with median, ensuring minimum value.
    
    Args:
        df: pandas DataFrame
        column: column name to replace NaN values
        min_value: minimum value to ensure (default: 0.01)
    
    Returns:
        DataFrame with NaN replaced by median
    """
    if column in df.columns:
        median_value = df[column].median()
        if not pd.isna(median_value):
            median_value = max(median_value, min_value)
            df[column].fillna(median_value, inplace=True)
    return df


def handle_nan_values(df, method='mean_plus_noise', noise_std=0.03, min_value=0.01):
    """
    Handle NaN values in all feature columns.
    Ensures no value falls below min_value to prevent zero weights.
    
    Args:
        df: pandas DataFrame with features
        method: 'mean', 'median', or 'mean_plus_noise' (default: 'mean_plus_noise')
        noise_std: standard deviation for noise if method='mean_plus_noise'
        min_value: minimum value to ensure (default: 0.01)
    
    Returns:
        DataFrame with NaN values replaced
    """
    feature_columns = ['post_sentiment', 'avg_comment_sentiment', 'upvote_ratio', 'post_recency']
    
    for col in feature_columns:
        if col in df.columns:
            if method == 'mean':
                df = replace_nan_with_mean(df, col, min_value)
            elif method == 'median':
                df = replace_nan_with_median(df, col, min_value)
            elif method == 'mean_plus_noise':
                df = replace_nan_with_mean_plus_noise(df, col, noise_std, min_value)
    
    return df


def get_nan_statistics(df, feature_columns=['post_sentiment', 'avg_comment_sentiment', 'upvote_ratio', 'post_recency']):
    """
    Get NaN statistics for feature columns.
    
    Args:
        df: pandas DataFrame
        feature_columns: list of feature column names
    
    Returns:
        Dictionary with NaN counts and percentages
    """
    stats = {}
    total_records = len(df)
    
    for col in feature_columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_percentage = (nan_count / total_records) * 100
            stats[col] = {
                'nan_count': nan_count,
                'nan_percentage': nan_percentage,
                'valid_count': total_records - nan_count
            }
    
    return stats

