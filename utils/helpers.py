from datetime import datetime, timezone
import math

# Timestamp formatting
def format_timestamp(utc):
    """
    UTC timestamp to ISO string.
    """
    if not utc:
        return ""
    return datetime.fromtimestamp(utc, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# Recency weight (uses fractional days for precision)
def recency_weight(created_utc, decay_days=30):
    """
    Returns a weight for a post or comment based on its age.
    Newer items have higher weight, decaying exponentially.
    
    Uses fractional days for precise recency calculation.
    """
    if not created_utc:
        return 0
    
    time_diff = datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)
    days_old = time_diff.total_seconds() / 86400  # Convert to fractional days
    
    return math.exp(-days_old / decay_days)


# Comment/Post age
def age_in_days(created_utc):
    """
    Returns age in days (integer) from UTC timestamp.
    For display purposes.
    """
    if not created_utc:
        return None
    return (datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)).days
