"""
CRITICAL FIX: Filter out posts with no comments to get REAL comment sentiment variance
NO ARTIFICIAL NOISE BOOSTING - JUST REAL DATA
"""

import praw
import pandas as pd
import numpy as np
import os
import time
import random
from dotenv import load_dotenv
import yaml
import sys
from transformers import pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.helpers import format_timestamp, recency_weight, age_in_days, handle_nan_values, get_nan_statistics
from utils.weight_validator import AutomatedWeightValidator, format_weights_string

load_dotenv()

with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

SUBREDDITS = config['scraper']['subreddits']
BATCH_SIZE = config['scraper']['batch_size']
MAX_BATCHES = config['scraper']['max_batches']
MAX_COMMENTS = config['scraper']['max_comments_per_post']
DELAY_MIN = config['scraper'].get('delay_min', 1.0)
DELAY_MAX = config['scraper'].get('delay_max', 3.0)
RETRY_WAIT = config['scraper'].get('retry_wait', 30)
RAW_DIR = config['paths']['raw_data_dir']
RAW_FILE_PREFIX = config['paths']['raw_data_file']
REPORT_DIR = config['paths']['report_dir']
GOOD_THRESHOLD = config['validation_thresholds']['good']
NEUTRAL_THRESHOLD = config['validation_thresholds']['neutral']
NAN_HANDLING_METHOD = config.get('nan_handling', {}).get('method', 'mean_plus_noise')
NOISE_STD = config.get('nan_handling', {}).get('noise_std', 0.03)
MIN_VALUE = config.get('nan_handling', {}).get('min_value', 0.01)

# CRITICAL: Minimum comments required per post
MIN_COMMENTS_REQUIRED = config.get('scraper', {}).get('min_comments_required', 3)

MAX_POST_AGE_DAYS = 183

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("\n" + "="*70)
print("LOADING SENTIMENT ANALYSIS MODEL")
print("="*70)
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("[OK] Sentiment model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading sentiment model: {e}")
    sentiment_analyzer = None
print("="*70)

print("\n" + "="*70)
print("INITIALIZING AUTOMATED WEIGHT VALIDATOR")
print("="*70)
validator = AutomatedWeightValidator(
    step_size=0.1,
    good_threshold=GOOD_THRESHOLD,
    neutral_threshold=NEUTRAL_THRESHOLD,
    min_weight=MIN_VALUE
)
print(f"[OK] Validator ready with {len(validator.weight_combinations)} weight combinations")
print(f"[OK] NaN handling: {NAN_HANDLING_METHOD.upper()}")
print(f"[OK] Minimum comments per post: {MIN_COMMENTS_REQUIRED}")
print(f"[CRITICAL] Filtering out posts with < {MIN_COMMENTS_REQUIRED} comments for REAL variance!")
print("="*70)

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

COMBINED_SUBS = "+".join(SUBREDDITS)
COLLECTED_POST_IDS = set()


def get_sentiment_score(text):
    """Get sentiment score - returns NaN if cannot analyze"""
    if not sentiment_analyzer:
        return np.nan
    
    if not text or len(text.strip()) < 10:
        return np.nan
    
    text = text[:512]
    
    try:
        result = sentiment_analyzer(text)[0]
        score = result['score']
        
        if result['label'] == 'POSITIVE':
            return min(score, 1.0)
        else:
            return max(1 - score, 0.0)
    except Exception as e:
        return np.nan


def get_post_generator(batch_number, subreddit):
    """Get different post sources with smart deduplication"""
    sources = [
        (subreddit.new(), "NEW_POSTS", "Most Recent Posts"),
        (subreddit.hot(), "HOT_POSTS", "Hot/Trending Posts"),
        (subreddit.top(time_filter='week'), "TOP_WEEK", "Top Posts Last Week"),
        (subreddit.top(time_filter='month'), "TOP_MONTH", "Top Posts Last Month"),
        (subreddit.controversial(time_filter='week'), "CONTROVERSIAL_WEEK", "Controversial Week"),
        (subreddit.top(time_filter='year'), "TOP_YEAR", "Top Posts All Time"),
        (subreddit.controversial(time_filter='month'), "CONTROVERSIAL_MONTH", "Controversial Month"),
    ]
    
    source_index = (batch_number - 1) % len(sources)
    generator, source_code, description = sources[source_index]
    
    return generator, source_code, description


def fetch_multi_subreddit_posts():
    """Main scraping function - FILTERS OUT posts with no comments"""
    total_fetched = 0
    total_skipped = 0
    total_duplicates = 0
    total_no_comments = 0  # NEW: Track posts skipped for no comments
    estimated_total = BATCH_SIZE * MAX_BATCHES
    
    source_distribution = {}
    
    for batch_number in range(1, MAX_BATCHES + 1):
        batch_records = []
        start_idx = (batch_number - 1) * BATCH_SIZE + 1
        end_idx = start_idx + BATCH_SIZE - 1
        
        subreddit = reddit.subreddit(COMBINED_SUBS)
        post_generator, source_code, source_description = get_post_generator(batch_number, subreddit)
        
        source_distribution[batch_number] = source_code
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_number}/{MAX_BATCHES}: COLLECTING POSTS {start_idx}-{end_idx}")
        print(f"{'='*70}")
        print(f"[INFO] Source: {source_code}")
        print(f"[INFO] Description: {source_description}")
        print(f"[FILTER] Only collecting posts with >= {MIN_COMMENTS_REQUIRED} comments")
        print(f"{'='*70}")
        
        retry_count = 0
        max_retries = 3
        
        while len(batch_records) < BATCH_SIZE:
            try:
                submission = next(post_generator)
            except StopIteration:
                print(f"\n[WARNING] No more posts available from {source_code}.")
                
                if len(batch_records) < BATCH_SIZE and retry_count < max_retries:
                    retry_count += 1
                    print(f"\n[INFO] Retrying with fallback source (Attempt {retry_count}/{max_retries})...")
                    post_generator = reddit.subreddit(COMBINED_SUBS).new(limit=None)
                    source_code = f"{source_code}_RETRY_{retry_count}"
                    continue
                break
            except Exception as e:
                print(f"\n[ERROR] Request failed: {e}. Retrying in {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                continue
            
            post_age = age_in_days(submission.created_utc)
            
            if post_age > MAX_POST_AGE_DAYS:
                total_skipped += 1
                continue
            
            if submission.id in COLLECTED_POST_IDS:
                total_duplicates += 1
                continue
            
            # CRITICAL FIX: Skip posts with too few comments
            num_comments = submission.num_comments
            if num_comments < MIN_COMMENTS_REQUIRED:
                total_no_comments += 1
                continue
            
            title = submission.title or ""
            post_body = submission.selftext or ""
            full_text = f"{title} {post_body}".strip()
            upvotes = submission.score
            upvote_ratio = getattr(submission, "upvote_ratio", np.nan)
            post_rec = recency_weight(submission.created_utc)
            
            post_sentiment = get_sentiment_score(full_text)
            
            # Fetch comments and calculate COMMENT SENTIMENT
            comment_sentiments = []
            try:
                submission.comments.replace_more(limit=0)
                comments_sorted = sorted(submission.comments.list(), 
                                       key=lambda x: x.created_utc, reverse=True)
                
                for comment in comments_sorted[:MAX_COMMENTS]:
                    comment_text = comment.body if hasattr(comment, 'body') else ""
                    if comment_text and len(comment_text.strip()) > 5:
                        comment_sent = get_sentiment_score(comment_text)
                        if not pd.isna(comment_sent):
                            comment_sentiments.append(comment_sent)
            except Exception as e:
                # If comment fetch fails, skip this post
                total_no_comments += 1
                continue
            
            # CRITICAL: If no valid comment sentiments, skip this post
            if len(comment_sentiments) == 0:
                total_no_comments += 1
                continue
            
            avg_comment_sentiment = np.mean(comment_sentiments)
            
            batch_records.append({
                "post_id": submission.id,
                "subreddit": submission.subreddit.display_name,
                "title": title,
                "text": full_text,
                "author": submission.author.name if submission.author else '[deleted]',
                "created_utc": format_timestamp(submission.created_utc),
                "num_comments": num_comments,
                "upvotes": upvotes,
                "upvote_ratio": upvote_ratio,
                "post_age_days": post_age,
                "post_sentiment": post_sentiment,
                "avg_comment_sentiment": avg_comment_sentiment,
                "post_recency": post_rec,
                "source_url": f"https://reddit.com{submission.permalink}",
                "source_type": source_code
            })
            
            COLLECTED_POST_IDS.add(submission.id)
            
            total_fetched += 1
            progress_percent = (total_fetched / estimated_total) * 100
            print(f"  -> Collecting: {len(batch_records)}/{BATCH_SIZE} | No comments skipped: {total_no_comments}", end="\r")
            
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        
        print(f"\n[OK] Batch {batch_number} collection complete: {len(batch_records)} posts")
        print(f"[INFO] Skipped {total_no_comments} posts with insufficient comments\n")
        
        if len(batch_records) == 0:
            print(f"[WARNING] Batch {batch_number} is empty, skipping...")
            continue
        
        df = pd.DataFrame(batch_records)
        
        print(f"{'─'*70}")
        print(f"NaN HANDLING: Using {NAN_HANDLING_METHOD.upper()}")
        print(f"{'─'*70}")
        
        nan_stats_before = get_nan_statistics(df)
        
        print(f"NaN counts BEFORE replacement:")
        for feat, stats in nan_stats_before.items():
            print(f"  - {feat}: {stats['nan_count']} ({stats['nan_percentage']:.1f}%)")
        
        if NAN_HANDLING_METHOD == 'mean_plus_noise':
            df = handle_nan_values(df, method=NAN_HANDLING_METHOD, noise_std=NOISE_STD, min_value=MIN_VALUE)
        else:
            df = handle_nan_values(df, method=NAN_HANDLING_METHOD, min_value=MIN_VALUE)
        
        print(f"[OK] NaN values replaced\n")
        
        batch_records = df.to_dict('records')
        
        print(f"{'─'*70}")
        print(f"VALIDATION: Finding best weights for Batch {batch_number}")
        print(f"{'─'*70}")
        
        best_weights, best_accuracy, _ = validator.find_best_weights(batch_records)
        
        print(f"\n[OK] Weight optimization complete!")
        print(f"  [ACCURACY] {best_accuracy:.2f}")
        print(f"  [WEIGHTS - ALL PARAMETERS USED]:")
        print(f"    - Post Sentiment:           {best_weights[0]:.2f} [USED]")
        print(f"    - Comment Sentiment (avg):  {best_weights[1]:.2f} [USED]")
        print(f"    - Upvote Ratio:             {best_weights[2]:.2f} [USED]")
        print(f"    - Post Recency:             {best_weights[3]:.2f} [USED]")
        
        print(f"\n{'─'*70}")
        print(f"LABELING: Applying best weights to Batch {batch_number}")
        print(f"{'─'*70}")
        
        labeled_records = validator.validate_and_label_batch(batch_records, best_weights)
        df = pd.DataFrame(labeled_records)
        
        label_counts = df['label'].value_counts().to_dict()
        print(f"[OK] Labeling complete!")
        print(f"  [DISTRIBUTION]")
        for label in ['good', 'neutral', 'bad']:
            count = label_counts.get(label, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"    - {label.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"  [STATISTICS]")
        print(f"    - Comment Sentiment Variance: {df['avg_comment_sentiment'].var():.6f}")
        
        print(f"\n{'─'*70}")
        print(f"SAVING: Batch {batch_number}")
        print(f"{'─'*70}")
        
        final_columns = [
            "post_id", "subreddit", "title", "text", "author", "created_utc",
            "num_comments", "upvotes", "upvote_ratio", "post_age_days",
            "post_sentiment", "avg_comment_sentiment", "post_recency",
            "validation_score", "label", "source_url", "source_type"
        ]
        
        batch_file = os.path.join(RAW_DIR, f"{RAW_FILE_PREFIX}_{batch_number}.csv")
        df[final_columns].to_csv(batch_file, index=False, encoding='utf-8')
        
        print(f"[OK] Saved CSV to: {batch_file}")
        
        weights_str = format_weights_string(best_weights)
        weights_file = os.path.join(REPORT_DIR, f"{RAW_FILE_PREFIX}_{batch_number}_report.txt")
        
        with open(weights_file, 'w', encoding='utf-8') as f:
            f.write(f"BATCH {batch_number} - WEIGHT OPTIMIZATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Source: {source_code}\n")
            f.write(f"Posts skipped (no comments): {total_no_comments}\n\n")
            f.write(f"Optimization Accuracy: {best_accuracy:.2f}\n\n")
            f.write("Best Weights:\n")
            f.write(f"  - Post Sentiment:           {best_weights[0]:.2f}\n")
            f.write(f"  - Comment Sentiment (avg):  {best_weights[1]:.2f}\n")
            f.write(f"  - Upvote Ratio:             {best_weights[2]:.2f}\n")
            f.write(f"  - Post Recency:             {best_weights[3]:.2f}\n\n")
            f.write(f"Weights String: {weights_str}\n\n")
            f.write(f"Comment Sentiment Variance: {df['avg_comment_sentiment'].var():.6f}\n")
        
        print(f"[OK] Saved report to: {weights_file}")
        
        print(f"\n{'='*70}")
        print(f"[SUCCESS] BATCH {batch_number} COMPLETE")
        print(f"{'='*70}")
    
    print(f"\n\n{'='*70}")
    print(f"[SUCCESS] SCRAPING COMPLETED")
    print(f"{'='*70}")
    print(f"  - Total posts collected: {total_fetched}")
    print(f"  - Posts skipped (no comments): {total_no_comments}")
    print(f"  - Duplicates skipped: {total_duplicates}")
    print(f"  - Unique posts: {len(COLLECTED_POST_IDS)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"REDDIT SAAS POST SCRAPER")
    print(f"REAL DATA ONLY - NO ARTIFICIAL NOISE")
    print(f"FILTERING POSTS WITH < {MIN_COMMENTS_REQUIRED} COMMENTS")
    print(f"{'='*70}\n")
    
    fetch_multi_subreddit_posts()
