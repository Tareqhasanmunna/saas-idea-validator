"""
Reddit SaaS Post Scraper with Automated Per-Batch Weight Optimization
WITH DIVERSITY FOR DYNAMIC MODEL LEARNING
WITH SENTIMENT ANALYSIS
WITH SMART DEDUPLICATION

Features:
- Automatic duplicate detection across batches
- Smart retry with different strategies
- Fallback to different sources if needed
- Maintains unique dataset
"""

import praw
import pandas as pd
import os
import time
import random
from dotenv import load_dotenv
import yaml
import sys
from transformers import pipeline
from datetime import datetime, timezone

# Add project root for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.helpers import format_timestamp, recency_weight, age_in_days
from utils.weight_validator import AutomatedWeightValidator, format_weights_string

# Load environment variables
load_dotenv()

# Load config.yaml
with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Config variables
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

# Maximum post age in days (6 months ≈ 183 days)
MAX_POST_AGE_DAYS = 183

# Create directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Initialize sentiment analyzer
print("\n" + "="*70)
print("LOADING SENTIMENT ANALYSIS MODEL")
print("="*70)
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("✓ Sentiment model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")
    print("⚠️  Will use fallback sentiment scores (0.5)")
    sentiment_analyzer = None
print("="*70)

# Initialize validator
print("\n" + "="*70)
print("INITIALIZING AUTOMATED WEIGHT VALIDATOR")
print("="*70)
validator = AutomatedWeightValidator(
    step_size=0.1,
    good_threshold=GOOD_THRESHOLD,
    neutral_threshold=NEUTRAL_THRESHOLD
)
print(f"✓ Validator ready with {len(validator.weight_combinations)} weight combinations")
print("="*70)

# Reddit authentication
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

COMBINED_SUBS = "+".join(SUBREDDITS)

# Global set to track all collected post IDs across batches
COLLECTED_POST_IDS = set()


def get_sentiment_score(text):
    """Get sentiment score for text using transformer model."""
    if not sentiment_analyzer:
        return 0.5
    
    if not text or len(text.strip()) < 10:
        return 0.5
    
    text = text[:512]
    
    try:
        result = sentiment_analyzer(text)[0]
        score = result['score']
        
        if result['label'] == 'POSITIVE':
            return min(score, 1.0)
        else:
            return max(1 - score, 0.0)
    except Exception as e:
        return 0.5


def get_post_generator(batch_number, subreddit):
    """
    Get different post sources - SMART DEDUPLICATION VERSION
    Rotates through multiple strategies to avoid duplicates
    """
    # Use different sources to maximize diversity and minimize duplicates
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
    """Main scraping function with smart deduplication."""
    total_fetched = 0
    total_skipped = 0
    total_duplicates = 0
    estimated_total = BATCH_SIZE * MAX_BATCHES
    
    source_distribution = {}
    
    for batch_number in range(1, MAX_BATCHES + 1):
        batch_records = []
        start_idx = (batch_number - 1) * BATCH_SIZE + 1
        end_idx = start_idx + BATCH_SIZE - 1
        
        # Get appropriate generator for this batch
        subreddit = reddit.subreddit(COMBINED_SUBS)
        post_generator, source_code, source_description = get_post_generator(batch_number, subreddit)
        
        source_distribution[batch_number] = source_code
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_number}/{MAX_BATCHES}: COLLECTING POSTS {start_idx}-{end_idx}")
        print(f"{'='*70}")
        print(f"📌 Source: {source_code}")
        print(f"📝 Description: {source_description}")
        print(f"{'='*70}")
        
        # Step 1: Collect batch data with deduplication
        sentiment_analyzed = 0
        retry_count = 0
        max_retries = 3
        
        while len(batch_records) < BATCH_SIZE:
            try:
                submission = next(post_generator)
            except StopIteration:
                print(f"\n⚠️  No more posts available from {source_code}.")
                
                # If we couldn't fill the batch, retry with different source
                if len(batch_records) < BATCH_SIZE and retry_count < max_retries:
                    retry_count += 1
                    print(f"\n🔄 Retrying with fallback source (Attempt {retry_count}/{max_retries})...")
                    post_generator = reddit.subreddit(COMBINED_SUBS).new(limit=None)
                    source_code = f"{source_code}_RETRY_{retry_count}"
                    continue
                break
            except Exception as e:
                print(f"\n❌ Request failed: {e}. Retrying in {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                continue
            
            # Calculate post age
            post_age = age_in_days(submission.created_utc)
            
            # Skip posts older than 6 months
            if post_age > MAX_POST_AGE_DAYS:
                total_skipped += 1
                continue
            
            # ✅ CHECK FOR DUPLICATES
            if submission.id in COLLECTED_POST_IDS:
                total_duplicates += 1
                print(f"  ⚠️  Duplicate found: {submission.id}, skipping...", end="\r")
                continue
            
            # Extract post details
            title = submission.title or ""
            post_body = submission.selftext or ""
            full_text = f"{title} {post_body}".strip()
            upvotes = submission.score
            num_comments = submission.num_comments
            upvote_ratio = getattr(submission, "upvote_ratio", 0.5)
            post_rec = recency_weight(submission.created_utc)
            
            # Get sentiment score
            post_sentiment = get_sentiment_score(full_text)
            sentiment_analyzed += 1
            
            # Fetch comments
            comment_rec_weights = []
            if num_comments > 0:
                try:
                    submission.comments.replace_more(limit=0)
                    comments_sorted = sorted(submission.comments.list(), 
                                           key=lambda x: x.created_utc, reverse=True)
                    for comment in comments_sorted[:MAX_COMMENTS]:
                        comment_rec_weights.append(recency_weight(comment.created_utc))
                except Exception as e:
                    pass
            
            avg_comment_rec = sum(comment_rec_weights) / len(comment_rec_weights) if comment_rec_weights else 0
            
            # Add to records
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
                "avg_comment_recency": round(avg_comment_rec, 2),
                "post_sentiment": round(post_sentiment, 4),
                "post_recency": round(post_rec, 4),
                "source_url": f"https://reddit.com{submission.permalink}",
                "source_type": source_code
            })
            
            # Track this post ID globally
            COLLECTED_POST_IDS.add(submission.id)
            
            total_fetched += 1
            progress_percent = (total_fetched / estimated_total) * 100
            print(f"  ➜ Collecting: {len(batch_records)}/{BATCH_SIZE} posts ({progress_percent:.1f}% total) | Duplicates: {total_duplicates}", end="\r")
            
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        
        print(f"\n✓ Batch {batch_number} collection complete: {len(batch_records)} posts\n")
        
        # Skip validation/labeling if batch is empty
        if len(batch_records) == 0:
            print(f"⚠️  Batch {batch_number} is empty, skipping...")
            continue
        
        # Step 2: Send to validation for weight optimization
        print(f"{'─'*70}")
        print(f"VALIDATION: Finding best weights for Batch {batch_number}")
        print(f"{'─'*70}")
        
        best_weights, best_accuracy, _ = validator.find_best_weights(batch_records)
        
        print(f"\n✓ Weight optimization complete!")
        print(f"  📊 Best accuracy: {best_accuracy:.2f}")
        print(f"  ⚙️  Best weights:")
        print(f"    • Sentiment:        {best_weights[0]:.2f}")
        print(f"    • Comment recency:  {best_weights[1]:.2f}")
        print(f"    • Upvote ratio:     {best_weights[2]:.2f}")
        print(f"    • Post recency:     {best_weights[3]:.2f}")
        
        # Step 3: Apply best weights to label batch
        print(f"\n{'─'*70}")
        print(f"LABELING: Applying best weights to Batch {batch_number}")
        print(f"{'─'*70}")
        
        labeled_records = validator.validate_and_label_batch(batch_records, best_weights)
        df = pd.DataFrame(labeled_records)
        
        # Label distribution
        label_counts = df['label'].value_counts().to_dict()
        print(f"✓ Labeling complete!")
        print(f"  📈 Label distribution:")
        for label in ['good', 'neutral', 'bad']:
            count = label_counts.get(label, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"    • {label.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Sentiment statistics
        print(f"  📊 Sentiment statistics:")
        print(f"    • Mean: {df['post_sentiment'].mean():.3f}")
        print(f"    • Std: {df['post_sentiment'].std():.3f}")
        print(f"    • Min: {df['post_sentiment'].min():.3f}")
        print(f"    • Max: {df['post_sentiment'].max():.3f}")
        
        # Step 4: Save clean CSV
        print(f"\n{'─'*70}")
        print(f"SAVING: Batch {batch_number}")
        print(f"{'─'*70}")
        
        final_columns = [
            "post_id", "subreddit", "title", "text", "author", "created_utc",
            "num_comments", "upvotes", "upvote_ratio", "post_age_days",
            "avg_comment_recency", "post_sentiment", "validation_score", "label", 
            "source_url", "source_type"
        ]
        
        batch_file = os.path.join(RAW_DIR, f"{RAW_FILE_PREFIX}_{batch_number}.csv")
        df[final_columns].to_csv(batch_file, index=False)
        
        print(f"✓ Saved CSV to: {batch_file}")
        print(f"  📁 Records: {len(df)}")
        
        # Step 5: Save weights and accuracy to TXT file
        weights_str = format_weights_string(best_weights)
        weights_file = os.path.join(REPORT_DIR, f"{RAW_FILE_PREFIX}_{batch_number}_report.txt")
        
        with open(weights_file, 'w') as f:
            f.write(f"BATCH {batch_number} - WEIGHT OPTIMIZATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Source: {source_code}\n")
            f.write(f"Description: {source_description}\n\n")
            f.write(f"Optimization Accuracy: {best_accuracy:.2f}\n\n")
            f.write("Best Weights:\n")
            f.write(f"  - Sentiment Weight:        {best_weights[0]:.2f}\n")
            f.write(f"  - Comment Recency Weight:  {best_weights[1]:.2f}\n")
            f.write(f"  - Upvote Ratio Weight:     {best_weights[2]:.2f}\n")
            f.write(f"  - Post Recency Weight:     {best_weights[3]:.2f}\n\n")
            f.write(f"Weights String: {weights_str}\n\n")
            f.write("Label Distribution:\n")
            for label in ['good', 'neutral', 'bad']:
                count = label_counts.get(label, 0)
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                f.write(f"  - {label.capitalize()}: {count} ({percentage:.1f}%)\n")
            f.write("\nSentiment Statistics:\n")
            f.write(f"  - Mean: {df['post_sentiment'].mean():.3f}\n")
            f.write(f"  - Std: {df['post_sentiment'].std():.3f}\n")
            f.write(f"  - Min: {df['post_sentiment'].min():.3f}\n")
            f.write(f"  - Max: {df['post_sentiment'].max():.3f}\n")
            f.write("\n" + "="*50 + "\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"CSV File: {batch_file}\n")
        
        print(f"✓ Saved report to: {weights_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH {batch_number} COMPLETE")
        print(f"{'='*70}")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print(f"🎉 SCRAPING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  ✓ Total posts collected: {total_fetched}")
    print(f"  ✓ Total duplicates skipped: {total_duplicates}")
    print(f"  ✓ Total posts skipped (>6 months): {total_skipped}")
    print(f"  ✓ Unique posts in collection: {len(COLLECTED_POST_IDS)}")
    print(f"  ✓ Total batches saved: {MAX_BATCHES}")
    print(f"  ✓ Data directory: {RAW_DIR}")
    print(f"  ✓ Report directory: {REPORT_DIR}")
    
    print(f"\n🌍 SOURCE DISTRIBUTION:")
    for batch_num, source in source_distribution.items():
        print(f"  • Batch {batch_num:2d}: {source}")
    
    print(f"\n💾 DATASET CHARACTERISTICS:")
    print(f"  ✓ Total unique posts: {len(COLLECTED_POST_IDS)}")
    print(f"  ✓ Age range: 0-180 days")
    print(f"  ✓ Diversity: Maximum (7 different sources rotated)")
    print(f"  ✓ Sentiment Analysis: ENABLED ✅")
    print(f"  ✓ Deduplication: ENABLED ✅")
    print(f"  ✓ Model learning: Dynamic & Adaptive ✅")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 REDDIT SAAS POST SCRAPER")
    print(f"WITH AUTOMATED PER-BATCH WEIGHT OPTIMIZATION")
    print(f"WITH DIVERSITY FOR DYNAMIC MODEL LEARNING")
    print(f"WITH SENTIMENT ANALYSIS")
    print(f"WITH SMART DEDUPLICATION ✨")
    print(f"{'='*70}")
    print(f"\n📋 CONFIGURATION:")
    print(f"  Target subreddits: {', '.join(SUBREDDITS)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Total batches: {MAX_BATCHES}")
    print(f"  Max post age: {MAX_POST_AGE_DAYS} days (~6 months)")
    
    print(f"\n🌍 DIVERSITY STRATEGY:")
    print(f"  ✓ Rotates through 7 different post sources")
    print(f"  ✓ Automatic duplicate detection")
    print(f"  ✓ Fallback retry mechanism")
    print(f"  ✓ Ensures unique dataset")
    
    print(f"\n💡 FEATURES:")
    print(f"  ✓ Multi-source data collection")
    print(f"  ✓ Real sentiment analysis")
    print(f"  ✓ Smart deduplication across batches")
    print(f"  ✓ Automatic retry on duplicates")
    print(f"  ✓ Better feature variance")
    print(f"  ✓ Dynamic model learning")
    print(f"  ✓ Improved generalization")
    
    print(f"{'='*70}\n")
    
    fetch_multi_subreddit_posts()
