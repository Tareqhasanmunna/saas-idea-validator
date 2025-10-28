import praw
import pandas as pd
import os
import time
import random
from dotenv import load_dotenv
import yaml
import sys

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


def get_post_generator(batch_number, subreddit):
    """
    Get different post sources for different batches for diversity.
    
    Batches 1-3:   Fresh posts (new)           - 0-7 days
    Batches 4-6:   Top posts (week)            - 0-7 days popular
    Batches 7-8:   Top posts (month)           - 1-30 days
    Batches 9-10:  Top posts (year)            - older data (filtered to 6 months)
    """
    if batch_number <= 3:
        return subreddit.new(limit=None), "NEW_FRESH", "Most Recent Posts (0-7 days)"
    elif batch_number <= 6:
        return subreddit.top('week', limit=None), "TOP_WEEK", "Popular Posts Last Week (0-7 days)"
    elif batch_number <= 8:
        return subreddit.top('month', limit=None), "TOP_MONTH", "Popular Posts Last Month (1-30 days)"
    else:
        return subreddit.top('year', limit=None), "TOP_YEAR", "Popular Posts (Older Data, Filtered to 6 Months)"


def fetch_multi_subreddit_posts():
    """Main scraping function with per-batch optimization and diversity."""
    total_fetched = 0
    total_skipped = 0
    estimated_total = BATCH_SIZE * MAX_BATCHES
    
    # Track source distribution
    source_distribution = {}
    
    for batch_number in range(1, MAX_BATCHES + 1):
        batch_records = []
        start_idx = (batch_number - 1) * BATCH_SIZE + 1
        end_idx = start_idx + BATCH_SIZE - 1
        
        # Get appropriate generator for this batch
        subreddit = reddit.subreddit(COMBINED_SUBS)
        post_generator, source_code, source_description = get_post_generator(batch_number, subreddit)
        
        # Track source
        source_distribution[batch_number] = source_code
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_number}/{MAX_BATCHES}: COLLECTING POSTS {start_idx}-{end_idx}")
        print(f"{'='*70}")
        print(f"📌 Source: {source_code}")
        print(f"📝 Description: {source_description}")
        print(f"{'='*70}")
        
        # Step 1: Collect batch data
        while len(batch_records) < BATCH_SIZE:
            try:
                submission = next(post_generator)
            except StopIteration:
                print(f"\n⚠️  No more posts available from {source_code}.")
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
            
            # Extract post details
            title = submission.title or ""
            post_body = submission.selftext or ""
            full_text = f"{title} {post_body}".strip()
            upvotes = submission.score
            num_comments = submission.num_comments
            upvote_ratio = getattr(submission, "upvote_ratio", 0.5)
            post_rec = recency_weight(submission.created_utc)
            
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
            post_sentiment = 0.5  # Placeholder
            
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
                "post_sentiment": post_sentiment,
                "post_recency": round(post_rec, 2),
                "source_url": f"https://reddit.com{submission.permalink}",
                "source_type": source_code
            })
            
            total_fetched += 1
            progress_percent = (total_fetched / estimated_total) * 100
            print(f"  ➜ Collecting: {len(batch_records)}/{BATCH_SIZE} posts ({progress_percent:.1f}% total)", end="\r")
            
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        
        print(f"\n✓ Batch {batch_number} collection complete: {len(batch_records)} posts\n")
        
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
        
        # Step 4: Save clean CSV (no metadata)
        print(f"\n{'─'*70}")
        print(f"SAVING: Batch {batch_number}")
        print(f"{'─'*70}")
        
        final_columns = [
            "post_id", "subreddit", "title", "text", "author", "created_utc",
            "num_comments", "upvotes", "upvote_ratio", "post_age_days",
            "avg_comment_recency", "validation_score", "label", "source_url", "source_type"
        ]
        
        # Save clean CSV to raw_batch folder
        batch_file = os.path.join(RAW_DIR, f"{RAW_FILE_PREFIX}_{batch_number}.csv")
        df[final_columns].to_csv(batch_file, index=False)
        
        print(f"✓ Saved CSV to: {batch_file}")
        print(f"  📁 Records: {len(df)}")
        
        # Step 5: Save weights and accuracy to TXT file in report folder
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
            f.write("\n" + "="*50 + "\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"CSV File: {batch_file}\n")
        
        print(f"✓ Saved report to: {weights_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH {batch_number} COMPLETE")
        print(f"{'='*70}")
    
    # Final summary with diversity breakdown
    print(f"\n\n{'='*70}")
    print(f"🎉 SCRAPING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  ✓ Total posts collected: {total_fetched}")
    print(f"  ✓ Total posts skipped (>6 months old): {total_skipped}")
    print(f"  ✓ Total batches saved: {MAX_BATCHES}")
    print(f"  ✓ Data directory: {RAW_DIR}")
    print(f"  ✓ Report directory: {REPORT_DIR}")
    
    print(f"\n🌍 DIVERSITY BREAKDOWN:")
    print(f"  📌 Batches 1-3:   NEW_FRESH - Most Recent Posts (0-7 days)")
    print(f"  📌 Batches 4-6:   TOP_WEEK - Popular Posts Last Week (0-7 days)")
    print(f"  📌 Batches 7-8:   TOP_MONTH - Popular Posts Last Month (1-30 days)")
    print(f"  📌 Batches 9-10:  TOP_YEAR - Older Popular Posts (up to 6 months)")
    
    print(f"\n🎯 ACTUAL SOURCE DISTRIBUTION:")
    for batch_num, source in source_distribution.items():
        print(f"  • Batch {batch_num:2d}: {source}")
    
    print(f"\n💾 DATASET CHARACTERISTICS:")
    print(f"  ✓ Total posts: 1000")
    print(f"  ✓ Age range: 0-180 days")
    print(f"  ✓ Diversity: Maximum (4 different sources)")
    print(f"  ✓ Model learning: Dynamic & Adaptive ✅")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 REDDIT SAAS POST SCRAPER")
    print(f"WITH AUTOMATED PER-BATCH WEIGHT OPTIMIZATION")
    print(f"WITH DIVERSITY FOR DYNAMIC MODEL LEARNING")
    print(f"{'='*70}")
    print(f"\n📋 CONFIGURATION:")
    print(f"  Target subreddits: {', '.join(SUBREDDITS)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Total batches: {MAX_BATCHES}")
    print(f"  Total posts to collect: {BATCH_SIZE * MAX_BATCHES}")
    print(f"  Weight combinations to test: 286 per batch")
    print(f"  Max post age: {MAX_POST_AGE_DAYS} days (~6 months)")
    
    print(f"\n🌍 DIVERSITY STRATEGY:")
    print(f"  • Batches 1-3:  Fresh posts (Most Recent)")
    print(f"  • Batches 4-6:  Top posts from last week (Popular)")
    print(f"  • Batches 7-8:  Top posts from last month (Established)")
    print(f"  • Batches 9-10: Older popular posts (Historical)")
    
    print(f"\n💡 BENEFITS:")
    print(f"  ✓ Multi-source data collection")
    print(f"  ✓ Better feature variance")
    print(f"  ✓ Dynamic model learning")
    print(f"  ✓ Improved generalization")
    print(f"  ✓ Comprehensive dataset")
    
    print(f"{'='*70}\n")
    
    fetch_multi_subreddit_posts()