import praw
import pandas as pd
import os
import time
import random
from dotenv import load_dotenv
import yaml
import sys

# Add project root to path for helpers import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.helpers import format_timestamp, recency_weight, age_in_days

load_dotenv()

# Load config
with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), "r") as f:
    config = yaml.safe_load(f)

# Config variables
SUBREDDIT = config['scraper']['subreddit']
BATCH_SIZE = config['scraper']['batch_size']
MAX_BATCHES = config['scraper']['max_batches']
MAX_COMMENTS = config['scraper']['max_comments_per_post']
DELAY_MIN = config['scraper'].get('delay_min', 1.0)
DELAY_MAX = config['scraper'].get('delay_max', 3.0)
RETRY_WAIT = config['scraper'].get('retry_wait', 30)

RAW_DIR = config['paths']['raw_data_dir']
RAW_FILE_PREFIX = config['paths']['raw_data_file']

GOOD_THRESHOLD = config['validation_thresholds']['good']
NEUTRAL_THRESHOLD = config['validation_thresholds']['neutral']

os.makedirs(RAW_DIR, exist_ok=True)

# Reddit authentication
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Scraper function
def fetch_subreddit_posts():
    subreddit = reddit.subreddit(SUBREDDIT)
    post_generator = subreddit.new(limit=None)
    total_fetched = 0

    for batch_number in range(1, MAX_BATCHES + 1):
        batch_records = []
        print(f"\nFetching batch {batch_number}...")

        while len(batch_records) < BATCH_SIZE:
            try:
                submission = next(post_generator)
            except StopIteration:
                print("No more posts available from Reddit.")
                break
            except Exception as e:
                print(f"Request failed: {e}. Retrying in {RETRY_WAIT} seconds...")
                time.sleep(RETRY_WAIT)
                continue

            # Post data
            title = submission.title or ""
            post_body = submission.selftext or ""
            full_text = f"{title} {post_body}".strip()
            upvotes = submission.score
            num_comments = submission.num_comments
            upvote_ratio = getattr(submission, "upvote_ratio", 0.5)
            post_rec = recency_weight(submission.created_utc)
            post_age = age_in_days(submission.created_utc)

            # Comment recency weights
            comment_rec_weights = []
            if num_comments > 0:
                submission.comments.replace_more(limit=0)
                comments_sorted = sorted(submission.comments.list(), key=lambda x: x.created_utc, reverse=True)
                for comment in comments_sorted[:MAX_COMMENTS]:
                    comment_rec_weights.append(recency_weight(comment.created_utc))

            avg_comment_rec = sum(comment_rec_weights) / len(comment_rec_weights) if comment_rec_weights else 0

            # Placeholder sentiment (replace with ML later)
            post_sentiment = 0.5

            # Validation score
            validation_score = (
                0.3 * post_sentiment +
                0.3 * avg_comment_rec +
                0.2 * upvote_ratio +
                0.2 * post_rec
            ) * 100
            validation_score = max(0, min(100, validation_score))

            # Final label
            if validation_score >= GOOD_THRESHOLD:
                label = "good"
            elif validation_score >= NEUTRAL_THRESHOLD:
                label = "neutral"
            else:
                label = "bad"

            # Append record
            batch_records.append({
                "post_id": submission.id,
                "title": title,
                "text": full_text,
                "author": submission.author.name if submission.author else '[deleted]',
                "created_utc": format_timestamp(submission.created_utc),
                "num_comments": num_comments,
                "upvotes": upvotes,
                "upvote_ratio": upvote_ratio,
                "post_age_days": post_age,
                "avg_comment_recency": round(avg_comment_rec, 2),
                "validation_score": round(validation_score, 2),
                "label": label,
                "source_url": f"https://reddit.com{submission.permalink}"
            })
            total_fetched += 1

            # Polite random delay
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        # Save batch
        batch_file = os.path.join(RAW_DIR, f"{RAW_FILE_PREFIX}_{batch_number}.csv")
        pd.DataFrame(batch_records).to_csv(batch_file, index=False)
        print(f"Saved batch {batch_number} with {len(batch_records)} records.")

    print(f"\nTotal posts fetched: {total_fetched}")

if __name__ == "__main__":
    fetch_subreddit_posts()
