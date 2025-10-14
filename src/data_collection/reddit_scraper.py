import praw
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timezone
import os
import math
from dotenv import load_dotenv
import yaml

# Load keys
load_dotenv()

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

scraper_cfg = config['scraper']
weights = config['validation_weights']
paths_cfg = config['paths']

# Reddit API Authentication
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Sentiment Analyzer
def get_sentiment(text):
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

# Format UTC timestamps
def format_timestamp(utc):
    if utc is None:
        return ""
    dt = datetime.fromtimestamp(utc, timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Recency weight
def recency_weight(created_utc):
    if not created_utc:
        return 0
    days_old = (datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)).days
    return math.exp(-days_old / 30)

def comment_age_days(created_utc):
    if not created_utc:
        return None
    return (datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, timezone.utc)).days

# Fetch posts
def fetch_subreddit_posts(subreddit_name, max_posts=2000):
    subreddit = reddit.subreddit(subreddit_name)
    data_records = []

    print(f"Fetching up to {max_posts} posts from r/{subreddit_name}...")

    for submission in subreddit.new(limit=None):
        title = submission.title or ""
        post_body = submission.selftext or ""
        full_text = f"{title} {post_body}".strip()
        upvotes = submission.score
        num_comments = submission.num_comments
        upvote_ratio = getattr(submission, "upvote_ratio", 0.5)

        # Post sentiment and recency
        post_sentiment = get_sentiment(full_text)
        post_recency = recency_weight(submission.created_utc)

        # Comment sentiment weighted by recency
        comment_sentiments = []
        comment_ages = []

        if num_comments > 0:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:scraper_cfg['max_comments_per_post']]:
                comment_sent = get_sentiment(comment.body or "")
                age_days = comment_age_days(comment.created_utc)
                comment_rec = recency_weight(comment.created_utc)
                comment_sentiments.append((comment_sent, comment_rec))
                if age_days is not None:
                    comment_ages.append(age_days)

        # Weighted average comment sentiment
        if comment_sentiments:
            numerator = sum(s * w for s, w in comment_sentiments)
            denominator = sum(w for _, w in comment_sentiments)
            avg_comment_sentiment = numerator / denominator if denominator != 0 else 0
        else:
            avg_comment_sentiment = 0

        # Average comment age
        avg_comment_age = sum(comment_ages) / len(comment_ages) if comment_ages else 0

        # Validation score (0–100%)
        validation_score = (
            post_sentiment * weights['post_sentiment'] +
            avg_comment_sentiment * weights['avg_comment_sentiment'] +
            upvote_ratio * weights['upvote_ratio'] +
            post_recency * weights['post_recency']
        ) * 100
        validation_score = max(0, min(100, validation_score))

        # Final label
        if validation_score >= 70:
            label = "good"
        elif validation_score >= 40:
            label = "neutral"
        else:
            label = "bad"

        # Add post row
        data_records.append({
            "post_id": submission.id,
            "title": title,
            "text": full_text,
            "author": submission.author.name if submission.author else '[deleted]',
            "created_utc": format_timestamp(submission.created_utc),
            "upvotes": upvotes,
            "num_comments": num_comments,
            "upvote_ratio": upvote_ratio,
            "post_sentiment": post_sentiment,
            "avg_comment_sentiment": avg_comment_sentiment,
            "post_age_days": (datetime.now(timezone.utc) - datetime.fromtimestamp(submission.created_utc, timezone.utc)).days,
            "avg_comment_age_days": round(avg_comment_age, 2),
            "validation_score": round(validation_score, 2),
            "label": label,
            "source_url": f"https://reddit.com{submission.permalink}"
        })

        if len(data_records) >= max_posts:
            break

    return data_records

def main():
    data_records = fetch_subreddit_posts(scraper_cfg['subreddit'], scraper_cfg['max_posts'])

    if data_records:
        df = pd.DataFrame(data_records)
        os.makedirs(paths_cfg['raw_data_dir'], exist_ok=True)
        csv_path = os.path.join(paths_cfg['raw_data_dir'], paths_cfg['raw_data_file'])
        df.to_csv(csv_path, index=False)

        print(f"\n Dataset saved as {csv_path}")
        print(f"Total records: {len(df)}")
        print("Columns:", list(df.columns))
        print(df.head(3))
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
