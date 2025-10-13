import praw
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timezone
import os

# -----------------------------
# Reddit API Authentication
# -----------------------------
reddit = praw.Reddit(
    client_id='p1Y3_wGlHjCWq47Z1hjJWg',
    client_secret='OQRu5IBSa0f9wEHtCIYeOjahgGLTtQ',
    user_agent='SaaSIdeaValidationBot/0.3 by Natural-Camera-4123'
)

# -----------------------------
# Sentiment Analyzer
# -----------------------------
def get_sentiment(text):
    if not text:
        return 0.0, "neutral"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return polarity, "positive"
    elif polarity < -0.3:
        return polarity, "negative"
    else:
        return polarity, "neutral"

# -----------------------------
# Format UTC timestamps
# -----------------------------
def format_timestamp(utc):
    if utc is None:
        return ""
    dt = datetime.fromtimestamp(utc, timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# -----------------------------
# Fetch posts + comments
# -----------------------------
def fetch_subreddit_posts(subreddit_name, max_posts=2000):
    subreddit = reddit.subreddit(subreddit_name)
    data_records = []

    print(f"Fetching up to {max_posts} posts from r/{subreddit_name}...")

    for submission in subreddit.new(limit=None):  # Generator fetches as much as possible
        title = submission.title or ""
        post_body = submission.selftext or ""
        full_text = f"{title} {post_body}".strip()
        upvotes = submission.score
        num_comments = submission.num_comments
        upvote_ratio = getattr(submission, "upvote_ratio", 0.5)

        # Sentiment for post
        post_sentiment, post_emotion = get_sentiment(full_text)

        # Collect comment sentiments
        comment_sentiments = []
        comments_data = []

        if num_comments > 0:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:20]:
                comment_text = comment.body or ""
                comment_sent, comment_emotion = get_sentiment(comment_text)
                comment_sentiments.append(comment_sent)
                comment_author = comment.author.name if comment.author else '[deleted]'

                comments_data.append({
                    "post_id": submission.id,
                    "comment_id": comment.id,
                    "comment_body": comment_text,
                    "author": comment_author,
                    "created_utc": format_timestamp(comment.created_utc),
                    "upvotes": comment.score,
                    "sentiment_score": comment_sent,
                    "emotion_label": comment_emotion,
                    "source_url": f"https://reddit.com{comment.permalink}" if hasattr(comment, 'permalink') else ""
                })

        avg_comment_sentiment = sum(comment_sentiments)/len(comment_sentiments) if comment_sentiments else 0

        # Auto-label post
        score_signal = (upvote_ratio * 0.6) + ((post_sentiment + avg_comment_sentiment)/2 * 0.4)
        if score_signal > 0.4:
            label = "good"
        elif score_signal < -0.2:
            label = "bad"
        else:
            label = "neutral"

        # Add main post record
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
            "label": label,
            "source_url": f"https://reddit.com{submission.permalink}"
        })

        # Add comments records
        data_records.extend(comments_data)

        # Stop after max_posts
        if len([r for r in data_records if r.get("post_id") == submission.id]) >= max_posts:
            break

    return data_records

# -----------------------------
# Main function
# -----------------------------
def main():
    subreddit = "SaaS"
    data_records = fetch_subreddit_posts(subreddit, max_posts=2000)

    if data_records:
        df = pd.DataFrame(data_records)

        # Ensure folder exists
        os.makedirs('data/raw', exist_ok=True)

        # Save to CSV
        csv_path = 'data/raw/supervised_dataset.csv'
        df.to_csv(csv_path, index=False)

        print(f"Dataset saved as {csv_path} with {len(df)} records.")
        print("Columns:", list(df.columns))
        print(df.head(3))
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
