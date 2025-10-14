# SaaS Idea Validation System

## Overview

This project automatically collects SaaS-related posts from Reddit and assigns a validation score and label (`good`, `neutral`, `bad`) for each post. The system considers post sentiment, comment sentiment, upvotes, and recency of posts and comments to generate the validation score. The dataset generated can be used for supervised learning and later integrated with a reinforcement learning system.

## Features

- Fetch posts and comments from a specific subreddit.
- Calculate post sentiment and weighted average comment sentiment.
- Consider post age and comment recency in validation.
- Generate a validation score (0–100%) and label (`good`, `neutral`, `bad`).
- Save the dataset in a structured CSV format for ML tasks.
- Configurable through `config.yaml` and `.env` for sensitive keys.

## Requirements

- Python 3.9+
- Packages:
  - praw
  - textblob
  - pandas
  - python-dotenv
  - pyyaml

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/saas-idea-validator.git
cd saas-idea-validator
```

2. **Create a `.env` file** with your Reddit API credentials:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

3. **Configure project settings** in `config.yaml`:

```yaml
scraper:
  subreddit: "SaaS"
  max_posts: 2000
  max_comments_per_post: 20

validation_weights:
  post_sentiment: 0.3
  avg_comment_sentiment: 0.3
  upvote_ratio: 0.2
  post_recency: 0.2

paths:
  raw_data_dir: "data/raw"
  raw_data_file: "supervised_dataset.csv"
```

4. **Run the scraper**:

```bash
python scraper.py
```

The dataset will be saved in the folder specified in `config.yaml`.

## Dataset Structure

| Column                  | Description |
|-------------------------|-------------|
| post_id                 | Reddit post ID |
| title                   | Post title |
| text                    | Full post text |
| author                  | Post author |
| created_utc             | Post timestamp (UTC) |
| upvotes                 | Number of upvotes |
| num_comments            | Number of comments |
| upvote_ratio            | Upvote ratio (if available) |
| post_sentiment          | Sentiment polarity of the post (-1 to 1) |
| avg_comment_sentiment   | Weighted average sentiment of comments |
| post_age_days           | Age of post in days |
| avg_comment_age_days    | Average age of comments in days |
| validation_score        | Computed validation score (0–100%) |
| label                   | Final label (`good`, `neutral`, `bad`) |
| source_url              | URL of the post |

## How it Works

1. The scraper fetches posts from the configured subreddit.
2. Each post is analyzed for sentiment using TextBlob.
3. Up to `max_comments_per_post` comments are considered, weighted by recency.
4. Post age is considered to give higher scores to recent posts.
5. A final validation score is calculated using configurable weights and mapped to a label.
6. The dataset is saved in a CSV file for downstream ML tasks.

## Future Work

- Integrate the dataset with a reinforcement learning system to continuously improve validation predictions.
- Develop a UI where users can input their own SaaS ideas to get a real-time validation score.
- Improve sentiment analysis using advanced models like Transformers for more accurate results.