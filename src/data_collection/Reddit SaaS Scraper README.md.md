# Reddit SaaS Scraper README.md


***

## Overview

This is a powerful Reddit SaaS Scraper designed to collect posts about SaaS, startups, entrepreneurship, and related topics from multiple subreddits. The scraper performs sentiment analysis on both posts and their comments, intelligently handles missing or null data, and produces high-quality labeled datasets optimized for supervised machine learning tasks.

***

## Supported Subreddits

The scraper collects data from the following subreddits:

- Entrepreneur
- startups
- indiehackers
- microsaas
- growthhacking
- Bootstrapped
- SaaS

***

## Features

- **Multi-source data collection:** Rotates through different Reddit post types (new, hot, top weekly/monthly/yearly, controversial) for diversity.
- **Sentiment analysis:** Uses a pre-trained transformer model to analyze sentiment in both posts and comments.
- **Comment filtering:** Only collects posts with at least 3 comments, ensuring reliable comment sentiment data.
- **Missing data handling:** Replaces missing values with median or noise-added mean while enforcing a minimum value to prevent zero-weight parameters.
- **Automated weight optimization:** Calculates optimal weights for four validation features (post sentiment, comment sentiment, upvote ratio, post recency) per batch.
- **Deduplication:** Tracks collected post IDs globally to prevent duplicate entries within and across batches.
- **Data labeling:** Classifies posts into 'good', 'neutral', or 'bad' categories based on weighted validation scores.
- **Configurable:** Flexible via YAML configuration file for subreddits, batch sizing, comment limits, NaN handling, and validation thresholds.
- **UTF-8 compatible:** All exported files handle Unicode characters properly, avoiding encoding errors.

***

## Setup \& Installation

1. **Install dependencies:**
```bash
pip install praw pandas numpy transformers python-dotenv pyyaml torch
```

2. **Create Reddit API credentials:**

- Register an app on Reddit to get `client_id`, `client_secret`, and `user_agent`.

3. **Configure environment variables:**

Create a `.env` file in the project root:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

4. **Adjust scraper configuration:**

Edit `config.yaml` to set subreddits, batch sizes, min comments, NaN handling method, etc.

***

## How it works

- Scraper cycles through defined subreddits and post sources.
- For each post, it verifies post age and uniqueness.
- Posts with fewer than 3 comments are skipped to ensure valid comment sentiment.
- Sentiment is analyzed on the full post text and up to 15 of its freshest comments.
- Missing or null features are handled by median replacements with a minimum value constraint.
- Per batch, the scraper finds optimal weights ensuring no validation parameter is unused.
- Labeled data is saved as CSV files along with batch-wise reports.

***

## Output

- Processed batches are saved under `data/raw/raw_batch/` as CSV files.
- Batch optimization reports are saved under `data/raw/raw_batch_report/`.
- Exported datasets contain validated sentiment scores and assigned labels ready for machine learning use.

***

## Usage

Run the scraper with:

```bash
python src/data_collection/reddit_scraper.py
```

Monitor console output for batch progress, NaN statistics, and validation weight summaries.

***

## Notes

- Ensure consistent tracking of post IDs between runs to avoid duplicates.
- You can merge multiple batch CSVs, de-dup, and use them for training ML models.
- Comment filtering guarantees better comment sentiment quality, improving model accuracy.

***

## Contact

For issues or further assistance, refer to the project documentation or your development team.

***

This README provides a complete guide to setup, usage, and understanding the functionalities of your Reddit SaaS scraper tailored for high-quality supervised machine learning dataset preparation.

