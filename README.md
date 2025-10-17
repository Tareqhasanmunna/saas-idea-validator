# SaaS Idea Validation System

## Overview

This project automatically collects SaaS-related posts from Reddit and assigns a validation score and label (`good`, `neutral`, `bad`) for each post. The system considers post sentiment, comment sentiment, upvotes, and recency of posts and comments to generate the validation score. The dataset generated can be used for supervised learning and later integrated with a reinforcement learning system.

## Merging raw scraped data

If you collected raw CSV batches with the scraper they are stored under `data/raw/raw_batch` by default. A small utility script is provided to merge those batch files into a single deduplicated CSV.

Example (PowerShell):

```powershell
# Merge using defaults from config.yaml
python .\src\data_collection\raw_data_marge.py

# Or specify input and output folders explicitly
python .\src\data_collection\raw_data_marge.py --input "E:\\saas-idea-validator\\data\\raw\\raw_batch" --output "E:\\saas-idea-validator\\data\\raw\\raw_marged"
```

The script will deduplicate rows by the `post_id` column if present and write an atomic merged CSV file into the specified output folder.