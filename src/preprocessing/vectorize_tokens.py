import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

def safe_token_join(token_string):
    """Safely converts string representation of tokens into a single clean string."""
    if not isinstance(token_string, str) or not token_string:
        return ""
    try:
        # Assuming the tokens are represented as a list-like string: '["token1", "token2"]'
        tokens = eval(token_string)
        if isinstance(tokens, list):
            return ' '.join(tokens)
        else:
            return "" 
    except:
        return ""

def vectorize_tokens_final_fix(input_csv_path, output_csv_path, token_column='token_text', target_column='label_numeric'):
    
    df = pd.read_csv(input_csv_path)
    initial_rows = len(df)
    print(f"Starting vectorization with {initial_rows} rows...")

    # 1. PREPARE TEXT AND GENERATE EMBEDDINGS
    df['clean_text'] = df[token_column].apply(safe_token_join)
    texts = df['clean_text'].tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    # 2. CREATE EMBEDDING DATAFRAME (THE CRITICAL STEP)
    # We must ensure the index of the embeddings matches the original DF index
    # We use .values for the original features to prevent index misalignment issues if the data was reordered
    
    emb_df = pd.DataFrame(embeddings, columns=[f'feature_{i}' for i in range(embeddings.shape[1])])
    
    # 3. CONCATENATE NEW FEATURES WITH ORIGINAL DATA
    
    # Select the original numeric features needed for the final model
    numeric_features = ['post_sentiment', 'avg_comment_sentiment', 'upvotes', 'upvote_ratio']
    target_and_features = df[[col for col in [target_column] + numeric_features if col in df.columns]]
    
    # Resetting index of both ensures they align perfectly for the concatenation
    final_df = pd.concat([emb_df.reset_index(drop=True), target_and_features.reset_index(drop=True)], axis=1)

    # 4. FINAL SAVE
    final_rows = len(final_df)
    
    final_df.to_csv(output_csv_path, index=False)
    
    print(f"✅ FINAL FIX Complete.")
    print(f"   Initial Rows: {initial_rows}, Final Rows: {final_rows}")
    print(f"   Data Integrity Check: {initial_rows == final_rows}")
    print(f"   Vectorized CSV saved to: {output_csv_path}")


# Usage (Ensure these paths are correct before running)
input_csv = 'E:/saas-idea-validator/data/processed/vectorized_dataset.csv'
output_csv = 'E:/saas-idea-validator/data/processed/vectorized_features_full_fixed.csv'

vectorize_tokens_final_fix(input_csv, output_csv)