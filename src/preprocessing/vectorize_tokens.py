import pandas as pd
from sentence_transformers import SentenceTransformer

def vectorize_tokens(input_csv_path, output_csv_path, token_column='token_text', target_column='label_numeric'):
    # Load original CSV
    df = pd.read_csv(input_csv_path)

    # Convert stringified token lists to actual Python lists
    df['tokens'] = df[token_column].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # Join tokens back into sentences for embedding
    texts = df['tokens'].apply(lambda tokens: ' '.join(tokens)).tolist()

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings with progress bar
    embeddings = model.encode(texts, show_progress_bar=True)

    # Create DataFrame from embeddings
    emb_df = pd.DataFrame(embeddings, columns=[f'feature_{i}' for i in range(embeddings.shape[1])])

    # Attach numeric features from original CSV if you want
    # For example, adding post_sentiment, avg_comment_sentiment, upvotes, upvote_ratio
    numeric_features = ['post_sentiment', 'avg_comment_sentiment', 'upvotes', 'upvote_ratio']
    for feat in numeric_features:
        if feat in df.columns:
            emb_df[feat] = df[feat]

    # Add the target column
    emb_df[target_column] = df[target_column]

    # Save the new vectorized CSV
    emb_df.to_csv(output_csv_path, index=False)
    print(f"Vectorized CSV saved to: {output_csv_path}")

# Usage
input_csv = 'E:/saas-idea-validator/data/processed/vectorized_dataset.csv'
output_csv = 'E:/saas-idea-validator/data/processed/vectorized_features.csv'
vectorize_tokens(input_csv, output_csv)
