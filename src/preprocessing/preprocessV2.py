"""Converted from preprocessV2.ipynb to preprocessV2.py"""

# NOTE: Markdown cells are converted to comment blocks. Code cells are preserved.

# ---- Code cell 1 ----
import os
import pandas as pd
import numpy as np
import re
import string
import emoji
import nltk

# Download necessary NLTK resources (first time only)
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Prepare stopwords
stop_words = set(stopwords.words('english'))

print("Libraries imported successfully.")



# ---- Code cell 2 ----
# Replace with your dataset path
df = pd.read_csv(r'E:saas-idea-validator\data\raw\raw_marged\merged_output.csv')

# Show first 10 rows
df.head(10)


# ---- Code cell 3 ----
columns_to_keep = [
    'title', 'text', 'post_sentiment', 'avg_comment_sentiment',
    'num_comments', 'upvotes', 'upvote_ratio', 'label'
]

# Keep only existing ones (some may be missing)
df = df[[c for c in columns_to_keep if c in df.columns]].copy()

print("Columns kept:", df.columns.tolist())



df.head(10)


# ---- Code cell 4 ----
print("Missing values per column:\n", df.isnull().sum())

# Drop rows with no label (since supervised learning needs labels)
if 'label' in df.columns:
    df = df.dropna(subset=['label'])

# Fill numeric NaNs with 0
num_cols = ['post_sentiment', 'avg_comment_sentiment', 'num_comments', 'upvotes', 'upvote_ratio']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fill text NaNs with empty strings
for col in ['title', 'text']:
    if col in df.columns:
        df[col] = df[col].fillna('')

print("After handling missing values, shape:", df.shape)
df.head(5)


# ---- Code cell 5 ----
df['text_no_url'] = df['text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', str(x)))

df[['text', 'text_no_url']].head(10)



# ---- Code cell 6 ----
df['text_no_emoji'] = df['text_no_url'].apply(lambda x: emoji.replace_emoji(x, replace=''))

df[['text_no_url', 'text_no_emoji']].head(10)


# ---- Code cell 7 ----
df['text_no_punct'] = df['text_no_emoji'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

df[['text_no_emoji', 'text_no_punct']].head(10)


# ---- Code cell 8 ----
df['text_no_num'] = df['text_no_punct'].apply(lambda x: re.sub(r'\d+', '', x))

df[['text_no_punct', 'text_no_num']].head(10)


# ---- Code cell 9 ----
df['text_no_special'] = df['text_no_num'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))

df[['text_no_num', 'text_no_special']].head(10)


# ---- Code cell 10 ----
df['text_cleaned'] = df['text_no_special'].apply(lambda x: re.sub(r'\s+', ' ', x.lower()).strip())

df[['text_no_special', 'text_cleaned']].head(10)


# ---- Code cell 11 ----
df['text_final'] = df['text_cleaned'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

df[['text_cleaned', 'text_final']].head(10)


# ---- Code cell 12 ----
df.head(10)


# ---- Code cell 13 ----
def remove_stopwords(text_final):
    return ' '.join(word for word in text_final.split() if word not in stop_words)
df.head(10)


# ---- Code cell 14 ----
df.drop(['title', 'text', 'text_no_url','text_no_emoji','text_no_punct','text_no_num','text_no_special','text_cleaned','num_comments'], axis=1, inplace=True)
df.head(10)


# ---- Code cell 15 ----
# Save the dataframe to a new CSV file
df.to_csv('E:\\saas-idea-validator\\data\\processed\\cleaned_dataset.csv', index=False)

print("Cleaned dataset saved as 'E:\\saas-idea-validator\\data\\processed\\cleaned_dataset.csv'")


# ---- Code cell 16 ----
import pandas as pd
import nltk
from nltk import word_tokenize
nltk.download('punkt')
df = pd.read_csv('E:\saas-idea-validator\data\processed\cleaned_dataset.csv')
df['token_text'] = df['text_final'].apply(lambda x: word_tokenize(str(x)))
print("Tokenized dataset saved as 'tokenized_dataset.csv'")
print(df.head())


# ---- Code cell 17 ----
df.drop(columns=['text_final'], inplace=True)
df.head(10)


# ---- Code cell 18 ----
# Save the dataframe to a new CSV file
df.to_csv('E:\\saas-idea-validator\\data\\processed\\tokenized_dataset.csv', index=False)

print("Tokenized dataset saved as 'E:\\saas-idea-validator\\data\\processed\\tokenized_dataset.csv'")


# ---- Code cell 19 ----
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
df = pd.read_csv(r'E:\saas-idea-validator\data\processed\tokenized_dataset.csv')
w2v_model = Word2Vec(sentences=df['token_text'], vector_size=100, window=5, min_count=1, workers=4)

# --- Step 5: Sentence vector (average of word vectors) ---
def get_sentence_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

df['vector'] = df['token_text'].apply(get_sentence_vector)
df.head(10)


# ---- Code cell 20 ----
# Save the dataframe to a new CSV file
df.to_csv('E:\\saas-idea-validator\\data\\processed\\vectorised_dataset.csv', index=False)

print("Vectorised dataset saved as 'E:\\saas-idea-validator\\data\\processed\\vectorised_dataset.csv'")



if __name__ == "__main__":
    print("This file is a straight conversion of preprocessV2.ipynb.")
    print("Inspect and adapt the functions/variables for your reinforcement learning workflow.")