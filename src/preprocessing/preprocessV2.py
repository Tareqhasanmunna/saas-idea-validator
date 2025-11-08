import os
import pandas as pd
import numpy as np
import re
import string
import emoji
import nltk
import logging
from datetime import datetime
import joblib
import yaml

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
STOP_WORDS = set(stopwords.words('english'))

class PreprocessingPipeline:
    def __init__(self, vectorizer_type='tfidf', max_features=200, logger_obj=None):
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.logger = logger_obj or logger
        self.vectorizer = None
    
    @staticmethod
    def remove_urls(text):
        if not isinstance(text, str):
            return text
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    @staticmethod
    def remove_emojis(text):
        if not isinstance(text, str):
            return text
        return emoji.replace_emoji(text, replace="")
    
    @staticmethod
    def remove_punctuation(text):
        if not isinstance(text, str):
            return text
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def remove_numbers(text):
        if not isinstance(text, str):
            return text
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def remove_special_characters(text):
        if not isinstance(text, str):
            return text
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    @staticmethod
    def lowercase(text):
        if not isinstance(text, str):
            return text
        return text.lower()
    
    @staticmethod
    def tokenize(text):
        if not isinstance(text, str):
            return []
        return word_tokenize(text)
    
    @staticmethod
    def remove_stopwords(tokens):
        return [token for token in tokens if token.lower() not in STOP_WORDS]
    
    def clean_text(self, text, remove_stopwords=True):
        if not isinstance(text, str):
            return []
        
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_special_characters(text)
        text = self.lowercase(text)
        
        tokens = self.tokenize(text)
        
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def fit_tfidf_vectorizer(self, texts):
        self.logger.info(f"[PREPROCESSING] Fitting TF-IDF ({self.max_features} features)...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        self.vectorizer.fit(texts)
        self.logger.info(f"[PREPROCESSING] TF-IDF fitted")
        return self.vectorizer
    
    def vectorize_tfidf(self, texts):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
        
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()
    
    @staticmethod
    def encode_labels(labels):
        unique_labels = sorted(set(labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = np.array([label_map[label] for label in labels])
        return encoded, label_map
    
    def process_dataset(self, input_csv, output_csv=None, save_vectorizer=None):
        self.logger.info(f"[PREPROCESSING] Processing {input_csv}")
        errors = []
        
        try:
            self.logger.info(f"[PREPROCESSING] Loading data...")
            df = pd.read_csv(input_csv)
            self.logger.info(f"[PREPROCESSING] Loaded {len(df)} rows")
            
            if 'text' not in df.columns or 'label' not in df.columns:
                error_msg = "CSV must have 'text' and 'label' columns"
                self.logger.error(f"[PREPROCESSING] {error_msg}")
                return {'success': False, 'output_file': None, 'row_count': 0, 'vector_dim': 0, 'errors': [error_msg]}
            
            texts = df['text'].fillna('').astype(str)
            labels = df['label']
            
            self.logger.info(f"[PREPROCESSING] Cleaning text...")
            cleaned_tokens = [self.clean_text(text) for text in texts]
            cleaned_texts = [' '.join(tokens) for tokens in cleaned_tokens]
            
            self.logger.info(f"[PREPROCESSING] Vectorizing...")
            self.fit_tfidf_vectorizer(cleaned_texts)
            vectors = self.vectorize_tfidf(cleaned_texts)
            
            self.logger.info(f"[PREPROCESSING] Vectors shape: {vectors.shape}")
            
            self.logger.info(f"[PREPROCESSING] Encoding labels...")
            encoded_labels, label_map = self.encode_labels(labels)
            self.logger.info(f"[PREPROCESSING] Labels: {label_map}")
            
            feature_cols = ['post_sentiment', 'avg_comment_sentiment', 'upvote_ratio', 'post_recency']
            X_numeric = df[feature_cols].fillna(0).values
            
            X_combined = np.hstack([X_numeric, vectors])
            
            output_df = df[['post_id', 'title', 'text']].copy() if 'post_id' in df.columns else pd.DataFrame()
            output_df['post_sentiment'] = df['post_sentiment']
            output_df['avg_comment_sentiment'] = df['avg_comment_sentiment']
            output_df['upvote_ratio'] = df['upvote_ratio']
            output_df['post_recency'] = df.get('post_recency', 0.5)
            output_df['label_numeric'] = encoded_labels
            
            vector_strs = [' '.join(map(str, vec)) for vec in vectors]
            output_df['vector'] = vector_strs
            
            if output_csv is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = 'data/processed'
                os.makedirs(output_dir, exist_ok=True)
                output_csv = os.path.join(output_dir, f"vectorised_dataset_{timestamp}.csv")
            
            output_df.to_csv(output_csv, index=False)
            self.logger.info(f"[PREPROCESSING] ✓ Saved: {output_csv}")
            
            if save_vectorizer:
                os.makedirs(os.path.dirname(save_vectorizer), exist_ok=True)
                joblib.dump(self.vectorizer, save_vectorizer)
                self.logger.info(f"[PREPROCESSING] Vectorizer saved")
            
            return {
                'success': True,
                'output_file': output_csv,
                'row_count': len(output_df),
                'vector_dim': vectors.shape[1],
                'label_map': label_map,
                'errors': errors
            }
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.logger.error(f"[PREPROCESSING] {error_msg}")
            return {
                'success': False,
                'output_file': None,
                'row_count': 0,
                'vector_dim': 0,
                'errors': [error_msg]
            }