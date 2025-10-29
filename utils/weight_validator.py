"""
Automated Weight Validation System for Reddit SaaS Post Validation

UPDATED FOR COMMENT SENTIMENT ANALYSIS

Weight Components:
    1. post_sentiment: Sentiment score of the post
    2. avg_comment_sentiment: Average sentiment of newest 10 comments (PUBLIC FEEDBACK)
    3. upvote_ratio: Reddit upvote ratio
    4. post_recency: Recency of the post itself
"""

import pandas as pd
import numpy as np


class AutomatedWeightValidator:
    """
    Automated validation system that tests different weight combinations
    to find the optimal weights for each batch of scraped data.
    
    Weight components: [post_sentiment, avg_comment_sentiment, upvote_ratio, post_recency]
    """
    
    def __init__(self, step_size=0.1, good_threshold=70, neutral_threshold=40):
        """
        Initialize the validator.
        
        Args:
            step_size: Granularity for weight generation (e.g., 0.1 means 0.0, 0.1, 0.2, ...)
            good_threshold: Score threshold for 'good' label
            neutral_threshold: Score threshold for 'neutral' label (below this is 'bad')
        """
        self.step_size = step_size
        self.good_threshold = good_threshold
        self.neutral_threshold = neutral_threshold
        self.weight_combinations = self._generate_weight_combinations()
        
    def _generate_weight_combinations(self):
        """Generate all possible weight combinations that sum to 1.0"""
        weights = []
        possible_values = [round(x * self.step_size, 2) for x in range(0, int(1/self.step_size) + 1)]
        
        for w1 in possible_values:
            for w2 in possible_values:
                for w3 in possible_values:
                    w4 = round(1.0 - w1 - w2 - w3, 2)
                    if 0 <= w4 <= 1.0 and abs(w1 + w2 + w3 + w4 - 1.0) < 0.01:
                        weights.append([w1, w2, w3, w4])
        
        return weights
    
    def calculate_validation_score(self, record, weights):
        """
        Calculate validation score for a single record using given weights.
        
        Args:
            record: Dictionary containing:
                   - post_sentiment
                   - avg_comment_sentiment (NEW! replaces comment recency)
                   - upvote_ratio
                   - post_recency
            weights: List of 4 weights [w_post_sent, w_comment_sent, w_upvote, w_post_rec]
        
        Returns:
            Validation score (0-100)
        """
        post_sentiment = record.get('post_sentiment', 0.5)
        avg_comment_sentiment = record.get('avg_comment_sentiment', 0.5)  # NEW!
        upvote_ratio = record.get('upvote_ratio', 0.5)
        post_rec = record.get('post_recency', 0)
        
        score = (
            weights[0] * post_sentiment +
            weights[1] * avg_comment_sentiment +  # NEW: Comment sentiment instead of recency
            weights[2] * upvote_ratio +
            weights[3] * post_rec
        ) * 100
        
        return max(0, min(100, score))
    
    def assign_label(self, score):
        """Assign label based on validation score"""
        if score >= self.good_threshold:
            return "good"
        elif score >= self.neutral_threshold:
            return "neutral"
        else:
            return "bad"
    
    def calculate_batch_accuracy(self, batch_data, weights, ground_truth_labels=None):
        """
        Calculate accuracy for a batch using given weights.
        
        Args:
            batch_data: List of dictionaries (batch records)
            weights: Weight combination to test
            ground_truth_labels: Optional ground truth labels for supervised accuracy
        
        Returns:
            Accuracy score (higher is better)
        """
        scores = []
        predicted_labels = []
        
        for record in batch_data:
            score = self.calculate_validation_score(record, weights)
            label = self.assign_label(score)
            scores.append(score)
            predicted_labels.append(label)
        
        # If ground truth is available, calculate actual accuracy
        if ground_truth_labels is not None:
            correct = sum(1 for pred, true in zip(predicted_labels, ground_truth_labels) if pred == true)
            accuracy = (correct / len(ground_truth_labels)) * 100
            return accuracy
        
        # Otherwise, use label distribution consistency as a heuristic
        label_counts = pd.Series(predicted_labels).value_counts()
        total = len(predicted_labels)
        
        # Calculate entropy-based score (higher entropy = more balanced)
        entropy = 0
        for count in label_counts.values:
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)
        
        # Normalize entropy (max entropy for 3 labels is log(3))
        max_entropy = np.log(3)
        normalized_entropy = (entropy / max_entropy) * 100
        
        # Also consider average score variance (reward discriminative scoring)
        score_variance = np.var(scores)
        variance_score = min(score_variance / 100, 1.0) * 100
        
        # Combined accuracy metric (70% entropy, 30% variance)
        accuracy = 0.7 * normalized_entropy + 0.3 * variance_score
        
        return accuracy
    
    def find_best_weights(self, batch_data, ground_truth_labels=None):
        """
        Test all weight combinations and find the one with best accuracy.
        
        Args:
            batch_data: List of dictionaries (batch records)
            ground_truth_labels: Optional ground truth labels
        
        Returns:
            Tuple of (best_weights, best_accuracy, all_results)
        """
        results = []
        
        print(f"Testing {len(self.weight_combinations)} weight combinations...")
        for i, weights in enumerate(self.weight_combinations):
            accuracy = self.calculate_batch_accuracy(batch_data, weights, ground_truth_labels)
            results.append({
                'weights': weights,
                'accuracy': accuracy
            })
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Tested {i + 1}/{len(self.weight_combinations)} combinations...", end='\r')
        
        print(f"  Tested {len(self.weight_combinations)}/{len(self.weight_combinations)} combinations.   ")
        
        # Sort by accuracy
        results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        best_result = results_sorted[0]
        best_weights = best_result['weights']
        best_accuracy = best_result['accuracy']
        
        return best_weights, best_accuracy, results_sorted
    
    def validate_and_label_batch(self, batch_data, best_weights):
        """
        Apply best weights to batch data and generate final labeled records.
        
        Args:
            batch_data: List of dictionaries (batch records)
            best_weights: Optimal weights found
        
        Returns:
            List of records with validation_score and label
        """
        labeled_records = []
        
        for record in batch_data:
            score = self.calculate_validation_score(record, best_weights)
            label = self.assign_label(score)
            
            record['validation_score'] = round(score, 2)
            record['label'] = label
            labeled_records.append(record)
        
        return labeled_records


def format_weights_string(weights):
    """Format weights as a comma-separated string"""
    return f"{weights[0]},{weights[1]},{weights[2]},{weights[3]}"


def parse_weights_string(weights_str):
    """Parse weights from comma-separated string"""
    return [float(w) for w in weights_str.split(',')]
