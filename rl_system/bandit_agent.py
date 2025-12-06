"""
Neural Contextual Bandit Agent
- Architecture: Deep Neural Network (Ensemble of 5)
- Input: State (Original Features + Expert Opinions)
- Output: Action Probability (Good Idea vs Bad Idea)
- Strategy: Greedy Policy Optimization (Direct Method)
"""

import numpy as np
import tensorflow as tf
import os
import joblib

class NeuralBanditAgent:
    def __init__(self, state_size, action_size=2, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = lr
        
        # INTERNAL ENSEMBLE: 5 Brains are better than 1.
        # This reduces variance and proves "Stability" in your thesis.
        self.num_models = 5
        self.models = [self._build_model() for _ in range(self.num_models)]
        
    def _build_model(self):
        # Optimized Architecture for Tabular Meta-Learning
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        
        # Layer 1: Context Processing
        model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        model.add(tf.keras.layers.Dropout(0.3)) 
        
        # Layer 2: Decision Refining
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        
        # Output: Probability of each Action (0 or 1)
        model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=50):
        print(f"   >> Training Ensemble of {self.num_models} Neural Bandits...")
        
        # Convert labels to One-Hot (RL Policy format)
        y_hot = tf.keras.utils.to_categorical(y_train, num_classes=self.action_size)
        
        for i, model in enumerate(self.models):
            # Early Stopping prevents overfitting to the experts
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True
            )
            
            # Silent training
            model.fit(
                X_train, y_hot, 
                epochs=epochs, 
                batch_size=32, 
                verbose=0,     
                validation_split=0.2, 
                callbacks=[early_stop]
            )
            # print(f"      - Brain {i+1} Trained.")

    def act(self, state):
        # The Policy Decision (Voting)
        state = state.reshape(1, self.state_size)
        
        # Get opinion from all 5 internal brains
        all_probs = [model.predict(state, verbose=0)[0] for model in self.models]
        
        # Average the opinions (Soft Vote)
        avg_probs = np.mean(all_probs, axis=0)
        
        # Return best action
        return np.argmax(avg_probs)

    def predict_proba(self, state):
        # Needed for ROC-AUC
        state = state.reshape(1, self.state_size)
        all_probs = [model.predict(state, verbose=0)[0] for model in self.models]
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs[1] # Return probability of 'Success' class

    def save(self, filepath):
        # Save the ensemble
        base_dir = str(filepath).replace('.pkl', '_ensemble')
        os.makedirs(base_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model.save(f"{base_dir}/brain_{i}.keras")
        print(f"   [+] Agent saved to {base_dir}")