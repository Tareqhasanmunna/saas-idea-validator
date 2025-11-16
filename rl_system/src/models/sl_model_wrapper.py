# rl_system/src/models/sl_model_wrapper.py
import joblib
import numpy as np
import lightgbm as lgb

class SLModelWrapper:
    """
    Robust wrapper that supports:
      - sklearn LGBMClassifier (has predict_proba)
      - lightgbm.Booster (predict returns probabilities)
    Returns: (prediction:int, confidence:float)
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # record type for branching
        self.is_sklearn_wrapper = hasattr(self.model, "predict_proba")
        self.is_booster = isinstance(self.model, lgb.basic.Booster)
        # If user saved sklearn wrapper but joblib gives Booster inside .booster_, handle both
        if not (self.is_sklearn_wrapper or self.is_booster):
            # Some LightGBM sklearn models have .booster_ attribute (sklearn wrapper)
            if hasattr(self.model, "booster_") and isinstance(self.model.booster_, lgb.basic.Booster):
                self.is_sklearn_wrapper = True
            else:
                # fallback: try to treat as Booster-like if it has predict
                self.is_booster = hasattr(self.model, "predict") and not self.is_sklearn_wrapper

        print(f"✓ SL Model loaded: {model_path} | sklearn_wrapper={self.is_sklearn_wrapper} | booster={self.is_booster}")

    def predict(self, features):
        """
        Input: features: sequence length = num_features
        Output: (pred_label:int, confidence:float)
        """
        X = np.array(features).reshape(1, -1)

        # Case 1: sklearn wrapper (LGBMClassifier)
        if self.is_sklearn_wrapper:
            # Some sklearn wrappers internally hold booster_.predict, but sklearn .predict_proba is safe to call
            try:
                proba = self.model.predict_proba(X)
            except Exception:
                # fallback to booster_.predict if sklearn wrapper fails
                if hasattr(self.model, "booster_"):
                    proba = self.model.booster_.predict(X)
                else:
                    raise
            # proba shape: (1, n_classes) or (1,) for binary in old versions
            proba = np.asarray(proba)
            if proba.ndim == 1:
                # binary probability as single value -> prob of positive class
                p_pos = float(proba[0])
                pred = int(p_pos > 0.5)
                conf = p_pos if pred == 1 else 1.0 - p_pos
            else:
                pred = int(np.argmax(proba[0]))
                conf = float(np.max(proba[0]))
            return pred, float(conf)

        # Case 2: direct Booster
        if self.is_booster:
            # Booster.predict returns:
            # - binary: array of shape (n_samples,) with prob for positive class
            # - multiclass: array (n_samples, n_class) with probabilities
            proba = self.model.predict(X)
            proba = np.asarray(proba)
            if proba.ndim == 1:
                p_pos = float(proba[0])
                pred = int(p_pos > 0.5)
                conf = p_pos if pred == 1 else 1.0 - p_pos
            else:
                pred = int(np.argmax(proba[0]))
                conf = float(np.max(proba[0]))
            return pred, float(conf)

        # Last fallback: generic predict -> treat as score
        if hasattr(self.model, "predict"):
            out = self.model.predict(X)
            out = np.asarray(out).ravel()
            pred = int(out[0])
            # no probability info, set confidence to 1.0 for deterministic prediction
            return pred, 1.0

        raise RuntimeError("Loaded SL model is not usable for prediction.")
