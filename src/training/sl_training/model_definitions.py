from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

class ModelFactory:
    @staticmethod
    def get_models_and_grids():
        models = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "svm": SVC(probability=True)
        }

        grids = {
            "logistic_regression": {"C":[0.01,0.1,1,10], "solver":["liblinear","lbfgs"]},
            "random_forest": {"n_estimators":[100,200], "max_depth":[6,10,None]},
            "gradient_boosting": {"n_estimators":[100,200], "learning_rate":[0.05,0.1], "max_depth":[3,5]},
            "xgboost": {"n_estimators":[200,300], "max_depth":[4,6,8], "learning_rate":[0.05,0.1], "subsample":[0.7,0.9]},
            "svm": {"C":[1,10,50], "kernel":["rbf","linear"]}
        }
        return models, grids
