from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class ModelFactory:
    @staticmethod
    def get_models_and_grids():
        models = {
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "lightgbm": lgb.LGBMClassifier(),
            "catboost": CatBoostClassifier(verbose=0, allow_writing_files=True, train_dir="E:/saas-idea-validator/reports/catboost_logs"),
            "logistic_regression": LogisticRegression(max_iter=500),
            "bernoulli_nb": BernoulliNB()
        }

        grids = {
            "decision_tree": {"max_depth":[None,5,10,20], "min_samples_split":[2,5,10]},
            "random_forest": {"n_estimators":[100,200], "max_depth":[6,10,None]},
            "gradient_boosting": {"n_estimators":[100,200], "learning_rate":[0.05,0.1], "max_depth":[3,5]},
            "xgboost": {"n_estimators":[200,300], "max_depth":[4,6,8], "learning_rate":[0.05,0.1], "subsample":[0.7,0.9]},
            "lightgbm": {"n_estimators":[100,200], "max_depth":[-1,5,10], "learning_rate":[0.05,0.1]},
            "catboost": {"iterations":[200,300], "depth":[4,6,8], "learning_rate":[0.05,0.1]},
            "logistic_regression": {"C":[0.01,0.1,1,10], "solver":["liblinear","lbfgs"]},
            "bernoulli_nb": {"alpha":[0.1,0.5,1]}
        }

        return models, grids
