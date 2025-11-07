from sklearn.model_selection import GridSearchCV

class HyperparameterTuner:
    def __init__(self, model, param_grid, cv_folds=5):
        self.model = model
        self.param_grid = param_grid
        self.cv_folds = cv_folds

    def tune(self, X, y):
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring='f1_weighted',
            cv=self.cv_folds,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X, y)
        return grid.best_estimator_, grid.best_params_, grid.best_score_

