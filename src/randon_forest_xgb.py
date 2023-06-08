from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForestXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_params=None, xgb_params=None):
        self.rf_params = rf_params
        self.xgb_params = xgb_params

        self.rf_classifier = RandomForestClassifier(**self.rf_params)
        self.xgb_classifier = XGBClassifier(**self.xgb_params)

    def fit(self, X, y):
        self.rf_classifier.fit(X, y)
        self.xgb_classifier.fit(X, y)

    def predict(self, X):
        rf_preds = self.rf_classifier.predict(X)
        xgb_preds = self.xgb_classifier.predict(X)

        # Voting scheme: take the mode of the predictions
        predictions = []
        for rf_pred, xgb_pred in zip(rf_preds, xgb_preds):
            if rf_pred == xgb_pred:
                predictions.append(rf_pred)
            else:
                predictions.append(1 if rf_pred == 1 else 0)

        return predictions
