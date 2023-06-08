from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class UnifiedEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classifiers = [
            RandomForestClassifier(),
            SVC()
        ]

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict(X))
        return predictions

    def predict_proba(self, X):
        probas = []
        for clf in self.classifiers:
            probas.append(clf.predict_proba(X))
        return probas
