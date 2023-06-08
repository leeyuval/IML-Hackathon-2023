from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class EnsembleClassifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ensemble_classifiers = [
            ('Random Forest',
             RandomForestClassifier(n_estimators=200, max_features='log2', min_samples_split=10, max_depth=10)),
            ('Logistic Regression', LogisticRegression()),
            ('SVM', SVC())
        ]

    def train(self):
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_features': ['auto', 'sqrt'],
            'min_samples_split': [2, 5, 10],
            'max_depth': [None, 10, 20]
        }

        param_grid_lr = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }

        param_grid_svc = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }

        grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, scoring='accuracy', cv=5)
        grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, scoring='accuracy', cv=5)
        grid_search_svc = GridSearchCV(SVC(), param_grid_svc, scoring='accuracy', cv=5)

        grid_search_rf.fit(self.X_train, self.y_train)
        grid_search_lr.fit(self.X_train, self.y_train)
        grid_search_svc.fit(self.X_train, self.y_train)

        self.best_rf = grid_search_rf.best_estimator_
        self.best_lr = grid_search_lr.best_estimator_
        self.best_svc = grid_search_svc.best_estimator_
        self.ensemble = VotingClassifier(estimators=self.ensemble_classifiers)
        self.ensemble.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.ensemble.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Ensemble Accuracy:", accuracy)
