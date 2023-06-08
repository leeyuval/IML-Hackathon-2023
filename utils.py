import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.classifiers_pool import get_classifiers
from sklearn.model_selection import RandomizedSearchCV


def corr_func(data, target):
    correlations = np.corrcoef(data, target, rowvar=False)[-1, :-1]
    feature_names = data.columns.tolist()[:-1]
    # Print the correlation values
    for feature_name, correlation in zip(feature_names, correlations):
        print(f"Correlation between {feature_name} and y: {correlation}")


def understand_features(data: pd.DataFrame):
    subset_data = data.sample(n=10000, random_state=42)  # Adjust the number as per your requirement
    # Separate the features and target variable from the subset
    mini_features = subset_data.drop("cancellation_datetime", axis=1)
    mini_target = subset_data["cancellation_datetime"]
    # Train a Random Forest classifier
    clf = RandomForestClassifier()
    clf.fit(mini_features, mini_target)
    # Get feature importances
    feature_importances = clf.feature_importances_
    # Print feature importance scores
    for feature, importance in zip(data.columns, feature_importances):
        print(f"{feature}: {round(importance, 4)}")


def union_grid_search(multi_classifers, data: pd.DataFrame,
                      target: pd.Series):
    param_grids = [
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
        {'max_depth': [None, 5, 10]},
        {'C': [0.1, 1, 10]},
        {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]},
        {'n_neighbors': [3, 5, 7]},
        {},
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
    ]
    # Create a results grid DataFrame
    results_grid = pd.DataFrame(columns=['Classifier', 'Parameters', 'Accuracy'])
    # Iterate over classifiers and parameter grids
    for classifier, param_grid in zip(multi_classifers.classifiers, param_grids):
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(classifier, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best parameters and accuracy score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Evaluate the classifier on the test set
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Append results to the grid DataFrame
        results_grid = results_grid.append({'Classifier': classifier.__class__.__name__,
                                            'Parameters': best_params,
                                            'Accuracy': accuracy}, ignore_index=True)
    # Display the results grid
    print(results_grid)


def tune_random_forest(X_train,y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

def find_model(X_train, y_train, split_size=.6):
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_f1 = -np.inf
    best_cls = None
    for classifier in get_classifiers():
        print(f"Estimator: {classifier.__class__.__name__}")
        # Perform train-test split
        trained_X, tested_X, trained_y, tested_y = train_test_split(X_train, y_train, train_size=split_size,
                                                                    test_size=(1 - split_size),
                                                                    stratify=y_train, random_state=42)
        # Train a Random Forest classifier
        classifier.fit(trained_X, trained_y)
        # Make predictions on the test set
        y_pred = np.round(classifier.predict(tested_X)).astype(int)
        # y_pred = np.round(y_pred).astype(int)
        # Evaluate the classifier

        f1 = f1_score(tested_y, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_cls = classifier
        print(f"f1: {round(f1, 6)}\n")
        fpr, tpr, thresholds = roc_curve(tested_y, y_pred)
        # Compute the area under the ROC curve (AUC-ROC)
        auc_roc = roc_auc_score(tested_y, y_pred)
        # Plot the ROC curve
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_roc))
        plt.plot([0, 1], [0, 1], 'k--')  # Plot the random classifier curve
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC + {classifier.__class__.__name__} + {split_size} and AUC value:'
                  f' {round(auc_roc, 6)}')
        plt.legend(loc='lower right')

        filename = f"{classifier.__class__.__name__}_split_{split_size}.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)

        plt.close()  # Close
    print(f"Best Estimator: {best_cls.__class__.__name__}")
    return best_cls


def search_best(X_train, y_train, estimator: BaseEstimator):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
    }
    rf = estimator
    random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, scoring='accuracy', cv=5)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    print(best_params)
    print(best_model)


def feature_selection(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize a Random Forest classifier
    rf_classifier = RandomForestClassifier()
    # Fit the model on the training data
    rf_classifier.fit(X_train, y_train)
    # Extract feature importance
    feature_importance = rf_classifier.feature_importances_
    # Sort the feature importance values in descending order
    sorted_indices = np.argsort(feature_importance)[::-1]
    # Select the top n features (e.g., top 5)
    top_features = sorted_indices[:5]
    # Print the top features
    print("Top 5 features for cancellation prediction:")
    for feature_index in top_features:
        print(X.columns[feature_index])


