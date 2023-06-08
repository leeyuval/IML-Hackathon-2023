from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


def get_classifiers():
    classifiers = [
        LinearRegression(),
        RandomForestClassifier(n_estimators=200,
                               max_features='log2',
                               min_samples_split=10,
                               max_depth=10),
        DecisionTreeClassifier(),
        LogisticRegression(),
        xgb.XGBClassifier(),
        SVC(),
        KNeighborsClassifier(),
        GaussianNB()
    ]
    return classifiers
