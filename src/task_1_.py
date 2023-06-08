import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.preprocess import parse_data
from src.randon_forest_xgb import RandomForestXGBClassifier
from utils import find_model


def run_task_one(data_set_path, test_set_path):
    data, target, test_data = parse_data(data_set_path, test_set_path)
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=.8,
                                                        test_size=.2, random_state=42)
    best_cls = find_model(X_train, y_train)
    predictions = pd.DataFrame(best_cls.predict(X_test),columns=['Predictions'])
    print(predictions)

    # ensemble_clf = RandomForestXGBClassifier(rf_params={'n_estimators': 100}, xgb_params={'n_estimators': 100})
    # ensemble_clf.fit(X_train, y_train)
    # predictions = ensemble_clf.predict(X_test)
    # print(f1_score(y_test,predictions))
    # report = classification_report(y_test, predictions)
    # print("Classification Report:\n", report)

if __name__ == '__main__':
    run_task_one("/Users/alon.frishberg/PycharmProjects/IML_Hackathon/data_src/agoda_cancellation_train.csv",
                  "/Users/alon.frishberg/PycharmProjects/IML_Hackathon/data_src/Agoda_Test_1.csv")
