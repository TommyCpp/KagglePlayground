import pandas as pd
from pandas import DataFrame
from sklearn import metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Age', 'Fare']
TRAIN_FEATURE = ['Survived', 'Pclass', 'Age', 'Fare']


def feature_engineer(train_data, raw_data):
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)  # type:DataFrame
    train_data['Title'] = raw_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    train_data['Title'] = train_data.loc[:, 'Title'].apply(
        lambda x: 0 if x == 'Mr' else 1 if x == 'Miss' else 2 if x == 'Ms' else 3 if x == 'Master' else 4)
    train_data['Fare'].fillna(0, inplace=True)
    for_dummy = train_data.pop('Title')
    train_data = pd.concat([train_data, pd.get_dummies(for_dummy, prefix='Title')], axis=1)
    return train_data


def _preprocessing(x):
    return preprocessing.scale(x)


def train_adaboost():
    data, raw_data = read_data("./train.csv", TRAIN_FEATURE)
    data = feature_engineer(data, raw_data)
    y = data.iloc[:, 0].as_matrix()
    x = data.iloc[:, 1:].as_matrix()
    x = _preprocessing(x)
    adaboost = AdaBoostClassifier()
    params = {
        "n_estimators": list(range(10, 600, 10))
    }
    gsAdaboost = GridSearchCV(estimator=adaboost, param_grid=params, cv=5,
                              scoring="accuracy", n_jobs=-1, verbose=1)
    gsAdaboost.fit(x, y)
    print(gsAdaboost.best_score_)
    print(gsAdaboost.best_params_)
    return gsAdaboost.best_estimator_


def write_test_adaboost():
    raw_data, x, result = read_test_data(TEST_FEATURE)  # type:DataFrame
    x = feature_engineer(x, raw_data)
    X = _preprocessing(x)
    model = train_adaboost()
    y = model.predict(X)
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    write_test_adaboost()
