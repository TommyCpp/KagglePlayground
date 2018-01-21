from sklearn import metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Sex', 'Age', 'Fare']


def _preprocessing(x):
    return preprocessing.scale(x)


def train_adaboost():
    data = read_data("./train.csv")
    data = feature_engineer(data, raw_data)
    test_data = data.sample(frac=0.2)
    data.drop(test_data.index)
    y = data.iloc[:, 0].as_matrix()
    x = data.iloc[:, 1:].as_matrix()
    x = _preprocessing(x)
    adaboost = AdaBoostClassifier(n_estimators=30)
    adaboost.fit(x, y)
    return adaboost, test_data


def test_adaboost():
    model, data = train_adaboost()
    data = data.as_matrix()
    y_true = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    y_pred = model.predict(x)
    test_result = metrics.f1_score(y_true, y_pred)
    print(test_result)
    return test_result


def write_test_adaboost():
    raw_data, x, result = read_test_data(TEST_FEATURE)  # type:DataFrame
    x = feature_engineer(x, raw_data)
    X = _preprocessing(x)
    model, _ = train_adaboost()
    y = model.predict(X).round()
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    test_adaboost()
