from sklearn import preprocessing, linear_model, metrics
from sklearn.ensemble import GradientBoostingClassifier

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Sex', 'Age', 'Fare']
TRAIN_FEATURE = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']


def _preprocessing(x):
    return x


def train():
    data = read_data("./train.csv")
    test_data = data.sample(frac=0.2)
    data.drop(test_data.index)
    data = data.as_matrix()
    label = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)

    gboost = GradientBoostingClassifier(n_estimators=150)
    gboost.fit(x, label)
    return gboost, test_data


def test():
    model, data = train()
    data = data.as_matrix()
    y_true = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    y_pred = model.predict(x).round()
    test_result = metrics.f1_score(y_true, y_pred)
    print(test_result)
    return test_result


def write_result():
    raw_data, x, result = read_test_data(TEST_FEATURE)  # type:DataFrame
    x = feature_engineer(x, raw_data)
    X = _preprocessing(x)
    gboost, _ = train()
    y = gboost.predict(X).round()
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    write_result()
