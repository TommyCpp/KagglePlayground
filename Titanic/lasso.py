from sklearn import preprocessing, linear_model, metrics

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Sex', 'Age', 'Fare']


def _preprocessing(x):
    return preprocessing.scale(x)


def train_Lasso():
    data = read_data("./train.csv")
    data = feature_engineer(data, raw_data)
    test_data = data.sample(frac=0.2)
    data.drop(test_data.index)
    data = data.as_matrix()
    label = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)

    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(x, label)
    return reg, test_data


def test_Lasso():
    model, data = train_Lasso()
    data = data.as_matrix()
    y_true = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    y_pred = model.predict(x).round()
    print(metrics.f1_score(y_true, y_pred))


def write_result_Lasso():
    raw_data, x, result = read_test_data(TEST_FEATURE)  # type:DataFrame
    x = feature_engineer(x, raw_data)
    X = _preprocessing(x)
    reg, _ = train_Lasso()
    y = reg.predict(X).round()
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    write_result_Lasso()
