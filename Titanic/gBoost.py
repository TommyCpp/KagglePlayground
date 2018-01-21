from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Age', 'Fare']
TRAIN_FEATURE = ['Survived', 'Pclass', 'Age', 'Fare']


def feature_engineer(train_data, raw_data):
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    # train_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    train_data['Title'] = raw_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    train_data['Title'] = train_data.loc[:, 'Title'].apply(
        lambda x: 0 if x == 'Mr' else 1 if x == 'Miss' or 'Ms' else 2 if x == 'Master' else 3)
    train_data['Fare'].fillna(0, inplace=True)
    return train_data


def _preprocessing(x):
    return x


def train():
    data, raw_data = read_data("./train.csv", TRAIN_FEATURE)
    data = feature_engineer(data, raw_data)
    test_data = data.sample(frac=0.3)
    data.drop(test_data.index)
    data = data.as_matrix()
    label = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)

    gboost = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=50)
    gboost.fit(x, label)
    return gboost, test_data


def test():
    model, data = train()
    data = data.as_matrix()
    y_true = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    y_pred = model.predict(x)
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
    test()
