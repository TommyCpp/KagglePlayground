import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
from sklearn import metrics, preprocessing


def read_data(location: str):
    df = pd.read_csv(location)
    train_data = df.loc[:, ['Survived', 'Pclass', 'Sex', 'Age']]  # type:DataFrame
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    return train_data


def train():
    data = read_data("./Titanic/train.csv")
    test_data = data.sample(frac=0.2)
    data.drop(test_data.index)
    data = data.as_matrix()
    label = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)

    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(x, label)
    return reg, test_data


def _preprocessing(x):
    return preprocessing.scale(x)


def test():
    reg, data = train()
    data = data.as_matrix()
    y_true = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    y_pred = reg.predict(x).round()
    print(metrics.f1_score(y_true, y_pred))


def write_result():
    test_data = read_data("./Titanic/test.csv")  # type:DataFrame
    raw_data = pd.read_csv("./Titanic/test.csv")
    passenger_id = raw_data.loc[:, 'PassengerId'].as_matrix()  # type:DataFrame
    x = test_data.loc[:, ['Pclass', 'Sex', 'Age']]  # type:DataFrame
    X = _preprocessing(x)
    reg, _ = train()
    y = reg.predict(X).round()
    result = DataFrame({"PassengerId": passenger_id.astype('int32'), "Survived": y.astype('int32')})
    result.to_csv("./Titanic/result.csv",index=False)


if __name__ == "__main__":
    write_result()
