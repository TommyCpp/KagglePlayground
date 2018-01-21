import pandas as pd
import numpy as np


def feature_engineer(train_data, raw_data):
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    train_data['Title'] = raw_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    train_data['Title'] = train_data.loc[:, 'Title'].apply(
        lambda x: 0 if x == 'Mr' else 1 if x == 'Miss' else 2 if x == 'Ms' else 3 if x == 'Master' else 4)
    train_data['Fare'].fillna(0, inplace=True)
    return train_data


def read_data(location: str, train_feature=['Survived', 'Pclass', 'Sex', 'Age', 'Fare']):
    df = pd.read_csv(location)
    train_data = df.loc[:, train_feature]  # type:DataFrame
    return train_data, df


def read_test_data(test_feature):
    test_data = pd.read_csv("./test.csv")
    length = test_data['PassengerId'].count()
    result = pd.DataFrame(
        {"PassengerId": test_data.loc[:, 'PassengerId'].as_matrix().astype('int32'), "Survived": [np.nan] * length})
    return test_data, test_data.loc[:, test_feature], result
