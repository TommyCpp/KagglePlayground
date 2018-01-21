from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Age', 'Fare']
TRAIN_FEATURE = ['Survived', 'Pclass', 'Age', 'Fare']


def feature_engineer(train_data, raw_data):
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    # train_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    train_data['Title'] = raw_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    train_data['Title'] = train_data.loc[:, 'Title'].apply(
        lambda x: 0 if x == 'Mr' else 1 if x == 'Miss' or x == 'Ms' else 2 if x == 'Master' else 3)
    train_data['Fare'].fillna(0, inplace=True)
    for_dummy = train_data.pop('Title')
    train_data = pd.concat([train_data, pd.get_dummies(for_dummy, prefix='Title')], axis=1)
    for_dummy = train_data.pop('Pclass')
    train_data = pd.concat([train_data, pd.get_dummies(for_dummy, prefix='Pclass')], axis=1)
    return train_data


def _preprocessing(x):
    return x


def train():
    data, raw_data = read_data("./train.csv", TRAIN_FEATURE)
    data = feature_engineer(data, raw_data)
    data = data.as_matrix()
    y = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    gboost = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }
    gsGBC = GridSearchCV(gboost, param_grid=gb_param_grid, cv=10,
                         scoring="accuracy", n_jobs=-1, verbose=1)
    gsGBC.fit(x, y)
    print(gsGBC.best_score_)
    return gsGBC.best_estimator_


def write_result():
    raw_data, x, result = read_test_data(TEST_FEATURE)  # type:DataFrame
    x = feature_engineer(x, raw_data)
    X = _preprocessing(x)
    gboost = train()
    y = gboost.predict(X).round()
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    write_result()
