from numpy.core.multiarray import ndarray
from sklearn import metrics, preprocessing, neighbors
from sklearn.model_selection import GridSearchCV

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
    # for_dummy = train_data.pop('Title')
    # train_data = pd.concat([train_data, pd.get_dummies(for_dummy, prefix='Title')], axis=1)
    # for_dummy = train_data.pop('Pclass')
    # train_data = pd.concat([train_data, pd.get_dummies(for_dummy, prefix='Pclass')], axis=1)
    return train_data


def _preprocessing(x):
    return x


def train_KNN():
    data, raw_data = read_data("./train.csv",TRAIN_FEATURE)
    data = feature_engineer(data, raw_data)
    y = data.iloc[:, 0].as_matrix()  # type:ndarray
    x = data.iloc[:, 1:].as_matrix()
    x = _preprocessing(x)

    knn_classifier = neighbors.KNeighborsClassifier()
    params = {
        "n_neighbors": [1, 200],
        "weights": ["uniform", "distance"]
    }
    gsKnn = GridSearchCV(estimator=knn_classifier, param_grid=params, scoring='accuracy', n_jobs=-1, verbose=1, cv=5)
    gsKnn.fit(x, y)
    print(gsKnn.best_score_)
    return gsKnn.best_estimator_


def write_result_Knn():
    raw_data, x, result = read_test_data(TEST_FEATURE)
    X = _preprocessing(x)
    knn = train_KNN()
    y = knn.predict(X)
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    train_KNN()
