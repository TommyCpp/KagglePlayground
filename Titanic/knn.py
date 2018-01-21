from numpy.core.multiarray import ndarray
from sklearn import metrics, preprocessing, neighbors

from Titanic.io import *

TEST_FEATURE = ['Pclass', 'Sex', 'Age', 'Fare']

def _preprocessing(x):
    return preprocessing.scale(x)


def train_KNN(n_neighbour=5):
    data = read_data("./train.csv")
    test_data = data.sample(frac=0.2)
    data.drop(test_data.index)
    data.iloc[:, 0] = data.iloc[:, 0].astype('int')
    y = data.iloc[:, 0].as_matrix()  # type:ndarray
    x = data.iloc[:, 1:].as_matrix()
    x = _preprocessing(x)

    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbour, weights='distance')
    knn_classifier.fit(x, y)
    return knn_classifier, test_data


def test_Knn(n_neighbour=5):
    model, data = train_KNN()
    data = data.as_matrix()
    y_true = data[:, 0]
    x = data[:, 1:]
    x = _preprocessing(x)
    y_pred = model.predict(x)
    test_result = metrics.f1_score(y_true, y_pred)
    print(test_result)
    return test_result


def write_result_Knn(neighbour=5):
    raw_data, x, result = read_test_data(TEST_FEATURE)
    X = _preprocessing(x)
    knn, _ = train_KNN(neighbour)
    y = knn.predict(X).round()
    result['Survived'] = y.astype(int)
    result.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    test_Knn(50)