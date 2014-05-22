

from sklearn.linear_model import SGDClassifier


def sgd_model(train, test, expected_outcomes):
    clf = SGDClassifier(alpha=0.001, n_iter=100, shuffle=True)
    clf = clf.fit(train.values, expected_outcomes.values)

    predictions = clf.predict(test.values)
    return predictions
