

from sklearn.ensemble import RandomForestClassifier


def random_forest_model(train, test, expected_outcomes):
    clf = RandomForestClassifier(n_estimators=10, n_jobs=4)
    clf = clf.fit(test.values, expected_outcomes.values)

    predictions = clf.predict(test.values)
    return predictions
