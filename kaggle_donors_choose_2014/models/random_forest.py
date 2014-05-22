

from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(object):

    def __init__(self, train, test, outcomes):
        self.train = train.copy()
        self.test = test.copy()
        self.outcomes = outcomes.copy()

    def train_model(self):
        # TODO split train into groups for training and testing

        clf = RandomForestClassifier(n_estimators=10, n_jobs=4)
        clf = clf.fit(self.train.values, self.outcomes.is_exciting.values)

        predictions = clf.predict(self.train.values)
        return predictions

    def test_model(self):
        clf = RandomForestClassifier(n_estimators=10, n_jobs=4)
        clf = clf.fit(self.train.values, self.outcomes.is_exciting.values)

        predictions = clf.predict(self.test.values)
        return predictions
