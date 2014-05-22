

from sklearn.linear_model import SGDClassifier


class SGDModel(object):

    def __init__(self, train, test, outcomes):
        self.train = train.copy()
        self.test = test.copy()
        self.outcomes = outcomes.copy()

    def train_model(self):
        # TODO split train into groups for training and testing

        clf = SGDClassifier(alpha=0.001, n_iter=100, shuffle=True)
        clf = clf.fit(self.train.values, self.outcomes.is_exciting.values)

        predictions = clf.predict(self.train.values)
        return predictions

    def test_model(self):
        clf = SGDClassifier(alpha=0.001, n_iter=100, shuffle=True)
        clf = clf.fit(self.train.values, self.outcomes.is_exciting.values)

        predictions = clf.predict(self.test.values)
        return predictions
