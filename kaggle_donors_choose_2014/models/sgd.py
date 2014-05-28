

from sklearn.linear_model import SGDClassifier
from kaggle_donors_choose_2014.util import to_numeric
import pandas as pd


class SGDModel(object):

    def __init__(self, train, test, outcomes):
        self.train = train.copy()
        self.test = test.copy()
        self.outcomes = outcomes.copy()
        self.model = None

    def prep(self, df):
        cp = df.copy()
        # print(self.train.head())
        # raise Exception()
        cp = cp.drop('projectid', 1)
        cp = cp.drop('date_posted', 1)
        cp = to_numeric(cp)
        cp = cp.fillna(-1)
        print(cp.columns.tolist())
        cp = cp[
            ['essay_length', 'total_price_including_optional_support']
        ].values
        return cp

    def fit(self, data):
        print('is_exciting', len(self.outcomes.is_exciting.values))
        self.model = self.model.fit(
            self.prep(data),
            self.outcomes.is_exciting.values
        )

    def predict(self, data):
        ids = data.projectid
        # noinspection PyUnresolvedReferences
        predictions = self.model.predict(self.prep(data))

        result = pd.DataFrame(ids, columns=['projectid'])
        result['score'] = predictions
        return result

    def train_model(self):
        # TODO split train into groups for training and testing
        self.model = SGDClassifier(alpha=0.0001, n_iter=100,
                                   penalty='l1', shuffle=True)
        self.fit(self.train)

        return self.predict(self.train)

    def test_model(self):
        self.model = SGDClassifier(alpha=0.001, n_iter=100, shuffle=True)
        self.fit(self.train)

        return self.predict(self.test)
