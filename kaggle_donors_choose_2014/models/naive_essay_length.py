

class NaiveEssayLengthModel(object):

    def __init__(self, train, test, _):
        self.train = train.copy()
        self.test = test.copy()

    def train_model(self):
        cp = self.train.copy()
        cp['score'] = cp.essay_length > 1600
        return cp[['projectid', 'score']]

    def test_model(self):
        cp = self.test.copy()
        cp['score'] = cp.essay_length > 1600
        return cp[['projectid', 'score']]
