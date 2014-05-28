

import csv
import gzip
from uuid import UUID
from nltk import SklearnClassifier, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from kaggle_donors_choose_2014 import util
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
sw = stopwords.words('english')
ssw = set(sw)


class TfidfModel(object):

    def __init__(self, train, test, outcomes):
        self.train = train.copy()
        self.test = test.copy()
        self.outcomes = outcomes.copy()
        self.model = None
        self.pidss = set(outcomes.index.tolist())

        pipeline = Pipeline([
            ('tfidf', TfidfTransformer()),
            ('chi2', SelectKBest(chi2, k=10000)),
            ('nb', MultinomialNB())
        ])
        self.classifier = SklearnClassifier(pipeline)

    def csv_to_tokens(self):
        for row in self.read_csv():
            pid = UUID(row[0])
            # in case of sampling
            if pid not in self.pidss:
                continue

            tk = util.get_tokens(row[5])

            # print(tk)

            # remove stopwords
            ret = []
            for i in tk:
                if i in ssw:
                    continue
                ret.append(i)

            # print(ret)

            yield ret, pid, self.outcomes.loc[pid].is_exciting

    def read_csv(self):
        with gzip.open('data/essays.csv.gz', 'r') as f:
            c = csv.reader(f, delimiter=',')
            c.next()
            for line in c:
                yield line

    def fit(self, data):
        freqs = [(FreqDist(tokens), exciting) for tokens, _, exciting in data]
        self.classifier.train(freqs)

    def predict(self, data):
        pos = np.array(self.classifier.batch_classify(
            [FreqDist(tokens) for tokens, _, exciting in data if exciting]
        ))
        neg = np.array(self.classifier.batch_classify(
            [FreqDist(tokens) for tokens, _, exciting in data if not exciting]
        ))

        # noinspection PyUnresolvedReferences
        print "Confusion matrix:\n->\tpos\tneg\npos\t%d\t%d\nneg\t%d\t%d" % (
            (pos == True).sum(), (pos == False).sum(),
            (neg == True).sum(), (neg == False).sum()
        )
        z = zip(
            np.concatenate((pos, neg), axis=0),
            [i for _, i, e in data if e] + [i for _, i, e in data if not e]
        )
        result = pd.DataFrame(z, columns=['score', 'projectid'])
        return result

    def train_model(self):
        # TODO split train into groups for training and testing
        self.tfidf = TfidfVectorizer(min_df=3, max_features=1000)
        self.model = LogisticRegression()
        res = [tup for tup in self.csv_to_tokens()]
        self.fit(res)

        return self.predict(res)

    def test_model(self):
        pass
