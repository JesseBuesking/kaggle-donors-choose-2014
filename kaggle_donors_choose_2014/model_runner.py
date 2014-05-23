

import csv
import gzip
import random
from datetime import date

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from models.naive_essay_length import NaiveEssayLengthModel
from models.random_forest import RandomForestModel
from models.sgd import SGDModel


models = {
    'naive-essay-length': NaiveEssayLengthModel,
    'sgd': SGDModel,
    'random-forest': RandomForestModel
}


class ModelRunner():

    def __init__(self):
        self.interesting_columns = [
            'projectid', 'school_metro', 'date_posted', 'essay_length',
            'school_charter', 'school_magnet', 'school_nlns', 'school_kipp',
            'school_charter_ready_promise', 'teacher_teach_for_america',
            'teacher_prefix', 'students_reached',
            'total_price_excluding_optional_support',
            'total_price_including_optional_support', 'poverty_level',
            'eligible_double_your_impact_match', 'eligible_almost_home_match',
            # 'essay'
        ]

    def init(self, model_name='naive-essay-length'):
        merged = self.load()
        merged = self.cleanup(merged)

        self.split_train_test(merged)

        #donations = pd.read_csv('donations.csv')
        self.outcomes = pd.read_csv('data/outcomes.csv.gz', compression='gzip')
        self.outcomes.loc[:, 'is_exciting'] = self.outcomes.is_exciting == 't'
        self.model = models[model_name](self.train, self.test, self.outcomes)

    def train_model(self):
        return self.model.train_model()

    def test_model(self):
        return self.model.test_model()

    def load(self):
        projects = pd.read_csv('data/projects.csv.gz', compression='gzip')
        essays = pd.DataFrame(
            [[l[0], len(l[5])] for l in self.read_csv()],
            columns=['projectid', 'essay_length']
        )
        # essays = pd.read_csv(
        #     'data/essays.csv.gz',
        #     compression='gzip'
        # )
        # noinspection PyCallingNonCallable
        merged = pd.merge(
            projects, essays, left_on='projectid', right_on='projectid',
            how='inner'
        )
        return merged

    def read_csv(self):
        with gzip.open('data/essays.csv.gz', 'r') as f:
            c = csv.reader(f, delimiter=',')
            c.next()
            for line in c:
                yield line

    def cleanup(self, merged):
        # merged.essay.fillna("", inplace=True)
        # merged['essay_length'] = merged.essay.map(lambda x: len(x))
        merged = merged[self.interesting_columns]
        merged.date_posted = pd.to_datetime(merged.date_posted)
        merged.fillna(-1, inplace=True)
        return merged

    def split_train_test(self, merged):
        self.train = merged[merged.date_posted < date(2014, 1, 1)]
        self.train = self.train.reset_index().drop('index', 1)
        self.test = merged[merged.date_posted >= date(2014, 1, 1)]
        self.test = self.test.reset_index().drop('index', 1)

    # def preprocess(self):
    #     # self.train = self.to_numeric(self.train)
    #     # self.test = self.to_numeric(self.test)
    #     tf = TfidfVectorizer(min_df=3, max_features=1000)
    #
    #     tf.fit(self.train.essay)
    #     self.tr = tf.transform(self.train.essay)
    #     self.ts = tf.transform(self.test.essay)
    #     print(self.tr)

    def auc_roc_score(self, predictions):
        # noinspection PyCallingNonCallable
        merged = pd.merge(
            self.outcomes[['projectid', 'is_exciting']],
            predictions,
            left_on='projectid',
            right_on='projectid',
            how='inner'
        )[['is_exciting', 'score']]
        return sklearn.metrics.roc_auc_score(
            merged.is_exciting.values,
            merged.score.values
        )

    def stats(self, predictions):
        pscore = self.auc_roc_score(predictions)

        # generate random predictions
        cp = predictions.copy()
        cp.loc[:, 'score'] = [random.randint(0, 1) for _ in range(cp.shape[0])]
        rscore = self.auc_roc_score(cp)

        def p(flt):
            return '{:.5f}'.format(flt)

        # statistics
        ret = [
            'Predicted score:      {}'.format(p(pscore)),
            'Random score:         {}'.format(p(rscore)),
            'Better than random:   {}'.format(p(pscore - rscore))
        ]
        return ret
