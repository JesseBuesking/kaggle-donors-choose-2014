

import csv
import gzip
import random
from datetime import date
from uuid import UUID

import pandas as pd
import sklearn
from kaggle_donors_choose_2014 import util
from kaggle_donors_choose_2014.models.tfidf import TfidfModel

from models.naive_essay_length import NaiveEssayLengthModel
from models.random_forest import RandomForestModel
from models.sgd import SGDModel


models = {
    'naive-essay-length': NaiveEssayLengthModel,
    'sgd': SGDModel,
    'random-forest': RandomForestModel,
    'tfidf': TfidfModel
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
        self.train = pd.DataFrame([])
        self.test = pd.DataFrame([])
        self.outcomes = pd.DataFrame([])

    def init(self, model_name='naive-essay-length', num_sample=None):
        self.load_projects()

        self.dated_ids = self.projects['projectid'].copy()

        # toss all before 2010-03-15 since that's the first recorded date when
        # at_least_1_teacher_referred_donor is True
        self.dated_ids = self.dated_ids[self.dated_ids.index >= '2010-03-15']

        # merged = self.load()
        # merged = self.cleanup(merged)
        #
        # self.split_train_test(merged)

        self.load_outcomes()
        self.outcomes = self.outcomes[
            self.outcomes.index.isin(self.dated_ids.values)
        ]

        # select sample if requested
        if num_sample is not None:
            def pidgen():
                for i in self.outcomes.index:
                    yield i
            pids = util.sample(pidgen(), num_sample)
        else:
            pids = list(self.outcomes.index)

        # filter down to samples (or full thing)
        self.dated_ids = self.dated_ids[self.dated_ids.isin(pids)]
        self.outcomes = self.outcomes[self.outcomes.index.isin(pids)]
        self.projects = self.projects[self.projects.projectid.isin(pids)]

        self.model = models[model_name](self.train, self.test, self.outcomes)

    def load_projects(self):
        converters = dict()

        def poverty_convert(x):
            return {
                'highest poverty': 'h+',
                'high poverty': 'h',
                'moderate poverty': 'm',
                'low poverty': 'l'
            }.get(x)

        converters['poverty_level'] = poverty_convert

        for i in ['projectid', 'teacher_acctid', 'schoolid']:
            converters[i] = lambda x: UUID(x)

        for i in [
            'school_charter', 'school_magnet', 'school_year_round',
            'school_nlns', 'school_kipp', 'school_charter_ready_promise',
            'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
            'eligible_double_your_impact_match', 'eligible_almost_home_match'
        ]:
            converters[i] = lambda x: x == 't'

        self.projects = pd.read_csv(
            'data/projects.csv.gz',
            index_col='date_posted',
            compression='gzip',
            parse_dates=['date_posted'],
            converters=converters
        )

    def load_outcomes(self):
        converters = dict()
        for i in [
            'is_exciting', 'fully_funded', 'at_least_1_teacher_referred_donor',
            'great_chat', 'at_least_1_green_donation',
            'three_or_more_non_teacher_referred_donors',
            'one_non_teacher_referred_donor_giving_100_plus',
            'donation_from_thoughtful_donor'
        ]:
            converters[i] = lambda x: x == 't'

        for i in ['great_messages_proportion']:
            converters[i] = lambda x: 0. if '' == x else float(x) / 100.

        for i in ['teacher_referred_count', 'non_teacher_referred_count']:
            converters[i] = lambda x: 0. if '' == x else int(float(x))

        converters['projectid'] = lambda x: UUID(x)

        self.outcomes = pd.read_csv(
            'data/outcomes.csv.gz',
            index_col='projectid',
            compression='gzip',
            converters=converters
        )

    def train_model(self):
        return self.model.train_model()

    def test_model(self):
        return self.model.test_model()

    def load(self):
        return None
        # self.projects = pd.read_csv('data/projects.csv.gz', compression='gzip')
        # self.essays = pd.DataFrame(
        #     [[l[0], len(l[5])] for l in self.read_csv()],
        #     columns=['projectid', 'essay_length']
        # )
        # # essays = pd.read_csv(
        # #     'data/essays.csv.gz',
        # #     compression='gzip'
        # # )
        # # noinspection PyCallingNonCallable
        # merged = pd.merge(
        #     self.projects, self.essays, on='projectid', how='inner')
        # return merged

    def csv_to_tokens(self):
        for row in self.read_csv():
            yield util.get_tokens(row[5])

    def read_csv(self):
        with gzip.open('data/essays.csv.gz', 'r') as f:
            c = csv.reader(f, delimiter=',')
            c.next()
            for line in c:
                yield line

    def cleanup(self, merged):
        # merged.essay.fillna("", inplace=True)
        # merged['essay_length'] = merged.essay.map(lambda x: len(x))
        merged = merged[self.interesting_columns].copy()
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
        self.outcomes.reset_index(inplace=True)
        # noinspection PyCallingNonCallable
        merged = pd.merge(
            self.outcomes[['projectid', 'is_exciting']],
            predictions,
            on='projectid',
            how='inner'
        )[['is_exciting', 'score']]
        return merged.shape[0], sklearn.metrics.roc_auc_score(
            merged.is_exciting.values,
            merged.score.values
        )

    def stats(self, predictions):
        size, pscore = self.auc_roc_score(predictions)

        # generate random predictions
        cp = predictions.copy()
        cp.loc[:, 'score'] = [random.randint(0, 1) for _ in range(cp.shape[0])]
        _, rscore = self.auc_roc_score(cp)

        def p(flt):
            return '{:.5f}'.format(flt)

        # statistics
        ret = [
            'Number of predictions:  {}'.format(size),
            'Predicted score:        {}'.format(p(pscore)),
            'Random score:           {}'.format(p(rscore)),
            'Better than random:     {}'.format(p(pscore - rscore))
        ]
        return ret
