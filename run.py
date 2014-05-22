

import pandas as pd
from datetime import date, datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from models.naive_essay_length import naive_essay_length_model
from models.random_forest import random_forest_model
from models.sgd import sgd_model


models = {
    'naive-essay-length': naive_essay_length_model,
    'svg': sgd_model,
    'random-forest': random_forest_model
}


class KaggleModel():

    def __init__(self):
        self.true_false = {np.nan: 2, 't': 1.0, 'f': 0.0}
        self.interesting_columns = [
            'projectid', 'school_metro', 'date_posted', 'essay_length',
            'school_charter', 'school_magnet', 'school_nlns', 'school_kipp',
            'school_charter_ready_promise', 'teacher_teach_for_america',
            'teacher_prefix', 'students_reached',
            'total_price_excluding_optional_support',
            'total_price_including_optional_support', 'poverty_level',
            'eligible_double_your_impact_match', 'eligible_almost_home_match',
            'essay'
        ]
        self.true_false_fields = [
            'school_charter', 'school_magnet', 'school_nlns', 'school_kipp',
            'school_charter_ready_promise', 'teacher_teach_for_america',
            'eligible_double_your_impact_match', 'eligible_almost_home_match'
        ]

    def run(self, model_name='naive-essay-length'):
        merged = self.load()
        merged = self.cleanup(merged)

        self.split_train_test(merged)

        #donations = pd.read_csv('donations.csv')
        outcomes = pd.read_csv('outcomes.csv')

        self.expected_outcomes = outcomes.is_exciting
        return models[model_name](self.train, self.test, self.expected_outcomes)

    def load(self):
        projects = pd.read_csv('data/projects.csv.gz', compression='gzip')
        essays = pd.read_csv(
            'data/essays.csv.gz',
            compression='gzip'
        )
        # noinspection PyCallingNonCallable
        merged = pd.merge(
            projects, essays, left_on='projectid', right_on='projectid',
            how='inner'
        )
        return merged

    def cleanup(self, merged):
        merged.essay.fillna("", inplace=True)
        merged['essay_length'] = merged.essay.map(lambda x: len(x))
        merged = merged[self.interesting_columns]
        merged.date_posted = pd.to_datetime(merged.date_posted)
        merged = merged.reset_index().drop('index', 1)
        merged.fillna(-1, inplace=True)
        return merged

    def split_train_test(self, merged):
        self.train = merged[merged.date_posted < date(2014, 1, 1)]
        self.test = merged[merged.date_posted >= date(2014, 1, 1)]

    def vectorize_target(self, df, columns):
        tf = df[columns]
        frame_as_dicts = [dict(x.iteritems()) for _, x in tf.iterrows()]
        vectorized_frame = pd.DataFrame(frame_as_dicts)
        result_frame = pd.concat([df, vectorized_frame], axis=1)
        result_frame = result_frame.drop(columns, axis=1)
        return result_frame

    def to_numeric(self, df):
        df = df.drop('projectid', 1)
        df = df.drop('date_posted', 1)

        for feature in self.true_false_fields:
            df[feature] = df[feature].map(lambda x: self.true_false[x])

        fields_to_vectorize = [
            'teacher_prefix', 'school_metro', 'poverty_level']

        df = self.vectorize_target(df, fields_to_vectorize)
        return df

    def preprocess(self):
       # self.train = self.to_numeric(self.train)
       # self.test = self.to_numeric(self.test)
        tf = TfidfVectorizer(min_df=3, max_features=1000)

        tf.fit(self.train.essay)
        self.tr = tf.transform(self.train.essay)
        self.ts = tf.transform(self.test.essay)
        print(self.tr)


def save_predictions(filename, predictions):
    with open(filename, 'wt') as f:
        f.write('projectid,is_exciting\n')
        for prediction in predictions:
            f.write("%s,%s\n" % (
                prediction[0],
                prediction[1]
            ))


if __name__ == '__main__':
    km = KaggleModel()

    model_name = 'naive-essay-length'
    predictions = km.run(model_name)

    oname = '{}-{}.csv'.format(
        datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        model_name
    )

    save_predictions(oname, predictions)
