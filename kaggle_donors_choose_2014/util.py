

import pandas as pd
import numpy as np


true_false = {np.nan: 2, 't': 1.0, 'f': 0.0}
true_false_fields = [
    'school_charter', 'school_magnet', 'school_nlns', 'school_kipp',
    'school_charter_ready_promise', 'teacher_teach_for_america',
    'eligible_double_your_impact_match', 'eligible_almost_home_match'
]


def to_numeric(df):
    for feature in true_false_fields:
        df[feature] = df[feature].map(lambda x: true_false[x])

    fields_to_vectorize = ['teacher_prefix', 'school_metro', 'poverty_level']
    df = vectorize_target(df, fields_to_vectorize)
    return df


def vectorize_target(df, columns):
    tf = df[columns]
    frame_as_dicts = [dict(x.iteritems()) for _, x in tf.iterrows()]
    vectorized_frame = pd.DataFrame(frame_as_dicts)
    result_frame = pd.concat([df, vectorized_frame], axis=1)
    result_frame = result_frame.drop(columns, axis=1)
    return result_frame
