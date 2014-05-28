# coding=utf-8
import random

import nltk
import string
import pandas as pd


replacements = {
    '“': '"',
    '”': '"',
    '’': "'"
}


def sample(iterator, k):
    """
    Samples k elements from an iterable object.

    :param iterator: an object that is iterable
    :param k: the number of items to sample
    """
    # fill the reservoir to start
    result = [next(iterator) for _ in range(k)]

    n = k
    for item in iterator:
        n += 1
        s = random.randint(0, n)
        if s < k:
            result[s] = item

    return result


def to_numeric(df):
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


def get_tokens(value):
    value = value.lower()

    value = value.replace('\r\n', ' ').replace('\n', ' ')
    value = value.replace(r'\r\n', ' ').replace(r'\r', ' ').replace(r'\n', ' ')
    for k, v in replacements.iteritems():
        value = value.replace(k, v)
    #remove the punctuation using the character deletion step of translate
    no_punctuation = value.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens
