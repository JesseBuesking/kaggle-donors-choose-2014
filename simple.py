# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from datetime import date
from sklearn.linear_model import SGDClassifier
import metrics

def clean(s):
    try:
        return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
    except:
        return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()


#donations = pd.read_csv('donations.csv')
projects = pd.read_csv('projects.csv')
#projects = projects[['projectid', 'students_reached', 'date_posted']]
outcomes = pd.read_csv('outcomes.csv')
#resources = pd.read_csv('resources.csv')
essays = pd.read_csv('essays.csv')
print "Loaded data"

outcomes = outcomes.sort('projectid')
outcomes = outcomes[['is_exciting', 'projectid',]]

essays.sort('projectid', inplace=True)
essays = essays[['projectid', 'essay']]
essays.essay = essays.essay.apply(clean)

projects.sort('projectid', inplace=True)
outcomes.sort('projectid', inplace=True)
essays.sort('projectid', inplace=True)

merged = pd.merge(essays, projects, on='projectid')
merged = merged[[
        'date_posted', 'projectid', 'essay',
        #'school_city','school_state','school_zip','school_metro','school_district','school_county',
        #'school_charter,school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise',
        #'teacher_prefix','teacher_teach_for_america','teacher_ny_teaching_fellow','primary_focus_subject',
        #'primary_focus_area','secondary_focus_subject','secondary_focus_area','resource_type','poverty_level',
        #'grade_level','fulfillment_labor_materials','total_price_excluding_optional_support',
        'total_price_including_optional_support','students_reached',#'eligible_double_your_impact_match',
        #'eligible_almost_home_match',
                ]]
merged['date_posted'] = pd.to_datetime(merged.date_posted)

print "merged data"

essays = None
projects = None

merged.essay.fillna("", inplace=True)
merged['essay_length'] = merged.essay.apply(lambda x: len(x))
merged.total_price_including_optional_support.fillna(0, inplace=True)
merged.students_reached.fillna(0, inplace=True)
merged.students_reached.fillna(0, inplace=True)

X_train = merged[merged.date_posted < date(2014, 1, 1)]
X_test = merged[merged.date_posted >= date(2014, 1, 1)]
X_train.drop('date_posted', axis=1, inplace=True)
X_test.drop('date_posted', axis=1, inplace=True)


merged = None
percent_pos = .75
percent_neg = .075

def sample_(percent_pos, percent_neg, df):
    temp_merge = pd.merge(df, outcomes, on='projectid')
    temp_merge = temp_merge.sort('projectid')
    
    temp_merge_pos = temp_merge[temp_merge.is_exciting == 't']
    rows_neg = np.random.choice(temp_merge_pos.index, int(len(temp_merge_pos) * percent_pos), replace=False)
    
    temp_merge_neg = temp_merge[temp_merge.is_exciting == 'f']
    rows_pos = np.random.choice(temp_merge_neg.index, int(len(temp_merge_neg) * percent_neg), replace=False)
    
    rows_final = np.concatenate((rows_pos, rows_neg), axis=1)

    return temp_merge.ix[rows_final]#.drop('is_exciting', axis=1)

X_train = sample_(percent_pos, percent_neg, X_train)
X_train.sort('projectid', inplace=True)

#X_train = pd.merge(X_train, outcomes, on='projectid')

vect = TfidfVectorizer(max_features=1000, stop_words='english', sublinear_tf=True)
vects = vect.fit_transform(X_train.essay)
svd_vects = TruncatedSVD(n_components=100).fit_transform(vects)
vects = None

features_nontext = X_train.drop(['projectid', 'essay', 'is_exciting'], axis=1).values
combined_features = np.c_[svd_vects, features_nontext]


#pipeline = Pipeline([
        #('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', sublinear_tf=True)),
        #('svd', TruncatedSVD(n_components=110)),
#        ('gbc', GradientBoostingClassifier()),
        #('lda', LDA()),
        #('lr', LogisticRegression()),
        #('lsvc', LinearSVC()),
        #('mnb', MultinomialNB()),
        #('sgd', SGDClassifier()),
        #('sgd', SGDClassifier()),
        #('svc', SVC()),
#            ])

print "Pipeline built"

#params = {
#    'tfidf__max_df': (.5, .75, 1.0),
#    'tfidf__stop_words': ('english', None)
#    'tfidf__ngram_range': ((1, 1), (1, 2)), 
#}

#pipeline.fit(X_train.essay.values, np.asarray(X_train.is_exciting, dtype="|S6"))
clf = LogisticRegression().fit(combined_features, np.asarray(X_train.is_exciting, dtype="|S6"))
print "Pipeline fit"

#X_test['is_exciting'] = pipeline.predict(X_test.essay.values)

vects = vect.fit_transform(X_test.essay)
svd_vects = TruncatedSVD(n_components=100).fit_transform(vects)
vects = None
features_nontext = X_test.drop(['projectid', 'essay',], axis=1).values
combined_features = np.c_[svd_vects, features_nontext]

X_test['is_exciting'] = clf.predict(combined_features)
tr = {'f': 0.0, 't': 1.0}

X_test['is_exciting'] = X_test['is_exciting'].apply(lambda x: tr[x])
X_test[['projectid', 'is_exciting']].to_csv("resultmixed.csv", index=False) # use last algos name as title

