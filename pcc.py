#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, metrics, cross_validation
import xgboost as xgb

def parse_arg(sys):
    config = {}
    for arg in sys.argv:
        if arg == '--test': config['Test']=True
        if arg == '--cv': config['CV']=True
    return config

config = parse_arg(sys)

print 'reading...'
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sampleSubmission.csv")

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# print 'transforming...'
# # transform counts to TFIDF features
# tfidf = feature_extraction.text.TfidfTransformer()
# train = tfidf.fit_transform(train).toarray()
# test = tfidf.transform(test).toarray()

# encode labels 
print 'encoding...'
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

train = np.array(train)
test = np.array(test)
labels = np.array(labels)

print 'shapes train/test/labels = %s / %s / %s' % (train.shape, test.shape, labels.shape)

# models
#clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=0)
clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=9, subsample=0.9, objective='multi:softprob')

if 'CV' in config:
    print 'cross validating...'

    kf = cross_validation.KFold(n=len(train), n_folds=3, shuffle=True)        

    scores = []
    for train_index, test_index in kf:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = labels[train_index], labels[test_index]    
        print '  - training...'
        clf.fit(X_train, y_train)
        print '  - preciting...'
        preds = clf.predict_proba(X_train)
        train_loss = metrics.log_loss(y_train, preds)
	preds = clf.predict_proba(X_test)
        test_loss = metrics.log_loss(y_test, preds)
        scores.append(test_loss)
        print "  - loss : train/test = %s / %s" % (train_loss, test_loss)

    print 'avg logloss = %s' % np.mean(np.array(scores))

if 'Test' in config:
    print 'training...'
    clf.fit(train, labels)

    print 'classification...'
    # predict on test set
    preds = clf.predict_proba(test)

    print 'writing...'
    # create submission file
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv('benchmark.csv', index_label='id')
