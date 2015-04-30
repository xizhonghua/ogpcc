#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, metrics, cross_validation
import xgboost as xgb

# fit all the models
def fit(models, X, y):
    for m in models:
        m.fit(X, y)

# predict
def predict(models, X, weights=None):
    n_classes = 9
    n_models = len(models)

    preds = np.zeros((len(X), n_classes))

    if weights == None:
        weights = np.zeros(n_models)
        weights.fill(1.0/n_models)    

    for m, weight in zip(models, weights):
        pred = m.predict_proba(X)
        preds = np.add(preds, np.multiply(pred, weight))    

    return preds

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
models = [
  xgb.XGBClassifier(n_estimators=2000, gamma=2, learning_rate=0.05, max_depth=15, subsample=0.9, seed=1, objective='multi:softprob', min_child_weight=10),
  xgb.XGBClassifier(n_estimators=2000, gamma=2, learning_rate=0.05, max_depth=15, subsample=0.9, seed=1, objective='multi:softprob', min_child_weight=10),
  xgb.XGBClassifier(n_estimators=2000, gamma=2, learning_rate=0.05, max_depth=15, subsample=0.9, seed=1, objective='multi:softprob', min_child_weight=10),
]

if 'CV' in config:
    print 'cross validating...'

    kf = cross_validation.KFold(n=len(train), n_folds=3, shuffle=True)        

    scores = []
    for train_index, test_index in kf:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        print '  - training...'
        fit(models, X_train, y_train)
        print '  - preciting...'
        preds = predict(models, X_train)
        train_loss = metrics.log_loss(y_train, preds)
        preds = predict(models, X_test)
        test_loss = metrics.log_loss(y_test, preds)
        scores.append(test_loss)
        print "  - loss : train/test = %s / %s" % (train_loss, test_loss)

    print 'avg logloss = %s' % np.mean(np.array(scores))

if 'Test' in config:
    print 'training...'
    fit(models, train, labels)
    preds = predict(models, train)
    train_loss = metrics.log_loss(labels, preds)
    print 'train loss on entire dataset = %s' % train_loss

    print 'classification...'
    # predict on test set
    preds = predict(models, test)

    print 'writing...'
    # create submission file
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv('benchmark.csv', index_label='id')
