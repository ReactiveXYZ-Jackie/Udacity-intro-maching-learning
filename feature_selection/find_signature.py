#!/usr/bin/python

import pickle
import numpy
import sys
import os
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = os.path.join(os.path.dirname(__file__), '..', 'feature_selection/word_data_overfit.pkl')
authors_file = os.path.join(os.path.dirname(__file__), '..', 'feature_selection/email_authors_overfit.pkl')
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )


### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = [x.replace("sshacklensf", '').replace("cgermannsf", '') for x in features_train]
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 100 events to put ourselves in this regime
features_train = features_train[:100].toarray()
labels_train   = labels_train[:100]

### your code goes here
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
preds = clf.predict(features_test)


from sklearn.metrics import accuracy_score
print "Accuracy on test: " + str(accuracy_score(preds, labels_test))
thres = 0.2
greatest_importance_list = sorted(filter(lambda x: x > thres, clf.feature_importances_), reverse = True)
print greatest_importance_list
greatest_importance = greatest_importance_list[0]
greatest_index = clf.feature_importances_.tolist().index(greatest_importance)
print "Greatest importance: " + str(greatest_importance)
feature_name = vectorizer.get_feature_names()[greatest_index]
print "Which is " + feature_name
