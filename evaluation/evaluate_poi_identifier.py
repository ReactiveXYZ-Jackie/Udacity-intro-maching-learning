#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from feature_format import featureFormat, targetFeatureSplit

fp = os.path.join(os.path.dirname(__file__), '../','final_project', 'final_project_dataset.pkl')
data_dict = pickle.load(open(fp, "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
# split training and testing data
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels ,test_size = .3, random_state = 42)
# create classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
preds = clf.predict(features_test)
from sklearn.metrics import precision_score, recall_score
print "Precison Score: " + str(precision_score(labels_test, preds))
print "Recall Score: " + str(recall_score(labels_test, preds))