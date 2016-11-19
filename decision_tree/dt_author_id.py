#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
from time import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "Length of feature: " + str(len(features_train[0]))



#########################################################
### your code goes here ###
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
preds = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print "Accuracy: " + str(accuracy_score(preds, labels_test))

#########################################################


