#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from feature_format import featureFormat, targetFeatureSplit

fp = os.path.join(os.path.dirname(__file__), '../','final_project', 'final_project_dataset.pkl')
data_dict = pickle.load(open(fp, "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
# split training and testing data
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels ,test_size = .3, random_state = 42)
# create classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)
