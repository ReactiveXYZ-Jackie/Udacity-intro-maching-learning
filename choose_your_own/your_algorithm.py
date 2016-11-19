#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.metrics import accuracy_score

#####
##### K nearest neighbour classifier
#####
from sklearn.neighbors import KNeighborsClassifier

# num of neighbors
numNeighbors = 15
# num of leaves
leafSize = 50

clf = KNeighborsClassifier(n_neighbors = numNeighbors, leaf_size = leafSize)

clf.fit(features_train, labels_train)

preds = clf.predict(features_test)

accuracy = accuracy_score(preds, labels_test)

print "K nearest neighbors using {0} neighbors & {1} leaves accuracy: {2}".format(numNeighbors, leafSize, accuracy)

##### 
##### Random Forest (ensemble)
#####
from sklearn.ensemble import RandomForestClassifier

# number of trees
numTrees = 10
# min number of samples before decision tree extends
minSamplesSplit = 50

clf = RandomForestClassifier(n_estimators = numTrees, min_samples_split = minSamplesSplit)

clf.fit(features_train,labels_train)

preds = clf.predict(features_test)

accuracy = accuracy_score(preds, labels_test)

print "Random forest (with {0} trees) accuracy: {1}".format(numTrees, accuracy)

##### 
##### Adaboost (ensemble)
#####
from sklearn.ensemble import AdaBoostClassifier

# number of 'weak' classifiers
numWeakClf = 30


# adaboost classifier using decision tree models
clf = AdaBoostClassifier(n_estimators = numWeakClf)

clf.fit(features_train, labels_train)

preds = clf.predict(features_test)

accuracy = accuracy_score(preds, labels_test)

print "Adaboost using decision tree weak classifier (with {0} workers) accuracy: {1}".format(numWeakClf, accuracy)



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
	pass
