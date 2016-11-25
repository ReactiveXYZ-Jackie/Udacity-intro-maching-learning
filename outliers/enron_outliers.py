#!/usr/bin/python

import pickle
import sys
import os
import matplotlib.pyplot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
file_path = os.path.join(os.path.dirname(__file__), '..', 'final_project','final_project_dataset.pkl')
data_dict = pickle.load( open(file_path, "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
max_bonus = data[0][0]
for point in data:
    salary = point[0]
    bonus = point[1]
    if bonus > max_bonus:
    	max_bonus = bonus
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print "Max bonus: " + str(max_bonus)
