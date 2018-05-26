#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from time import time

clf = svm.SVC(kernel = "rbf", C=10000.)

# uncomment the following two lines to cut down training datasets to 0.1% of their sizes, for quicker training (but lesser accuracy)
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time() # training time begins
clf.fit(features_train, labels_train)
t1 = time () # training time ends
print "Training done \nTime taken:", round(t1-t0, 3), "s"

t0 = time() # prediction time begins
pred = clf.predict(features_test)
t1 = time() # prediction time ends
print "Prediction over \nTime taken:", round(t1-t0, 3), "s"

# Output the first 50 predicitons
chris = [i for i in pred if i == 1]
print "Chirs emails:", len(chris)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy
#########################################################


