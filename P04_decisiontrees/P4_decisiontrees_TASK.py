#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:25:04 2017

@author: tugg
"""

import pandas as pa
import numpy as np
import pdb
import sys

################################
#    Do NOT edit header        #
################################

################################
#    Helper Funcitons          #
################################

class Tree_node:
    original_Data = None
    
    def __init__(self,split=None,right_child=None, left_child=None):
        self.split=split
        self.right_child = right_child
        self.left_child = left_child


    def return_child(self,obs):
        column = self.split[0]
                        
        if self.is_categorical(column):
            if obs[column] == self.split[1]:
                return self.right_child
            else:
                return self.left_child
            
        else:
            if obs[column] >= self.split[1]:
                return self.right_child
            else:
                return self.left_child
            
    def classify(self, obs):
        
        child = self.return_child(obs)
        
        if child.__class__.__name__ == 'Tree_node':
            return child.classify(obs)
    
        target_col = self.split[2]
        if self.is_categorical(target_col):
            #print "majority vote"
            return child[target_col].value_counts().keys()[0]
        else:
            #print "average"
            return np.average(child[~child[target_col].isnull()]["age"])
 

    def is_categorical(self, column):
        category=True
        if not Tree_node.original_Data[column].dtype.name == "category":
            category = False
        return category
    
    
def gini_impurity(data,column, weights=None):
    try:
        counts = uniquecounts(data, column)
        probs = counts/data.shape[0]
        if len(probs) == 1:
            prob_obs = np.ones(data.shape[0])
        else:
            la1 = lambda x: probs[probs.index == x][0]
            prob_obs = np.array(map(la1, data[column]))
            prob_obs = np.square(prob_obs)

        if weights is None:
            weights = np.ones(data.shape[0])
        weights = weights/sum(weights)
        return 1-sum(weights*prob_obs)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
       

# Count ocurrences of goal variable
def uniquecounts(data, column):
   val_cnt = data[column].value_counts()
   return val_cnt.drop(val_cnt[val_cnt == 0].index)

################################
#    Data Precprocessing       #
################################
dat_car = pa.read_csv('car.data.csv', sep=",")
dat_car.dtypes
for i in dat_car.columns.values:
    dat_car[i] = dat_car[i].astype('category')

dat = dat_car
target_col = "rating"

# shuffle data
np.random.seed(42)
dat = dat.reindex(np.random.permutation(dat.index))

# split data into training and test set
split = dat.shape[0]/100*20
                         
dat_test = dat.iloc[0:split]
dat_train = dat.iloc[(split+1):dat.shape[0]]

Tree_node.original_Data = dat_train


################################
#    Start Editing here        #
################################

# You only have to edit this file at locations which have the marker
# print("edit here"). 
# For technical problems please use the lab sessions. If the task is unclear please email to tugg@zhaw.ch

################################
#  Train a tree stump          #
################################

# Task 1:
# Complete the function "find_best_split" and train a decision tree stump. You can compute the Information gain
# with respect to gini impurity using the code below
# p=float(set1.shape[0])/in_set.shape[0]
# gain=in_score-p*gini_impurity(set1, target_col, weights)-(1-p)*gini_impurity(set2, target_col, weights) 

# Find best split with respect to target column
def find_best_split(in_set, target_col, weights=None):
    in_score = gini_impurity(in_set, target_col,weights)

    best_gain = 0
    best_split = None
    best_sets = None
    
    print("edit here")
    
    return best_split, best_sets

# Partitions set according to variable and cutoff value
def divideset(in_set, column, value):
   # Make a function that tells us if a row is in
   # the first group (true) or the second group (false)
   split_function=None
   if not in_set[column].dtype.name == "category":
      # assume it to be numerical if not category
      split_function=lambda in_set:in_set[column]>=value
   else:
      split_function=lambda in_set:in_set[column]==value
                                   
   # Divide the rows into two sets and return them
   set1= in_set[split_function(in_set)].copy()
   set2= in_set[np.invert(split_function(in_set))].copy()
   return (set1,set2)


# Train the stump
split, sets = find_best_split(dat_train, target_col)
stump = Tree_node(split,sets[0],sets[1])
print stump.split



################################
#   Build Confusion Matrix     #
################################

# Task 2:
# Complete the function "conf_matrix" and compute the confusion matrices using the training set 
# and the test set

def conf_matrix(in_data, target_col, tree):
  
    levels = uniquecounts(in_data, target_col).keys()
    conf_mat =  np.zeros((len(levels),len(levels)))
    p_correct = None
    
    print("edit here")
    
    return conf_mat, p_correct

# Build confusion Matrix with training data
conf_mat_stump_train, p_correct_stump_train = conf_matrix(dat_train, target_col,stump)

# Build confusion Matrix with test data
conf_mat_stump_test, p_correct_stump_test = conf_matrix(dat_test, target_col,stump)


################################
#    Tree with depth 5         #
################################

# Task 3:
# write a funciton which recursively trains a decision tree. Stop the recursion if 
# - incoming data is homogenious in the target column
# - max depth is reached
# - data cannot be split any further
# Then train a tree of depth 5, does this perfom better than the tree stump?

# Recursively train tree
def train_tree(in_data, target_col, max_depth=99, weigths = None):
    print("edit here")
    return None

# Tree of depth 5
depth5_tree = train_tree(dat_train, target_col, 5)


# Build confusion Matrix with training data
conf_mat_5_train, p_correct_5_train = conf_matrix(dat_train, target_col,depth5_tree)

# Build confusion Matrix with test data 
conf_mat_5_test, p_correct_5_test = conf_matrix(dat_test, target_col,depth5_tree)


    
    
################################
#    Build tree using Sklearn  #
################################

# Task 4:
# Train a decision tree using the widely used library scikit-learn

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import sys 


d = defaultdict(LabelEncoder)

# Encoding the variable for use with sklearn
dat_train_encoded = dat_train.apply(lambda x: d[x.name].fit_transform(x))
dat_test_encoded = dat_test.apply(lambda x: d[x.name].transform(x))

# use DecisionTreeClassifier
print("edit")
d_tree = None
predictions = None


# Inverse the encoded
predictions = d[target_col].inverse_transform(predictions)

sum(dat_test[target_col] == predictions)/float(len(predictions))




################################
#   Boost Trees using AdaBoost #
################################

# Task 5:
# Train an ensemble of Trees using our own AdaBoost implementation
# as a base classifier use the DecisionTree from sk-learn other wise it will be too slow
# again, use trees of depth 5

# you can use these code snippets in your implementation

# compute model importance:
# model_imp = np.log(1-weighted_err)/(weighted_err)

# update observation weights:
# for i in range(N):
#    if predictions[i] == in_data[column].iloc[i]:
#        w[i] = w[i]*weighted_err/(1-weighted_err)

# compute weighted error
#  weighted_err = w.dot(predictions != in_data[column])
#  if weighted_err < 1e-200:
#     break


def ada_boost_trees(in_data, column, depth, m):
    trees = []
    importance = []
    
    N, _ = in_data.shape
    # initialize weights
    w = np.ones(in_data.shape[0]) * float(1)/in_data.shape[0]
    
    for k in range(m):
	print("edit")
  
        
        trees.append(d_tree)
        importance.append(model_imp)
        
    return trees, importance


trees, importance = ada_boost_trees(dat_train_encoded, target_col, 5, 50)


def predict_boosted_trees(trees, importance, obs):
    N, _ = obs.shape
    
    predictions_dir = dict()
    
    for (tree, model_imp) in zip(trees, importance):
        if model_imp == 0: continue
        print model_imp
        # majority vote
        predictions = tree.predict(obs)
        levels = set(predictions)

        for level in levels:
            if level in predictions_dir.keys():
                predictions_dir[level] += (predictions == level)*(model_imp)
            else:
                predictions_dir[level] = (predictions == level)*(model_imp)
    
    pred = np.zeros((N,len(predictions_dir.keys())))
    
    for k in predictions_dir.keys():
        pred[:,k]=predictions_dir[k]
    
    return np.argmin(pred, axis=1)
    


predictions = predict_boosted_trees(trees, importance,dat_test_encoded[dat_test_encoded.columns.difference([target_col])])
predictions = d[target_col].inverse_transform(predictions)
sum(dat_test[target_col] == predictions)/float(len(predictions))



##########################################
#   Boost Trees using AdaBoost v sklearn #
##########################################

# Task 6:
# Repeat task 5 using the AdaBoost implementation of scikit-learn

seed = 42
from sklearn.ensemble import AdaBoostClassifier

print("edit")
predictions = None
predictions = d[target_col].inverse_transform(predictions)

sum(dat_test[target_col] == predictions)/float(len(predictions))

