import random
from collections import Counter
from statistics import mode
import numpy as np
import pandas as pd
from sklearn import metrics

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature # feature index
        self.threshold = threshold # feature threshold
        self.value = value # feature index majority
        self.left = left # child nodes
        self.right = right

def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Grow a decision tree based on the input data

    Args:
        x (list of lists): List of data points
        y (list): List of labels corresponding to data points
        nmin (int): Minimum number of data points required for a node
        minleaf (int): Minimum number of data points required for a leaf node
        nfeat (int): Number of random features to consider for splitting

    Returns:
        Node: Root node of the decision tree
    """

    if pure(y): # if pure return majority class
        return Node(value=mode(y))

    if len(y) < nmin: # if fewer cases than nmin majority class
        return Node(value=mode(y))

    feature, threshold, leftx, rightx, lefty, righty = split(x, y, minleaf, nfeat) # GINI search
    if feature == None: # no split so node becomes leaf
        return Node(value=mode(y))

    left_child = tree_grow(leftx, lefty, nmin, minleaf, nfeat)
    right_child = tree_grow(rightx, righty, nmin, minleaf, nfeat)

    parent = Node()
    parent.feature = feature
    parent.threshold = threshold 
    parent.left = left_child
    parent.right = right_child
    return parent

        
def tree_pred(x, tr):
    """
    Make predictions using a decision tree

    Args:
        x (list of lists): List of data points to make predictions on
        tr (Node): Root node of the decision tree

    Returns:
        list: Predicted labels for the input data points
    """
    predictions = []
    
    for i in range(len(x)):
        current_node = tr
        while current_node.value == None:
            if x[i][current_node.feature] <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        predictions.append(current_node.value)
    
    return predictions


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    Grow multiple decision trees using bootstrapped samples

    Args:
        x (list of lists): List of data points
        y (list): List of labels corresponding to data points
        nmin (int): Minimum number of data points required for a node
        minleaf (int): Minimum number of data points required for a leaf node
        nfeat (int): Number of random features to consider for splitting
        m (int): Number of trees to grow

    Returns:
        list: List of root nodes of the grown decision trees
    """
    trees = []

    for _ in range(m):
        bootstrap_i = np.random.choice(len(x), len(x), replace=True)
        x_b = [x[i] for i in bootstrap_i]
        y_b = [y[i] for i in bootstrap_i]
        trees.append(tree_grow(x_b, y_b, nmin, minleaf, nfeat))

    return trees


def tree_pred_b(trees, x):
    """
    Make predictions using multiple decision trees

    Args:
        trees (list): List of root nodes of decision trees
        x (list of lists): List of data points to make predictions on

    Returns:
        list: Predicted labels for the input data points
    """
    outcomes = []
    new_y = []
    for tree in trees:
        outcomes.append(tree_pred(x, tree))

    for i in range(len(outcomes[0])):
        new_y.append([a[i] for a in outcomes])
        
    new_y = [mode(a) for a in new_y]
    return new_y

def pure(y):
    """
    Check if all elements in the input list are the same

    Args:
        y (list): List of elements to check.

    Returns:
        bool: True if all elements are the same, otherwise False.
    """
    if len(set(y)) == 1: # if all the same classes in remaining 
        return True
    else:
        return False
    
def gini(y):
    """
    Calculate the Gini impurity for a given list of labels

    Args:
        y (list): List of labels.

    Returns:
        float: Gini impurity value.
    """
    counts = list(Counter(y).values())
    p = 0
    for count in counts:
        p += (count / len(y))**2
    return 1 - p

def split(x, y, minleaf, nfeat):
    """
    Find the best feature and threshold for splitting data points

    Args:
        x (list of lists): List of data points.
        y (list): List of labels corresponding to data points.
        minleaf (int): Minimum number of data points required for a leaf node.
        nfeat (int): Number of random features to consider for splitting.

    Returns:
        tuple: Best feature index, best threshold value,
               left child data points, right child data points,
               left child labels, right child labels.
    """
    features = random.sample(range(len(x[0])), nfeat) # random features selecting
    
    best_gini = 1
    best_feature = None
    best_threshold = None
    best_left_indices = []
    best_right_indices = []

    for feature_index in features:
        values = [a[feature_index] for a in x]
        for t in range(len(values)):
            left_indices = []
            right_indices = []
            right_values = []
            for i, a in enumerate(x): # indices lower than threshold
                if a[feature_index] <= values[t]:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
                    right_values.append(a[feature_index])

            if len(left_indices) >= minleaf and len(right_indices) >= minleaf: # minleaf criteria
                left_gini = gini([y[i] for i in left_indices])
                right_gini = gini([y[i] for i in right_indices])

                weighted_gini = (len(left_indices) / len(y)) * left_gini + (len(right_indices) / len(y)) * right_gini # weighted based on size

                if weighted_gini < best_gini:
                    if weighted_gini == best_gini and random() == 0: ## if gini same and random == 0, skip if random == 1 replace
                        continue
                    best_gini = weighted_gini
                    best_feature = feature_index
                    right_values.sort()
                    best_threshold = (values[t] + right_values[0]) / 2
                    best_left_indices = left_indices
                    best_right_indices = right_indices

    # from indices to rows
    best_left_x = [a for i, a in enumerate(x) if i in best_left_indices]
    best_right_x = [a for i, a in enumerate(x) if i in best_right_indices]
    best_left_y = [a for i, a in enumerate(y) if i in best_left_indices]
    best_right_y = [a for i, a in enumerate(y) if i in best_right_indices]

    return best_feature, best_threshold, best_left_x, best_right_x, best_left_y, best_right_y
