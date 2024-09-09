'''RANDOM FOREST CLASSIFIER CLASS

Name        :  RandomForest_Catogerizer
Attributes  : 1. n_factors
              2. n_trees
              3. max_depth
              4. min_data
              5. mode
              6. trees
Methods     : 1. bootstrap
              2. fit_to_forest
              3. make_Prediction
              4. Predict
              5. find_xfactors'''

'''IMPORTING THE DECISION TREE CLASS I CREATED FROM SCRATCH'''
from DecisionTree import DecisionTree_Catogerizer

# other dependencies
import numpy as np
from random import choice
from collections import Counter

class RandomForest_Catogerizer:
    def __init__(self, n_factors = None, n_trees = 20, max_depth = 20, min_data = 30, mode = 'random') :
        
        self.n_factors = n_factors
        # the number of features to be considered while splitting data for each tree in the forest
        # instead of using all features in the dataset for training trees in the forest, a random subset of n_factors features will be used 
        self.n_trees = n_trees
        # total number of decision trees in the forest
        self.max_depth = max_depth
        # limiting value of depth for any tree in the forest
        self.min_data = min_data
        # minimum number of data points in a tree node before splitting that node
        self.mode = mode
        # introduced for telling the trees to select random subset of features for training purpose
        self.trees = []
        # a list to store trained trees in the forest

    '''METHOD 1 - bootstrap
    used to implement bootstrapping which is an important addition in Random Forest'''

    def bootstrap(self, X, y):
        '''X : the feature matrix, a 2D numpy array
           y : the corresponding target column matrix, a 2D numpy array '''
        
        # finding number of rows in X
        n_records = np.shape(X)[0]
        # choosing as many row indices as the number of rows in the dataset, randomly with replacement
        idxs = np.random.choice(n_records, n_records, replace = True)
        # returning bootstrapped subset from the original dataset
        return X[idxs], y[idxs]
    
    '''METHOD 2 - fit_to_forest
    used for fitting the training data into our random forest classifier'''
        
    def fit_to_forest(self, X, y):
        '''X : the feature matrix, a 2D numpy array
           y : the corresponding target column matrix, a 2D numpy array '''

        # fitting the data into each of the trees in the forest
        for i in range(self.n_trees):
            tree = DecisionTree_Catogerizer(self.n_factors, self.mode, self.min_data, self.max_depth)
            X_sampled, y_sampled = self.bootstrap(X,y)
            tree.fit(X_sampled, y_sampled)
            # collecting the trained trees
            self.trees.append(tree)

    '''METHOD 3 - make_Prediction
    for classifiying a single feature vector. Implements the concept of aggregation viz. characteristic of random forest'''        

    def make_Prediction(self, x):
        'x : single feature vector'
        pred = []
        for tree in self.trees:
            pred.append(tree.find_type(x, tree.root)) 
        # final class predicted is the class predicted by majority of the trees in the forest    
        return max(pred, key = pred.count)
    
    '''METHOD 4 - Predict
    for classifying all feature vectors in a feature matrix'''

    def Predict(self, X):
        '''X : the feature matrix'''

        predictions = [self.make_Prediction(x) for x in X]
        # returns a list of predictions corresponding to each feature vector in the matrix
        return predictions
    
    
    '''METHOD 5 - find_xfactors
    gives a dictionary of counts of all factors used by the decision trees in the forest'''

    def find_xfactors(self):
        determiners = []
        for tree in self.trees:
            determiners.extend(tree.used_factors)
        factor_importance = Counter(determiners)
        return factor_importance
        