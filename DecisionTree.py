
'''DECISION TREE CLASSIFIER CLASS

Name       :  DecsionTree_Catogerizer
Attributes : 1. n
             2. mode
             3. min_data
             4. max_depth
Methods    : 1. entropy
             2. gini_index
             3. info_gain
             4. cleave
             5. perfect_cleave
             6. make_tree
             7. fit
             8. find_type
             9. predict
             10. find_leaf_type'''

import numpy as np


# defining the class for a node in our decision tree
class Node():
    def __init__(self, factor_idx = None, cut_off = None, first = None, second = None, impurity_loss = None, type = None ):
        # for decision node

        self.factor_idx = factor_idx 
        # index of the column in which the deciding factor viz. a feature is present
        self.cut_off = cut_off 
        # the value of the deciding feature upon which the splitting is done
        self.first = first 
        # first child of the node. any data point with value of deciding factor less than or equal to cut off value will be sent to this child
        self.second = second 
        # second child of the node
        self.impurity_loss = impurity_loss 
        # the loss in impurity of the data while moving from the node to it's children

        # for leaf node
        self.type = type 
        # the catogery/type/class in which majority of the datapoints in the node belongs to
        self.used_factors = []


# defining the class for the decision tree which does the job of classification
class DecisionTree_Catogerizer():
    def __init__(self, n = None, mode = None, min_data = 2, max_depth = 2):

        '''min_data : the minimum number of datapoints that should be present in a node for splitting the node
           max_depth : the maximum depth of the decsion tree'''

        # initialize the root node of the tree
        self.root = None
        self.n = n # used only when self.mode is 'random'. It's value is None by default
        self.mode = mode # introduced for feature subset selection.
        # give non-None value only when using alongwith random forest algorithm.
        self.min_data = min_data
        self.max_depth = max_depth
        self.used_factors = [] # collects a factor index each time it is used for splitting 
        

        
        # defining the class methods 

    '''METHOD 1 - entropy
    used for calculating the entropy in the data within a node'''
    def entropy(self, y):

        '''y represents the 2D column matrix (a numpy array). 
        It is basically the column of the target variable of the data inside a node.
         No. of rows in this array is same as the number of datapoints present in the node. '''
        
        types = np.unique(y) # a list of catogeries to which the data points in the node belong
        entropy = 0
        # iterate over each catogery
        for cls in types:
            cls_proportion = len(y[ y==cls ])/len(y)
            entropy += -cls_proportion* np.log2(cls_proportion)
        return entropy
    
    '''METHOD 2 - gini_index
    used for calculating the ginin index of data within a node. Method similar to entropy'''
    def gini_index(self, y):
        types = np.unique(y)
        gini = 0
        for cls in types:
            cls_proportion = len(y[ y==cls ])/len(y)
            gini += cls_proportion**2
        return 1-gini
    
    '''METHOD 3 - cleave
    used to divide the data in a node into it's child nodes based on the cut-off value of the deciding factor'''
    def cleave(self, data, factor_idx, cut_off):
        
        '''data : a 2D numpy array corresponding to the subset of the dataset
          present in the node which is to be split'''
        data_first = np.array([row for row in data if row[ factor_idx ] <= cut_off])
        data_second = np.array([row for row in data if row[ factor_idx ] > cut_off])
        
        return data_first, data_second

    '''METHOD 4 - info_gain
    used for calculating the information gain value associated with a specific spliting of a node.
    Each split is characterised by a specific deciding factor and its specific cut-off value'''
    def info_gain(self, parent, f_child, s_child, mode = 'gini'):

        '''parent  : represents the target variable column of the data in the node which was split
           f_child : represents the target variable column of the data in the first_child node obtained after split
           s_child : represents the target variable column of the data in the second_child node obtained after the split'''
        # assigning weights to the child nodes
        w_f = len(f_child)/len(parent)
        w_s = len(s_child)/len(parent)

        if mode == 'gini':
            # calculating the weighted average of impurity in the child nodes
            weighted_avg_impurity = w_f*self.gini_index(f_child) + w_s*self.gini_index(s_child)

            # subtracting this from impurity of the parent node
            gain = self.gini_index(parent) - weighted_avg_impurity
            return gain
        else :
            weighted_avg_impurity = w_f*self.entropy(f_child) + w_s*self.entropy(s_child)
            gain = self.entropy(parent) - weighted_avg_impurity
            return gain
        
    '''METHOD 5 - perfect_cleave
    returns a dictionary of information regarding the best possible cleavage of a node in the decision tree'''
    def perfect_cleave(self, data, num_factors):

        '''data : a 2D numpy array corresponding to a subset of the dataset predsent in a node.
                  It's last column consists of the target variable values. Rest of the columns
                  corresponds to various features    '''

        # initializes an empty dictionary
        perfect_cleave = {}
        # sets value for maximum information gain which will be updated later
        max_info_gain = -float('inf')

        if self.mode == 'random':
            # This is the case when a subset of features is used for spliiting node data, specially while using the tree for random forest classification
            idxs = np.random.choice(num_factors, self.n, replace=False)
            for idx in idxs:
                factor_values = data[:,idx]
                cut_off_values = np.unique(factor_values)
                # inner loop to iterate over possible cut_off values of the current deciding factor
                for cut_off in cut_off_values:
                    # splitting the data in the node based on the current factor and current threshold
                    data_f, data_s = self.cleave(data, idx, cut_off)
                    if len(data_f)>0 and len(data_s)>0:
                        parent_y = data[:,-1]
                        f_child_y = data_f[:,-1]
                        s_child_y = data_s[:,-1]
                        current_info_gain = self.info_gain(parent_y, f_child_y, s_child_y)
                        # checking if the information gain of the current split is more than that of previous split
                        if current_info_gain > max_info_gain:
                            # update the dictionary values
                            perfect_cleave['factor_idx'] = idx
                            perfect_cleave['cut_off'] = cut_off
                            perfect_cleave['data_f'] = data_f
                            perfect_cleave['data_s'] = data_s
                            perfect_cleave['impurity_loss'] = current_info_gain
                            max_info_gain = current_info_gain

        

        else :
            # outer loop to iterate over possible deciding factors
            for idx in range(num_factors):
                factor_values = data[:,idx]
                cut_off_values = np.unique(factor_values)
                
                # inner loop to iterate over possible cut_off values of the current deciding factor
                for cut_off in cut_off_values:
                    # splitting the data in the node based on the current factor and current threshold
                    data_f, data_s = self.cleave(data, idx, cut_off)
                    if len(data_f)>0 and len(data_s)>0:
                        parent_y = data[:,-1]
                        f_child_y = data_f[:,-1]
                        s_child_y = data_s[:,-1]
                        current_info_gain = self.info_gain(parent_y, f_child_y, s_child_y)
                        # checking if the information gain of the current split is more than that of previous split
                        if current_info_gain > max_info_gain:
                            # update the dictionary values
                            perfect_cleave['factor_idx'] = idx
                            perfect_cleave['cut_off'] = cut_off
                            perfect_cleave['data_f'] = data_f
                            perfect_cleave['data_s'] = data_s
                            perfect_cleave['impurity_loss'] = current_info_gain
                            max_info_gain = current_info_gain

        return perfect_cleave
    
    '''METHOD 6 - find_leaf_type
    to determine the catogery represented by a leaf node in the decision tree'''
    def find_leaf_type(self, y):
        y = list(y)
        return max(y, key = y.count)

    '''METHOD 7 - make_tree
    for building the decision tree by making nodes, splitting nodes and so on.
    This is a recursive function'''
    def make_tree(self, data, depth = 0):

        '''data : a 2D array. Its last column consists of target variable value.
                  Rest of the columns correspond to various features in the dataset'''
        
        # seperating feature matrix and target column from the data
        X, y = data[:,:-1], data[:,-1]
        num_samples, num_factors = np.shape(X)

        # checking for stopping conditions
        if num_samples >= self.min_data and depth <= self.max_depth:
            split_dict = self.perfect_cleave(data, num_factors)
            if split_dict['impurity_loss']>0:
                # add the factor_idx each time it is used for splitting
                self.used_factors.append(split_dict['factor_idx'])
                first_subtree = self.make_tree(split_dict['data_f'],depth = depth + 1)
                second_subtree = self.make_tree(split_dict['data_s'], depth = depth + 1)
                return Node(split_dict['factor_idx'], split_dict['cut_off'], first_subtree, second_subtree, split_dict['impurity_loss'])
        
        # create leaf node
        leaf_type = self.find_leaf_type(y)
        return Node(type = leaf_type)
    
    '''METHOD 8 - fit
    the method for training the catogerizer model using the training dataset'''
    
    def fit(self, X, y):
        '''X : represents the feature matrix of training dataset.
               It is a 2D numpy array.
           y : represents the column matrix of target variable in the training set.
               It is a 2D numpy array'''
        
        # joining X,y to get complete dataset
        dataset = np.concatenate((X,y), axis = 1)
        # setting up root node of the tree by feeding entire training set into make_tree function
        self.root = self.make_tree(dataset)

    '''METHOD 9 - find_type
    used for predicting the catogery of a given feature vector'''
    
    def find_type(self, x, tree):
        if tree.type != None:
            return tree.type
        factor_value = x[tree.factor_idx]
        if factor_value <= tree.cut_off:
            return self.find_type(x, tree.first)
        else:
            return self.find_type(x, tree.second)

    '''METHOD 10 - predict
    used for predicting the types of data points for a given dataset'''
    def predict(self, X):
        predictions = [self.find_type(x, self.root) for x in X]

        # return a list of predictions corresponding to each feature vector in the dataset
        return predictions 
    

