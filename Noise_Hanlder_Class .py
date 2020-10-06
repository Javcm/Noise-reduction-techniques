import warnings
warnings.filterwarnings("ignore")
import glob
import pandas as pd
import numpy as np
from random import sample,seed,shuffle
from numpy import zeros, sum as sumnp
from scipy.stats import mode
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


#Class that implement 6 methods to handle the noise instances
class NoiseHandling(BaseEstimator):s
    def __init__(self, method = 'ENN', n_neighbors = 1, n_splits = 5, 
                 filter_type = 'majority', random_state = None):
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_splits = n_splits
        self.filter_type = filter_type
        self.random_state = random_state
        
    def __filtering_rule(self, y_pred, y):
        if self.filter_type == 'majority':
            y_pred_mode = mode(y_pred, axis = 1)[0].ravel()
            return y_pred_mode == y
        elif self.filter_type == 'consensus':
            y_pred_bool = y_pred == y[:, None]
            return sumnp(y_pred_bool, axis = 1) > 0
        else:
            raise ValueError('Undefined rule')

    #Iterative edited-K Nearest Centroid Neighborhood
    def __iekncn_fit(self, X, y):
        f = True
        y=pd.DataFrame(y)
        y_aux=y

        while f == True:
            clf = NearestCentroid()
            clf.fit(X, y_aux.values.ravel())
            labels = clf.predict(X)
            y_pred =labels.ravel()

            if (labels==y_aux.values.ravel()).sum()!=len(y_aux):
                X=X[labels==y_aux.values.ravel()]
                y_aux=y_aux[labels==y_aux.values.ravel()]
            else:
                f=False
        indexes = np.arange(len(y))
        j = 0
        Filter=[]

        for i in range(len(indexes)):
            if indexes[i]==y.index[j]:
                Filter.append(True)
                j=j+1
            else:
                Filter.append(False)
        self.filter_ = Filter
        return self
            
            
    #Complementary Neural Network
    def __cmtnn_fit(self,X,y):  
        
        y_comp=y
        shuffle(y_comp)
        mt = MLPClassifier(random_state=1, max_iter=500).fit(X,y)
        mc = MLPClassifier(random_state=1, max_iter=500).fit(X,y_comp)
        y_t = mt.predict(X)
        y_c = mc.predict(X)        
        self.filter_ = (y!=y_t)&(y!=y_c)
        
        return self

    #Decremental Reduction Optimization Procedure
    def __drop_fit(self, X, y):
        S = X
        nn_search = NearestNeighbors(n_neighbors = 5 + 1)
        nn_search.fit(S)
        neigh_ind = nn_search.kneighbors(S, return_distance = False)
        Filter=[]

        for i in range(np.shape(neigh_ind)[0]):
            #associates.append(X[neigh_ind[i,1:]])
            neigh = KNeighborsClassifier(n_neighbors=5+1)
            neigh.fit(X, y)
            ni =  (neigh.predict(X[neigh_ind[i,:]])==y[neigh_ind[i,:]]).sum()
            ne =  (neigh.predict(X[neigh_ind[i,1:]])==y[neigh_ind[i,1:]]).sum()
            Filter.append(ni<=ne)
        self.filter_ = list(map(operator.not_, Filter))
        return self
    
    # Cross Validates Commites Filter 
    def __cvcf_fit(self,X,y):
        skf = StratifiedKFold(n_splits = self.n_splits, shuffle = True,
                              random_state = self.random_state)
        predictions = zeros((X.shape[0], 3))
        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X)
            
        self.filter_ = predictions == y
        return self   
    
    #Edited Nearest Neighbor
    def __enn_fit(self, X, y):
        nn_search = NearestNeighbors(n_neighbors = self.n_neighbors + 1)
        nn_search.fit(X)
        neigh_ind = nn_search.kneighbors(X, return_distance = False)[:, 1:]
        labels = y[neigh_ind]
        y_pred = mode(labels, axis = 1)[0].ravel()
        self.filter_ = y == y_pred
        return self
    
    #Ensemble Filter
    def __ef_fit(self, X, y):
        skf = StratifiedKFold(n_splits = self.n_splits, shuffle = True,
                              random_state = self.random_state)
        predictions = zeros((X.shape[0], 3))
        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            
            learning_algorithms = [DecisionTreeClassifier(), \
                                  KNeighborsClassifier(), \
                                  LogisticRegression()]
            for index, model in enumerate(learning_algorithms):
                model.fit(X_train, y_train)
                predictions[test_idx, index] = model.predict(X_test)
                
        self.filter_ = self.__filtering_rule(predictions, y)
        return self 
    
    def __filter_resample(self, X, y):
        return X[self.filter_], y[self.filter_]
            
    def __resample(self, X, y):
        X = check_array(X)
        
        if (self.method == 'ENN') or (self.method == 'EF') or (self.method == 'IEKNCN') or (self.method == 'DROP') or (self.method == 'CMTNN') or (self.method == 'CVCF'):
            return self.__filter_resample(X, y)
        else:
            raise ValueError('Undefined method')
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        if self.method == 'ENN':
            return self.__enn_fit(X, y)
        elif self.method == 'EF':
            return self.__ef_fit(X, y)
        elif self.method == 'IEKNCN':
            return self.__iekncn_fit(X, y)
        elif self.method == 'DROP':
            return self.__drop_fit(X, y)        
        elif self.method == 'CMTNN':
            return self.__cmtnn_fit(X, y)
        elif self.method == 'CVCF':
            return self.__cvcf_fit(X, y)
        else:
            raise ValueError('Undefined method')
            
    def fit_resample(self, X, y):
        return self.fit(X, y).__resample(X, y)

