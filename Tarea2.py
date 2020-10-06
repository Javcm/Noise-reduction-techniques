import warnings
warnings.filterwarnings("ignore")
import glob
import pandas as pd
import numpy as np
from random import sample,seed,shuffle
np.random.seed(42)
random_state = np.random.RandomState(42)
seed(a=42)

files = glob.glob("/Users/javie/OneDrive/Documentos/CIMAT/Sem 3/Preprocesamiento/Tarea 2/csv/*.csv")
Data_frames=[]
for file in files:    
    Data_frames.append(pd.read_csv(file))
Dfs5  = list( Data_frames[i] for i in [3,7,11])
Dfs10 = list( Data_frames[i] for i in [0,4,8])
Dfs15 = list( Data_frames[i] for i in [1,5,9])
Dfs20 = list( Data_frames[i] for i in [2,6,10])

#Import the libraries we are going to use
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
from imblearn.pipeline  import Pipeline
from sklearn.preprocessing import StandardScaler
import operator
import scikit_posthocs as sp

#Class that implement 6 methods to handle the noise instances
class NoiseHandling(BaseEstimator):
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


#This function help us to make the classificacion and returns the score of the task, 
#takes as parameters the train and test sets, de classificator and the noise handle method

def classification(X_train,y_train,X_test,y_test,clf,method):
    classifier = clf
    y_score = classifier.fit(X_train, y_train).predict(X_test)
    pipe = Pipeline([('standardize', StandardScaler()),
                  ('noise_handler', NoiseHandling(method = method, filter_type = 'consensus',n_neighbors = 5)),
                  ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    score=pipe.score(X_test, y_test)
    return score
        
#This function makes k-fold to compute the different socres to all the methods to handle the noise, returns an array        
def scores_rsfk(X,y,clf):
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    scores1,scores2,scores3,scores4,scores5=[],[],[],[],[]
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #Classifiers "ENN","EF","IEKNCN","DROP","CVCF"
        scores1.append(classification(X_train,y_train,X_test,y_test,clf,"ENN"))
        scores2.append(classification(X_train,y_train,X_test,y_test,clf,"EF"))
        scores3.append(classification(X_train,y_train,X_test,y_test,clf,"IEKNCN"))
        scores4.append(classification(X_train,y_train,X_test,y_test,clf,"DROP"))
        scores5.append(classification(X_train,y_train,X_test,y_test,clf,"CVCF"))
        
    scores = np.array([scores1,scores2,scores3,scores4,scores5]).T
    return scores

#This function compute the Nemenyi Friedman statistics of a series of classifcators 
def statistics(df,clf):
    X = df.drop('Class', 1).to_numpy()
    y = df['Class']
    scores=scores_rsfk(X,y,clf)
    return sp.posthoc_nemenyi_friedman(scores).mean()

#Block of code that returns the AUC obtained from the statistical test for every base of a given level of noise
clf1 = DecisionTreeClassifier(random_state=42)
clf2 = SVC(kernel='linear', probability=True,max_iter=1000)
clf3 = KNeighborsClassifier(4)
classifiers=[clf1,clf2,clf3]
#table = pd.DataFrame([], columns=["ENN", "EF","IEKNCN","DROP","CVCF"])
DFS=[Dfs5,Dfs10,Dfs15,Dfs20]
AUC=[]
for df in DFS: 
  for i in range(3):    
    for clf in classifiers:
        AUC.append(statistics(df[i],clf))


#Block of code auxiliar to make the tables
#aux=aux15
#aux.append(np.reshape(aux,(9,5)).mean(axis=0))
#aux=np.around(aux,3)
#table = pd.DataFrame(aux, columns=["ENN", "EF","IEKNCN","DROP","CVCF"])
#table.index=["Decision Tree","Decision Tree","Decision Tree","SVC","SVC","SVC", "KNC","KNC","KNC","Promedio"]
#table["Database"]=["Heart","Heart","Heart","Pima","Pima","Pima","Ring","Ring","Ring",""]
#table
#print(table.to_latex())
#ranks=[]
#for i in range(9):    
#    ranks.append(list(ss.rankdata(aux[i])))
#print(np.around(np.array(ranks).mean(axis=0),2))
