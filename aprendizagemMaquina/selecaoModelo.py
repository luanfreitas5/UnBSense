#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from aprendizagemMaquina.machineLearning import MachineLearning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import MinMaxScaler
from warnings import filterwarnings
filterwarnings('ignore')


class SelecaoModelo(MachineLearning):

    def __init__(self):
        super().__init__()
        
    def selecaoParametros(self):
        
        self.inicializacao()
        
        logspace_range = [0.0001, 0.001,  0.01,   0.1]
        c_range = [0.001,  0.01,   0.1,   1] 
        max_depth_range = np.arange(5, 16, 5)  # [ 5 10 15]
        n_neighbors_range = np.arange(3, 20, 2)  # [ 3  5  7  9 11 13 15 17 19]
        
        parametrosLinearDiscriminantAnalysis = {'solver':['svd', 'lsqr', 'eigen'], 'shrinkage': logspace_range}
        parametrosDecisionTreeClassifier = {'splitter':["best", "random"], 'max_depth':max_depth_range}
        parametrosRandomForestClassifier = {'max_depth':max_depth_range}
        parametrosGradientBoostingClassifier = {'loss': ['deviance', 'exponential'], 'max_depth':max_depth_range}
        parametrosKNeighborsClassifier = {'n_neighbors':n_neighbors_range, 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean', 'manhattan']}
        parametrosGaussianNB = {'var_smoothing':logspace_range}
        parametrosMLPClassifier = {'hidden_layer_sizes':[(10, 30, 10), (20,)], 'activation':['tanh', 'relu'], 'solver':['sgd', 'adam']}
        parametrosLogisticRegression = {'penalty':['l1', 'l2', 'elasticnet'], 'C':c_range}
        parametrosSVC = {'C':c_range, 'gamma':logspace_range}
        
        dicionarioParametros = {LinearDiscriminantAnalysis(): parametrosLinearDiscriminantAnalysis,
                                DecisionTreeClassifier(criterion='entropy'): parametrosDecisionTreeClassifier,
                                RandomForestClassifier(n_estimators=100, criterion='entropy'): parametrosRandomForestClassifier,
                                GradientBoostingClassifier(criterion='squared_error'): parametrosGradientBoostingClassifier,
                                KNeighborsClassifier(): parametrosKNeighborsClassifier,
                                GaussianNB(): parametrosGaussianNB,
                                MLPClassifier(max_iter=50): parametrosMLPClassifier,
                                LogisticRegression(): parametrosLogisticRegression,
                                SVC(kernel='rbf', probability=True): parametrosSVC
                                }
        
        resultado = ""
        
        for classificador, listaParametros in dicionarioParametros.items():
            resultado += self.gridSearch(classificador, listaParametros)
        
        if not os.path.exists('results/'):
            os.makedirs('results/')
            
        selecaoModelos = open('results/selecaoModelos.txt', "w")
        selecaoModelos.write(resultado)
        selecaoModelos.close()
        
    def gridSearch(self, classificador, listaParametros):
        clf = GridSearchCV(classificador, listaParametros, cv=self.cv, verbose=1, scoring='balanced_accuracy')
        clf.fit(self.x_train, self.y_train)
            
        resultado = f"{clf.best_estimator_.__class__.__name__}\nBest Params {clf.best_params_}\n\n"
            
        print(resultado, "")
        
        return resultado
