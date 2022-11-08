#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import random
# from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from unidecode import unidecode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from sklearn.model_selection import cross_val_score
from utilitarios.utilitarios import getDiretorioPlots, getDiretorioMatrizesConfusoes
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection._validation import validation_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.ensemble._gb import GradientBoostingClassifier
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.svm._classes import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


class MachineLearning():

    def __init__(self):
        '''
        Constructor
        '''
        
        self.metricas = ['Precisão', 'Recall', 'F1-Score',
                         'Especificidade', 'Valor Preditivo Negativo',
                         "Media Validação Cruzada", "Desvio Padrão Validação Cruzada",
                         'Acurácia', 'Acurácia Balanceada']
        
        self.labelEncoder = LabelEncoder()
        self.cv = KFold(n_splits=10, shuffle=True, random_state=42)
        
    def inicializacao(self):
        
        '''
        random_state
        Controla o embaralhamento aplicado aos dados antes de aplicar a divisão. 
        Passe um int para saída reproduzível em várias chamadas de função.
        Se um valor fixo for atribuído como random_state = 42, 
        não importa quantas vezes você execute seu código, 
        o resultado seria o mesmo, ou seja, 
        os mesmos valores nos conjuntos de dados de treinamento e teste.
        '''
        
        random.seed(42)
        
        # Carregamento da Base de Dados
        #self.df = pd.read_pickle('datasets/twitterbase_reduced.pickle')
        self.df = pd.read_pickle('datasets/twitterbase.pickle')
        
        self.features_names = self.df.drop(columns=['classe']).columns
        
        self.df['classe'] = self.labelEncoder.fit_transform(self.df['classe'])
        
        # Separando as features e rotulos de classe
        self.x = np.ascontiguousarray(self.df.drop(columns=['classe']).copy())
        self.y = np.ascontiguousarray(self.df['classe'])
        
        # Dividindo a base de dados em Treino (50%), Validação (30%) e Teste (20%)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=0.15, random_state=42)
        
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                              test_size=0.15, random_state=42)
        
        self.x_train, self.y_train = np.ascontiguousarray(self.x_train), np.ascontiguousarray(self.y_train)
        self.x_val, self.y_val = np.ascontiguousarray(self.x_val), np.ascontiguousarray(self.y_val)
        self.x_test, self.y_test = np.ascontiguousarray(self.x_test), np.ascontiguousarray(self.y_test)
        
        # Dicionario de pesos das classes
        self.class_weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)))
        
        # Dataframe dos resultado de metricas dos algoritmos
        self.df_resultados = pd.DataFrame(columns=self.metricas)
        self.df_resultados.index.name = "Classificador"
        
        # Matrizes Confusão
        self.listaMatrizConfusao = []
        self.listaEstimadoresEnsemble = []
        self.dicionarioModelos = {}
        
    def plotarMatrizesConfusoes(self, conjunto):
        '''
        Plota as matrizes de confusão balanceadas
        '''
        
        diretorio = getDiretorioMatrizesConfusoes(conjunto)
        
        for nome_classificador, matriz_confusao_relativa in self.listaMatrizConfusao:
            plt.figure(figsize=(10, 8))
            plt.title(f'Matriz de confusão {nome_classificador} --- {conjunto}')
            sns.heatmap(matriz_confusao_relativa, linewidths=.5, cmap='coolwarm', annot=True, fmt='.1%', vmin=0, vmax=1,
                        xticklabels=self.labelEncoder.classes_, yticklabels=self.labelEncoder.classes_)
            plt.xlabel("Predito")
            plt.ylabel("Real")
            plt.savefig(f'{diretorio}{conjunto}_matrizConfusao_{unidecode(nome_classificador.replace(" ", "_")).lower()}')
            # plt.show()
            plt.close()
    
    def plotarCurvaRoc(self, x, y, set):
        
        '''
        Plota Curva de Roc
        '''
        
        diretorio = getDiretorioPlots()
            
        ax = plt.gca()
            
        for nome_classificador, classificador in self.dicionarioModelos.items():
            if not 'Ensemble' in nome_classificador:
                display = RocCurveDisplay.from_estimator(classificador, x, y, ax=ax, name=nome_classificador)
        
        _ = display.ax_.set_title(f"Roc Curve --- {set}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(f'{diretorio}/{set}_curvaRoc')
        # plt.show()
        plt.close()
    
    def plotarCurvaPrecisaoRecall(self, x, y, set):
        
        '''
        Plota Curva de Precisao e Recall
        '''
        
        diretorio = getDiretorioPlots()
            
        ax = plt.gca()
            
        for nome_classificador, classificador in self.dicionarioModelos.items():
            if not 'Ensemble' in nome_classificador:
                display = PrecisionRecallDisplay.from_estimator(classificador, x, y, ax=ax, name=nome_classificador)
        
        _ = display.ax_.set_title(f"Precision-Recall Curve --- {set}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(f'{diretorio}{set}_curvaPrecisaoRecall')
        # plt.show()
        plt.close()
        
    def avaliacao(self, classificador, predicao, x, y, nome_classificador):
        '''
        Computa as mericas de avaliação dos clasficadores
        '''
        '''
        precisao = np.round(precision_score(y, predicao) * 100, 2)
        revocacao = np.round(recall_score(y, predicao) * 100, 2)
        f1score = np.round(f1_score(y, predicao) * 100, 2)
        acuracia = np.round(accuracy_score(y, predicao) * 100, 2)
        acuracia_balanceada = np.round(balanced_accuracy_score(y, predicao) * 100, 2)
        '''
        
        matriz_confusao = confusion_matrix(y, predicao)
        matriz_confusao_relativa = matriz_confusao / matriz_confusao.sum(axis=1, keepdims=True)
        
        self.listaMatrizConfusao.append((nome_classificador, matriz_confusao_relativa))
        
        tp, fn, fp, tn = matriz_confusao.ravel()
        
        precisao = np.round((tp / (tp + fp)) * 100, 2)
        revocacao = np.round((tp / (tp + fn)) * 100, 2)
        f1score = np.round((2 * precisao * revocacao) / (precisao + revocacao), 2)     
        especificidade = np.round((tn / (tn + fp)) * 100, 2)
        valor_preditivo_negativo = np.round((tn / (tn + fn)) * 100, 2)
        
        acuracia = np.round(((tp + tn) / (tp + fn + fp + tn)) * 100, 2)
        acuracia_balanceada = np.round((revocacao + especificidade) / 2, 2)
        
        mediaValidacaoCruzada = np.round(cross_val_score(classificador, x, y).mean() * 100, 2)
        desvioPadraoValidacaoCruzada = np.round(cross_val_score(classificador, x, y).std() * 100, 2)
        
        return [precisao, revocacao, f1score,
                especificidade, valor_preditivo_negativo,
                mediaValidacaoCruzada, desvioPadraoValidacaoCruzada,
                acuracia, acuracia_balanceada]
        
    def plotarCurvaValidacao(self):
        self.inicializacao()
        analiseDiscriminanteLinear = LinearDiscriminantAnalysis(solver='lsqr')
        arvoreDecisao = DecisionTreeClassifier(class_weight=self.class_weight, criterion='entropy', splitter="best")
        florestaRandomica = RandomForestClassifier(class_weight=self.class_weight, criterion='entropy', n_estimators=250)
        gradientBoosting = GradientBoostingClassifier(criterion='mse', loss='exponential')
        knn = KNeighborsClassifier(metric='manhattan', weights='distance')
        naiveBayes = GaussianNB()
        perceptronMulticamadas = MLPClassifier(activation='tanh', hidden_layer_sizes=(10, 30, 10), solver='adam')
        regressaoLogistica = LogisticRegression(class_weight=self.class_weight, penalty='l2')
        svm = SVC(kernel='rbf', class_weight=self.class_weight, probability=True)
        
        logspace_range = np.logspace(-10, -1, 10)
        max_depth_range = np.arange(1, 16)
        n_neighbors_range = np.arange(1, 20, 2)
        max_iter_range = np.arange(5, 201, 5)
        
        listaModelos = [[analiseDiscriminanteLinear, logspace_range, 'shrinkage', 'Análise Discriminante Linear'],
                        # [arvoreDecisao, max_depth_range, 'max_depth', 'Arvore de Decisão'],
                        # [florestaRandomica, max_depth_range, 'max_depth', 'Floresta Randômica'],
                        # [gradientBoosting, max_depth_range, 'max_depth', 'Gradient Boosting'],
                        [knn, n_neighbors_range, 'n_neighbors', 'KNN'],
                        [naiveBayes, logspace_range, 'var_smoothing', 'Naive Bayes'],
                        [perceptronMulticamadas, max_iter_range, 'max_iter', 'Perceptron Multicamadas'],
                        [regressaoLogistica, logspace_range, 'C', 'Regressão Logistica'],
                        [svm, logspace_range, 'gamma', 'SVM']
            ]
        
        for modelo, param_range, param_name, nomeModelo in listaModelos:
            print(nomeModelo)
            self.curvaValidacao(modelo, param_range, param_name, nomeModelo)
                
    def curvaValidacao(self, modelo, param_range, param_name, nomeModelo, logx=False):

        # Validate the param_range
        param_range = np.asarray(param_range)
        
        # draw the curves on the current axes
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        
        # Get the colors for the train and test curves
        colors = resolve_colors(n_colors=2)
        
        train_scores, test_scores = validation_curve(modelo, self.x, self.y, cv=self.cv, n_jobs=1,
                                                     scoring='balanced_accuracy',
                                                     param_range=param_range, param_name=param_name)
        
        # compute the mean and standard deviation of the training data
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        
        # compute the mean and standard deviation of the test data
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Specify the curves to draw and their labels
        labels = ("Acurácia Balenceada de Treinamento", "Acurácia Balenceada da Validação Cruzada")
        curves = ((train_scores_mean, train_scores_std), (test_scores_mean, test_scores_std))
        
        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            ax[0].fill_between(param_range, mean - std, mean + std, alpha=0.25, color=colors[idx])
        
        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            ax[0].plot(param_range, mean, "o-", color=colors[idx], label=labels[idx])
        
        if logx:
            ax[0].set_xscale("log")
            
        # Set the title of the figure
        ax[0].set_title(f"Acurácia Balenceada --- {nomeModelo}")
        
        # Add the legend
        ax[0].legend(frameon=True, loc="best")
        
        # Set the axis labels
        ax[0].set_xlabel(param_name)
        ax[0].set_ylabel("Acurácia Balenceada")
        
        ###################################################################################################
        
        train_scores, test_scores = validation_curve(modelo, self.x, self.y, cv=self.cv, n_jobs=1,
                                                     scoring='neg_mean_squared_error',
                                                     param_range=param_range, param_name=param_name)
        
        # compute the mean and standard deviation of the training data
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        
        # compute the mean and standard deviation of the test data
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Specify the curves to draw and their labels
        labels = ("Loss de Treinamento", "Loss da Validação Cruzada")
        curves = ((train_scores_mean, train_scores_std), (test_scores_mean, test_scores_std))    
        
        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            ax[1].fill_between(param_range, mean - std, mean + std, alpha=0.25, color=colors[idx])
        
        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            ax[1].plot(param_range, mean, "o-", color=colors[idx], label=labels[idx])
            
        if logx:
            ax[1].set_xscale("log")
            
        # Set the title of the figure
        ax[1].set_title(f"Loss --- {nomeModelo}")
        
        # Add the legend
        ax[1].legend(frameon=True, loc="best")
        
        # Set the axis labels
        ax[1].set_xlabel(param_name)
        ax[1].set_ylabel("Loss")
        # plt.grid(True)
        diretorio = 'plots/curvasValidadoes/'
        plt.savefig(f'{diretorio}curvasValidadoes_{unidecode(nomeModelo.replace(" ", "_")).lower()}')
        # plt.show()

