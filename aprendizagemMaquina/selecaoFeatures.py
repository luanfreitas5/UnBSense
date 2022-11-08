#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import ticker
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from utilitarios.utilitarios import getDiretorioPlots
from warnings import filterwarnings
filterwarnings('ignore')


class SelecaoFeatures():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def inicializacao(self):
        df = pd.read_csv('datasets/twitterbase_pre_pandemia_zscore.csv', delimiter=';')
        
        # checa se todo datafframe possui valor ausente
        #print(df.isnull().any().any())
        
        # checa cada columa do datafframe possui valor ausente
        # print(df.isnull().any())
        
        self.x = df.drop(columns=['classe']).copy()
        self.y = df['classe']
        
    def plotarPCA(self, pca):
        # https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
        
        plt.rcParams["figure.figsize"] = (12, 6)
        
        fig, ax = plt.subplots()
        xi = np.arange(1, pca.n_components_ + 1, step=1)
        y = np.cumsum(pca.explained_variance_ratio_)
        
        plt.ylim(0.0, 1.1)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')
        
        plt.xlabel('Number of Components')
        plt.xticks(np.arange(1, pca.n_components_ + 1, step=1))  # change from 0-based array index to 1-based human-readable label
        plt.ylabel('Cumulative variance (%)')
        plt.title('The number of components needed to explain variance')
        
        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.text(0.5, 0.85, f'95% cut-off threshold\nn_components_ {pca.n_components_}', color='red', fontsize=16)
        
        #ax.grid(axis='x')
        plt.grid(True)
        diretorio = getDiretorioPlots()
        
        plt.savefig(f'{diretorio}pca_variance_sums')
        
        plt.show()
        
    def pca(self):
        self.inicializacao()
        pca = PCA(n_components=0.98).fit(self.x)
        data = pca.transform(self.x)
        
        features = [ f'Feature {index+1}' for index in range(data.shape[1]) ]
        df_reduzido = pd.DataFrame(data=data, columns=features)#.abs()
        df_reduzido['classe'] = self.y
    
        df_reduzido.to_csv('datasets/twitterbase_pre_pandemia_zscore_reduced.csv', index=False, sep=';')
            
        self.plotarPCA(pca)
        

        
