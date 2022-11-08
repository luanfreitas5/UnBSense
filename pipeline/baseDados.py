#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
import pickle
import pandas as pd
from utilitarios.utilitarios import getDiretorio
from sklearn.preprocessing._data import StandardScaler, MinMaxScaler
from utilitarios.utilitarios import pastaCandidatos, pastaTweets, pastaTweetsMerge, pastaTweetsMergeNormalizados, pastaCaracteristicas


class BaseDados():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        pass
                
    def normalizarVetoresCaracteristicas(self, listaGrupos, periodo):
        """_summary_

        Args:
            listaGrupos (list): _description_
            periodo (str): _description_
        """
        for grupo in listaGrupos:
            diretorio = getDiretorio(pastaCaracteristicas, periodo, grupo)
            df = pd.read_pickle('{}featurevector_df.pickle'.format(diretorio))
            df = self.normalizar(df)
            df['classe'] = [grupo] * len(df)
            df = df.sort_index(axis=0)
            df.to_csv('{}featurevector_df_normalized.csv'.format(diretorio), sep=';')
        
    def normalizarBaseDados(self, df, periodo):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            periodo (str): _description_
        """
        zscore = StandardScaler()
        data_zscore = zscore.fit_transform(df.iloc[:,:-1])
        df_zscore = pd.DataFrame(data=data_zscore)
        df_zscore['classe'] = df['classe']
        df_zscore.columns = df.columns
        
        df_zscore.to_csv(f'datasets/twitterbase_{periodo}_zscore.csv', sep=';', index=False)
        print(f'datasets/twitterbase_{periodo}_zscore.csv criada')
        
        df_minmax = MinMaxScaler()
        data_minmax = df_minmax.fit_transform(df.iloc[:,:-1])
        df_minmax = pd.DataFrame(data=data_minmax)
        df_minmax['classe'] = df['classe']
        df_minmax.columns = df.columns
        
        df_minmax.to_csv(f'datasets/twitterbase_{periodo}_minmax.csv', sep=';', index=False)
        print(f'datasets/twitterbase_{periodo}_minmax.csv criada')
    
    def construirBaseDadosTwitter(self, periodo, diretorioDepressao, diretorioControle):
        """_summary_

        Args:
            periodo (str): _description_
            diretorioDepressao (str): _description_
            diretorioControle (str): _description_
        """
        dfDepressao = pd.read_csv(f'{diretorioDepressao}{periodo}_depressao_vetoresCaracteristicas.csv', sep=';', index_col=0)
        dfDepressao['classe'] = ['depressao'] * len(dfDepressao)
        
        dfControle = pd.read_csv(f'{diretorioControle}{periodo}_controle_vetoresCaracteristicas.csv', sep=';', index_col=0)
        dfControle['classe'] = ['controle'] * len(dfControle)
        
        df = pd.concat([dfDepressao, dfControle])
        df.sort_index(inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(f'datasets/twitterbase_{periodo}.csv', sep=';', index=False)
        print(f'Dataset twitterbase_{periodo}.csv criada')
        
        self.normalizarBaseDados(df, periodo)
