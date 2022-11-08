#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
import re, os, pickle
from unidecode import unidecode
import pandas as pd
from warnings import filterwarnings
from utilitarios.utilitarios import getDiretorio
from utilitarios.utilitarios import pastaTweets, pastaTweetsMerge
import tqdm
filterwarnings('ignore')


class ProcessamentoArquivos():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        self.parts = 50
        
    def gravar_tweet_lake(self, part, tweet_lake, diretorioTweetsMerge):
        """_summary_

        Args:
            part (_type_): _description_
            tweet_lake (_type_): _description_
            diretorioTweetsMerge (_type_): _description_
        """
        
        # build a new datalake from scratch
        if part in range(1, 10):
            tweet_lake.to_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part), protocol=pickle.HIGHEST_PROTOCOL)
        elif part in range(10, 100):
            tweet_lake.to_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            tweet_lake.to_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part), protocol=pickle.HIGHEST_PROTOCOL)
            
    def converter_minuscula(self, processing_series):
        """_summary_

        Args:
            processing_series (_type_): _description_

        Returns:
            _type_: _description_
        """

        # removing all diacritical marks and lowering the case
        processing_seriesNova = pd.Series([str(tweet).lower() for tweet in processing_series.values],
                                      index=processing_series.index)
        
        # Verifica se existe letra maiucula
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'[\u0041-\u005A]', text) != None else False 
                                                     for text in processing_seriesNova.values]]
        if not serieProcessada.empty:
            print("Converção de letras maiuscula para miniculas deu erro")
            exit(1)
        else:
            pass
            # print("Converção de letras maiuscula para miniculas realizado com sucesso")
            
        return processing_seriesNova
    
    def remover_links_hastags_espacos_risadas(self, processing_seriesNova) -> pd.Series:
        """_summary_

        Args:
            processing_seriesNova (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        # removing links
        processing_seriesNova = pd.Series([tweet.split('http', 1)[0] for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'.http\S*', tweet) != None else False for tweet in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de links deu erro")
            exit(1)
        else:
            pass
            
        # removing hashtags 
        processing_seriesNova = pd.Series(processing_seriesNova.apply(lambda tweet: re.sub(r'#\S+', ' ', tweet)).values,
                              index=processing_seriesNova.index)
        processing_seriesNova = pd.Series(processing_seriesNova.apply(lambda tweet: re.sub('#', ' ', tweet)).values,
                                      index=processing_seriesNova.index)
        
        serieProcessada = processing_seriesNova.loc[[True if '#' in text else False for text in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de hashtags deu erro")
            exit(1)
        else:
            pass
            
        # cleaning irregular laughter marks
        processing_seriesNova = pd.Series([re.sub(r'.*kkk*\S*', ' haha ', tweet) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        processing_seriesNova = pd.Series([re.sub(r'.*(rs)(rs)+\S*', ' haha ', tweet) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        processing_seriesNova = pd.Series([re.sub(r'.*(ha|hu)\S*(ha|hu)+\S*', ' haha ', tweet) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        # removing all non-regular whitespaces
        processing_seriesNova = pd.Series([' '.join(tweet.split()) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        processing_seriesNova = pd.Series([' '.join(tweet.split()) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        return processing_seriesNova
    
    def remover_nao_lantinos(self, processing_seriesNova) -> pd.Series:
        """_summary_

        Args:
            processing_seriesNova (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        # removing all non-latin characters
        processing_seriesNova = pd.Series([re.sub(r'[^\u0061-\u007A\u0020]', ' ', unidecode(str(tweet).lower())) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'[^\u0061-\u007A\u0020]', tweet) != None else False for tweet in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de caracteres não latinos deu erro")
            exit(1)
        else:
            pass
            
        return processing_seriesNova
    
    
