#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from utilitarios.utilitarios import getDiretorio, pastaTweets, pastaTweetsMerge
import os
import tqdm
import pandas as pd
import pickle
import re
import unidecode
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import nltk
import string
from wordcloud.wordcloud import STOPWORDS
from spacy.lang.pt.stop_words import STOP_WORDS


class Etapa_3_Arquivos():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        self.parts = 100       
    
        
    def etapa_3_1_mesclarDatasets(self, periodo, grupo, diretorioTweetsMerge):
        """Etapa 3.1 Mesclar arquivos

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioTweetsMerge (str): _description_
        """        
        diretorioTweets = getDiretorio(pastaTweets, periodo, grupo)
        listaTweets = [f"{diretorioTweets}{candidato}" for candidato in os.listdir(diretorioTweets)]
        listaTweets.sort()
        
        pbar = tqdm.tqdm(range(0, self.parts), colour='green')

        for part in pbar:
            '''
            Concat all database (only full json) files into a DataFrame
            Kernel is dying becuse the there are too many files to concat?
            '''
            
            pbar.set_description(f"Mesclando Arquivos - {periodo} {grupo}")
            
            # defines block indexing of files_list: start and end indexis
            start = int(len(listaTweets) / self.parts) * part
            
            if part == self.parts - 1:
                end = len(listaTweets) - 1
            else:
                end = int(len(listaTweets) / self.parts) * (part + 1) - 1
            
            flist = [fname for fname in listaTweets[start:end]]
            
            try:
                base = pd.concat([pd.read_pickle(file)
                                  for file in flist if '_fulljson' in file], sort=False)
        
            except Exception as e:
                print(e)
                
            try:
                base.index = pd.to_datetime(base['created_at'], infer_datetime_format=True, utc=True).dt.tz_localize(None)
                tweet_lake = base.sort_index().copy()
        
                del base
        
            except Exception as e:
                print(base.screen_name.unique(), e)

            # build a new datalake from scratch
            if part + 1 in range(0, 10):
                # tweet_lake.to_csv('{}processed_datalake_part00{}.csv'.format(diretorioTweetsMergeNormalizados, part + 1), sep=';')
                tweet_lake.to_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            elif part + 1 in range(10, 100):
                # tweet_lake.to_csv('{}processed_datalake_part0{}.csv'.format(diretorioTweetsMergeNormalizados, part + 1), sep=';')
                tweet_lake.to_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # tweet_lake.to_csv('{}processed_datalake_part{}.csv'.format(diretorioTweetsMergeNormalizados, part + 1), sep=';')
                tweet_lake.to_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            
    def etapa_3_2_limparDatasets(self, periodo, grupo, diretorioTweetsMergeNormalizados):
        """Etapa 3.2 Limpar arquivos
        unicode https://www.unicode.org/charts/PDF/

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioTweetsMergeNormalizados (str): _description_
        """
        
        diretorioTweetsMerge = getDiretorio(pastaTweetsMerge, periodo, grupo)
        pbar = tqdm.tqdm(range(1, self.parts + 1), colour='green')
        
        for part in pbar:
            
            pbar.set_description(f"Limpando Arquivos - {periodo} {grupo}")
                            
            if part in range(1, 10):
                tweet_lake = pd.read_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part))
            elif part in range(10, 100):
                tweet_lake = pd.read_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part))
            else:
                tweet_lake = pd.read_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part))
        
            full_text = tweet_lake.full_text
                
            full_text.name = 'full_text'
            
            full_text_novo = self.converter_minuscula(full_text)
                
            full_text_limpo = self.remover_links_hastags_espacos_risadas(full_text_novo)
            
            full_text_original = full_text_limpo.values.copy()
            
            tweet_lake.insert(loc=2, column='full_text_original', value=full_text_original)
            
            full_text_sem_lantino = self.remover_nao_lantinos(full_text_limpo)
                
            tweet_lake['full_text'] = full_text_sem_lantino.values.copy()
            
            tweet_lake.drop(list(tweet_lake.loc[[True if '' == text else False for text in tweet_lake.full_text.values]].index), inplace=True, errors='ignore')
            
            if part in range(1, 10):
                tweet_lake.to_pickle('{}processed_datalake_part00{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            elif part in range(10, 100):
                tweet_lake.to_pickle('{}processed_datalake_part0{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                tweet_lake.to_pickle('{}processed_datalake_part{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
    
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
        processing_seriesNova = pd.Series([re.sub(r'[^\u0061-\u007A\u0020]', ' ', unidecode.unidecode(str(tweet).lower())) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'[^\u0061-\u007A\u0020]', tweet) != None else False for tweet in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de caracteres não latinos deu erro")
            exit(1)
        else:
            pass
            
        return processing_seriesNova