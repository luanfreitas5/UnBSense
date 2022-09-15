#!/usr/bin/env python
# coding: utf-8

import re, os, pickle

from unidecode import unidecode
import xlrd
import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from threading import Lock, Thread
from warnings import filterwarnings
from utilitarios.utilitarios import  abrirBaseSerie, limparTela, getDiretorio
from utilitarios.utilitarios import pastaCandidatos, pastaTweets, pastaTweetsMerge, pastaTweetsMergeNormalizados, pastaCaracteristicas
import tqdm
filterwarnings('ignore')


class ProcessamentoArquivos():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.parts = 100
            
    def mergeDatasets(self, periodo, grupo, diretorioTweetsMerge):
        '''
        Etapa 3.1 Mesclar arquivos
        '''
        
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
                tweet_lake.to_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            elif part + 1 in range(10, 100):
                tweet_lake.to_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                tweet_lake.to_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            del tweet_lake
            
    def limparDataset(self, periodo, grupo, diretorioTweetsMergeNormalizados):
        
        """
        Etapa 3.2 Limpar arquivos
        unicode https://www.unicode.org/charts/PDF/
        """
        
        diretorioTweetsMerge = getDiretorio(pastaTweetsMerge, periodo, grupo)
        pbar = tqdm.tqdm(range(1, self.parts + 1), colour='green')
        
        for part in pbar:
            
            pbar.set_description(f"Limpando Arquivos - {periodo} {grupo}")
                            
            if part in range(1, 10):
                df = pd.read_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part))
            elif part in range(10, 100):
                df = pd.read_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part))
            else:
                df = pd.read_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part))
        
            processing_series = df.full_text
                
            processing_series.name = 'full_text'
            
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
                
            processing_seriesNovaRemovidaParte1 = self.removerCaracteresInrelevantesParte1(processing_seriesNova)
            
            full_text_original = processing_seriesNovaRemovidaParte1.values.copy()
            df.insert(loc=2, column='full_text_original', value=full_text_original)
            processing_seriesNovaRemovidaParte2 = self.removerCaracteresInrelevantesParte2(processing_seriesNovaRemovidaParte1)
                
            df['full_text'] = processing_seriesNovaRemovidaParte2.values.copy()
            df.drop(list(df.loc[[True if '' == text else False for text in df.full_text.values]].index), inplace=True, errors='ignore')
            
            if part in range(1, 10):
                df.to_pickle('{}processed_datalake_part00{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            elif part in range(10, 100):
                df.to_pickle('{}processed_datalake_part0{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                df.to_pickle('{}processed_datalake_part{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            
        del df
    
    def removerCaracteresInrelevantesParte1(self, processing_seriesNova):
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
    
    def removerCaracteresInrelevantesParte2(self, processing_seriesNova):
        # removing all non-latin characters
        processing_seriesNova = pd.Series([re.sub(r'[^\u0061-\u007A\u0020]', ' ', unidecode(str(tweet).lower())) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'[^\u0061-\u007A\u0020]', tweet) != None else False for tweet in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de caracteres não latinos deu erro")
            exit(1)
        else:
            pass
            
        return processing_seriesNova
    
