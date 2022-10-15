#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from utilitarios.utilitarios import getDiretorio, pastaTweets, pastaTweetsMerge
import pandas as pd
import os
import tqdm
from pipeline.processamentoArquivos import ProcessamentoArquivos
import re


class Etapa_3(ProcessamentoArquivos):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
    
    def merge_datasets(self, periodo, grupo, diretorioTweetsMerge):
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

            self.gravar_tweet_lake(part + 1, tweet_lake, diretorioTweetsMerge)
            
    def limpar_dataset(self, periodo, grupo, diretorioTweetsMergeNormalizados):
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
            
            self.gravar_tweet_lake(part, tweet_lake, diretorioTweetsMergeNormalizados)   
