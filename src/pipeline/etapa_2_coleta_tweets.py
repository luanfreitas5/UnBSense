#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from snscrape.base import ScraperException
import snscrape.modules.twitter as sntwitter
import tqdm
import itertools
import pandas as pd
import os
import pickle
from utilitarios.utilitarios import getDiretorioInacessivel, getDiretorio, pastaCandidatos, pastaTweets
from pipeline.etapa_1_busca_usuarios import Etapa_1_Busca_Usuarios


class Etapa_2_Coleta_Tweets(Etapa_1_Busca_Usuarios):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
        
    def etapa_2_coletarTweets(self, periodo, grupo, diretorioTweets):
        """Etapa 2 Coletar Tweets de Usuarios

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioTweets (str): _description_
        """        
        since, until = self.get_intervalo_periodo(periodo)
        listaCandidatos = self.get_lista_usuarios(periodo, grupo)
        pbar = tqdm.tqdm(listaCandidatos, colour='green')
        
        for candidato in pbar:
            pbar.set_description(f"Extraindo Tweets - {periodo} {since} - {until} {grupo}")
            self.etapa_2_coletar_tweets_recursiva(candidato, grupo, diretorioTweets, since, until)
            
    def etapa_2_coletar_tweets_recursiva(self, candidato, grupo, diretorioTweets, since, until):
        """_summary_

        Args:
            candidato (str): _description_
            grupo (str): _description_
            diretorioTweets (str): _description_
            since (str): _description_
            until (str): _description_
        """
        try:
            if not os.path.isfile(f'{diretorioTweets}{candidato}_fulljson.csv'):
                tweets = sntwitter.TwitterSearchScraper(f'from:{candidato} since:{since} until:{until} -filter:retweets').get_items()
                tweets, tweetsCopy = itertools.tee(tweets)
                sliced_scraped_tweets = itertools.islice(tweets, len(list(tweetsCopy)))
                tweets_df = pd.DataFrame(sliced_scraped_tweets)
                
                if not tweets_df.empty:
                    
                    tweets_df = self.renomear_atributos_df(tweets_df)
                    tweets_df.to_pickle(f'{diretorioTweets}{candidato}_fulljson.pickle', protocol=pickle.HIGHEST_PROTOCOL)
                    
                else:
                    file = open(f'{getDiretorioInacessivel(grupo)}candidatosBloqueados.txt', "a")
                    file.write(f'{candidato}\n')
                    file.close()
                            
        except ScraperException:
            self.etapa_2_coletar_tweets_recursiva(candidato, grupo, diretorioTweets, since, until)
            
    def get_lista_usuarios(self, periodo, grupo) -> list:
        """_summary_

        Args:
            periodo (str): _description_
            grupo (str): _description_

        Returns:
            list: _description_
        """
        diretorioCandidatos = getDiretorio(pastaCandidatos)
        diretorioTweets = getDiretorio(pastaTweets, periodo, grupo)
        colunasSelecionadas = ['id', 'id_screen_name', 'screen_name']
        df = pd.read_csv(f'{diretorioCandidatos}{periodo}_{grupo}.csv', delimiter=';', parse_dates=['created_at'], index_col='created_at', low_memory=False)[colunasSelecionadas]
        
        candidatosCadastrados = [file for file in os.listdir(diretorioTweets) if file.endswith('.csv')]
        candidatosCadastrados = set([file.rsplit('_', 1)[0] for file in candidatosCadastrados])
        
        if os.path.isfile(f'{getDiretorioInacessivel(grupo)}candidatosBloqueados.txt'):
            file = open(f'{getDiretorioInacessivel(grupo)}candidatosBloqueados.txt', 'r')
            candidatosinacessiveis = set(file.read().splitlines())
            file.close()
        else:
            candidatosinacessiveis = set()
        
        df.drop_duplicates(subset='id_screen_name', keep='first', inplace=True)
        df = df[~df['screen_name'].isin(candidatosCadastrados)]
        df = df[~df['screen_name'].isin(candidatosinacessiveis)]
        
        listaCandidatos = df['screen_name'].tolist()
        
        return listaCandidatos