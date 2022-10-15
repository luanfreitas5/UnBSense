#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from snscrapeApp.twitter import Twitter
from snscrape.base import ScraperException
import snscrape.modules.twitter as sntwitter
import os
import pandas as pd
import tqdm
import itertools


class Etapa_1(Twitter):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
        
    def buscar_usuarios(self, periodo, grupo, diretorioCandidatos):
        """Etapa 1 Buscar Usuarios

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioCandidatos (str): _description_
        """        
        since, until = self.get_intervalo_periodo(periodo)
        stringBuscas = self.get_palavras_chaves(grupo)
        pbar = tqdm.tqdm(stringBuscas, colour='green')
        
        for stringBusca in pbar:
            pbar.set_description(f"Buscando usuarios - {periodo} {grupo}")
            self.executar_busca_usuarios(stringBusca, periodo, grupo, diretorioCandidatos, since, until)
            
    def executar_busca_usuarios(self, stringBusca, periodo, grupo, diretorioCandidatos, since, until):
        """_summary_

        Args:
            stringBusca (str): _description_
            periodo (str): _description_
            grupo (str): _description_
            diretorioCandidatos (str): _description_
            since (str): _description_
            until (str): _description_
        """
        try:
            tweets = sntwitter.TwitterSearchScraper(f'{stringBusca} since:{since} until:{until} -filter:retweets lang:pt').get_items()
            if grupo == 'controle':
                sliced_scraped_tweets = itertools.islice(tweets, 10 ** 5)
            else:
                tweets, tweetsCopy = itertools.tee(tweets)
                sliced_scraped_tweets = itertools.islice(tweets, len(list(tweetsCopy)))
                
            tweets_df = pd.DataFrame(sliced_scraped_tweets)
            if not tweets_df.empty:
                
                tweets_df = self.renomear_atributos_df(tweets_df)
                
                if not os.path.isfile(f'{diretorioCandidatos}{periodo}_{grupo}.csv'):
                    tweets_df.to_csv(f'{diretorioCandidatos}{periodo}_{grupo}.csv', index=False, encoding='utf-8', sep=';')
                else:
                    tweets_df.to_csv(f'{diretorioCandidatos}{periodo}_{grupo}.csv', index=False, encoding='utf-8', sep=';', mode='a', header=False)
                
        except ScraperException:
            self.pesquisarCandidatos(stringBusca, periodo, grupo, diretorioCandidatos, since, until)