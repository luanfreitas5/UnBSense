#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from snscrapeApp.twitter import Twitter
from snscrape.base import ScraperException
import snscrape.modules.twitter as sntwitter
from utilitarios.utilitarios import getDiretorioInacessivel
import os
import pandas as pd
import pickle
import tqdm
import itertools


class Etapa_2(Twitter):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
        
    def coletar_tweets(self, periodo, grupo, diretorioTweets):
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
            self.executar_coleta_tweets(candidato, grupo, diretorioTweets, since, until)
            
    def executar_coleta_tweets(self, candidato, grupo, diretorioTweets, since, until):
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
            self.extrairTweetsLinhaTempoCandidato(candidato, grupo, diretorioTweets, since, until)