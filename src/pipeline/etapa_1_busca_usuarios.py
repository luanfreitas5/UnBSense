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
# https://github.com/igorbrigadir/twitter-advanced-search


class Etapa_1_Busca_Usuarios():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        # nomes de atributos do snscrape twitter
        self.atributos = ['date', 'id', 'content', 'idUser', 'username', 'user',
                         'likeCount', 'lang', 'media', 'outlinks', 'tcooutlinks',
                         'mentionedUsers', 'replyCount', 'inReplyToTweetId', 'inReplyToIdUser', 'inReplyToUsername', 'inReplyToUser',
                         'hashtags', 'cashtags', 'retweetCount', 'retweetedTweet',
                         'quoteCount', 'quotedTweet', 'coordinates', 'place',
                         'conversationId', 'renderedContent', 'source', 'sourceUrl', 'sourceLabel', 'url']
        
        # nomes de atributos do twitter API
        self.novos_atributos = ['created_at', 'id', 'full_text', 'id_screen_name', 'screen_name', 'user',
                               'favorite_count', 'lang', 'media', 'links', 'tcooutlinks',
                               'mentioned_users', 'reply_count', 'in_reply_to_tweet_id', 'in_reply_to_id_screen_name', 'in_reply_to_screen_name', 'in_reply_to_user',
                               'hashtags', 'cashtags', 'retweet_count', 'retweeted_tweet',
                               'quote_count', 'quoted_tweet', 'coordinates', 'place',
                               'conversation_id', 'rendered_content', 'source', 'source_url', 'source_label', 'url']
    
    def etapa_1_buscarUsuarios(self, periodo, grupo, diretorioCandidatos):
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
            self.etapa_1_buscar_usuarios_recursiva(stringBusca, periodo, grupo, diretorioCandidatos, since, until)
            
    def etapa_1_buscar_usuarios_recursiva(self, stringBusca, periodo, grupo, diretorioCandidatos, since, until):
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
            self.etapa_1_buscar_usuarios_recursiva(stringBusca, periodo, grupo, diretorioCandidatos, since, until)
            
    def get_intervalo_periodo(self, periodo) -> list:
        """_summary_

        Args:
            periodo (str): _description_

        Returns:
            list: _description_
        """
        if periodo == 'pre-pandemia':
            since, until = '2018-01-01', '2019-12-31'
        else:
            since, until = '2020-01-01', '2021-12-31'
            
        return [since, until]
    
    def renomear_atributos_df(self, tweets_df):
        tweets_df['idUser'] = tweets_df['user'].apply(lambda x: int(x['id']))
        tweets_df['username'] = tweets_df['user'].apply(lambda x: x['username'])
        tweets_df['inReplyToIdUser'] = tweets_df['inReplyToUser'].apply(lambda x: int(x['id']) if x else None)
        tweets_df['inReplyToUsername'] = tweets_df['inReplyToUser'].apply(lambda x: x['username'] if x else None)
        
        tweets_df = tweets_df[self.atributos]
        rename_atributos = dict(zip(self.atributos, self.novos_atributos))
        tweets_df.rename(columns=rename_atributos, inplace=True)
        
        return tweets_df
    
    def get_usuarios_unicos(self, df_depressao, df_controle) -> list:
        """_summary_

        Args:
            df_depressao (pd.DataFrame): _description_
            df_controle (pd.DataFrame): _description_

        Returns:
            list: _description_
        """
        df_merge = pd.merge(df_depressao, df_controle, how='inner', on='id_screen_name')
        listaCandidatosDuplicados = set(df_merge['id_screen_name'].to_list())
        df_depressao = df_depressao[~df_depressao['id_screen_name'].isin(listaCandidatosDuplicados)]
        df_controle = df_controle[~df_controle['id_screen_name'].isin(listaCandidatosDuplicados)]
        
        return [df_depressao, df_controle]
    
    def excluir_tweets_duplicados(self, df) -> pd.DataFrame:
        """excluindo tweets duplicados

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df.drop_duplicates(subset='id', keep='first', inplace=True)
        return df
    
    def excluir_usuarios_duplicados(self, listaPeriodos, diretorioCandidatos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            diretorioCandidatos (str): _description_
        """
        print('\nLimpando tweets e candidatos duplicados')
        for periodo in listaPeriodos:
            df_depressao = pd.read_csv(diretorioCandidatos + f'{periodo}_depressao.csv', delimiter=';', low_memory=False)
            df_controle = pd.read_csv(diretorioCandidatos + f'{periodo}_controle.csv', delimiter=';', low_memory=False)
            
            df_depressao, df_controle = self.get_usuarios_unicos(df_depressao, df_controle)
            df_depressao = self.excluir_tweets_duplicados(df_depressao)
            df_controle = self.excluir_tweets_duplicados(df_controle)
            
            df_depressao.to_csv(f'{diretorioCandidatos}{periodo}_depressao.csv', index=False, encoding='utf-8', sep=';')
            df_controle.to_csv(f'{diretorioCandidatos}{periodo}_controle.csv', index=False, encoding='utf-8', sep=';')

    def get_palavras_chaves(self, grupo) -> list:
        """_summary_
    
        Args:
            grupo (str): _description_
    
        Returns:
            list: _description_
        """
        operador_and = 'AND'
        operador_or = 'OR'
        a = '\u00E3'
        e = '\u00EA'
        o = '\u00F4'
        listaStringBusca = [
                      f'(\"fui diagnosticado depressivo\")', f'(\"fui diagnosticado depressiva\")',
                      f'(\"sou depressivo\")', f'(\"sou depressiva\")', f'(\"fui depressivo\")', f'(\"fui depressiva\")',
                      f'(\"tenho depress{a}o\")', f'(\"tinha depress{a}o\")',
                      f'(\"fui diagnosticado com depress{a}o\")', f'(\"fui diagnosticada com depress{a}o\")',
                      f'(\"fui diagnosticada depressivo\")', f'(\"fui diagnosticada depressiva\")',
                      f'(\"estou com depress{a}o\")', f'(\"estava com depress{a}o\")',
                      f'(\"minha depress{a}o\")', f'(\"t{o} depressivo\")', f'(\"t{o} depressiva\")',
                      
                      f'(\"fui diagnosticado depr{e}\")', f'(\"fui diagnosticada depr{e}\")',
                      f'(\"sou depr{e}\")', f'(\"fui depr{e}\")',
                      f'(\"tenho depr{e}\")', f'(\"tinha depr{e}\")',
                      f'(\"fui diagnosticado com depr{e}\")', f'(\"fui diagnosticada com depr{e}\")',
                      f'(\"estou com depr{e}\")', f'(\"estava com depr{e}\")',
                      f'(\"minha depr{e}\")',
                      
                      f'(eu {operador_and} sou {operador_and} depressivo)', f'(eu {operador_and} sou {operador_and} depressiva)',
                      f'(eu {operador_and} fui {operador_and} depressivo)', f'(eu {operador_and} fui {operador_and} depressiva)',
                      f'(eu {operador_and} tenho {operador_and} depress{a}o)', f'(eu {operador_and} tinha {operador_and} depress{a}o)',
                      f'(eu {operador_and} fui {operador_and} diagnosticado {operador_and} com {operador_and} depress{a}o)',
                      f'(eu {operador_and} fui {operador_and} diagnosticada {operador_and} com {operador_and} depress{a}o)',
                      f'(eu {operador_and} fui {operador_and} diagnosticado {operador_and} depressivo)',
                      f'(eu {operador_and} fui {operador_and} diagnosticado {operador_and} depressiva)',
                      f'(eu {operador_and} estou {operador_and} com {operador_and} depress{a}o)',
                      f'(eu {operador_and} estava {operador_and} com {operador_and} depress{a}o)',
                      f'(sou {operador_and} depressivo)', f'(sou {operador_and} depressiva)',
                      f'(fui {operador_and} depressivo)', f'(fui {operador_and} depressiva)',
                      f'(tenho {operador_and} depress{a}o)', f'(tinha {operador_and} depress{a}o)',
                      f'(fui {operador_and} diagnosticado {operador_and} com {operador_and} depress{a}o)',
                      f'(fui {operador_and} diagnosticada {operador_and} com {operador_and} depress{a}o)',
                      f'(fui {operador_and} diagnosticado {operador_and} depressivo)',
                      f'(fui {operador_and} diagnosticado {operador_and} depressiva)',
                      f'(minha {operador_and} depress{a}o)', f'(t{o} {operador_and} depressivo)', f'(t{o} {operador_and} depressiva)',
                      
                      f'(eu {operador_and} sou {operador_and} depr{e})', f'(eu {operador_and} fui {operador_and} depr{e})',
                      f'(eu {operador_and} tenho {operador_and} depr{e})', f'(eu {operador_and} tinha {operador_and} depr{e})',
                      f'(eu {operador_and} fui {operador_and} diagnosticado {operador_and} com {operador_and} depr{e})',
                      f'(eu {operador_and} fui {operador_and} diagnosticado {operador_and} depr{e})',
                      f'(eu {operador_and} fui {operador_and} diagnosticada {operador_and} com {operador_and} depr{e})',
                      f'(eu {operador_and} fui {operador_and} diagnosticada {operador_and} depr{e})',
                      f'(eu {operador_and} estou {operador_and} com {operador_and} depr{e})',
                      f'(eu {operador_and} estava {operador_and} com {operador_and} depr{e})',
                      f'(sou {operador_and} depr{e})', f'(fui {operador_and} depr{e})',
                      f'(tenho {operador_and} depr{e})', f'(tinha {operador_and} depr{e})',
                      f'(fui {operador_and} diagnosticado {operador_and} com {operador_and} depr{e})',
                      f'(fui {operador_and} diagnosticada {operador_and} com {operador_and} depr{e})',
                      f'(fui {operador_and} diagnosticado {operador_and} depr{e})',
                      f'(fui {operador_and} diagnosticada {operador_and} depr{e})',
                      f'(minha {operador_and} depr{e}\")', f'(t{o} {operador_and} depressivo)', f'(t{o} {operador_and} depressiva)',
                      ]
        
        if grupo == 'controle':
            listaStringBusca = [f'-{string}' for string in listaStringBusca]
            # listaStringBusca = [f'-depress{a}o {operador_or} -depressivo {operador_or} -depressiva {operador_or} -depr{e}']
            return listaStringBusca
        elif grupo == 'depressao':
            return listaStringBusca
        else:
            print('Group ' + grupo + ' is not a valid grupo')
