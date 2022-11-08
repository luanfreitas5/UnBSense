#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from wordcloud.wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from utilitarios.utilitarios import getDiretorioPlots
from sklearn.feature_extraction.text import CountVectorizer
from utilitarios.colorPalettes import colorblind, paired

from processamentoLinguagemNatural.nlp import ProcessamentoLinguagemNatural


class NuvemPalavras():
    """_summary_
    """
    def __init__(self):
        """Constructor
        """
        pass

    
        
    def gerarNuvemPalavra(self, periodo):
        """_summary_

        Args:
            periodo (str): _description_
        """
        grupoDepressao = 'depressao'
        grupoControle = 'controle'
        grupoGeral = 'geral'
        
        nlp = ProcessamentoLinguagemNatural()
        listaStopwords = nlp.getStopwords()
        
        counterDepressao = pd.read_pickle(f'datasets/{periodo}/dicionariosFequenciasPalavras/counterDepressao.pickle')
        
        counterControle = pd.read_pickle(f'datasets/{periodo}/dicionariosFequenciasPalavras/counterControle.pickle')
        
        for stoppword in listaStopwords:
            if stoppword in counterDepressao:
                del counterDepressao[stoppword]
            if stoppword in counterControle:
                del counterControle[stoppword]
            
        counterGeral = counterDepressao + counterControle
        
        arquivoSaidaNuvemDepressao = f'{periodo}_depressao_nuvem_palavras_tweets'
        arquivoSaidaNuvemControle = f'{periodo}_controle_nuvem_palavras_tweets'
        arquivoSaidaNuvemGeral = f'{periodo}_geral_nuvem_palavras_tweets'
        
        arquivoSaidaNuvemDepressaoBarras = f'{periodo}_barras_depressao_nuvem_palavras_tweets'
        arquivoSaidaNuvemControleBarras = f'{periodo}_barras_controle_nuvem_palavras_tweets'
        arquivoSaidaNuvemGeralBarras = f'{periodo}_barras_geral_nuvem_palavras_tweets'
        
        arquivoSaidaNuvemDepressaoPizza = f'{periodo}_pizza_depressao_nuvem_palavras_tweets'
        arquivoSaidaNuvemControlePizza = f'{periodo}_pizza_controle_nuvem_palavras_tweets'
        arquivoSaidaNuvemGeralPizza = f'{periodo}_pizza_geral_nuvem_palavras_tweets'
        
        tituloNuvemDepressao = f'Nuvem de Palavras de Tweets de Candidatos de Depressão {periodo}'
        tituloNuvemControle = f'Nuvem de Palavras de Tweets de Candidatos de Controle {periodo}'
        tituloNuvemGeral = f'Nuvem de Palavras de Tweets de Candidatos {periodo}'
        
        tituloDepressao = f'Frequência de Palavras de Candidatos de Depressão {periodo}'
        tituloControle = f'Frequência de Palavras de Candidatos de Controle {periodo}'
        tituloGeral = f'Frequência de Palavras de Candidatos {periodo}'
        
        diretorioNuvemDepressao = getDiretorioPlots('nuvemPalavras', periodo, grupoDepressao)
        diretorioNuvemControle = getDiretorioPlots('nuvemPalavras', periodo, grupoControle)
        diretorioNuvemGeral = getDiretorioPlots('nuvemPalavras', periodo, grupoGeral)
        
        self.plotarNuvemPalavras(counterDepressao, periodo, grupoDepressao, tituloNuvemDepressao, arquivoSaidaNuvemDepressao, diretorioNuvemDepressao)
        self.plotarNuvemPalavras(counterControle, periodo, grupoControle, tituloNuvemControle, arquivoSaidaNuvemControle, diretorioNuvemControle)
        self.plotarNuvemPalavras(counterGeral, periodo, grupoGeral, tituloNuvemGeral, arquivoSaidaNuvemGeral, diretorioNuvemGeral)
        
        top = 10
        topCounterDepressao = counterDepressao.most_common(top)
        topCounterControle = counterControle.most_common(top)
        topCounterGeral = counterGeral.most_common(top)
        
        del counterDepressao
        del counterControle
        del counterGeral
        
        self.plotaGraficoBarras(topCounterDepressao, tituloDepressao, arquivoSaidaNuvemDepressaoBarras, diretorioNuvemDepressao)
        self.plotaGraficoBarras(topCounterControle, tituloControle, arquivoSaidaNuvemControleBarras, diretorioNuvemControle)
        self.plotaGraficoBarras(topCounterGeral, tituloGeral, arquivoSaidaNuvemGeralBarras, diretorioNuvemGeral)
       
        self.plotarGraficoPizza(topCounterDepressao, tituloDepressao, arquivoSaidaNuvemDepressaoPizza, diretorioNuvemDepressao)
        self.plotarGraficoPizza(topCounterControle, tituloControle, arquivoSaidaNuvemControlePizza, diretorioNuvemControle)
        self.plotarGraficoPizza(topCounterGeral, tituloGeral, arquivoSaidaNuvemGeralPizza, diretorioNuvemGeral)

    def get_top_n_gram1(self, corpus, ngram_range, vocabulary=None, n=None) -> list:
        """_summary_

        Args:
            corpus (list): _description_
            ngram_range (tuple): _description_
            vocabulary (dict, optional): _description_. Defaults to None.
            n (int, optional): _description_. Defaults to None.

        Returns:
            list: _description_
        """
        if vocabulary:
            vec = CountVectorizer(ngram_range=ngram_range, vocabulary=vocabulary).fit(corpus)
        else:
            vec = CountVectorizer(ngram_range=ngram_range, vocabulary=vocabulary, stop_words=self.listaStopwords).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    def get_top_n_gram(self, corpus, n=None) -> list:
        """_summary_

        Args:
            corpus (list): _description_
            n (int, optional): _description_. Defaults to None.

        Returns:
            list: _description_
        """
        words_freq = list(corpus.items())
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]
        
    def obterFrequenciaPalavras1(self, serie, grupo, periodo, ngram_range=(1, 1), vocabulary=None, n=None) -> pd.DataFrame:
        """_summary_

        Args:
            serie (pd.Series): _description_
            grupo (str): _description_
            periodo (str): _description_
            ngram_range (tuple, optional): _description_. Defaults to (1, 1).
            vocabulary (list, optional): _description_. Defaults to None.
            n (int, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """        
        print(f'Calculando Frequencia de Palavras {grupo} {periodo}')
        
        frequencia = self.get_top_n_gram1(serie.astype(str), ngram_range, vocabulary=vocabulary, n=n)
        
        dfFrequencia = pd.DataFrame(frequencia, columns=['text' , 'count'])
        dfFrequencia.set_index('text', inplace=True)
        return frequencia
    
    def obterFrequenciaPalavras(self, serie, grupo, periodo, n=None) -> pd.DataFrame:
        """_summary_

        Args:
            serie (pd.Series): _description_
            grupo (str): _description_
            periodo (str): _description_
            n (int, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        print(f'Calculando Frequencia de Palavras {grupo} {periodo}')
        
        frequencia = self.get_top_n_gram(serie, n=n)
        
        dfFrequencia = pd.DataFrame(frequencia, columns=['text' , 'count'])
        dfFrequencia.set_index('text', inplace=True)
        return dfFrequencia

    def obterVocabulario(self, grupo) -> list:
        """_summary_

        Args:
            grupo (str): _description_

        Returns:
            list: _description_
        """
        if grupo == 'depressao':
            
            listaBusca = ['fui diagnosticado depressivo', 'fui diagnosticado depressiva',
                  'sou depressivo', 'sou depressiva', 'fui depressivo', 'fui depressiva',
                  'tenho depress\u00E3o', 'tinha depress\u00E3o',
                  'fui diagnosticado com depress\u00E3o', 'fui diagnosticada com depress\u00E3o',
                  'fui diagnosticada depressivo', 'fui diagnosticada depressiva',
                  'estou com depress\u00E3o', 'estava com depress\u00E3o',
                  'minha depress\u00E3o', 't\u00F4 depressivo', 't\u00F4 depressiva',
                  
                  'fui diagnosticado depr\u00EA', 'fui diagnosticada depr\u00EA',
                  'sou depr\u00EA', 'fui depr\u00EA',
                  'tenho depr\u00EA', 'tinha depr\u00EA',
                  'fui diagnosticado com depr\u00EA', 'fui diagnosticada com depr\u00EA',
                  'estou com depr\u00EA', 'estava com depr\u00EA',
                  'minha depr\u00EA'
                          ]
            return listaBusca
        elif grupo == 'controle':
            listaBusca = ['feliz', 'alegre', 'divertido', 'divertida',
                          'grato', 'grata', 'animada', 'animado',
                          'contente', 'satisfeito', 'satisfeita',
                          'aben\u00E7oado', 'abencoado', 'aben\u00E7oada', 'abencoada']
            return listaBusca
            
    def plotarNuvemPalavras(self, counter, periodo, grupo, arquivoSaida, diretorio):
        """_summary_

        Args:
            counter (Counter): _description_
            periodo (str): _description_
            grupo (str): _description_
            arquivoSaida (str): _description_
            diretorio (str): _description_
        """
        nlp = ProcessamentoLinguagemNatural()
        listaStopwords = nlp.getStopwords()
        
        print(f'Plotando Frequencia de Palavras Grafico de Barras --- {periodo} {grupo}')
        
        # create the wordcloud object
        wordcloud = WordCloud(width=800, height=400, stopwords=listaStopwords, background_color='white', max_font_size=150, max_words=800,
                              min_font_size=1, collocation_threshold=2, collocations=False).generate_from_frequencies(counter)
        
        # plot the wordcloud object
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(diretorio + f'{arquivoSaida}.png', bbox_inches='tight', format="png")
        plt.close()
            
    def plotaGraficoBarras(self, most_common, arquivoSaida, diretorio):
        """_summary_

        Args:
            most_common (list): _description_
            arquivoSaida (str): _description_
            diretorio (str): _description_
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        keys = [key for key, count in most_common]
        counts = [count for key, count in most_common]
        plt.barh(keys, counts, color=colorblind[0])
        ax.bar_label(ax.containers[0])
        ax.grid(axis='x')
        ax.invert_yaxis()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        plt.savefig(f'{diretorio}{arquivoSaida}_barras.png', bbox_inches='tight', format="png")
        plt.close()
        
    def plotarGraficoPizza(self, most_common, arquivoSaida, diretorio):
        """_summary_

        Args:
            most_common (list): _description_
            arquivoSaida (str): _description_
            diretorio (str): _description_
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        keys = [key for key, count in most_common]
        counts = [count for key, count in most_common]
        plt.pie(counts, labels=keys, autopct="%.2f%%", colors=paired)
        plt.savefig(diretorio + arquivoSaida + '.png', bbox_inches='tight', format="png")
        plt.close()
