#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from utilitarios.utilitarios import getDiretorio
from utilitarios.utilitarios import pastaCandidatos, pastaTweets, pastaTweetsMerge, pastaTweetsMergeNormalizados, pastaCaracteristicas
from utilitarios.utilitarios import limparTela
import pandas as pd
from threading import Thread
import os
from pipeline.baseDados import BaseDados
from processamentoLinguagemNatural.nuvemPalavras import NuvemPalavras
from pipeline.exploracaoDados import ExploracaoDados
from utilitarios.utilitarios import start_threads
from pipeline.etapa_1_busca_usuarios import Etapa_1_Busca_Usuarios
from pipeline.etapa_2_coleta_tweets import Etapa_2_Coleta_Tweets
from pipeline.etapa_3_arquivos import Etapa_3_Arquivos
from pipeline.etapa_4_caracteristicas import Etapa_4_Caracteristicas
from pipeline.etapa_5_qualidade_dados import Etapa_5_Qualidade_Dados
from pipeline.etapa_6_vetores_caracteristicas import Etapa_6_Vetores_Caracteristicas

class Metodologia(object):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        pass
    
    def etapa_1_buscar_usuarios(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        listaThreads = []
        pipeline = Etapa_1_Busca_Usuarios()
        
        for periodo in listaPeriodos: 
            for grupo in listaGrupos:
                diretorioCandidatos = getDiretorio(pastaCandidatos)
                thread = Thread(target=pipeline.etapa_1_buscarUsuarios, args=(periodo, grupo, diretorioCandidatos), daemon=True)
                listaThreads.append(thread)
        
        start_threads(listaThreads)
            
        pipeline.excluir_usuarios_duplicados(listaPeriodos, diretorioCandidatos)
            
    def etapa_2_coletar_tweets(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        listaThreads = []
        pipeline = Etapa_2_Coleta_Tweets()
        
        for periodo in listaPeriodos:
            for grupo in listaGrupos:
                diretorioTweets = getDiretorio(pastaTweets, periodo, grupo)
                thread = Thread(target=pipeline.etapa_2_coletarTweets, args=(periodo, grupo, diretorioTweets), daemon=True)
                listaThreads.append(thread) 

        start_threads(listaThreads)
            
    def etapa_3_1_mesclar_arquivos(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        pipeline = Etapa_3_Arquivos()
        
        for periodo in listaPeriodos:
            for grupo in listaGrupos:
                diretorioTweetsMerge = getDiretorio(pastaTweetsMerge, periodo, grupo)
                pipeline.etapa_3_1_mesclarDatasets(periodo, grupo, diretorioTweetsMerge)
            
                
    def etapa_3_2_limpar_arquivos(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        pipeline = Etapa_3_Arquivos()
        
        for periodo in listaPeriodos:
            for grupo in listaGrupos:
                diretorioTweetsMergeNormalizados = getDiretorio(pastaTweetsMergeNormalizados, periodo, grupo)
                pipeline.etapa_3_2_limparDatasets(periodo, grupo, diretorioTweetsMergeNormalizados)
            
    def etapa_4_extrair_caracteristicas(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        pipeline = Etapa_4_Caracteristicas()
        
        for periodo in listaPeriodos:
            for grupo in listaGrupos:
                diretorioTweetsMergeNormalizados = getDiretorio(pastaTweetsMergeNormalizados, periodo, grupo)
                diretorioCaracteristicas = getDiretorio(pastaCaracteristicas, periodo, grupo)
                pipeline.etapa_4_extracaoCaracteristicas(periodo, grupo, diretorioCaracteristicas, diretorioTweetsMergeNormalizados)
                
    def etapa_5_qualidade_Dados(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        pipeline = Etapa_5_Qualidade_Dados()
        
        for periodo in listaPeriodos:
            for grupo in listaGrupos:
                diretorioCaracteristicas = getDiretorio(pastaCaracteristicas, periodo, grupo)
                pipeline.etapa_5_remocaoOutliers(periodo, grupo, diretorioCaracteristicas)
                
    def etapa_6_vetores_caracteristicas(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        pipeline = Etapa_6_Vetores_Caracteristicas()

        for periodo in listaPeriodos:
            for grupo in listaGrupos:
                diretorioCaracteristicas = getDiretorio(pastaCaracteristicas, periodo, grupo)
                pipeline.etapa_6_VetoresCaracteristicas(periodo, grupo, diretorioCaracteristicas)
            
    def etapa_7_criar_base_dados(self, listaPeriodos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        limparTela()
        baseDados = BaseDados()
        
        for periodo in listaPeriodos:
            diretorioCaracteristicasDepressao = getDiretorio(pastaCaracteristicas, periodo, 'depressao')
            diretorioCaracteristicasControle = getDiretorio(pastaCaracteristicas, periodo, 'controle')
            baseDados.construirBaseDadosTwitter(periodo, diretorioCaracteristicasDepressao, diretorioCaracteristicasControle)
            
    def etapaexploracaoBaseDados(self, listaPeriodos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
        """
        limparTela()
        exploracaoDados = ExploracaoDados()
        
        exploracaoDados.graficoBarras(listaPeriodos)
        exploracaoDados.graficoPizza(listaPeriodos)
            
    def etapaCriarBaseDadosTexto(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        nuvemPalavras = NuvemPalavras()
        limparTela()
        listaThreads = []
        for periodo in listaPeriodos:
            # listaThreads = []
            for grupo in listaGrupos:
                
                candidatosBusca = [getDiretorio(pastaCandidatos) + f'{grupo}_{periodo}.csv']
                # nuvemPalavras.criarBaseDadosTexto(candidatosBusca, f'twitterTextos_candidatos_{grupo}_{periodo}.csv', 'datasets/basesDadosTexto/')
                
                thread = Thread(target=nuvemPalavras.criarBaseDadosTexto, args=(candidatosBusca, f'twitterTextos_candidatos_{grupo}_{periodo}.csv', 'datasets/basesDadosTexto/'), daemon=True)
                listaThreads.append(thread)
                
        for thread in listaThreads:
            thread.start()
        for thread in listaThreads:
            thread.join()
            
        listaThreads = []
        for periodo in listaPeriodos:
            listaThreads = []
            for grupo in listaGrupos:
                
                diretorioTweets = getDiretorio(pastaTweets, periodo, grupo)
                listaTweets = [f'{diretorioTweets}{file}' for file in os.listdir(diretorioTweets)]
                # nuvemPalavras.criarBaseDadosTexto(listaTweets, f'twitterTextos_tweets_{grupo}_{periodo}.csv', 'datasets/basesDadosTexto/')
                
                thread = Thread(target=nuvemPalavras.criarBaseDadosTexto, args=(listaTweets, f'twitterTextos_tweets_{grupo}_{periodo}.csv', 'datasets/basesDadosTexto/'), daemon=True)
                listaThreads.append(thread)
                
            for thread in listaThreads:
                thread.start()
            for thread in listaThreads:
                thread.join()
            
    def etapaNuvemPalavras(self, listaPeriodos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
        """
        nuvemPalavras = NuvemPalavras()
        for periodo in listaPeriodos:
            diretorio = 'datasets/basesDadosTexto/'
            nuvemPalavras.gerarNuvemPalavra(diretorio, periodo)
            
    def etapaFequenciaPalavras(self, listaPeriodos, listaGrupos):
        """_summary_

        Args:
            listaPeriodos (list): _description_
            listaGrupos (list): _description_
        """
        nuvemPalavras = NuvemPalavras()
        
        for periodo in listaPeriodos:
            listaDf = []
            for grupo in listaGrupos:
                df = pd.read_csv(f'datasets/basesDadosTexto/twitterTextos_candidatos_{grupo}_{periodo}.csv',
                                             sep=';', encoding='utf-8', usecols=['full_text'])
                
                vocabulary = nuvemPalavras.obterVocabulario(grupo)
                
                arquivoSaida = f'{periodo}_{grupo}_frequencia_stringsBusca'
                titulo = f'Frequencia das stringsBusca de {grupo} no periodo {periodo}'
                
                dfFequencia = nuvemPalavras.obterFrequenciaPalavras1(df['full_text'], grupo, periodo, vocabulary=vocabulary, ngram_range=(1, 5), n=12)
                nuvemPalavras.graficoPizza(dfFequencia, grupo, periodo, titulo, arquivoSaida)
                
                listaDf.append(dfFequencia)
