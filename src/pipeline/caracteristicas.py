#!/usr/bin/env python
# coding: utf-8

import os, pickle
import re
import pandas as pd
import unidecode
from utilitarios.utilitarios import gravarBaseSerie, abrirBaseSerie, atualizarBaseSerie, \
    getExpressaoRegural, carregarBaseAnewBR, \
    carregarBaseMedicamentosAntiDepressivos, tratarValoresAusentes, \
    pastaTweetsMergeNormalizados, getDiretorio
from utilitarios.utilitarios import candidatosExtraidosCaracteristicas
import tqdm
import sys


class Caracteristicas(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.part = 100
        self.atributo_volume_tweets, self.atributo_volume_tweets_noite = 'volumeTweets', 'volumeTweetsNoite'
        self.atributo_indice_insonia, self.atributo_pronome_1_pessoa = 'indiceInsonia', 'pronome1Pessoa'
        self.atributo_pronome_2_pessoa, self.atributo_pronome_3_pessoa = 'pronome2Pessoa', 'pronome3Pessoa'
        self.atributo_valencia, self.atributo_ativacao, self.atributo_termos_depressivos = 'valencia', 'ativacao', 'termosDepressivos'
        self.atributo_medicamentos_anti_depressivo, self.atributo_grafo_social = 'medicamentosAntiDepressivo', 'grafoSocial'
        self.atributo_caracteres_orientais, self.atributo_emojis = 'caracteresOrientais', 'emojis'
        self.atributo_links, self.atributo_midia, self.atributo_curtidas = 'links', 'midia', 'curtidas'
        
        self.listaAtributos = [self.atributo_volume_tweets,
                               self.atributo_indice_insonia, self.atributo_pronome_1_pessoa,
                               self.atributo_pronome_2_pessoa, self.atributo_pronome_3_pessoa,
                               self.atributo_valencia, self.atributo_ativacao,
                               self.atributo_termos_depressivos, self.atributo_grafo_social,
                               self.atributo_medicamentos_anti_depressivo,
                               self.atributo_caracteres_orientais, self.atributo_emojis,
                               self.atributo_links, self.atributo_midia, self.atributo_curtidas]
        
    def gerenciarCaracteristicas(self, df, caracteristica, listaNovosCandidatos, periodo='', grupo='', part=0, serieTemporal_volume_tweets={}):
        serieTemporal = {}
        
        if caracteristica == self.atributo_volume_tweets:
            serieTemporal = self.extrairVolumeTweets(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
            
        elif caracteristica == self.atributo_volume_tweets_noite:
            serieTemporal = self.extrairVolumeTweetsNoite(df, caracteristica, listaNovosCandidatos)
            
        elif caracteristica == self.atributo_indice_insonia:
            serieTemporal = self.extrairIndiceInsomia(df, listaNovosCandidatos, serieTemporal_volume_tweets, periodo, grupo, part)
        
        elif caracteristica == self.atributo_pronome_1_pessoa:
            serieTemporal = self.extrairPronome1Pessoa(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_pronome_2_pessoa:
            serieTemporal = self.extrairPronome2Pessoa(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_pronome_3_pessoa:
            serieTemporal = self.extrairPronome3Pessoa(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
            
        elif caracteristica == self.atributo_valencia:
            serieTemporal = self.extrairValencia(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
            
        elif caracteristica == self.atributo_ativacao:
            serieTemporal = self.extrairAtivacao(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
            
        elif caracteristica == self.atributo_termos_depressivos:
            serieTemporal = self.extrairTermosDepressivos(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_medicamentos_anti_depressivo:
            serieTemporal = self.extrairMedicamentosAntiDepressivos(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_grafo_social:
            serieTemporal = self.extrairGrafoSocial(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_caracteres_orientais:
            serieTemporal = self.extrairCaracteresOrientais(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_emojis:
            serieTemporal = self.extrairEmojis(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
            
        elif caracteristica == self.atributo_links:
            serieTemporal = self.extrairLinks(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)  
        
        elif caracteristica == self.atributo_midia:
            serieTemporal = self.extrairMidia(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)
        
        elif caracteristica == self.atributo_curtidas:
            serieTemporal = self.extrairCurtidas(df, caracteristica, listaNovosCandidatos, periodo, grupo, part)

        else:
            print('Caracteristica não definida')
            sys.exit()
        
        return serieTemporal
    
    def setMedia(self, df):
        for dicionario in df['media']:
            if dicionario:
                for media in dicionario:
                    return media['fullUrl']
            else:
                return None 
            
    def setLinks(self, df):
        for lista in df['links']:
            if lista:
                for link in lista:
                    return link
            else:
                return None         
    
    def extracaoCaracteristicas(self, periodo, grupo, diretorioCaracteristicas):
        
        diretorioTweetsMergeNormalizados = getDiretorio(pastaTweetsMergeNormalizados, periodo, grupo)
        
        usecols = ['screen_name', 'full_text', 'full_text_original',
                   'favorite_count', 'media', 'links', 'reply_count']
        
        diretorioOutliens = f'{diretorioCaracteristicas}outliers.pickle'
        
        if os.path.isfile(diretorioOutliens):
            listaOutliers = list(pd.read_pickle(diretorioOutliens).keys())
        else:
            listaOutliers = list()

        listaCandidatosExtraidosCaracteristicas = candidatosExtraidosCaracteristicas(self.listaAtributos, diretorioCaracteristicas)
        listaArquivosMerge = [fname for fname in os.listdir(diretorioTweetsMergeNormalizados) if 'processed_datalake_part' in fname]
        
        for i, part in enumerate(listaArquivosMerge, 1):
            # print(f"Processando arquivo {periodo} --- {grupo} parte {i+1}")
            
            df = pd.read_pickle(f'{diretorioTweetsMergeNormalizados}{part}')[usecols]
            df['screen_name'] = df['screen_name'].astype(str)
            
            # Removendo candidatos já extraindo caracteristicas e outliers
            df = df[~df['screen_name'].isin(listaCandidatosExtraidosCaracteristicas)]
            df = df[~df['screen_name'].isin(listaOutliers)]
            df = df.where(pd.notnull(df), None)  # preenchendo nan com none
            df['media'] = self.setMedia(df)
            df['links'] = self.setLinks(df)
            
            if not df.empty:  # dataframe vazio significa que todos os candiadtos da parte foram extraidos suas caracteristicas
                
                # Cria uma lista de candidatos não extraido as 15 caracteristicas
                listaNovosCandidatos = list(set(df['screen_name'].to_list()))
                listaNovosCandidatos.sort()
                
                serieTemporal_volume_tweets = self.gerenciarCaracteristicas(df, self.atributo_volume_tweets, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_volume_tweets:
                    atualizarBaseSerie(serieTemporal_volume_tweets, self.atributo_volume_tweets, diretorioCaracteristicas)
                
                serieTemporal_indice_insonia = self.gerenciarCaracteristicas(df, self.atributo_indice_insonia, listaNovosCandidatos, periodo, grupo, i, serieTemporal_volume_tweets)
                if serieTemporal_indice_insonia:
                    atualizarBaseSerie(serieTemporal_indice_insonia, self.atributo_indice_insonia, diretorioCaracteristicas)
                del serieTemporal_volume_tweets
                del serieTemporal_indice_insonia
                
                serieTemporal_pronome_1_pessoa = self.gerenciarCaracteristicas(df, self.atributo_pronome_1_pessoa, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_pronome_1_pessoa:
                    atualizarBaseSerie(serieTemporal_pronome_1_pessoa, self.atributo_pronome_1_pessoa, diretorioCaracteristicas)
                del serieTemporal_pronome_1_pessoa
                
                serieTemporal_pronome_2_pessoa = self.gerenciarCaracteristicas(df, self.atributo_pronome_2_pessoa, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_pronome_2_pessoa:
                    atualizarBaseSerie(serieTemporal_pronome_2_pessoa, self.atributo_pronome_2_pessoa, diretorioCaracteristicas)
                del serieTemporal_pronome_2_pessoa
                
                serieTemporal_pronome_3_pessoa = self.gerenciarCaracteristicas(df, self.atributo_pronome_3_pessoa, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_pronome_3_pessoa:
                    atualizarBaseSerie(serieTemporal_pronome_3_pessoa, self.atributo_pronome_3_pessoa, diretorioCaracteristicas)
                del serieTemporal_pronome_3_pessoa
                
                serieTemporal_caracteres_orientais = self.gerenciarCaracteristicas(df, self.atributo_caracteres_orientais, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_caracteres_orientais:
                    atualizarBaseSerie(serieTemporal_caracteres_orientais, self.atributo_caracteres_orientais, diretorioCaracteristicas)
                del serieTemporal_caracteres_orientais
                
                serieTemporal_emojis = self.gerenciarCaracteristicas(df, self.atributo_emojis, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_emojis:
                    atualizarBaseSerie(serieTemporal_emojis, self.atributo_emojis, diretorioCaracteristicas)
                del serieTemporal_emojis
                
                serieTemporal_curtidas = self.gerenciarCaracteristicas(df, self.atributo_curtidas, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_curtidas:
                    atualizarBaseSerie(serieTemporal_curtidas, self.atributo_curtidas, diretorioCaracteristicas)
                del serieTemporal_curtidas
                
                serieTemporal_midia = self.gerenciarCaracteristicas(df, self.atributo_midia, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_midia:
                    atualizarBaseSerie(serieTemporal_midia, self.atributo_midia, diretorioCaracteristicas)
                del serieTemporal_midia
                
                serieTemporal_links = self.gerenciarCaracteristicas(df, self.atributo_links, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_links:
                    atualizarBaseSerie(serieTemporal_links, self.atributo_links, diretorioCaracteristicas)
                del serieTemporal_links
                
                serieTemporal_grafo_social = self.gerenciarCaracteristicas(df, self.atributo_grafo_social, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_grafo_social:
                    atualizarBaseSerie(serieTemporal_grafo_social, self.atributo_grafo_social, diretorioCaracteristicas)
                del serieTemporal_grafo_social
                
                serieTemporal_medicamentos_anti_depressivo = self.gerenciarCaracteristicas(df, self.atributo_medicamentos_anti_depressivo, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_medicamentos_anti_depressivo:
                    atualizarBaseSerie(serieTemporal_medicamentos_anti_depressivo, self.atributo_medicamentos_anti_depressivo, diretorioCaracteristicas)
                del serieTemporal_medicamentos_anti_depressivo
                
                serieTemporal_termos_depressivos = self.gerenciarCaracteristicas(df, self.atributo_termos_depressivos, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_termos_depressivos:
                    atualizarBaseSerie(serieTemporal_termos_depressivos, self.atributo_termos_depressivos, diretorioCaracteristicas)
                del serieTemporal_termos_depressivos
                
                serieTemporal_ativacao = self.gerenciarCaracteristicas(df, self.atributo_ativacao, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_ativacao:
                    atualizarBaseSerie(serieTemporal_ativacao, self.atributo_ativacao, diretorioCaracteristicas)
                del serieTemporal_ativacao
                
                serieTemporal_valencia = self.gerenciarCaracteristicas(df, self.atributo_valencia, listaNovosCandidatos, periodo, grupo, i)
                if serieTemporal_valencia:
                    atualizarBaseSerie(serieTemporal_valencia, self.atributo_valencia, diretorioCaracteristicas)
                del serieTemporal_valencia
                
                del df
                
    def criarSerieTemporal(self, df, candidato, caracteristica, expressaoRegular='', baseAnewBR={}):
        
        serieTemporal = pd.Series()
        
        if caracteristica in [self.atributo_volume_tweets, self.atributo_volume_tweets_noite]:
            '''volume de tweets
            volume de tweetss noite - indice de insonia'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['full_text'].astype(str).resample('D').count(), name=candidato)
            
        elif caracteristica == self.atributo_curtidas:
            '''numero de curtidas'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['favorite_count'].resample('D').sum(), name=candidato)
            
        elif caracteristica == self.atributo_midia:
            '''midia'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['media'].apply(lambda x: len(x) if x else 0).resample('D').sum(), name=candidato)
            
        elif caracteristica == self.atributo_links:
            '''links'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['links'].apply(lambda x: len(x) if x else 0).resample('D').sum(), name=candidato)
            
        elif caracteristica == self.atributo_grafo_social:
            '''grafo social'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['reply_count'].resample('D').sum(), name=candidato)
            
        elif caracteristica in [self.atributo_caracteres_orientais, self.atributo_emojis] and expressaoRegular:
            '''caracteres orientais 
            emojis'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['full_text_original'].apply(lambda x: len(re.findall(expressaoRegular, x))).resample('D').sum(), name=candidato)
        
        elif caracteristica in [self.atributo_valencia, self.atributo_ativacao] and expressaoRegular:
            '''emocao valencia
            emocao ativacao'''
            if caracteristica == self.atributo_valencia:
                key = 'Valencia-Media'
            else:
                key = 'Alerta-Media'
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['full_text'].apply(lambda x: sum([baseAnewBR[word][key] for word in re.findall(expressaoRegular, x)])).resample('D').mean(), name=candidato)
        
        elif caracteristica in [self.atributo_pronome_1_pessoa, self.atributo_pronome_2_pessoa, self.atributo_pronome_3_pessoa,
                                self.atributo_termos_depressivos, self.atributo_medicamentos_anti_depressivo] and expressaoRegular:
            '''estilo linguisco de 1ª pessoa, 
            estilo linguisco 2ª pessoa, 
            estilo linguisco 3ª pessoa, 
            termos de depressao, 
            medicamento anti-depressao'''
            serieTemporal = pd.Series(df.loc[df['screen_name'] == candidato]['full_text'].astype(str).apply(lambda x: len(re.findall(expressaoRegular, unidecode.unidecode(x.lower())))).resample('D').sum(), name=candidato)

        else:
            print('atributo nao definido')

        return serieTemporal

    def extrairVolumeTweets(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 1 - Volume de Tweets
        '''
        serieTemporal = {}
        
        if listaNovosCandidatos:
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_volume_tweets} {periodo} {grupo} parte {part}")
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, agrupa por dia e conta a quantdade de texto  
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato].fillna(value=serieTemporal[candidato].mean(skipna=True), inplace=True)
                
        return serieTemporal
    
    def extrairVolumeTweetsNoite(self, df, caracteristica, listaNovosCandidatos):
        '''
        Extração da Caracteristica 2 - Indice de Insonia (Parte 1) Volume de Tweets durante o turno da noite 22:00 - 6:00
        '''
        serieTemporal = {}
        
        if listaNovosCandidatos:
            
            # cria um sub-conjunto dataframe do index datetme no intervalodas 22H as 6H
            df_noite = df.loc[[True if (index.to_pydatetime().hour >= 22 or index.to_pydatetime().hour <= 6) else False for index in df.index]]
            
            for candidato in listaNovosCandidatos:
                
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, agrupa por dia e conta a quantdade de texto
                serieTemporal[candidato] = self.criarSerieTemporal(df_noite, candidato, caracteristica)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairIndiceInsomia(self, df, listaNovosCandidatos, serieTemporal_volume_tweets, periodo, grupo, part):
        '''
        Extração da Caracteristica 2 - Indice de Insonia (Parte 2) Relação de Volume de Tweets durante o turno da noite 22:00 - 6:00
        '''
        serieTemporal = {}
        
        if listaNovosCandidatos:
            serieTemporal_volume_tweets_noite = self.gerenciarCaracteristicas(df, self.atributo_volume_tweets_noite, listaNovosCandidatos)
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_indice_insonia} {periodo} {grupo} parte {part}")
            
                # Para cada index datetime do usuario faz uma relação com o numero de postagens da noite pelo numero de postagens do dia todo
                serieTemporal[candidato] = serieTemporal_volume_tweets[candidato] / serieTemporal_volume_tweets_noite[candidato]
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
        return serieTemporal
    
    def extrairPronome1Pessoa(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 3 - Estilo Linguistico de Pronome de 1ª Pessoa
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            pronomes1pessoa = ['eu', 'nos']

            expressaoRegular = getExpressaoRegural(pronomes1pessoa)
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_pronome_1_pessoa} {periodo} {grupo} parte {part}")
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, 
                # busca no texto palavras na 1ª pessoa conta a frequencia, soma os valores e armazenha no horario do dia,
                # agrupa por dia e soma os valores do dia
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairPronome2Pessoa(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 4 - Estilo Linguistico de Pronome de 2ª Pessoa
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            
            pronomes2pessoa = ['tu', 'voce', 'vc']

            expressaoRegular = getExpressaoRegural(pronomes2pessoa)
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_pronome_2_pessoa} {periodo} {grupo} parte {part}")
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, 
                # busca no texto palavras na 2ª pessoa conta a frequencia, soma os valores e armazenha no horario do dia,
                # agrupa por dia e soma os valores do dia
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairPronome3Pessoa(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 5 - Estilo Linguistico de Pronome de 3ª Pessoa
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            
            pronomes3pessoa = ['ele', 'ela', 'eles', 'elas']

            expressaoRegular = getExpressaoRegural(pronomes3pessoa)
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_pronome_3_pessoa} {periodo} {grupo} parte {part}")
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, 
                # busca no texto palavras na 3ª pessoa conta a frequencia, soma os valores e armazenha no horario do dia,
                # agrupa por dia e soma os valores do dia
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
        
    def extrairTermosDepressivos(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 8 - Termos de Depressão
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            
            baseAnewBR = carregarBaseAnewBR()
        
            listaTermosDepressivos = baseAnewBR.loc[baseAnewBR['Valencia-Media'] <= 4].index.to_list()
            listaTermosDepressivos.sort()
            
            expressaoRegular = getExpressaoRegural(listaTermosDepressivos)  
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_termos_depressivos} {periodo} {grupo} parte {part}")
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, 
                # busca no texto palavras estão na lista depression_terms armazenha numa lista de frequencia,
                # armazenha cada lista de frequencia numa lista (lista de lista)
                # adiciona cada elemento da lista de lista numa unica lista  
                # conta a frequencia e armazenha no horario do dia,
                # agrupa por dia e soma os valores do dia
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                    
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairMedicamentosAntiDepressivos(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 9 - Medicamentos Anti-Depressivos
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            baseMedicamentosAntiDepressivos = carregarBaseMedicamentosAntiDepressivos()
        
            listaMedicamentosAntiDepressivos = baseMedicamentosAntiDepressivos['Medicamento'].to_list()
            listaMedicamentosAntiDepressivos.sort()
            
            expressaoRegular = getExpressaoRegural(listaMedicamentosAntiDepressivos)  
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_medicamentos_anti_depressivo} {periodo} {grupo} parte {part}")
                # Para cada index datetime do usuario cria um sub-conjunto com full_text, 
                # busca no texto palavras estão na lista depression_terms armazenha numa lista de frequencia,
                # armazenha cada lista de frequencia numa lista (lista de lista)
                # adiciona cada elemento da lista de lista numa unica lista  
                # conta a frequencia e armazenha no horario do dia,
                # agrupa por dia e soma os valores do dia
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                    
        return serieTemporal
    
    def extrairCaracteresOrientais(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 11 - Caracteres Orientais Japônes, Chinês e Coreano
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            # https://medium.com/the-artificial-impostor/detecting-chinese-characters-in-unicode-strings-4ac839ba313a
            # korean japanese chinese
            unicodeOrientais = ['[\uac00-\ud7a3]', '[\u3040-\u30ff]', '[\u4e00-\u9FFF]']
            
            expressaoRegular = '{}'.format('|'.join([r'{}'.format(i) for i in unicodeOrientais]))
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_caracteres_orientais} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairEmojis(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 12 - Emojis
        '''
        serieTemporal = {}
        
        if listaNovosCandidatos:
            expressaoRegular = r'[\U0001F300-\U000E007F]'
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_emojis} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairCurtidas(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 15 - Numero de Curtidas
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_curtidas} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairMidia(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 14 - Midia (fotos, videos e gifs)
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
      
        if listaNovosCandidatos:
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_midia} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairLinks(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 13 - Frequência de Links
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
      
        if listaNovosCandidatos:
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_links} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairGrafoSocial(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part):
        '''
        Extração da Caracteristica 10 - grafo social
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
  
        if listaNovosCandidatos:
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_grafo_social} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairValencia(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part) -> dict:
                
        '''
        Extração da Caracteristica 6 - Emoção Valencia
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            baseAnewBR = carregarBaseAnewBR().T.to_dict()
        
            expressaoRegular = getExpressaoRegural(list(baseAnewBR.keys())) 
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_valencia} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular, baseAnewBR)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def extrairAtivacao(self, df, caracteristica, listaNovosCandidatos, periodo, grupo, part) -> dict:
                
        '''
        Extração da Caracteristica 7 - Emoção Ativacao
        '''
        serieTemporal = {}
        
        # listaNovosCandidatos = getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas)
        
        if listaNovosCandidatos:
            baseAnewBR = carregarBaseAnewBR().T.to_dict()
            
            expressaoRegular = getExpressaoRegural(list(baseAnewBR.keys())) 
            
            pbar = tqdm.tqdm(listaNovosCandidatos, colour='green')
            for candidato in pbar:
                pbar.set_description(f"Extraindo {self.atributo_ativacao} {periodo} {grupo} parte {part}")
                serieTemporal[candidato] = self.criarSerieTemporal(df, candidato, caracteristica, expressaoRegular, baseAnewBR)
                
                # Preencher valores Nan com a media dos dados do candidato
                serieTemporal[candidato] = tratarValoresAusentes(serieTemporal[candidato])
                
        return serieTemporal
    
    def remocaoOutliers(self, periodo, grupo, diretorioCaracteristicas):
        
        serieTemporal_volume_tweets = abrirBaseSerie(self.atributo_volume_tweets, diretorioCaracteristicas)
        
        if not os.path.exists(f'{diretorioCaracteristicas}outliers.pickle'):
            outliers = {}
        else:
            outliers = pd.read_pickle(f'{diretorioCaracteristicas}outliers.pickle')
        
        for key in serieTemporal_volume_tweets.keys():
            valorMaxTweetDia = serieTemporal_volume_tweets[key].max(axis=0)
            somaTotalTweets = serieTemporal_volume_tweets[key].sum()
        
            if valorMaxTweetDia > 300:
                outliers[key] = valorMaxTweetDia
                
            if somaTotalTweets < 30:
                outliers[key] = somaTotalTweets
            else:
                pass
            
        del serieTemporal_volume_tweets
        
        listaSeriesTemporais = [x for x in os.listdir(diretorioCaracteristicas) if '_serieTemporal.pickle' in x]
        
        pbar = tqdm.tqdm(listaSeriesTemporais, colour="green")
            
        for csvfile in pbar:
            
            pbar.set_description(f"Removendo Outliers {periodo} {grupo}")
            caracteristica = csvfile.rsplit('_', 1)[0]
            serieTemporal = abrirBaseSerie(caracteristica, diretorioCaracteristicas)
            
            for candidate in outliers.keys():
                try:
                    serieTemporal.pop(candidate)
        
                except KeyError:
                    pass
        
                except Exception as e:
                    print(f"Erro na remocao de outliens\n{e}")
                    break
            
            gravarBaseSerie(serieTemporal, caracteristica, diretorioCaracteristicas)
        
        with open(diretorioCaracteristicas + 'outliers.pickle', 'wb') as f:
            pickle.dump(outliers, f, protocol=pickle.HIGHEST_PROTOCOL)
