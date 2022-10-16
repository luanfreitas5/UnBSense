'''
Created on 15 de out de 2022

@author: luanm
'''
from utilitarios.utilitarios import getDiretorio, pastaTweetsMergeNormalizados
from utilitarios.utilitarios import candidatosExtraidosCaracteristicas, atualizarBaseSerie
import os
import pandas as pd
from pipeline.caracteristicas import Caracteristicas


class Etapa_4(Caracteristicas):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
        
    def extracao_caracteristicas(self, periodo, grupo, diretorioCaracteristicas):
        """Etapa 4 Extrair Atributos

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioCaracteristicas (str): _description_
        """
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
                
                self.aplicar_extracao(df, listaNovosCandidatos, periodo, grupo, i, diretorioCaracteristicas)
    
                del df
