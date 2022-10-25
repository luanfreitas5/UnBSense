#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
import os, pickle
import pandas as pd
import numpy as np
from utilitarios.utilitarios import abrirBaseSerie
import tqdm
from scipy.stats import entropy
from pipeline.caracteristicas import Caracteristicas


class VetoresCaracteristicas():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        pass
            
    def parseMediaPonderada(self, series) -> float:
        """Calcula a media dos valores
        
        skipna=True
        Exclua NA / valores nulos. Se uma linha / coluna inteira for NA, o resultado será NA.

        Args:
            series (pd.Series): _description_

        Returns:
            float: _description_
        """
        try:
            series_media = series.mean(skipna=True)
            return series_media
        except Exception:
            return series
    
    def parseVariancia(self, series) -> float:
        """
        '''
        Calcula a variaça imparcial

        Normalizado por N-1 por padrão
        
        ddof=0
        https://stackoverflow.com/questions/62938495/difference-between-numpy-var-and-pandas-var

        Args:
            series (pd.Series): _description_

        Returns:
            float: _description_
        """        
        try:
            series_var = series.var(ddof=0, skipna=True)
            return series_var
        except Exception:
            series_media = self.parseMediaPonderada(series)
            return series_media
        
    def parseMediaMovelPonderada(self, series) -> float:
        """Calcula a media ponterada dos valores
        
        if not np.isnan(series[i]) else 0 
        pula posicoes da serie igual a nan 

        Args:
            series (pd.Series): _description_

        Returns:
            float: _description_
        """        
        try:
            if len(series) > 7:
                convolucao = np.convolve(series, np.ones(7), 'valid')
                mediaMovel = sum(convolucao) / (len(series) - 7)
                return mediaMovel
            else:
                series_media = self.parseMediaPonderada(series)
                return series_media
            
        except Exception:
            series_media = self.parseMediaPonderada(series)
            return series_media
    
    def parseEntropia(self, series) -> float:
        """Calcule a entropia de valores de probabilidade.
        S = -sum(pk * log(pk), axis=axis)

        Args:
            series (pd.Series): _description_

        Returns:
            float: _description_
        """
        try:
            series_entropy = entropy(series.value_counts())
            return series_entropy
        except Exception:
            series_media = self.parseMediaPonderada(series)
            return series_media

    def calcularVetoresCaracteristicas(self, periodo, grupo, diretorioCaracteristicas):
        """Etapa 6 Sumarizacao de Vetores de Caracteristicas

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioCaracteristicas (str): _description_
        """        
        caracteristicas = Caracteristicas()
        
        arquivoVetoresCaracteristicas = f'{diretorioCaracteristicas}{periodo}_{grupo}_vetoresCaracteristicas.csv'
        
        listaCaracteristicas = [[f'{i}_media', f'{i}_variancia', f'{i}_mediaMovelPonterada', f'{i}_entropia'] for i in caracteristicas.listaAtributos]
        listaCaracteristicas = [i for lista in listaCaracteristicas for i in lista]
        
        listaCandidatos = set(abrirBaseSerie(caracteristicas.atributo_volume_tweets, diretorioCaracteristicas))
        
        if os.path.isfile(arquivoVetoresCaracteristicas):
            df = pd.read_csv(arquivoVetoresCaracteristicas, sep=';', index_col=0, encoding='utf-8')
            listaCandidatosProcessados = set(df.index)
        else:
            df = pd.DataFrame(columns=listaCaracteristicas, index=listaCandidatos, data=0.0)
            listaCandidatosProcessados = set()
        
        listaSeriesTemporais = [fnames for fnames in os.listdir(diretorioCaracteristicas) if '_serieTemporal' in fnames]
        
        listaNovosCandidatos = list(set(listaCandidatos) - set(listaCandidatosProcessados))
        listaNovosCandidatos.sort()
        
        del listaCandidatos
        del listaCandidatosProcessados
        
        if listaNovosCandidatos:
            
            for candidato in listaNovosCandidatos:
                df.loc[candidato] = 0.0
                
            df.sort_index(axis=0, inplace=True)
            
            pbar = tqdm.tqdm(listaSeriesTemporais, colour="green")
                
            # carregando series temporais de cada caracteristicas
            for serie in pbar:
                seriesTemporais = {}
                atributo = serie.rsplit('_', 1)[0]
                pbar.set_description(f"Sumarizando atributos {periodo} {grupo}")
                
                with open(diretorioCaracteristicas + serie, 'rb') as f:
                    seriesTemporais[atributo] = pickle.load(f)
                print()  
                for candidato in listaNovosCandidatos:
                    
                    for atributo in seriesTemporais.keys():
                        df.loc[candidato][f'{atributo}_media'] = self.parseMediaPonderada(seriesTemporais[atributo][candidato])
                        df.loc[candidato][f'{atributo}_variancia'] = self.parseVariancia(seriesTemporais[atributo][candidato])
                        df.loc[candidato][f'{atributo}_mediaMovelPonterada'] = self.parseMediaMovelPonderada(seriesTemporais[atributo][candidato])
                        df.loc[candidato][f'{atributo}_entropia'] = self.parseEntropia(seriesTemporais[atributo][candidato])
                            
                    if not os.path.exists(arquivoVetoresCaracteristicas):
                        df.to_csv(arquivoVetoresCaracteristicas, sep=';', encoding='utf-8')
                    else:
                        df.to_csv(arquivoVetoresCaracteristicas, sep=';', header=False, mode='a', encoding='utf-8')
    
