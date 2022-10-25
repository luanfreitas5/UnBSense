#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from utilitarios.utilitarios import gravarBaseSerie, abrirBaseSerie
import os
import pandas as pd
import tqdm
import pickle
from pipeline.caracteristicas import Caracteristicas

class Etapa_5_Qualidade_Dados(Caracteristicas):
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
        
    def etapa_5_remocaoOutliers(self, periodo, grupo, diretorioCaracteristicas):
        """Etapa 5 Qualidade de dados - remover outliers
        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioCaracteristicas (str): _description_
        """
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