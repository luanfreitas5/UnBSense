#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser
import os 
import pickle
import pandas as pd
import collections
import platform 
import tqdm
import unidecode
import re
import numpy as np

pastaCandidatos = 'candidatos'
pastaTweets = 'tweets'
pastaTweetsMerge = 'tweets_merge'
pastaTweetsMergeNormalizados = 'tweets_merge_normalizados'
pastaCaracteristicas = 'caracteristicas'


def strfdelta(tdelta, fmt="{hours}:{minutes}:{seconds}"):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def getDiretorioPlots():
    '''
    Obtem o diretorio dos graficos gerados
    '''
    diretorio = 'plots/'
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    return diretorio


def getDiretorioInacessivel(grupo):
    diretorio = f'datasets/inacessivel/{grupo}/'
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    return diretorio


def getDiretorio(diretorio, periodo='', grupo=''):
    if diretorio == 'candidatos':
        diretorio = 'datasets/candidatos/'
    elif diretorio == 'tweets':
        diretorio = f'datasets/{periodo}/tweets/{grupo}/'
    elif diretorio == 'tweets_merge':
        diretorio = f'datasets/{periodo}/tweets_merge/{grupo}/'
    elif diretorio == 'tweets_merge_normalizados':
        diretorio = f'datasets/{periodo}/tweets_merge_normalizados/{grupo}/'
    elif diretorio == 'caracteristicas':
        diretorio = f'datasets/{periodo}/caracteristicas/{grupo}/'
    else:
        print("diretorio nao encontrado")
        
    if diretorio:
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
    
    return diretorio


def abrirBaseSerie(caracteristica, diretorioCaracteristicas):
    '''
    Loads a series collection from a pickle file.
    NEVER load pickle objects that are not trusted.
    '''
    arquivoPickle = f'{diretorioCaracteristicas}{caracteristica}_serieTemporal.pickle'
    
    try:
        with open(arquivoPickle, 'rb') as f:
            return pickle.load(f, fix_imports=False)
        
    except FileNotFoundError:
        return {}
            
    except Exception as e:
        print(f"Arquivo Pickle com Erro \n{e}")
        return {}


def gravarBaseSerie(serieTemporal, caracteristica, diretorioCaracteristicas):
    '''
    Saves feature series collection to disk, under the current group (e.g, depression)
    '''
    try:
        with open(f'{diretorioCaracteristicas}{caracteristica}_serieTemporal.pickle', 'wb') as f:
            pickle.dump(serieTemporal, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    except FileNotFoundError:
        pass
            
    except Exception as e:
        print(f"Arquivo Pickle com Erro \n{e}")

    
def atualizarBaseSerie(serieTemporal, caracteristica, diretorioCaracteristicas):
    '''
    Saves feature series collection to disk, under the current group (e.g, depression)
    '''
    arquivoPickle = f'{diretorioCaracteristicas}{caracteristica}_serieTemporal.pickle'
    
    try:
        if os.path.exists(arquivoPickle):
            serie = pd.read_pickle(arquivoPickle)
            serie.update(serieTemporal)
            serie = dict(collections.OrderedDict(sorted(serie.items())))
            with open(arquivoPickle, 'wb') as f:
                pickle.dump(serie, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(arquivoPickle, 'wb') as f:
                pickle.dump(serieTemporal, f, protocol=pickle.HIGHEST_PROTOCOL)
                
    except Exception as e:
        print(f"Arquivo Pickle com Erro \n{e}")

    
def candidatosExtraidosCaracteristicas(listaAtributos, diretorioCaracteristicas):
    
    listaCandidatosExtraidos = [set(abrirBaseSerie(caracteristica, diretorioCaracteristicas).keys()) for caracteristica in listaAtributos]
    
    if listaCandidatosExtraidos:
        listaCandidatosExtraidos = list(set.intersection(*listaCandidatosExtraidos))
    else:
        listaCandidatosExtraidos = []
    
    return listaCandidatosExtraidos


def getListaNovosCandidatos(caracteristica, listaNovosCandidatos, diretorioCaracteristicas):
    
    listaCandidatosExtraidos = set(abrirBaseSerie(caracteristica, diretorioCaracteristicas).keys())
    
    # Cria uma lista de candidatos não extraido a caracteristica selecionada
    # Diferenca entre duas listas
    listaNovosCandidatos = list(set(listaNovosCandidatos) - set(listaCandidatosExtraidos))
    listaNovosCandidatos.sort()
    
    return listaNovosCandidatos


def getExpressaoRegural(lista):
    
    expressaoRegular = '{}'.format('|'.join([r'\b{}\b'.format(i) for i in lista]))
    
    return expressaoRegular


def carregarBaseAnewBR():
    
    baseAnewBR = pd.read_excel('datasets/baseUtilitarias/anew_br.xlsx')
    baseAnewBR['Palavra'] = [unidecode.unidecode(x).lower() for x in baseAnewBR['Palavra'].astype(str).values]
    baseAnewBR.dropna(inplace=True)
    baseAnewBR.drop_duplicates('Palavra', keep='first', inplace=True)
    baseAnewBR.sort_values('Palavra', inplace=True)
    baseAnewBR.set_index('Palavra', inplace=True)
    
    return baseAnewBR


def carregarBaseMedicamentosAntiDepressivos():
    
    baseMedicamentosAntiDepressivos = pd.read_excel('datasets/baseUtilitarias/medicamentosAntidepressivo_br.xlsx')
    baseMedicamentosAntiDepressivos['Medicamento'] = baseMedicamentosAntiDepressivos['Medicamento'].apply(lambda x: re.search("^(\w+|.*\s\w+)\s\(.*$", x).group(1) \
                                                                                              if re.search("^(\w+|.*\s\w+)\s\(.*$", x) != None else None)
    baseMedicamentosAntiDepressivos['Medicamento'] = [unidecode.unidecode(x).lower() for x in baseMedicamentosAntiDepressivos['Medicamento'].astype(str).values]
    baseMedicamentosAntiDepressivos.dropna(inplace=True)
    baseMedicamentosAntiDepressivos.drop_duplicates('Medicamento', keep='first', inplace=True)
    baseMedicamentosAntiDepressivos.sort_values('Medicamento', inplace=True)
    
    return baseMedicamentosAntiDepressivos


def tratarValoresAusentes(serie: pd.Series):
    
    serie.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    media = serie.mean(skipna=True)
    
    if media == np.inf or media - np.inf:
        media = 0
    
    # Preencher valores Nan com a media dos dados do candidato
    serie.fillna(value=media, inplace=True)
    
    # Se existir valor inf infinito sera substituido por zero 0
    # serie.replace([np.inf, -np.inf], 0.0, inplace=True)
    
    return serie

    
def carregarParametros():
    '''
    '''
    parser = ArgumentParser(description='Collect data from the Twitter API for a specified listaGrupos')
    parser.add_argument('-g', '--group', dest='group')
    parser.add_argument('-m', '--mode', dest='mode', help='(Update) candidates\' datasets to the present or \n(Extend) candidates\' datasets further into the past')
    parser.add_argument('-p', '--periodo', dest='periodo')

    argumentos = parser.parse_args()

    if argumentos.group != None:
        listaGrupos = [argumentos.group]
    else:
        listaGrupos = ['depressao', 'controle']
        # listaGrupos = ['depressao']
        # listaGrupos = ['controle']
        
    if argumentos.periodo != None:
        listaPeriodos = [argumentos.periodo]
    else:
        listaPeriodos = ['pre-pandemia', 'pandemia']
        # listaPeriodos = ['pre-pandemia']
        # listaPeriodos = ['pandemia']
        
    if argumentos.mode in ['buscar']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['coletar']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['mesclar']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['limpar']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['atributos']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['qualidade']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['vetores']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['dataset']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['apredizagem']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['exploracao']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['basetexto']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['nuvempalavras']:
        mode = argumentos.mode.lower()
    elif argumentos.mode in ['frequencia']:
        mode = argumentos.mode.lower()
    else:
        mode = None
        
    if mode != None:
        return [listaPeriodos, listaGrupos, mode]
    else:
        return [None, None, None]


def limparTela():
    
    plt = platform.system()
    
    if plt == "Windows":
        clear = lambda: os.system('cls')
        clear()
    elif plt == "Linux":
        print("Your system is Linux")
        clear = lambda: os.system('clear')
        clear()
    else:
        print("Unidentified system")
