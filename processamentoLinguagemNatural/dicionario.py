'''
Created on 24 de set de 2022

@author: luanm
'''

import os
import pickle
import tqdm
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize

class MyClass(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
    def criarBaseDadosTexto(self, listaUsuario, arquivoSaida, diretorio):
        """_summary_

        Args:
            listaUsuariolistaUsuario (list): _description_
            arquivoSaida (str): _description_
            diretorio (str): _description_
        """
        usecols = ['created_at', 'id', 'id_screen_name', 'screen_name', 'full_text']
        

        for usuario in tqdm.tqdm(listaUsuario):
            
            df = pd.read_csv(usuario, sep=';',
                                        usecols=usecols,
                                        index_col='created_at',
                                        parse_dates=['created_at']).drop_duplicates(subset='id',
                                                                                      keep='first',
                                                                                      inplace=False).sort_index(inplace=False)
                                                                                      
            df['full_text_clean'] = df['full_text'].apply(str).apply(lambda x: self.processamentoGeral(x))
            df['full_text_len'] = df['full_text_clean'].astype(str).apply(len)
            df['full_text_word_count'] = df['full_text_clean'].apply(lambda x: len(str(x).split()))
            
            df.drop_duplicates(subset='id', keep='first', inplace=True)
            df.rename_axis('created_at').reset_index(inplace=True)
            
            if not os.path.isfile(diretorio + arquivoSaida):
                df.to_csv(diretorio + arquivoSaida, sep=';', encoding='utf-8')
            else:
                df.to_csv(diretorio + arquivoSaida, sep=';', encoding='utf-8', mode='a', header=False)
        
    def gerarDicionariosFequenciaPalavras(self, periodo, diretorio):
        """_summary_

        Args:
            periodo (str): _description_
            diretorio (str): _description_
        """
        arquivoDepressao = f'{diretorio}twitterTextos_tweets_depressao_{periodo}.csv'
        arquivoControle = f'{diretorio}twitterTextos_tweets_controle_{periodo}.csv'
        
        chunksize = 10 ** 5
        
        counterDepressao = Counter()
        counterControle = Counter()
        
        dfDeprpessao = pd.read_csv(arquivoDepressao, sep=';', encoding='utf-8', usecols=['full_text_clean'], iterator=True, chunksize=chunksize)
        dfControle = pd.read_csv(arquivoControle, sep=';', encoding='utf-8', usecols=['full_text_clean'], iterator=True, chunksize=chunksize)
        
        for i, df in enumerate(dfDeprpessao):
            print(f'{periodo} dfDeprpessao parte {i+1}')
            counterDepressao.update(word_tokenize(' '.join(df['full_text_clean'].astype(str).tolist())))
            
        with open(f'{periodo}_counterDepressao.pickle', 'wb') as f:
            pickle.dump(counterDepressao, f, protocol=pickle.HIGHEST_PROTOCOL)

        for i, df in enumerate(dfControle):
            print(f'{periodo} dfControle parte {i+1}')
            counterControle.update(word_tokenize(' '.join(df['full_text_clean'].astype(str).tolist())))
            
        with open(f'{periodo}_counterControle.pickle', 'wb') as f:
            pickle.dump(counterControle, f, protocol=pickle.HIGHEST_PROTOCOL)
            