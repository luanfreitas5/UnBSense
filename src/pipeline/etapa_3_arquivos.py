#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
from utilitarios.utilitarios import getDiretorio, pastaTweets, pastaTweetsMerge
import os
import tqdm
import pandas as pd
import pickle
import re
import unidecode
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import nltk
import string
from wordcloud.wordcloud import STOPWORDS
from spacy.lang.pt.stop_words import STOP_WORDS


class Etapa_3_Arquivos():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        self.parts = 100
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        self.wl = WordNetLemmatizer()
        
    def getStopwords(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        portugueseInglesStopwords = []
        
        portugueseInglesStopwords.extend([unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in nltk.corpus.stopwords.words('portuguese')])

        portugueseInglesStopwords.extend([unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in nltk.corpus.stopwords.words('english')])
        portugueseInglesStopwords.extend([unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in STOPWORDS])
        portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in STOP_WORDS])
        
        '''with open('datasets/baseUtilitarias/portugueseST.txt', 'r', encoding='latin-1') as words:
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in words])
            
        with open('datasets/baseUtilitarias/stopwords-pt.txt', 'r', encoding='utf-8') as words:
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in words])
            
        with open('datasets/baseUtilitarias/chave.MF300.txt', 'r', encoding='latin-1') as words:
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in words])
            
        with open('datasets/baseUtilitarias/folha.MF300.txt', 'r', encoding='latin-1') as words:
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in words])
            
        with open('datasets/baseUtilitarias/publico.MF300.txt', 'r', encoding='latin-1') as words:
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in words])
            
        with open('datasets/baseUtilitarias/stopwords_ptbr.txt', 'r', encoding='utf-8') as words:
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for bigram in words for word in bigram.split(" ")])'''
        
        stopwords = '''
                    a, agora, ainda, alguém, algum, alguma, algumas, alguns, ampla, amplas, amplo, amplos, ante, antes, ao, aos, após, aquela, 
                    aquelas, aquele, aqueles, aquilo, as, até, através, cada, coisa, coisas, com, como, contra, contudo, da, daquele, daqueles, 
                    das, de, dela, delas, dele, deles, depois, dessa, dessas, desse, desses, desta, destas, deste, deste, destes, deve, devem, 
                    devendo, dever, deverá, deverão, deveria, deveriam, devia, deviam, disse, disso, disto, dito, diz, dizem, do, dos, e, é, 
                    ela, elas, ele, eles, em, enquanto, entre, era, essa, essas, esse, esses, esta, está, estamos, estão, estas, estava, estavam, 
                    estávamos, este, estes, estou, eu, fazendo, fazer, feita, feitas, feito, feitos, foi, for, foram, fosse, fossem, grande, 
                    grandes, há, isso, isto, já, la, lá, lhe, lhes, lo, mas, me, mesma, mesmas, mesmo, mesmos, meu, meus, minha, minhas, muita,
                    muitas, muito, muitos, na, não, nas, nem, nenhum, nessa, nessas, nesta, nestas, ninguém, no, nos, nós, nossa, nossas, nosso,
                    nossos, num, numa, nunca, o, os, ou, outra, outras, outro, outros, para, pela, pelas, pelo, pelos, pequena, pequenas, pequeno, 
                    pequenos, per, perante, pode, pude, podendo, poder, poderia, poderiam, podia, podiam, pois, por, porém, porque, posso, pouca, 
                    poucas, pouco, poucos, primeiro, primeiros, própria, próprias, próprio, próprios, quais, qual, quando, quanto, quantos, que, 
                    quem, são, se, seja, sejam, sem, sempre, sendo, será, serão, seu, seus, si, sido, só, sob, sobre, sua, suas, talvez, também, 
                    tampouco, te, tem, tendo, tenha, ter, teu, teus, ti, tido, tinha, tinham, toda, todas, todavia, todo, todos, tu, tua, tuas, 
                    tudo, última, últimas, último, últimos, um, uma, umas, uns, vendo, ver, vez, vindo, vir, vos, vós a, able, about, across, 
                    after, all, almost, also, am, among, an, and, any, are, as, at, be, because, been, but, by, can, cannot,
                    could, dear, did, do, does, either, else, ever, every, for, from, get, got, had, has, have, he, her, hers, him, his, how,
                    however, i, if, in, into, is, it, its, just, least, let, like, likely, may, me, might, most, must, my, neither, no, nor, not,
                    of, off, often, on, only, or, other, our, own, rather, said, say, says, she, should, since, so, some, than, that, the, their,
                    them, then, there, these, they, this, tis, to, too, twas, us, wants, was, we, were, what, when, where, which, while, who,
                    whom, why, will, with, would, yet, you, your
                    '''
        
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        portugueseInglesStopwords.extend([unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in tokenizer.tokenize(stopwords)])
        
        stopwords = ['haha', 'ai', 'acho', 'nan', 'fica',
                    'vao', 'quer', 'queria', 'querer',
                    'achei', 'fica', 'ficou', 'deixa', 'deixou', 'pra', 'to', 'vc',
                    'tá', 'pq', 'tô', 'ta', 'mt', 'pro',
                    'né', 'eh', 'tbm', 'ja', 'ah', 'vcs', 'hj', 'so', 'mto',
                    'agr', 'oq', 'la', 'tou', 'td', 'voce', 'ne', 'obg', 'tb',
                    'pra', 'to', 'vc', 'tá', 'pq', 'tô', 'ta', 'mt', 'pro',
                    'né', 'eh', 'tbm', 'ja', 'ah', 'vcs', 'hj', 'so', 'mto',
                    'agr', 'oq', 'la', 'tou', 'td', 'voce', 'ne', 'obg', 'tb',
                    'pra', 'vc', 'pra', 'to', 'os', 'rappi', 'vcs', 'nao', 'pq',
                    'mim', 'ai', 'ta', 'ja', 'ter', 'fazer', 'lá', 'deu', 'dado',
                    'então', 'vou', 'vai', 'veze', 'ficar', 'tá', 'apena', 'apenas',
                    'melhor', 'cara', 'gente', 'casa', 'pessoa', 'tocada',
                    'tava', 'falar', 'serum', 'gt', 'bts', 'ia', 'preciso', 'vox',
                    'fico', 'sair', 'tomar', 'quase', 'conta', 'al'
                    'falar', 'falando', 'amo', 'amor'
                    ]
        
        portugueseInglesStopwords.extend([unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in stopwords])
        
        portugueseInglesStopwords = list(set(portugueseInglesStopwords))
        portugueseInglesStopwords.sort()
        
        return portugueseInglesStopwords
        
    def normalizacao(self, text) -> str:
        """limpeza texto

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        text = text.lower().strip() 
        text = re.compile('<.*?>').sub('', text) 
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
        text = re.sub('\s+', ' ', text)  
        text = re.sub(r'\[[0-9]*\]', ' ', text) 
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text) 
        text = re.sub(r'\s+', ' ', text) 
        text = re.sub("@\S+", "", text)
        text = re.sub("\$", "", text)
        text = re.sub("https?:\/\/.*[\r\n]*", "", text)
        text = re.sub("#", "", text)
        text = re.sub(r'https?:\/\/[\r\n],"[\r\n]"', '', text, flags=re.MULTILINE) 
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub('[^a-zà-ù ]', ' ', text)
        return text
     
    def obter_pos_tag(self, token) -> str:
        """obter a categoria gramatica 

        Args:
            token (str): _description_

        Returns:
            str: _description_
        """
        if token.startswith('J'):
            return wordnet.ADJ
        elif token.startswith('V'):
            return wordnet.VERB
        elif token.startswith('N'):
            return wordnet.NOUN
        elif token.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    def lematizacao(self, string) -> str:
        """reduz uma palavra ao seu radical

        Args:
            string (str): _description_

        Returns:
            str: _description_
        """
        token = word_tokenize(string)
        word_pos_tags = nltk.pos_tag(token)  # classificação da classe gramatica do token ("agora" ,"RV")
        a = [self.wl.lemmatize(tag[0], self.obter_pos_tag(tag[1])) for idx, tag in enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
        return " ".join(a)
    
    def remocao_stopword(self, string, listaStopwords) -> str:
        """remoção de stopwords

        Args:
            string (str): _description_
            listaStopwords (list): _description_

        Returns:
            str: _description_
        """
        #a = [i for i in string.split() if i not in listaStopwords]
        #return ' '.join(a)
        pattern = re.compile(r'\b(' + r'|'.join(listaStopwords) + r')\b\s*')
        a = pattern.sub('', string)
        return a
    
    def preprocessamentoTexto(self, texto_limpo) -> str:
        """Pré-processamento de texto

        Args:
            texto_limpo (str): _description_

        Returns:
            str: _description_
        """
        listaStopwords = self.getStopwords()
        
        texto_limpo = self.normalizacao(texto_limpo)
        texto_limpo = self.remocao_stopword(texto_limpo, listaStopwords)
        texto_limpo = self.lematizacao(texto_limpo)
        
        return texto_limpo
        
    
        
    def etapa_3_1_mesclarDatasets(self, periodo, grupo, diretorioTweetsMerge):
        """Etapa 3.1 Mesclar arquivos

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioTweetsMerge (str): _description_
        """        
        diretorioTweets = getDiretorio(pastaTweets, periodo, grupo)
        listaTweets = [f"{diretorioTweets}{candidato}" for candidato in os.listdir(diretorioTweets)]
        listaTweets.sort()
        
        pbar = tqdm.tqdm(range(0, self.parts), colour='green')

        for part in pbar:
            '''
            Concat all database (only full json) files into a DataFrame
            Kernel is dying becuse the there are too many files to concat?
            '''
            
            pbar.set_description(f"Mesclando Arquivos - {periodo} {grupo}")
            
            # defines block indexing of files_list: start and end indexis
            start = int(len(listaTweets) / self.parts) * part
            
            if part == self.parts - 1:
                end = len(listaTweets) - 1
            else:
                end = int(len(listaTweets) / self.parts) * (part + 1) - 1
            
            flist = [fname for fname in listaTweets[start:end]]
            
            try:
                base = pd.concat([pd.read_pickle(file)
                                  for file in flist if '_fulljson' in file], sort=False)
        
            except Exception as e:
                print(e)
                
            try:
                base.index = pd.to_datetime(base['created_at'], infer_datetime_format=True, utc=True).dt.tz_localize(None)
                tweet_lake = base.sort_index().copy()
        
                del base
        
            except Exception as e:
                print(base.screen_name.unique(), e)

            # build a new datalake from scratch
            if part + 1 in range(0, 10):
                # tweet_lake.to_csv('{}processed_datalake_part00{}.csv'.format(diretorioTweetsMergeNormalizados, part + 1), sep=';')
                tweet_lake.to_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            elif part + 1 in range(10, 100):
                # tweet_lake.to_csv('{}processed_datalake_part0{}.csv'.format(diretorioTweetsMergeNormalizados, part + 1), sep=';')
                tweet_lake.to_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # tweet_lake.to_csv('{}processed_datalake_part{}.csv'.format(diretorioTweetsMergeNormalizados, part + 1), sep=';')
                tweet_lake.to_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part + 1), protocol=pickle.HIGHEST_PROTOCOL)
            
    def etapa_3_2_limparDatasets(self, periodo, grupo, diretorioTweetsMergeNormalizados):
        """Etapa 3.2 Limpar arquivos
        unicode https://www.unicode.org/charts/PDF/

        Args:
            periodo (str): _description_
            grupo (str): _description_
            diretorioTweetsMergeNormalizados (str): _description_
        """
        
        diretorioTweetsMerge = getDiretorio(pastaTweetsMerge, periodo, grupo)
        pbar = tqdm.tqdm(range(1, self.parts + 1), colour='green')
        
        for part in pbar:
            
            pbar.set_description(f"Limpando Arquivos - {periodo} {grupo}")
                            
            if part in range(1, 10):
                tweet_lake = pd.read_pickle('{}datalake_part00{}.pickle'.format(diretorioTweetsMerge, part))
            elif part in range(10, 100):
                tweet_lake = pd.read_pickle('{}datalake_part0{}.pickle'.format(diretorioTweetsMerge, part))
            else:
                tweet_lake = pd.read_pickle('{}datalake_part{}.pickle'.format(diretorioTweetsMerge, part))
        
            full_text = tweet_lake.full_text
                
            full_text.name = 'full_text'
            
            full_text_novo = self.converter_minuscula(full_text)
                
            full_text_limpo = self.remover_links_hastags_espacos_risadas(full_text_novo)
            
            full_text_original = full_text_limpo.values.copy()
            
            tweet_lake.insert(loc=2, column='full_text_original', value=full_text_original)
            
            full_text_sem_lantino = self.remover_nao_lantinos(full_text_limpo)
                
            tweet_lake['full_text'] = full_text_sem_lantino.values.copy()
            
            tweet_lake.drop(list(tweet_lake.loc[[True if '' == text else False for text in tweet_lake.full_text.values]].index), inplace=True, errors='ignore')
            
            if part in range(1, 10):
                tweet_lake.to_pickle('{}processed_datalake_part00{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            elif part in range(10, 100):
                tweet_lake.to_pickle('{}processed_datalake_part0{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                tweet_lake.to_pickle('{}processed_datalake_part{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
                
    def remover_palavras_frequentes(self, string):
        lista = ['deus', 'vida', 'mundo', 'mano', 'semana', 'video', 'brasil', 'amigo', 'merda', 'porra', 
                 'caralho', 'amanha', 'vontade', 'gosto', 'gostei', 'feliz', 'mae', 'cu', 'linda']
        
        pattern = re.compile(r'\b(' + r'|'.join(lista) + r')\b\s*')
        a = pattern.sub('', string)
        return a
    
    
    def parte2(self, periodo, grupo, diretorioTweetsMergeNormalizados):
        
        pbar = tqdm.tqdm(range(1, self.parts + 1), colour='green')
        
        for part in pbar:
            
            pbar.set_description(f"Parte 2 Limpando Arquivos - {periodo} {grupo}")
                            
            if part in range(1, 10):
                tweet_lake = pd.read_pickle('{}processed_datalake_part00{}.pickle'.format(diretorioTweetsMergeNormalizados, part))
            elif part in range(10, 100):
                tweet_lake = pd.read_pickle('{}processed_datalake_part0{}.pickle'.format(diretorioTweetsMergeNormalizados, part))
            else:
                tweet_lake = pd.read_pickle('{}processed_datalake_part{}.pickle'.format(diretorioTweetsMergeNormalizados, part))
        
            tweet_lake['full_text'] = tweet_lake['full_text'].apply(str).apply(lambda x: self.preprocessamentoTexto(x))
                
            tweet_lake['full_text_original'] = tweet_lake['full_text_original'].apply(str).apply(lambda x: self.preprocessamentoTexto(x))
            
            if part in range(1, 10):
                tweet_lake.to_pickle('{}processed_datalake_part00{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            elif part in range(10, 100):
                tweet_lake.to_pickle('{}processed_datalake_part0{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                tweet_lake.to_pickle('{}processed_datalake_part{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
                
            if not os.path.exists(diretorioTweetsMergeNormalizados):
                os.makedirs(diretorioTweetsMergeNormalizados)
                
            tweet_lake['full_text'] = tweet_lake['full_text'].apply(str).apply(lambda x: self.remover_palavras_frequentes(x))
                
            tweet_lake['full_text_original'] = tweet_lake['full_text_original'].apply(str).apply(lambda x: self.remover_palavras_frequentes(x))
                
            if part in range(1, 10):
                tweet_lake.to_pickle('{}processed_datalake_part00{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            elif part in range(10, 100):
                tweet_lake.to_pickle('{}processed_datalake_part0{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                tweet_lake.to_pickle('{}processed_datalake_part{}.pickle'.format(diretorioTweetsMergeNormalizados, part), protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def processamento_lematizacao(self, string):
        
        token = word_tokenize(string)
        word_pos_tags = nltk.pos_tag(token)  # classificação da classe gramatica do token ("agora" ,"RV")
        a = [self.wl.lemmatize(tag[0], self.obter_pos_tag(tag[1])) for idx, tag in enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
        return " ".join(a)
    
    
    def converter_minuscula(self, processing_series):
        """_summary_

        Args:
            processing_series (_type_): _description_

        Returns:
            _type_: _description_
        """

        # removing all diacritical marks and lowering the case
        processing_seriesNova = pd.Series([str(tweet).lower() for tweet in processing_series.values],
                                      index=processing_series.index)
        
        # Verifica se existe letra maiucula
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'[\u0041-\u005A]', text) != None else False 
                                                     for text in processing_seriesNova.values]]
        if not serieProcessada.empty:
            print("Converção de letras maiuscula para miniculas deu erro")
            exit(1)
        else:
            pass
            # print("Converção de letras maiuscula para miniculas realizado com sucesso")
            
        return processing_seriesNova
    
    def remover_links_hastags_espacos_risadas(self, processing_seriesNova) -> pd.Series:
        """_summary_

        Args:
            processing_seriesNova (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        # removing links
        processing_seriesNova = pd.Series([tweet.split('http', 1)[0] for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'.http\S*', tweet) != None else False for tweet in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de links deu erro")
            exit(1)
        else:
            pass
            
        # removing hashtags 
        processing_seriesNova = pd.Series(processing_seriesNova.apply(lambda tweet: re.sub(r'#\S+', ' ', tweet)).values,
                              index=processing_seriesNova.index)
        processing_seriesNova = pd.Series(processing_seriesNova.apply(lambda tweet: re.sub('#', ' ', tweet)).values,
                                      index=processing_seriesNova.index)
        
        serieProcessada = processing_seriesNova.loc[[True if '#' in text else False for text in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de hashtags deu erro")
            exit(1)
        else:
            pass
            
        # cleaning irregular laughter marks
        processing_seriesNova = pd.Series([re.sub(r'.*kkk*\S*', ' haha ', tweet) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        processing_seriesNova = pd.Series([re.sub(r'.*(rs)(rs)+\S*', ' haha ', tweet) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        processing_seriesNova = pd.Series([re.sub(r'.*(ha|hu)\S*(ha|hu)+\S*', ' haha ', tweet) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        # removing all non-regular whitespaces
        processing_seriesNova = pd.Series([' '.join(tweet.split()) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        processing_seriesNova = pd.Series([' '.join(tweet.split()) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        return processing_seriesNova
    
    def remover_nao_lantinos(self, processing_seriesNova) -> pd.Series:
        """_summary_

        Args:
            processing_seriesNova (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        # removing all non-latin characters
        processing_seriesNova = pd.Series([re.sub(r'[^\u0061-\u007A\u0020]', ' ', unidecode.unidecode(str(tweet).lower())) for tweet in processing_seriesNova.values], index=processing_seriesNova.index)
        
        serieProcessada = processing_seriesNova.loc[[True if re.search(r'[^\u0061-\u007A\u0020]', tweet) != None else False for tweet in processing_seriesNova.values]]
        
        if not serieProcessada.empty:
            print("Remoção de caracteres não latinos deu erro")
            exit(1)
        else:
            pass
            
        return processing_seriesNova