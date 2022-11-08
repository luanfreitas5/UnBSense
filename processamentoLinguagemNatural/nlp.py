#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
import nltk
from wordcloud.wordcloud import STOPWORDS
from spacy.lang.pt.stop_words import STOP_WORDS
import unidecode
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
import copy

class ProcessamentoLinguagemNatural():
    """_summary_
    """
    def __init__(self):
        """Constructor
        """
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        self.wl = WordNetLemmatizer
    
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
        
        with open('datasets/baseUtilitarias/portugueseST.txt', 'r', encoding='latin-1') as words:
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
            portugueseInglesStopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for bigram in words for word in bigram.split(" ")])
        
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
        
    def remocao_stopword(self, string, listaStopwords) -> str:
        """remoção de stopwords

        Args:
            string (str): _description_
            listaStopwords (list): _description_

        Returns:
            str: _description_
        """
        a = [i for i in string.split() if i not in listaStopwords]
        return ' '.join(a)
    
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

