#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from aprendizagemMaquina.machineLearning import MachineLearning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pickle
from utilitarios.utilitarios import getDiretorioResultados
from warnings import filterwarnings
filterwarnings('ignore')

class AlgoritmosSupervisionado(MachineLearning):

    def __init__(self):
        super().__init__()
        
    # Main -------------------------------------------------------------------------- 
        
    def main_validacao(self):
        
        self.inicializacao()
        
        # dabl.SimpleClassifier().fit(self.x_train, self.y_train)
        
        self.modeloBaseline()
        
        self.modeloAnaliseDiscriminanteLinear(self.x_val, self.y_val)
        
        self.modeloArvoreDecisao(self.x_val, self.y_val)
        
        self.modeloFlorestaRandomica(self.x_val, self.y_val)
        self.modeloGradientBoosting(self.x_val, self.y_val)
        self.modeloKnn(self.x_val, self.y_val)
        
        self.modeloNaiveBayes(self.x_val, self.y_val)
        self.modeloPerceptronMulticamadas(self.x_val, self.y_val)
        #self.modeloRegressaoLogistica(self.x_val, self.y_val)
        self.modeloSvm(self.x_val, self.y_val)
        
        self.modeloEnsembleVotacao(self.x_val, self.y_val, voting='hard')
        self.modeloEnsembleVotacao(self.x_val, self.y_val, voting='soft')
        
        self.plotarCurvaRoc(self.x_val, self.y_val, "validacao")
        self.plotarCurvaPrecisaoRecall(self.x_val, self.y_val, "validacao")
        self.plotarMatrizesConfusoes("validacao")
        
        diretorioResultados = getDiretorioResultados()
        
        self.df_resultados.to_csv(f'{diretorioResultados}df_resultados_validacao.csv', sep=';', encoding='utf-8', quotechar='"', doublequote=True)    
        
        with open(f'{diretorioResultados}modelosTreinados.pickle', 'wb') as f:
            pickle.dump(self.dicionarioModelos, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def main_ensemble(self):
        
        self.inicializacao()
        
        self.modelosTreinados = pd.read_pickle('resultados/modelosTreinados.pickle')
        
        nomeMelhorModelo = 'Floresta Randômica'
        print(nomeMelhorModelo)
        melhorModelo = self.modelosTreinados[nomeMelhorModelo]
        
        self.modeloAdaBoost(self.x_val, self.y_val, melhorModelo)
        self.modeloBaggingBoost(self.x_val, self.y_val, melhorModelo)
        
        diretorioResultados = getDiretorioResultados()
        
        self.df_resultados.to_csv(f'{diretorioResultados}df_resultados_validacao.csv', sep=';', encoding='utf-8', quotechar='"', doublequote=True, mode='a')
        
        with open(f'{diretorioResultados}modelosTreinados.pickle', 'wb') as f:
            pickle.dump(self.modelosTreinados, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def main_teste(self):
        
        '''
        Dicionario de modelosTreinados chaves:  Baseline Regressão Logistica
                                                Análise Discriminante Linear
                                                Arvore de Decisão
                                                Floresta Randômica
                                                Gradient Boosting
                                                KNN
                                                Naive Bayes
                                                Perceptron Multicamadas
                                                SVM
                                                Ensemble Votação Hard
                                                Ensemble Votação Soft
        '''
        
        self.inicializacao()
        modelosTreinados = pd.read_pickle('resultados/modelosTreinados.pickle')
        
        nomeMelhorModelo = 'Arvore de Decisão'
        print(nomeMelhorModelo)
        melhorModelo = modelosTreinados[nomeMelhorModelo]
        predicao = melhorModelo.predict(self.x_test)
        self.df_resultados.loc[nomeMelhorModelo] = self.avaliacao(melhorModelo, predicao, self.x_test, self.y_test, nomeMelhorModelo)
        self.dicionarioModelos[nomeMelhorModelo] = melhorModelo
        
        nomeMelhorEnsemble = 'Ensemble Votação Soft'
        print(nomeMelhorEnsemble)
        melhorEnsemble = modelosTreinados[nomeMelhorEnsemble]
        predicao = melhorEnsemble.predict(self.x_test)
        self.df_resultados.loc[nomeMelhorEnsemble] = self.avaliacao(melhorEnsemble, predicao, self.x_test, self.y_test, nomeMelhorEnsemble)
        
        self.plotarCurvaRoc(self.x_test, self.y_test, "teste")
        self.plotarCurvaPrecisaoRecall(self.x_test, self.y_test, "teste")
        self.plotarMatrizesConfusoes("teste")

        diretorioResultados = getDiretorioResultados()
        
        self.df_resultados.to_csv(f'{diretorioResultados}df_resultados_teste.csv', sep=';', encoding='utf-8', quotechar='"', doublequote=True)
    
    # Algoritmo Baseline -------------------------------------------------------------------------- 
    
    def modeloBaseline(self):
        
        nome_classificador = "Baseline Regressão Logistica"
        print(nome_classificador)
        #baseline = DummyClassifier(strategy="most_frequent", random_state=183212, constant=None)
        baseline = LogisticRegression(class_weight=self.class_weight, C=1, penalty='l2')
        baseline.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = baseline
        
        predicao = baseline.predict(self.x_val)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(baseline, predicao, self.x_val, self.y_val, nome_classificador)
    
    # Algoritmos Classficadores -------------------------------------------------------------------------- 
    
    def modeloAnaliseDiscriminanteLinear(self, x, y):  
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
        
        Análise Discriminante Linear
        Um classificador com um limite de decisão linear, 
        gerado ajustando densidades condicionais  de classe aos dados e usando a regra de Bayes.
        
        solver : {'svd', 'lsqr', 'eigen'}, default='svd'
            Solver para usar, valores possíveis:
               - 'svd': Decomposição de valor singular (padrão).
                Não calcula a matriz de covariância, portanto, este solver é
                recomendado para dados com um grande número de recursos.
               - 'lsqr': solução de mínimos quadrados.
                Pode ser combinado com o estimador de encolhimento ou de covariância personalizado.
               - 'eigen': Decomposição de valores próprios.
                Pode ser combinado com o estimador de encolhimento ou de covariância personalizado.
    
        shrinkage : 'auto' or float, default=None
            Parâmetro shrinkage, valores possíveis:
               - Nenhum: sem encolhimento (padrão).
               - 'auto': encolhimento automático usando o lema de Ledoit-Wolf.
               - flutuar entre 0 e 1: parâmetro de retração fixo.
        '''
        
        nome_classificador = "Análise Discriminante Linear"
        print(nome_classificador)
        analiseDiscriminanteLinear = LinearDiscriminantAnalysis(shrinkage=0.01, solver='lsqr')
        analiseDiscriminanteLinear.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = analiseDiscriminanteLinear
        self.listaEstimadoresEnsemble.append((nome_classificador, analiseDiscriminanteLinear))
        
        predicao = analiseDiscriminanteLinear.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(analiseDiscriminanteLinear, predicao, x, y, nome_classificador)
        
    def modeloArvoreDecisao(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        
        Arvore de Decisão
        Um classificador de árvore de decisão.
        
        criterion : {"gini", "entropy"}, default="gini"
            A função para medir a qualidade de uma divisão. Os critérios suportados são
            "gini" para a impureza de Gini e "entropy" para o ganho de informação.
            
        splitter : {"best", "random"}, default="best"
            A estratégia utilizada para escolher a divisão em cada nó. Compatível
            estratégias são "best" para escolher a melhor divisão e "random" para escolher.
            
        max_depth : int, default=None
            A profundidade máxima da árvore. Se Nenhum, então os nós são expandidos até
            todas as folhas são puras ou até que todas as folhas contenham menos de
            amostras min_samples_split.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de várias saídas, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "Arvore de Decisão"
        print(nome_classificador)
        arvoreDecisao = DecisionTreeClassifier(class_weight=self.class_weight, criterion='entropy', max_depth=15, splitter="best")
        arvoreDecisao.fit(self.x_train, self.y_train)

        self.dicionarioModelos[nome_classificador] = arvoreDecisao
        self.listaEstimadoresEnsemble.append((nome_classificador, arvoreDecisao))

        predicao = arvoreDecisao.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(arvoreDecisao, predicao, x, y, nome_classificador)
        
    def modeloFlorestaRandomica(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        
        Classificador Floresta Randômica / Floresta Aleatória - Random forest - RF
        Uma floresta aleatória é um meta estimador que ajusta vários classificadores de árvore de decisão 
        em várias subamostras do conjunto de dados e usa a média para melhorar a precisão preditiva e 
        controlar o ajuste excessivo. O tamanho da subamostra é controlado com o 
        parâmetro max_samples se bootstrap=True(padrão), caso contrário, 
        todo o conjunto de dados é usado para construir cada árvore.
        
        n_estimators : int, default=100
            O número de árvores na floresta.
            
        criterion : {"gini", "entropy"}, default="gini"
            A função para medir a qualidade de uma divisão. Os critérios suportados são
            "gini" para a impureza de Gini e "entropy" para o ganho de informação.
            
        splitter : {"best", "random"}, default="best"
            A estratégia utilizada para escolher a divisão em cada nó. Compatível
            estratégias são "best" para escolher a melhor divisão e "random" para escolher.
            
        max_depth : int, default=None
            A profundidade máxima da árvore. Se Nenhum, então os nós são expandidos até
            todas as folhas são puras ou até que todas as folhas contenham menos de
            amostras min_samples_split.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de várias saídas, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "Floresta Randômica"
        print(nome_classificador)
        florestaRandomica = RandomForestClassifier(class_weight=self.class_weight, criterion='entropy', max_depth=15, n_estimators=500)
        florestaRandomica.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = florestaRandomica
        self.listaEstimadoresEnsemble.append((nome_classificador, florestaRandomica))
        
        predicao = florestaRandomica.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(florestaRandomica, predicao, x, y, nome_classificador)
        
    def modeloGradientBoosting(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        
        Gradient Boosting
        GB constrói um modelo aditivo de forma progressiva; 
        ele permite a otimização de funções de perda diferenciáveis ​​arbitrárias. 
        Em cada estágio as n_classes_ árvores de regressão são 
        ajustadas no gradiente negativo da função de perda de desvio binomial ou multinomial. 
        A classificação binária é um caso especial em que apenas uma única árvore de regressão é induzida.
        
        loss : {'deviance', 'exponential'}, default='deviance'
            A função de perda a ser otimizada. 'deviance' refere-se a
             desvio (= regressão logística) para classificação
             com saídas probabilísticas. Para gradiente 'exponential' de perda
             impulsionar recupera o algoritmo AdaBoost.
             
        criterion : {'friedman_mse', 'squared_error', 'mse', 'mae'}, default='friedman_mse'
            A função para medir a qualidade de uma divisão. Critérios compatíveis
            são 'friedman_mse' para o erro quadrático médio com melhoria
            pontuação de Friedman, 'squared_error' para erro quadrático médio e 'mae'
            para o erro absoluto médio. O valor padrão de 'friedman_mse' é
            geralmente o melhor, pois pode fornecer uma melhor aproximação em alguns casos.
             
        max_depth : int, default=None
            A profundidade máxima da árvore. Se Nenhum, então os nós são expandidos até
            todas as folhas são puras ou até que todas as folhas contenham menos de
            amostras min_samples_split.
        '''
        
        nome_classificador = 'Gradient Boosting'
        print(nome_classificador)
        gradientBoosting = GradientBoostingClassifier(criterion='squared_error', loss='exponential', max_depth=10)
        gradientBoosting.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = gradientBoosting
        self.listaEstimadoresEnsemble.append((nome_classificador, gradientBoosting))
        
        predicao = gradientBoosting.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(gradientBoosting, predicao, x, y, nome_classificador)
        
    def modeloKnn(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        
        KNN - k-vizinhos mais próximos
        Classificador implementando o voto dos k-vizinhos mais próximos.
        
        n_neighbors : int, default=5
            Número de vizinhos a serem usados ​​por padrão para consultas :meth:`kneighbors`.
            
        weights : {'uniform', 'distance'} or callable, default='uniform'
            Função de peso usada na previsão. Valores possíveis:
            - 'uniform' : pesos uniformes. Todos os pontos em cada bairro
              são ponderadas igualmente.
            - 'distance' : pontos de peso pelo inverso de sua distância.
              neste caso, os vizinhos mais próximos de um ponto de consulta terão um
              maior influência do que os vizinhos mais distantes.
            - [callable] : uma função definida pelo usuário que aceita um
              array de distâncias e retorna um array da mesma forma contendo os pesos.
              
        metric : str or callable, default='minkowski'
            A métrica de distância a ser usada para a árvore. A métrica padrão é
            minkowski, e com p=2 é equivalente ao padrão euclidiano
            métrica. 
            Se a métrica for "precomputed", assume-se que X é uma matriz de distância e
            deve ser quadrado durante o ajuste. X pode ser um :term:`sparse graph`,
            nesse caso, apenas elementos "nonzero" podem ser considerados vizinhos.
        '''
        
        nome_classificador = 'KNN'
        print(nome_classificador)
        knn = KNeighborsClassifier(metric='manhattan', n_neighbors=13, weights='distance')
        knn.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = knn
        self.listaEstimadoresEnsemble.append((nome_classificador, knn))
        
        predicao = knn.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(knn, predicao, x, y, nome_classificador)
        
    def modeloNaiveBayes(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        
        Naive Bayes - Gaussian Naive Bayes
        
        var_smoothing : float, default=1e-9
            Parte da maior variação de todos os recursos que é adicionado ao
            desvios para estabilidade de cálculo.
        '''
        
        nome_classificador = "Gaussian Naive Bayes"
        print(nome_classificador)
        #naiveBayes = MultinomialNB(alpha=0.01, fit_prior=True)
        naiveBayes = GaussianNB(var_smoothing=0.0001)
        naiveBayes.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = naiveBayes
        self.listaEstimadoresEnsemble.append((nome_classificador, naiveBayes))
        
        predicao = naiveBayes.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(naiveBayes, predicao, x, y, nome_classificador)
        
    def modeloPerceptronMulticamadas(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        
        Classificador Perceptron Multicamadas - Multi Layer Perceptron - MLP
        Este modelo otimiza a função log-loss usando LBFGS ou gradiente descendente estocástico.
        
        hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
            O i-ésimo elemento representa o número de neurônios no i-ésimo camada oculta.
            
        activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
            Função de ativação para a camada oculta.
            - 'identity', ativação sem operação, útil para implementar gargalo linear, retorna f(x) = x
            - 'logistic', a função sigmóide logística, retorna f(x) = 1 / (1 + exp(-x)).
            - 'tanh', a função tan hiperbólica, retorna f(x) = tanh(x).
            - 'relu', a função de unidade linear retificada, retorna f(x) = max(0, x)
            
        solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
            TO solucionador para otimização de peso.
            - 'lbfgs' é um otimizador na família de métodos quase-Newton.
            - 'sgd' refere-se à descida de gradiente estocástica.
            - 'adam' refere-se a um otimizador baseado em gradiente estocástico proposto por Kingma, Diederik e Jimmy Ba
            
        max_iter : int, default=200
            Número máximo de iterações (épocas). O solver itera até a convergência
            (determinado por 'tol') ou este número de iterações. Para estocástico
            solucionadores ('sgd', 'adam'), observe que isso determina o número de épocas
            (quantas vezes cada ponto de dados será usado), não o número de
            etapas de gradiente.
        '''
        
        nome_classificador = "Perceptron Multicamadas"
        print(nome_classificador)
        perceptronMulticamadas = MLPClassifier(max_iter=200, activation='tanh', hidden_layer_sizes=(10, 30, 10), solver='adam')
        perceptronMulticamadas.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = perceptronMulticamadas
        self.listaEstimadoresEnsemble.append((nome_classificador, perceptronMulticamadas))
        
        predicao = perceptronMulticamadas.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(perceptronMulticamadas, predicao, x, y, nome_classificador)
        
    def modeloRegressaoLogistica(self, x, y):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        
        Regressão Logistica
        
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Especifique a norma da pena:
            - `'none'`: nenhuma penalidade é adicionada;
            - `'l2'`: adiciona um termo de penalidade L2 e é a escolha padrão;
            - `'l1'`: adiciona um termo de penalidade L1;
            - `'elasticnet'`: ambos os termos de penalidade L1 e L2 são adicionados.
            
        C : float, default=1.0
            Parâmetro de regularização. A força da regularização é
            inversamente proporcional a C. Deve ser estritamente positivo. A penalidade
            é uma penalidade de l2 ao quadrado.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de várias saídas, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "Regressão Logistica"
        print(nome_classificador)
        regressaoLogistica = LogisticRegression(class_weight=self.class_weight, C=1, penalty='l2')
        regressaoLogistica.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = regressaoLogistica
        self.listaEstimadoresEnsemble.append((nome_classificador, regressaoLogistica))
        
        predicao = regressaoLogistica.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(regressaoLogistica, predicao, x, y, nome_classificador)
    
    def modeloSvm(self, x, y, kernel='rbf'):      
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        
        SVM - Support Vector Machine.
        A implementação é baseada em libsvm. 
        O tempo de ajuste escala pelo menos quadraticamente com o número de amostras e 
        pode ser impraticável além de dezenas de milhares de amostras. 
        
        C : float, default=1.0
            Parâmetro de regularização. A força da regularização é
            inversamente proporcional a C. Deve ser estritamente positivo. A penalidade
            é uma penalidade de l2 ao quadrado.
            
        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  default='rbf'
            Especifica o tipo de kernel a ser usado no algoritmo.
            Se nenhum for fornecido, 'rbf' será usado. Se um callable é dado, é
            usado para pré-computar a matriz do kernel a partir de matrizes de dados; aquela matriz
            deve ser um array de formato ``(n_samples, n_samples)``.
            
        gamma : {'scale', 'auto'} or float, default='scale'
            Coeficiente de kernel para 'rbf', 'poli' e 'sigmoid'.
            - se ``gamma='scale'`` (default) é passado então ele usa
              1 / (n_features * X.var()) como valor de gama,
            - se 'auto', usa 1 / n_features.
            
        probability : bool, default=False
            Se deve habilitar estimativas de probabilidade. Isso deve ser habilitado antes
            para chamar `fit`, irá desacelerar esse método, pois ele usa internamente
            Validação cruzada de 5 vezes, e `predict_proba` pode ser inconsistente com `predict`.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de várias saídas, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "SVM"
        print(nome_classificador)
        svm = SVC(kernel=kernel, class_weight=self.class_weight, C=1, gamma=0.1, probability=True)
        
        svm = svm.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = svm
        self.listaEstimadoresEnsemble.append((nome_classificador, svm))
        
        predicao = svm.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(svm, predicao, x, y, nome_classificador)
    
    # Algoritmos Ensemble -------------------------------------------------------------------------- 
    
    def modeloEnsembleVotacao(self, x, y, voting='hard'):
        
        '''
        Ensemble Votação
        estimators : list of (str, estimator) tuples
            Invocar o método ``fit`` no ``VotingClassifier`` ajustará clones
            desses estimadores originais que serão armazenados no atributo de classe
            ``self.estimators_``. Um estimador pode ser definido como ``'drop'``
            usando ``set_params``.
    
        voting : {'hard', 'soft'}, default='hard'
            Se 'hard', usa rótulos de classe previstos para votação da regra da maioria.
            Caso contrário, se 'soft', prevê o rótulo da classe com base no argmax de
            as somas das probabilidades previstas, o que é recomendado para
            um conjunto de classificadores bem calibrados.
        '''

        nome_classificador = f"Ensemble Votação {voting.title()}"
        print(nome_classificador)
        votingClassifier = VotingClassifier(estimators=self.listaEstimadoresEnsemble, voting=voting)
        votingClassifier.fit(self.x_train, self.y_train)
        
        self.dicionarioModelos[nome_classificador] = votingClassifier
        
        predicao = votingClassifier.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(votingClassifier, predicao, x, y, nome_classificador)
        
    def modeloAdaBoost(self, x, y, modelo):
        
        '''
        '''

        nome_classificador = "AdaBoost"
        print(nome_classificador)
        
        adaClassifier = AdaBoostClassifier(base_estimator=modelo, n_estimators=100, random_state=42)
        adaClassifier.fit(self.x_train, self.y_train)
        
        self.modelosTreinados[nome_classificador] = adaClassifier
        
        predicao = adaClassifier.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(adaClassifier, predicao, x, y, nome_classificador)
        
    def modeloBaggingBoost(self, x, y, modelo):
        
        '''
        '''

        nome_classificador = "BaggingBoost"
        print(nome_classificador)
        
        baggingClassifier = BaggingClassifier(base_estimator=modelo, n_estimators=100, random_state=42)
        baggingClassifier.fit(self.x_train, self.y_train)
        
        self.modelosTreinados[nome_classificador] = baggingClassifier
        
        predicao = baggingClassifier.predict(x)
        self.df_resultados.loc[nome_classificador] = self.avaliacao(baggingClassifier, predicao, x, y, nome_classificador)
