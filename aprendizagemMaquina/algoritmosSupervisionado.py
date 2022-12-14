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
        
        nomeMelhorModelo = 'Floresta Rand??mica'
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
        Dicionario de modelosTreinados chaves:  Baseline Regress??o Logistica
                                                An??lise Discriminante Linear
                                                Arvore de Decis??o
                                                Floresta Rand??mica
                                                Gradient Boosting
                                                KNN
                                                Naive Bayes
                                                Perceptron Multicamadas
                                                SVM
                                                Ensemble Vota????o Hard
                                                Ensemble Vota????o Soft
        '''
        
        self.inicializacao()
        modelosTreinados = pd.read_pickle('resultados/modelosTreinados.pickle')
        
        nomeMelhorModelo = 'Arvore de Decis??o'
        print(nomeMelhorModelo)
        melhorModelo = modelosTreinados[nomeMelhorModelo]
        predicao = melhorModelo.predict(self.x_test)
        self.df_resultados.loc[nomeMelhorModelo] = self.avaliacao(melhorModelo, predicao, self.x_test, self.y_test, nomeMelhorModelo)
        self.dicionarioModelos[nomeMelhorModelo] = melhorModelo
        
        nomeMelhorEnsemble = 'Ensemble Vota????o Soft'
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
        
        nome_classificador = "Baseline Regress??o Logistica"
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
        
        An??lise Discriminante Linear
        Um classificador com um limite de decis??o linear, 
        gerado ajustando densidades condicionais  de classe aos dados e usando a regra de Bayes.
        
        solver : {'svd', 'lsqr', 'eigen'}, default='svd'
            Solver para usar, valores poss??veis:
               - 'svd': Decomposi????o de valor singular (padr??o).
                N??o calcula a matriz de covari??ncia, portanto, este solver ??
                recomendado para dados com um grande n??mero de recursos.
               - 'lsqr': solu????o de m??nimos quadrados.
                Pode ser combinado com o estimador de encolhimento ou de covari??ncia personalizado.
               - 'eigen': Decomposi????o de valores pr??prios.
                Pode ser combinado com o estimador de encolhimento ou de covari??ncia personalizado.
    
        shrinkage : 'auto' or float, default=None
            Par??metro shrinkage, valores poss??veis:
               - Nenhum: sem encolhimento (padr??o).
               - 'auto': encolhimento autom??tico usando o lema de Ledoit-Wolf.
               - flutuar entre 0 e 1: par??metro de retra????o fixo.
        '''
        
        nome_classificador = "An??lise Discriminante Linear"
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
        
        Arvore de Decis??o
        Um classificador de ??rvore de decis??o.
        
        criterion : {"gini", "entropy"}, default="gini"
            A fun????o para medir a qualidade de uma divis??o. Os crit??rios suportados s??o
            "gini" para a impureza de Gini e "entropy" para o ganho de informa????o.
            
        splitter : {"best", "random"}, default="best"
            A estrat??gia utilizada para escolher a divis??o em cada n??. Compat??vel
            estrat??gias s??o "best" para escolher a melhor divis??o e "random" para escolher.
            
        max_depth : int, default=None
            A profundidade m??xima da ??rvore. Se Nenhum, ent??o os n??s s??o expandidos at??
            todas as folhas s??o puras ou at?? que todas as folhas contenham menos de
            amostras min_samples_split.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de v??rias sa??das, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "Arvore de Decis??o"
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
        
        Classificador Floresta Rand??mica / Floresta Aleat??ria - Random forest - RF
        Uma floresta aleat??ria ?? um meta estimador que ajusta v??rios classificadores de ??rvore de decis??o 
        em v??rias subamostras do conjunto de dados e usa a m??dia para melhorar a precis??o preditiva e 
        controlar o ajuste excessivo. O tamanho da subamostra ?? controlado com o 
        par??metro max_samples se bootstrap=True(padr??o), caso contr??rio, 
        todo o conjunto de dados ?? usado para construir cada ??rvore.
        
        n_estimators : int, default=100
            O n??mero de ??rvores na floresta.
            
        criterion : {"gini", "entropy"}, default="gini"
            A fun????o para medir a qualidade de uma divis??o. Os crit??rios suportados s??o
            "gini" para a impureza de Gini e "entropy" para o ganho de informa????o.
            
        splitter : {"best", "random"}, default="best"
            A estrat??gia utilizada para escolher a divis??o em cada n??. Compat??vel
            estrat??gias s??o "best" para escolher a melhor divis??o e "random" para escolher.
            
        max_depth : int, default=None
            A profundidade m??xima da ??rvore. Se Nenhum, ent??o os n??s s??o expandidos at??
            todas as folhas s??o puras ou at?? que todas as folhas contenham menos de
            amostras min_samples_split.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de v??rias sa??das, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "Floresta Rand??mica"
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
        GB constr??i um modelo aditivo de forma progressiva; 
        ele permite a otimiza????o de fun????es de perda diferenci??veis ??????arbitr??rias. 
        Em cada est??gio as n_classes_ ??rvores de regress??o s??o 
        ajustadas no gradiente negativo da fun????o de perda de desvio binomial ou multinomial. 
        A classifica????o bin??ria ?? um caso especial em que apenas uma ??nica ??rvore de regress??o ?? induzida.
        
        loss : {'deviance', 'exponential'}, default='deviance'
            A fun????o de perda a ser otimizada. 'deviance' refere-se a
             desvio (= regress??o log??stica) para classifica????o
             com sa??das probabil??sticas. Para gradiente 'exponential' de perda
             impulsionar recupera o algoritmo AdaBoost.
             
        criterion : {'friedman_mse', 'squared_error', 'mse', 'mae'}, default='friedman_mse'
            A fun????o para medir a qualidade de uma divis??o. Crit??rios compat??veis
            s??o 'friedman_mse' para o erro quadr??tico m??dio com melhoria
            pontua????o de Friedman, 'squared_error' para erro quadr??tico m??dio e 'mae'
            para o erro absoluto m??dio. O valor padr??o de 'friedman_mse' ??
            geralmente o melhor, pois pode fornecer uma melhor aproxima????o em alguns casos.
             
        max_depth : int, default=None
            A profundidade m??xima da ??rvore. Se Nenhum, ent??o os n??s s??o expandidos at??
            todas as folhas s??o puras ou at?? que todas as folhas contenham menos de
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
        
        KNN - k-vizinhos mais pr??ximos
        Classificador implementando o voto dos k-vizinhos mais pr??ximos.
        
        n_neighbors : int, default=5
            N??mero de vizinhos a serem usados ??????por padr??o para consultas :meth:`kneighbors`.
            
        weights : {'uniform', 'distance'} or callable, default='uniform'
            Fun????o de peso usada na previs??o. Valores poss??veis:
            - 'uniform' : pesos uniformes. Todos os pontos em cada bairro
              s??o ponderadas igualmente.
            - 'distance' : pontos de peso pelo inverso de sua dist??ncia.
              neste caso, os vizinhos mais pr??ximos de um ponto de consulta ter??o um
              maior influ??ncia do que os vizinhos mais distantes.
            - [callable] : uma fun????o definida pelo usu??rio que aceita um
              array de dist??ncias e retorna um array da mesma forma contendo os pesos.
              
        metric : str or callable, default='minkowski'
            A m??trica de dist??ncia a ser usada para a ??rvore. A m??trica padr??o ??
            minkowski, e com p=2 ?? equivalente ao padr??o euclidiano
            m??trica. 
            Se a m??trica for "precomputed", assume-se que X ?? uma matriz de dist??ncia e
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
            Parte da maior varia????o de todos os recursos que ?? adicionado ao
            desvios para estabilidade de c??lculo.
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
        Este modelo otimiza a fun????o log-loss usando LBFGS ou gradiente descendente estoc??stico.
        
        hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
            O i-??simo elemento representa o n??mero de neur??nios no i-??simo camada oculta.
            
        activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
            Fun????o de ativa????o para a camada oculta.
            - 'identity', ativa????o sem opera????o, ??til para implementar gargalo linear, retorna f(x) = x
            - 'logistic', a fun????o sigm??ide log??stica, retorna f(x) = 1 / (1 + exp(-x)).
            - 'tanh', a fun????o tan hiperb??lica, retorna f(x) = tanh(x).
            - 'relu', a fun????o de unidade linear retificada, retorna f(x) = max(0, x)
            
        solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
            TO solucionador para otimiza????o de peso.
            - 'lbfgs' ?? um otimizador na fam??lia de m??todos quase-Newton.
            - 'sgd' refere-se ?? descida de gradiente estoc??stica.
            - 'adam' refere-se a um otimizador baseado em gradiente estoc??stico proposto por Kingma, Diederik e Jimmy Ba
            
        max_iter : int, default=200
            N??mero m??ximo de itera????es (??pocas). O solver itera at?? a converg??ncia
            (determinado por 'tol') ou este n??mero de itera????es. Para estoc??stico
            solucionadores ('sgd', 'adam'), observe que isso determina o n??mero de ??pocas
            (quantas vezes cada ponto de dados ser?? usado), n??o o n??mero de
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
        
        Regress??o Logistica
        
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Especifique a norma da pena:
            - `'none'`: nenhuma penalidade ?? adicionada;
            - `'l2'`: adiciona um termo de penalidade L2 e ?? a escolha padr??o;
            - `'l1'`: adiciona um termo de penalidade L1;
            - `'elasticnet'`: ambos os termos de penalidade L1 e L2 s??o adicionados.
            
        C : float, default=1.0
            Par??metro de regulariza????o. A for??a da regulariza????o ??
            inversamente proporcional a C. Deve ser estritamente positivo. A penalidade
            ?? uma penalidade de l2 ao quadrado.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de v??rias sa??das, uma lista de dicts pode ser fornecida no mesmo
            ordem como as colunas de y.
        '''
        
        nome_classificador = "Regress??o Logistica"
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
        A implementa????o ?? baseada em libsvm. 
        O tempo de ajuste escala pelo menos quadraticamente com o n??mero de amostras e 
        pode ser impratic??vel al??m de dezenas de milhares de amostras. 
        
        C : float, default=1.0
            Par??metro de regulariza????o. A for??a da regulariza????o ??
            inversamente proporcional a C. Deve ser estritamente positivo. A penalidade
            ?? uma penalidade de l2 ao quadrado.
            
        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  default='rbf'
            Especifica o tipo de kernel a ser usado no algoritmo.
            Se nenhum for fornecido, 'rbf' ser?? usado. Se um callable ?? dado, ??
            usado para pr??-computar a matriz do kernel a partir de matrizes de dados; aquela matriz
            deve ser um array de formato ``(n_samples, n_samples)``.
            
        gamma : {'scale', 'auto'} or float, default='scale'
            Coeficiente de kernel para 'rbf', 'poli' e 'sigmoid'.
            - se ``gamma='scale'`` (default) ?? passado ent??o ele usa
              1 / (n_features * X.var()) como valor de gama,
            - se 'auto', usa 1 / n_features.
            
        probability : bool, default=False
            Se deve habilitar estimativas de probabilidade. Isso deve ser habilitado antes
            para chamar `fit`, ir?? desacelerar esse m??todo, pois ele usa internamente
            Valida????o cruzada de 5 vezes, e `predict_proba` pode ser inconsistente com `predict`.
            
        class_weight : dict, list of dict or "balanced", default=None
            Pesos associados a classes no formato ``{class_label: weight}``.
            Se None, todas as classes devem ter peso um. Por
            problemas de v??rias sa??das, uma lista de dicts pode ser fornecida no mesmo
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
        Ensemble Vota????o
        estimators : list of (str, estimator) tuples
            Invocar o m??todo ``fit`` no ``VotingClassifier`` ajustar?? clones
            desses estimadores originais que ser??o armazenados no atributo de classe
            ``self.estimators_``. Um estimador pode ser definido como ``'drop'``
            usando ``set_params``.
    
        voting : {'hard', 'soft'}, default='hard'
            Se 'hard', usa r??tulos de classe previstos para vota????o da regra da maioria.
            Caso contr??rio, se 'soft', prev?? o r??tulo da classe com base no argmax de
            as somas das probabilidades previstas, o que ?? recomendado para
            um conjunto de classificadores bem calibrados.
        '''

        nome_classificador = f"Ensemble Vota????o {voting.title()}"
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
