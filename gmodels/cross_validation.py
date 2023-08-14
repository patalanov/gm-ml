import pandas as pd
import numpy as np
import pickle
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import auc

from statsmodels.formula.api import ols

#import gm module
from gmodels.binary_classification import BinaryClassifier
from gmodels.regression_valorizacao import FormulaRegression
from gmodels.xG import xG



class CrossValidate:

    def __init__(self, edicao_cartola, curr_rodada_id):
        self.edicao_cartola = edicao_cartola
        self.curr_rodada_id = curr_rodada_id

        self.resultados = {
            "Algoritmo" :[],
            "Parametrização" :[],
            "Acuracia" :[],
            "Recall" :[],
            #"Roc_auc": []
        }
        self.resultados_modelos_nao_padronizaveis = {
            "Algoritmo" :[],
            "Parametrização" :[],
            "Acuracia" :[],
            "Recall" :[],
            #"Roc_auc" :[]
        }
        self.resultados_regressoes = {
            "Algoritmo" :[],
            "Parametrização" :[],
            "R2" :[],
            "MAE" :[],
            "MSE" :[],
        }
        # definindo as diferentes configurações para o método SVC
        self.modelos_svm = {
            "Kernel rbf, gamma auto": SVC(kernel = 'rbf', gamma='auto', probability=True),
            "Kernel linear": SVC(kernel = 'linear', probability=True),
            "Kernel sigmoide": SVC(kernel = 'sigmoid', probability=True),
            "Kernel polinomial grau 2": SVC(kernel = 'poly', degree=2, probability=True),
            "Kernel polinomial grau 3": SVC(kernel = 'poly', degree=3, probability=True)
        }
        # definindo as diferentes configurações para o método NB
        self.modelos_gaussian = {
            "Var smoothing 1e-9": GaussianNB(var_smoothing=1e-9),
            "Var smoothing 1e-8": GaussianNB(var_smoothing=1e-8),
            "Var smoothing 1e-7": GaussianNB(var_smoothing=1e-7),
            "Var smoothing 1e-6": GaussianNB(var_smoothing=1e-6),
            "Var smoothing 1e-5": GaussianNB(var_smoothing=1e-5)
        }
        # definindo as diferentes configurações para o método KNN
        self.modelos_knn = {
            "N=3": KNeighborsClassifier(n_neighbors=3),
            "N=5": KNeighborsClassifier(n_neighbors=5),
            "N=7": KNeighborsClassifier(n_neighbors=7),
            "N=9": KNeighborsClassifier(n_neighbors=9),
            "N=11": KNeighborsClassifier(n_neighbors=11),
        }
        # definindo as diferentes configurações para o método Decision Tree
        self.modelos_dt = {
            "Depth=3": DecisionTreeClassifier(max_depth=3),
            "Depth=4": DecisionTreeClassifier(max_depth=4),
            "Depth=5": DecisionTreeClassifier(max_depth=5),
            "Depth=6": DecisionTreeClassifier(max_depth=6),
            "Depth=7": DecisionTreeClassifier(max_depth=7)
        }
        # definindo as diferentes configurações para o método Regressão Logística
        self.modelos_logis_regression = {
            "Logistic Regression": LogisticRegression(),
            "Logistic Regression, fit_intercetp=False": LogisticRegression(fit_intercept=False,solver='lbfgs'),
            "Logistic Regression, Weights Balanced": LogisticRegression(class_weight='balanced',solver='lbfgs'),
        }

        self.modelos_xgboost = {
            "XGBoost Default": XGBClassifier(),
            "XGBoost with max_depth=3, learning_rate=0.1": XGBClassifier(max_depth=3, learning_rate=0.1),
            "XGBoost with subsample=0.7, colsample_bytree=0.8": XGBClassifier(subsample=0.7, colsample_bytree=0.8),
            "XGBoost with scale_pos_weight for imbalanced": XGBClassifier(scale_pos_weight=3)
            # ... add more configurations as needed
        }

        self.modelos_linear_regression = {
            "linear_regression": [(None, None),(None, None),('linear_regression', LinearRegression())],
            "polynomial_features_degree 2": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=2, include_bias=False)),('linear_regression', LinearRegression())],
            "polynomial_features_degree 3": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=3, include_bias=False)),('linear_regression', LinearRegression())],
            "ridge_alpha_0.00_degree_2": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=2, include_bias=False)),('linear_regression', Ridge(alpha=0.00))],
            "ridge_alpha_0.01_degree_2": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=2, include_bias=False)),('linear_regression', Ridge(alpha=0.01))],
            "ridge_alpha_0.1_degree_2": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=2, include_bias=False)),('linear_regression', Ridge(alpha=0.1))],
            "ridge_alpha_0.5_degree_2": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=2, include_bias=False)),('linear_regression', Ridge(alpha=0.5))],
            "ridge_alpha_1.0_degree_2": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=2, include_bias=False)),('linear_regression', Ridge(alpha=1.0))],
            "ridge_alpha_0.00_degree_3": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=3, include_bias=False)),('linear_regression', Ridge(alpha=0.00))],
            "ridge_alpha_0.01_degree_3": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=3, include_bias=False)),('linear_regression', Ridge(alpha=0.01))],
            "ridge_alpha_0.1_degree_3": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=3, include_bias=False)),('linear_regression', Ridge(alpha=0.1))],
            "ridge_alpha_0.5_degree_3": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=3, include_bias=False)),('linear_regression', Ridge(alpha=0.5))],
            "ridge_alpha_1.0_degree_3": [('scaler', StandardScaler()), ('polynomial_features',PolynomialFeatures(degree=3, include_bias=False)),('linear_regression', Ridge(alpha=1.0))],
        }


    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)


    def avaliar_modelos(self, tipo, parametrizacoes, X, y, resultados):

        """Avalia modelos utilizando 10-fold cross-validation

        Essa função recebe um conjunto de parametrizações, um conjunto de 
        atributos e labels e popula uma estrutura de resultados.    
        """
        # Vamos iterar sobre cada parametrização no dicionário.
        # Ao adicionar .items(), vamos iterar sobre todos os pares
        # (chave, valor) do dicionário:
        for nome, modelo in parametrizacoes.items():
            seed = 0
            np.random.seed(seed)
            print("Avaliando parametrização:", nome)
            print("\tProgresso: [", end = '')
            # Vamos padronizar nossos dados com o StandardScaler
            scaler = StandardScaler()
            # StratifiedKFold irá separar nossos dados em K splits estratificados,
            # ou seja, K splits onde a distribuição das classes será igual ao dos
            # dados originais. Shuffle irá embaralhar nossos dados antes de efetuar
            # o split.
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            # As 3 listas a seguir irão armazenar as métricas de acurácia, recall e
            # área sob a curva ROC de cada fold. Usaremos essas listas para calcular
            # a média de cada métrica.
            accs_scores = []
            recall_scores = []        
            roc_aucs_scores = []
            # O método split irá retornar índices que nós usaremos pra indexar os
            # conjuntos X e y. Por exemplo, se tivermos 10 exemplos no nosso conjunto
            # de dados e quisermos realizar 5 splits, uma possível divisão de splits
            # seria [1, 2], [0, 3], [4, 9], [5, 8], [6, 7]. Então para cada iteração
            # do for, o método split separa 1 desses splits para treino e o resto 
            # para teste (ex. teste = [1, 2], treino = [0, 3, 4, 9, 5, 8, 6, 7]). O 
            # loop for acaba depois que todos os splits forem usados para teste.        
            for treino, teste in cv.split(X, y):
                # Fit apenas no conjunto de treino:
                scaler.fit(X[treino])
                # Vamos escalar tanto os dados de treino quanto de teste.
                X_treino_escalado = scaler.transform(X[treino])
                X_teste_escalado = scaler.transform(X[teste])
                # Fit do modelo nos dados de treino:
                modelo.fit(X_treino_escalado, y[treino])
                # Calculo das métricas do fold. Armazenamos elas nas listas que
                # definimos anteriormente.
                y_pred = modelo.predict(X_teste_escalado)                    
                accs_scores.append(accuracy_score(y[teste], y_pred))
                recall_scores.append(recall_score(y[teste], y_pred, average=None))
                # y_score calculado como especificado em:
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
                y_score = modelo.predict_proba(X_teste_escalado)
                #roc_aucs_scores.append(roc_auc_score(y[teste], y_score, multi_class='ovr'))
                # Barra de progresso             
                print("#", end = '')
            print("]")
            # Adicionando média dos folds aos resultados:
            self.resultados['Algoritmo'].append(tipo)
            self.resultados["Parametrização"].append(nome)
            self.resultados["Acuracia"].append(np.mean(accs_scores))
            self.resultados["Recall"].append(np.mean(recall_scores))
            #self.resultados["Roc_auc"].append(np.mean(roc_aucs_scores))

    def avaliar_modelos_nao_padronizados(self, tipo, parametrizacoes, X, y, resultados):

        """Avalia modelos utilizando 10-fold cross-validation

        Essa função recebe um conjunto de parametrizações, um conjunto de 
        atributos e labels e popula uma estrutura de resultados.    
        """
        # Vamos iterar sobre cada parametrização no dicionário.
        # Ao adicionar .items(), vamos iterar sobre todos os pares
        # (chave, valor) do dicionário:
        for nome, modelo in parametrizacoes.items():
            seed = 0
            np.random.seed(seed)
            print("Avaliando parametrização:", nome)
            print("\tProgresso: [", end = '')
            # Vamos padronizar nossos dados com o StandardScaler
            #scaler = StandardScaler()
            # StratifiedKFold irá separar nossos dados em K splits estratificados,
            # ou seja, K splits onde a distribuição das classes será igual ao dos
            # dados originais. Shuffle irá embaralhar nossos dados antes de efetuar
            # o split.
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            # As 3 listas a seguir irão armazenar as métricas de acurácia, recall e
            # área sob a curva ROC de cada fold. Usaremos essas listas para calcular
            # a média de cada métrica.
            accs_scores = []
            recall_scores = []        
            roc_aucs_scores = []
            # O método split irá retornar índices que nós usaremos pra indexar os
            # conjuntos X e y. Por exemplo, se tivermos 10 exemplos no nosso conjunto
            # de dados e quisermos realizar 5 splits, uma possível divisão de splits
            # seria [1, 2], [0, 3], [4, 9], [5, 8], [6, 7]. Então para cada iteração
            # do for, o método split separa 1 desses splits para treino e o resto 
            # para teste (ex. teste = [1, 2], treino = [0, 3, 4, 9, 5, 8, 6, 7]). O 
            # loop for acaba depois que todos os splits forem usados para teste.        
            for treino, teste in cv.split(X, y):
                # Fit do modelo nos dados de treino:
                modelo.fit(X[treino], y[treino])
                # Calculo das métricas do fold. Armazenamos elas nas listas que
                # definimos anteriormente.
                y_pred = modelo.predict(X[teste])                    
                accs_scores.append(accuracy_score(y[teste], y_pred))
                recall_scores.append(recall_score(y[teste], y_pred, average=None))
                # y_score calculado como especificado em:
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
                y_score = modelo.predict_proba(X[teste])
                #roc_aucs_scores.append(roc_auc_score(y[teste], y_score))
                # Barra de progresso             
                print("#", end = '')
            print("]")
            # Adicionando média dos folds aos resultados:
            self.resultados_modelos_nao_padronizaveis['Algoritmo'].append(tipo)
            self.resultados_modelos_nao_padronizaveis["Parametrização"].append(nome)
            self.resultados_modelos_nao_padronizaveis["Acuracia"].append(np.mean(accs_scores))
            self.resultados_modelos_nao_padronizaveis["Recall"].append(np.mean(recall_scores))
            #self.resultados_modelos_nao_padronizaveis["Roc_auc"].append(np.mean(roc_aucs_scores))

    def avaliar_modelos_de_regressao(self, tipo, parametrizacoes, X, y, resultados):

        for nome, parametro in parametrizacoes.items():

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #holdout

            if parametro[0] == (None, None):
                model = LinearRegression()
            else:
                model = Pipeline(steps=[parametro[0],parametro[1], parametro[2]])
                #fit model
                model.fit(X_train, y_train)
                # Extraindo a métrica R2 para treino:
                y_pred_train_model = model.predict(X_train)
                r2_train = r2_score(y_train, y_pred_train_model)
                print(f'R2 treino: {r2_train : .6f}')

                # Extraindo a métrica R2 para teste:
                y_pred_test_model = model.predict(X_test)
                r2_test = r2_score(y_test, y_pred_test_model)
                print(f'R2 teste: {r2_test : .6f}')
                print('R2:', r2_score(y_test, y_pred_test_model))
                print('MAE: ', mean_absolute_error(y_test, y_pred_test_model))
                print('MSE:', mean_squared_error(y_test, y_pred_test_model))

                # Adicionando resultados:
                self.resultados_regressoes['Algoritmo'].append(nome)
                self.resultados_regressoes['Parametrização'].append(parametro)
                self.resultados_regressoes["R2"].append(r2_score(y_test, y_pred_test_model))
                self.resultados_regressoes["MAE"].append(mean_absolute_error(y_test, y_pred_test_model))
                self.resultados_regressoes["MSE"].append(mean_squared_error(y_test, y_pred_test_model))


    def model_persistence(self, edicao_cartola, rodada, model):

        if model == 'binary':
        
            atletas = BinaryClassifier(edicao_cartola, rodada)
            try:
                df_position, X, y = atletas.select_features_and_targets()
            except ValueError:
                print("Input contains NaN, infinity, or a value too large for dtype('float64').")
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())


            #print (df_position, X, y)

            self.avaliar_modelos('SVM', self.modelos_svm, X, y, self.resultados)
            self.avaliar_modelos("GaussianNB", self.modelos_gaussian, X, y, self.resultados)
            self.avaliar_modelos_nao_padronizados('KNN', self.modelos_knn, X, y, self.resultados_modelos_nao_padronizaveis)
            self.avaliar_modelos_nao_padronizados("Decision Trees", self.modelos_dt, X, y, self.resultados_modelos_nao_padronizaveis)

            resultados_df = pd.DataFrame.from_dict(self.resultados)
            #print(resultados_df)
            resultados_nao_padronizados_df = pd.DataFrame.from_dict(self.resultados_modelos_nao_padronizaveis)
            #print(resultados_nao_padronizados_df)    
            all_parameters = pd.concat([resultados_df, resultados_nao_padronizados_df]).sort_values(by='Acuracia', ascending=False).reset_index(drop=True)
            print (all_parameters)
            
            algoritmo = (all_parameters.loc[0]['Algoritmo'])
            parametrizacao = (all_parameters.loc[0]['Parametrização'])

            print('Algoritmo',algoritmo)
            print('Parametrizacao',parametrizacao)
            # now you can save it to a file
            all_parameters.to_csv(f'gmodels/csv/R{rodada}_binary_parametrization.csv')

            # unpack all model dictionaries into one
            all_models = {**self.modelos_svm, **self.modelos_gaussian, **self.modelos_knn, **self.modelos_dt}

            print (all_models[parametrizacao])
            # now you can save best round model to a file
            with open(f'gmodels/pickle/{model}/R{rodada}_{algoritmo}_{parametrizacao}.pkl', 'wb') as f:
                pickle.dump(all_models[parametrizacao], f)

            # reiniciar
            self.resultados_modelos_nao_padronizaveis = {
                "Algoritmo" :[],
                "Parametrização" :[],
                "Acuracia" :[],
                "Recall" :[],
                #"Roc_auc" :[]
            }

            # best_roc_aucs = resultados_df.groupby(["Algoritmo"])["Roc_auc"].agg([ ("Roc_auc", max) ])
            # print(best_roc_aucs)

            # melhores_parametrizacoes = {}
            # for linha in best_roc_aucs.itertuples():
            #     tipo_algo = linha[0]
            #     melhor_valor = linha[1]    
            #     # Colocamos iloc[0] no final para pegar a primeira ocorrencia, pois podemos
            #     # ter mais de uma mesma parametrização com mesmo valor roc_aoc:
            #     melhores_parametrizacoes[tipo_algo] = resultados_df.query(
            #         "(Algoritmo==@tipo_algo) & (Roc_auc==@melhor_valor)").iloc[0]["Parametrização"]
                
            #     print("Melhor parametrização do", tipo_algo, "=", melhores_parametrizacoes[tipo_algo])
                
            # melhor_gaussiannb = modelos_gaussian[melhores_parametrizacoes["GaussianNB"]]
            # print(melhor_gaussiannb)

            # melhor_KNN = modelos_knn[melhores_parametrizacoes["KNN"]]
            # print(melhor_KNN)

            # melhor_svm = modelos_svm[melhores_parametrizacoes["SVM"]]
            # print(melhor_svm)

            # max_roc_auc_posse = resultados_posse_df["Roc_auc"].max()
            # print(max_roc_auc_posse)

            # # Usamos o iloc para pegar apenas o primeiro resultado
            # melhor_parametrizacao_posse = resultados_posse_df.query("Roc_auc == @max_roc_auc").iloc[0]
            # print(melhor_parametrizacao_posse)

            # melhor_parametrizacao_posse = resultados_posse_df.loc[resultados_posse_df["Roc_auc"].idxmax()]
            # print(melhor_parametrizacao_posse)


        elif model == 'xG':

            xg = xG(self.edicao_cartola)

            X = xg.df_finalizacoes[['Distance','Angle Radians','header']].copy().values
            y = xg.df_finalizacoes[['Goal']].values.reshape(-1)

            print(X.shape)
            print(y.shape)

            #self.avaliar_modelos_nao_padronizados('Logistic Regression', self.modelos_logis_regression, X, y, self.resultados_modelos_nao_padronizaveis)
            self.avaliar_modelos_nao_padronizados('Logistic Regression', self.modelos_xgboost, X, y, self.resultados_modelos_nao_padronizaveis)

            resultados_nao_padronizados_df = pd.DataFrame.from_dict(self.resultados_modelos_nao_padronizaveis)
            
            algoritmo = (resultados_nao_padronizados_df.loc[0]['Algoritmo'])
            parametrizacao = (resultados_nao_padronizados_df.loc[0]['Parametrização'])

            print('Algoritmo',algoritmo)
            print('Parametrizacao',parametrizacao)
            # now you can save it to a file
            resultados_nao_padronizados_df.to_csv(f'gmodels/csv/R{rodada}_xG_parametrization.csv')

            print (self.modelos_logis_regression[parametrizacao])
            # now you can save best round model to a file
            with open(f'gmodels/pickle/{model}/R{rodada}_{algoritmo}_{parametrizacao}.pkl', 'wb') as f:
                pickle.dump(self.modelos_logis_regression[parametrizacao], f)

            # reiniciar
            self.resultados_modelos_nao_padronizaveis = {
                "Algoritmo" :[],
                "Parametrização" :[],
                "Acuracia" :[],
                "Recall" :[],
                #"Roc_auc" :[]
            }


        elif model == 'valorizacao':

            valorizacao = FormulaRegression(self.edicao_cartola)

            # Carregando dataset
            allCartola = valorizacao.carregaDataset()
            # Selecionando ano e rodada de interesse
            ano = self.edicao_cartola
            rodada = self.curr_rodada_id
            # selecionar atributos
            dfRecorte = valorizacao.selecionaAtributos(rodada, allCartola).fillna(0)
            # filtrar
            dfRecorte = valorizacao.filtro(dfRecorte, ano, rodada).reset_index(drop=True)
            dfRecorte.head()

            #Ajusta o modelo de regressão linear múltipla com variacao_num como variável resposta
            mod = ols('dfRecorte.variacao_num ~ dfRecorte.pontos_num + dfRecorte.variacao_anterior + dfRecorte.preco_anterior + dfRecorte.pontuacao_anterior + dfRecorte.media_num_anterior',data=dfRecorte)
            res = mod.fit()
            print(res.summary())

            variables = [
            'pontos_num',
            'variacao_anterior',
            'preco_anterior',
            'pontuacao_anterior',
            'media_num_anterior',
            ]

            X = dfRecorte[variables] 
            y = dfRecorte['variacao_num']

            self.avaliar_modelos_de_regressao('regressao', self.modelos_linear_regression, X, y, self.resultados_regressoes)

            resultados_regressoes_df = pd.DataFrame.from_dict(self.resultados_regressoes).sort_values(by='MAE', ascending=False).reset_index(drop=True)
            print(resultados_regressoes_df)

            resultados_regressoes_df.to_csv(f'gmodels/csv/R{rodada}_valorizacao_parametrization.csv')

            algoritmo = (resultados_regressoes_df.loc[0]['Algoritmo'])
            parametrizacao = (resultados_regressoes_df.loc[0]['Parametrização'])

            print(algoritmo,parametrizacao)

            # now you can save best round model to a file
            with open(f'gmodels/pickle/{model}/R{rodada}_{algoritmo}.pkl', 'wb') as f:
                pickle.dump(parametrizacao, f)


if __name__ == "__main__":

    models = CrossValidate()
    #models.model_persistence('2021', 37, 'binary')
    #models.model_persistence('2021', 37, 'xG')
    models.model_persistence('2021', 37, 'valorizacao')



















