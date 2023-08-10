import pandas as pd
import numpy as np
import glob
import pickle

import seaborn as sns
from matplotlib import pyplot as plt

import math
import statistics

from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.dummy import DummyClassifier

from sklearn.tree import plot_tree
from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import auc
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer
#import xgboost as xgb

import scipy
from scipy import stats
import scipy.stats
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import shapiro
from scipy.stats import kstest
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import chisquare
from scipy.stats import skew, kurtosis 

from scipy.spatial import distance_matrix 
from scipy.cluster import hierarchy 
from scipy.spatial.distance import pdist



class FormulaRegression():

	def __init__(self, edicao_cartola):
		self.edicao_cartola = edicao_cartola
		# importar cartola
		self.file_name = f'database/2021/Cartola_2021_Individual'
		self.df_cartola = pd.read_csv(self.file_name, compression='gzip', low_memory=False)
		self.rodada = self.df_cartola['rodada_id'].max()-1

	# Carrega Dataset com histórico atletas
	def carregaDataset(self):
	    folder = '/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/ferramentas_auto_gm/Ferramentas_Automatizadas_Gato_Mestre/database/All/historico-atletas/'
	    filename = 'historico-atletas-rodada_atleta_rodada_'
	    allCartola = pd.DataFrame()
	    for i in range(2011,2020):
	        database = f'{folder}{filename}{str(i)}.csv'
	        dfano = pd.read_csv(database)
	        dfano['ano'] = i
	        allCartola = allCartola.append(dfano)
	    cartola_atual = pd.read_csv('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/ferramentas_auto_gm/Ferramentas_Automatizadas_Gato_Mestre/database/2021/Cartola_2021_Individual', compression='gzip')
	    cartola_atual = cartola_atual[['atleta_id', 'rodada_id', 'clube_id',
	           'posicao_id', 'status_id', 'pontos_num', 'preco_num', 'variacao_num',
	           'media_num', 'jogos_num']]
	    cartola_atual['ano'] = self.edicao_cartola
	    cartola_atual['atleta_rodada_id'] = self.edicao_cartola
	    allCartola = allCartola.append(cartola_atual)

	    return allCartola


	def get_ultima_pontuacao_valorizacao_jogou(self,df):
	    df_res = pd.DataFrame()
	    for i in df['atleta_id'].unique():
	        df_atleta = dfRecorte[dfRecorte['atleta_id']==i]
	        df_columns = df_atleta[['pontos_num','variacao_num']]
	        res = df_columns.mask(df_columns==0).ffill().iloc[[-1]].astype(float)
	        df_atleta['ultima_pontuacao_jogou'] = res.pontos_num.iloc[0]
	        df_atleta['ultima_valorizacao_jogou'] = res.variacao_num.iloc[0]
	        df_res = df_res.append(df_atleta).fillna(0)
	    return df_res


	def avaliaCorrelacoes(self,df):
	    plt.figure(figsize=(12,7))
	    correlacao = df.corr()
	    # sns.heatmap(correlacao, annot = True);

	    correlations = df.corr()['variacao_num'].drop('variacao_num')
	    print(correlations.sort_values())


	def selecionaAtributos(self, rodada, df):
	    df['preco_open'] = df['preco_num'] - df['variacao_num']
	    df['preco_open'] =  df['preco_open'].fillna(0)

	    df['variacao_anterior'] = df.groupby(['atleta_id'])['variacao_num'].shift(+1).replace(to_replace=0, method='ffill')
	    df['pontuacao_anterior'] = df.groupby(['atleta_id'])['pontos_num'].shift(+1).replace(to_replace=0, method='ffill')
	    df['preco_anterior'] = df.groupby(['atleta_id'])['preco_open'].shift(+1).replace(to_replace=0, method='ffill')
	    df['media_num_anterior'] = df.groupby(['atleta_id'])['media_num'].shift(+1).replace(to_replace=0, method='ffill')

	    columns = [
	        'ano',
	        'rodada_id',
	        'atleta_id',
	        'jogos_num',
	        'pontos_num',
	        'variacao_num',
	        'preco_num',
	        'media_num',
	        'variacao_anterior',
	        'preco_anterior',
	        'media_num_anterior',
	        'pontuacao_anterior'
	    ]
	    
	    return df[columns].dropna(axis=0)


	def filtro(self, allCartola, ano, rodada, minimo_jogos=False, rodada_anterior=True):
	    # Jogadores que tem mais do que 50% dos jogos e atuaram na rodada anterio
	    dfRecorte = allCartola[(allCartola['ano']==ano) & (allCartola['rodada_id']<=rodada)]
	    dfRecorte = dfRecorte[(dfRecorte['pontos_num']!=0) | (dfRecorte['variacao_num']!=0)]
	    if minimo_jogos:
	        dfRecorte = dfRecorte[dfRecorte['jogos_num']>=int(rodada/2)]
	    else:
	        dfRecorte = dfRecorte[dfRecorte['jogos_num']>=5]
	    if rodada_anterior:
	        dfRecorte = dfRecorte[(dfRecorte['pontuacao_anterior']!=0.0) | (dfRecorte['variacao_anterior']!=0.0)]
	    return dfRecorte


	def best_model(self, rodada):
		# match file
		model_selection = glob.glob(f'gmodels/pickle/valorizacao/R{rodada}*.pkl')[0]
		# load best binary classifier saved in cross_validation.py
		with open(model_selection, 'rb') as f:
			parametrizacao = pickle.load(f)
			print('REG MODEL',parametrizacao)

		# Carregando dataset
		allCartola = self.carregaDataset()
		# Selecionando ano e rodada de interesse
		ano = self.edicao_cartola
		rodada = self.rodada
		dfRecorte = self.selecionaAtributos(rodada, allCartola).fillna(0).reset_index(drop=True)
		dfRecorte = self.filtro(dfRecorte, ano, rodada).reset_index(drop=True)
		dfRecorte.head()
		# independent variables
		variables = [
            'pontos_num',
            'variacao_anterior',
            'preco_anterior',
            'pontuacao_anterior',
            'media_num_anterior',
            ]
        # features, target
		X = dfRecorte[variables] 
		y = dfRecorte['variacao_num']
		# split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #holdout
		# set pipiline from persisted model
		model = Pipeline(steps=[parametrizacao[0],parametrizacao[1], parametrizacao[2]])
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

		
	# # regressão simples
	# def linearRegression(self, X, y, recursive=False, resultados={}, production=False):
	        
	# 	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33) #holdout
	# 	gm_model = LinearRegression()
	# 	gm_model.fit(X_train, Y_train)
	# 	y_hat = gm_model.predict(X_test)
	# 	print(gm_model)
	# 	print(gm_model.coef_)
	# 	print(gm_model.intercept_)
	# 	r2 = r2_score(Y_test, y_hat)
	# 	if recursive:
	# 	    if r2 < 0.99:
	# 	        print('\nProcurando resultado melhor')
	# 	        linearRegression(X,y, True)

	# 	        print('R2:', r2_score(Y_test, y_hat))
	# 	        print('MAE: ', mean_absolute_error(Y_test, y_hat))
	# 	        print('MSE:', mean_squared_error(Y_test, y_hat))
	# 	else:
	# 	    print('R2:', r2_score(Y_test, y_hat))
	# 	    print('MAE: ', mean_absolute_error(Y_test, y_hat))
	# 	    print('MSE:', mean_squared_error(Y_test, y_hat))

	# 	return gm_model, None


	# def PolynomialRegression(self, X=None, y=None, degree=None, recursive=False, resultados={}, production=False):
	# 	# split sets
	# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #holdout

	# 	model = Pipeline(steps=[
	# 		('scaler',StandardScaler()),
	# 		('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)), 
	# 		('linear_regression', LinearRegression())])

	# 	#fit model
	# 	model.fit(X_train, y_train)

	# 	print(model['linear_regression'].coef_)
	# 	print(model['linear_regression'].intercept_)

	# 	# Extraindo a métrica R2 para treino:
	# 	y_pred_train_model = model.predict(X_train)
	# 	r2_train = r2_score(y_train, y_pred_train_model)
	# 	print(f'R2 treino: {r2_train : .6f}')

	# 	# Extraindo a métrica R2 para teste:
	# 	y_pred_test_model = model.predict(X_test)
	# 	r2_test = r2_score(y_test, y_pred_test_model)
	# 	# print(f'R2 teste: {r2_test : .6f}')
	# 	print('R2:', r2_score(y_test, y_pred_test_model))
	# 	print('MAE: ', mean_absolute_error(y_test, y_pred_test_model))
	# 	print('MSE:', mean_squared_error(y_test, y_pred_test_model))

	# 	return model['linear_regression'], model['polynomial_features']


	# #polynomial regression with regularization
	# def RigdeRegression(self, X=None, y=None, degree=None, alpha=None, recursive=False, resultados={}, production=False):
	# 	# split sets
	# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #holdout

	# 	model = Pipeline(steps=[
	# 		('scaler',StandardScaler()),
	# 		('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)), 
	# 		('ridge', Ridge(alpha=alpha))])

	# 	#reg_model = Ridge(alpha=alpha)
	# 	print(f'Regularization with alpha = {alpha}')

	# 	#fit model
	# 	model.fit(X_train, y_train)

	# 	print(model['ridge'].coef_)
	# 	print(model['ridge'].intercept_)

	# 	# Extraindo a métrica R2 para treino:
	# 	y_pred_train_model = model.predict(X_train)
	# 	r2_train = r2_score(y_train, y_pred_train_model)
	# 	print(f'R2 treino: {r2_train : .6f}')

	# 	# Extraindo a métrica R2 para teste:
	# 	y_pred_test_model = model.predict(X_test)
	# 	r2_test = r2_score(y_test, y_pred_test_model)
	# 	print(f'R2 teste: {r2_test : .6f}')
	# 	print('R2:', r2_score(y_test, y_pred_test_model))
	# 	print('MAE: ', mean_absolute_error(y_test, y_pred_test_model))
	# 	print('MSE:', mean_squared_error(y_test, y_pred_test_model))

	# 	return model['ridge'], model['polynomial_features']


	# def regressao(self, dfRecorte, ano, rodada, minimo_jogos=False, rodada_anterior=True, polynomial=False, degree=None, regularization=False, alpha=False, production=False, resultados={}):
	    
	#     print('Recorte em regressao ANTES',dfRecorte)

	#     dfRecorte = self.filtro(dfRecorte, ano, rodada, minimo_jogos, rodada_anterior)
	#     print('Recorte em regressao',dfRecorte)
	#     #dfRecorte = removeOutliers(dfRecorte, 0.2, 0.98)
	#     self.avaliaCorrelacoes(dfRecorte)
	    
	#     dfNormalizado = dfRecorte
	    
	#     variables = [
	#         'pontos_num',
	#         'variacao_anterior',
	#         'preco_anterior',
	#         'pontuacao_anterior',
	#         'media_num_anterior',
	#     ]

	#     print(variables)
	    
	#     # quadratic functions
	#     if polynomial:
	#         if regularization:
	#             # if we already have picked the model
	#             if production:
	#                 regression, polynomial = self.RigdeRegression(X=dfNormalizado[variables], y=dfNormalizado['variacao_num'], degree=degree, alpha=alpha, recursive=False, production=True)
	#             else:
	#                 # test for best results
	#                 # lista de valores a serem testados para Ridge Regression - eles penalizam o theta das equações quadráticas
	#                 alpha_values = [0.00, 0.01, 0.1, 0.5, 1.0]
	#                 for degree in [2,3]:
	#                     for alpha in alpha_values:
	#                         regression, polynomial = self.RigdeRegression(X=dfNormalizado[variables], y=dfNormalizado['variacao_num'], degree=degree, alpha=alpha, recursive=False, resultados=resultados)
	#         else:
	#             if production:
	#                 # if we already have picked the model
	#                 regression, polynomial = self.PolynomialRegression(X=dfNormalizado[variables], y=dfNormalizado['variacao_num'], degree=degree, recursive=False, production=True)
	#             else:
	#                 # test for best results
	#                 for degree in [2,3]:
	#                     regression, polynomial = self.PolynomialRegression(X=dfNormalizado[variables], y=dfNormalizado['variacao_num'], degree=degree, recursive=False, resultados=resultados)
	#     # linear functions
	#     else:
	#         if production:
	#              # if we already have picked the model
	#             regression, polynomial = self.linearRegression(dfNormalizado[variables], dfNormalizado['variacao_num'], False, production=True)
	#         else:
	#             # test results
	#             regression, polynomial = self.linearRegression(dfNormalizado[variables], dfNormalizado['variacao_num'], False, resultados=resultados)
	    
	#     return regression, polynomial


if __name__ == "__main__":

	valorizacao = FormulaRegression(2021)

	valorizacao.best_model(rodada=37)
























