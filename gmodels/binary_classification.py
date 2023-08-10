import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from icecream import ic
import glob
import pickle



class BinaryClassifier:

	def __init__(self, edicao_cartola, curr_rodada_id):
		self.edicao_cartola = edicao_cartola
		self.curr_rodada_id = curr_rodada_id
		# importar cartola
		self.file_name = f'database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}_Individual'
		self.df_cartola = pd.read_csv(self.file_name, compression='gzip', low_memory=False)
		self.rodada = self.df_cartola['rodada_id'].max()


	def preprocess_dataset(self, curr_rodada_id):
		# extrair jgadores que não jogaram nas rodadas
		df_cartola = self.df_cartola[(self.df_cartola['variacao_num'] != 0.0) | (self.df_cartola['pontos_num'] != 0.0)] 
		# df_cartola.info()
		# df_cartola.describe()
		############################################## dataset #######################################
		#rodada atual
		rodada = curr_rodada_id
		players_attrs = pd.DataFrame()
		# get dataset, file by file and append to container
		for r in range(1,rodada+1):
		    df_rodada = pd.read_csv(f'gmodels/dataset/{self.edicao_cartola}/R{r}.csv')
		    df_rodada['rodada_id'] = r
		    players_attrs = players_attrs.append(df_rodada,sort=False)
		    #print (f'UPLOADED ROUND {r}')
		print(players_attrs.info())
		print(players_attrs.describe())

		# o parâmetro how poderia ter outros valores que também funcionam
		jogadores = df_cartola.merge(players_attrs, how='right', on='atleta_id')

		# Usando a moda ou média dependendo do tipo do atributo.
		jogadores = jogadores.apply(lambda x:x.fillna(
		    x.mode()[0] if x.dtype == 'object' else x.mean(), axis=0))

		for column in jogadores.columns:
			if 'atleta_id' not in column:
				try:
					print('Atributo: ', column, ' Moda: ', jogadores[column].mode()[0])
				except:
					pass

		df_players_train = players_attrs.copy()
		# testar com melhor jogador do campeonato
		#print(df_train[df_train['apelido']=='Arrascaeta'])
		# tratar dados faltantes
		df_players_train['prob_vitoria'] = df_players_train.apply(lambda x: x['prob_vitoria_mandante'] if not np.isnan(x['prob_vitoria_mandante']) else x['prob_vitoria_visitante'], axis=1)
		df_players_train['media_movel'] = df_players_train.apply(lambda x: x['media_movel_mandante'] if not np.isnan(x['media_movel_mandante']) else x['media_movel_visitante'], axis=1)

		if 'XP_min_para_valorizacao' not in df_players_train.columns:
			df_players_train['XP_min_para_valorizacao'] = 0
		if 'ranking_clube' not in df_players_train.columns:
			df_players_train['ranking_clube'] = 0
		if 'odds_home' not in df_players_train.columns:
			df_players_train['odds_home'] = 0
		if 'odds_away' not in df_players_train.columns:
			df_players_train['odds_away'] = 0
		if 'prob_sg_mandante' not in df_players_train.columns:
			df_players_train['prob_sg_mandante'] = 0
		if 'prob_sg_visitante' not in df_players_train.columns:
			df_players_train['prob_sg_visitante'] = 0
		if 'XP' not in df_players_train.columns:
			df_players_train['XP'] = 0

		df_players_train['odds'] = df_players_train.apply(lambda x: x['odds_home'] if not np.isnan(x['odds_home']) else x['odds_away'], axis=1)
		df_players_train['prob_SG'] = df_players_train.apply(lambda x: x['prob_sg_mandante'] if not np.isnan(x['prob_sg_mandante']) else x['prob_sg_visitante'], axis=1)


		col_nans = ['fase','prob_vitoria_mandante','prob_vitoria_visitante','prob_vitoria', 'XP_min_para_valorizacao','prob_vitoria_visitante',\
		            'ranking_clube', 'media_movel', 'odds_home', 'odds_away', 'odds','prob_sg_mandante', 'prob_sg_visitante','prob_SG','XP']

		for col in col_nans:
		    df_players_train[col] = df_players_train[col].fillna(df_players_train[col].mean())

		# replace all nan with mean
		df_players_train = df_players_train.fillna(df_players_train.mean())
		# get points achieved by player the following round 
		df_players_train['points_next'] = df_players_train.groupby('atleta_id')['pontos_num'].shift(-1)
		# drop null
		#df_players_train = df_players_train[df_players_train['points_next'].notnull()]

		print('Binary Train',df_players_train.info())

		# extract only relevant features from dataframe
		df_features = df_players_train[['rodada_id', 
										'atleta_id',
										'apelido', 
										'clube',
										'posicao',
										'preco_open',
										'pontos_num', 
										'variacao_num', 
										'ieo_gm_mean', 
										'iao_gm_mean',
										'ied_gm_mean', 
										'ieg_gm_mean',
										'iro_gm', 
										'irp_gm', 
										'ippc_gm_mean', 
										'ranking_clube', 
										'fase', 
										'prob_vitoria_mandante',
										'prob_vitoria_visitante', 
										'prob_vitoria', 
										'mando', 
										'XP_mandante', 
										'XP_visitante', 
										'XP',
										'media_movel_mandante',
										'media_movel_visitante',
										'media_movel',
										'odds',
										'points_next']].copy()

		print('CHECK IF NULL',df_features.isnull().any())
		print(df_features['points_next'])

		################################################################################################
		
		# separate training data by position
		ATA_train = df_features[df_features['posicao']=='ATA'].reset_index(drop=True).copy()
		MEI_train = df_features[df_features['posicao']=='MEI'].reset_index(drop=True).copy()
		LAT_train = df_features[df_features['posicao']=='LAT'].reset_index(drop=True).copy()
		ZAG_train = df_features[df_features['posicao']=='ZAG'].reset_index(drop=True).copy()
		GOL_train = df_features[df_features['posicao']=='GOL'].reset_index(drop=True).copy()
		TEC_train = df_features[df_features['posicao']=='TEC'].reset_index(drop=True).copy()

		return ATA_train, MEI_train, LAT_train, ZAG_train, GOL_train, TEC_train

	################################################################################################

	# vamos criar uma função que recebe um threshold como argumento e retorna y binario
	def binarizar_y_hat(self, df=None, threshold=5.0):
		#df = df_train.copy()
		#print (df['points_next'])
		'''
		1 = acima da média
		0 = abaxo da média
		'''
		df['points_next'] = df['points_next'].apply(lambda x: np.where(x > threshold, 1, 0))
		print (df)
		return df


	def sample_balanced_dataset(self, inputs, targets):
		# O número de elementos desejado é goal_size (isso também poderia vir por parâmetro)
		goal_size = int(len(targets) / 3)

		# Separando os ids dos elementos que pertecem à cada classe
		classe_0_ids = np.where(targets == 0)[0]
		classe_1_ids = np.where(targets == 1)[0]

		tamanhos_classes = [len(x) for x in [classe_0_ids, classe_1_ids]]

		# Assim como no oversample, decidimos se faremos amostragem com ou sem repetição comparando
		# o número de elementos da classe com o objetivo. Agora queremos que cada classe tenha goal_size elementos
		replace = False
		if tamanhos_classes[0] < goal_size:
		  replace = True
		classe_0_sampled_ids = np.random.choice(classe_0_ids, size=goal_size, replace=replace)

		replace = False
		if tamanhos_classes[1] < goal_size:
		  replace = True
		classe_1_sampled_ids = np.random.choice(classe_1_ids, size=goal_size, replace=replace)

		# Até agora estávamos apenas operando nos IDs agora vamos pegar os elementos selecionados no inputs
		dataset_sampled_data = np.concatenate([inputs[classe_0_sampled_ids],
		                                   inputs[classe_1_sampled_ids]])
		# O mesmo para o target
		dataset_sampled_targets = np.concatenate([targets [classe_0_sampled_ids],
		                                    targets [classe_1_sampled_ids]])

		return dataset_sampled_data, dataset_sampled_targets 


	def select_features_and_targets(self, posicao=None):
		
		if posicao   == 'ATA': 
			index=0
			df_position = self.preprocess_dataset(self.curr_rodada_id)[index]
		elif posicao == 'MEI': 
			index=1
			df_position = self.preprocess_dataset(self.curr_rodada_id)[index]
		elif posicao == 'LAT': 
			index=2
			df_position = self.preprocess_dataset(self.curr_rodada_id)[index]
		elif posicao == 'ZAG': 
			index=3
			df_position = self.preprocess_dataset(self.curr_rodada_id)[index]
		elif posicao == 'GOL': 
			index=4
			df_position = self.preprocess_dataset(self.curr_rodada_id)[index]
		elif posicao == 'TEC': 
			index=5
			df_position = self.preprocess_dataset(self.curr_rodada_id)[index]
		else:
			ATA_train, MEI_train, LAT_train, ZAG_train, GOL_train, TEC_train = self.preprocess_dataset(self.curr_rodada_id)
			df_position = pd.concat([ATA_train, MEI_train, LAT_train, ZAG_train, GOL_train, TEC_train ])

		# obter desvio padrão de pontos, m geral maior que a média, cerca de 3 pts
		threshold = df_position['pontos_num'].std()
		print (f'Desvio padrão da média de Pontos na rodada {self.curr_rodada_id}: {threshold}')
		# passar média de pontos atual como divisor de classes target
		df_players_train_features = self.binarizar_y_hat(df=df_position, threshold=5.0)
		# trabalhar com cópia do dataframe
		df_temp = df_players_train_features.copy()
		# todas as variáveis
		cols = ['atleta_id', 'apelido', 'clube', 'posicao', 'status_pre', 'preco_open',
		 'pontos_num', 'variacao_num', 'ieo_gm_mean', 'iao_gm_mean',
		 'ied_gm_mean', 'ieg_gm_mean', 'ippc_gm_mean', 'iro_gm', 'irp_gm',
		 'adversário', 'ranking', 'fase', 'media_movel_mandante',
		 'media_movel_visitante', 'prob_vitoria_mandante',
		 'prob_vitoria_visitante', 'mando', 'XP_mandante', 'XP_visitante',
		 'rodada_id', 'XP_min_para_valorizacao', 'ranking_clube', 'media_movel',
		 'prob_vitoria', 'odds_home', 'odds_away', 'odds', 'XP',
		 'prob_sg_mandante', 'prob_sg_visitante', 'prob_SG', 'points_next']
		# flexibilizar a escolha de variáveis
		features_cols = [
		 'preco_open',
		 'pontos_num', 
		 'variacao_num', 
		 #'ieo_gm_mean', 
		 #'iao_gm_mean',
		 #'ied_gm_mean', 
		 #'ieg_gm_mean',
		 'irp_gm',
		 #'iro_gm',
		 'ippc_gm_mean', 
		 'fase', 
		 'prob_vitoria', 
		 'XP',
		 'odds',
		 'media_movel',
		 'points_next'
		]
		# selecionar colunas relevantes
		df_temp = df_temp[features_cols]
		# separar features e target
		inputs = df_temp.iloc[:, :-1].to_numpy()
		targets = df_temp.iloc[:, -1].to_numpy()

		print(np.bincount(targets))
		# balancear os dados com o método criado anteriormente
		players_dataset_sampled_data, players_dataset_sampled_targets = self.sample_balanced_dataset(inputs, targets)
		# reinicializar X, y
		X = players_dataset_sampled_data
		y = players_dataset_sampled_targets
		
		return df_position, X, y


	def best_model(self, rodada=None, develop=False):

		posicoes = ['ATA','MEI','LAT','ZAG','GOL','TEC']

		seed = 0
		np.random.seed(seed)
		
		# match file
		model = glob.glob(f'gmodels/pickle/binary/R{rodada}*.pkl')[0]
		print (model)

		# load best binary classifier saved in cross_validation.py
		with open(model, 'rb') as f:
			clf = pickle.load(f)
			print(f'Model {clf} classes: ', clf.classes_)

		#################################################################
		if develop:
			# verificar precisão por posição
			for pos in posicoes:
				# obter dataset balanceado
				df_position, X, y = self.select_features_and_targets(pos)
				df_all_positions = df_all_positions.append(df_position)
				# split em treino e teste
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
				# fit
				clf.fit(X_train, y_train)
				# predict
				y_pred = clf.predict(X_test)
				# make report
				print(classification_report(y_test, y_pred))
		
		#################################################################
		# obeter dataframe completo do dataset e X,y balacenados
		df_position, X, y = self.select_features_and_targets()
		# features em tela
		df_temp = df_position[[
			 'preco_open',
			 'pontos_num', 
			 'variacao_num', 
			 #'ieo_gm_mean', 
			 #'iao_gm_mean',
			 #'ied_gm_mean', 
			 #'ieg_gm_mean',
			 'irp_gm',
			 #'iro_gm',
			 'ippc_gm_mean', 
			 'fase', 
			 'prob_vitoria', 
			 'XP',
			 'odds',
			 'media_movel']]
		# split em treino e teste
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
		# fit no conjunto
		clf.fit(X_train, y_train)
		# predição para todo o dataframe com base no clasificador
		y_hat = clf.predict_proba(df_temp)[:,1]
		print (y_hat.shape)
		# adicionar probabilidades ao dataframe completo
		df_position['binary_prob'] = pd.Series(list(y_hat)).values
		# separar apenas as probabilidades da rodada atual
		df = df_position[df_position['rodada_id']==rodada]\
								.sort_values(by='binary_prob', ascending=False)

		print (df.head(50))
		df.to_csv(f'temp/csv/df_binary_model_R{self.rodada}.csv')
		# retornar predições para dataframe original
		return df


	def knn_model(self):

		# obter dataset balanceado
		df_position, X, y = self.select_features_and_targets()
		# split em treino e teste
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)

		# estamos explicitamente selecionando os atributos de interesse
		knn = KNeighborsClassifier(n_neighbors=7)

		#######################
		# padronizar os dados

		# X_train = df_pl_train[x_features_cols]
		# y_train = df_pl_train[['points_next']]
		knn.fit(X_train, y_train)

		# X_test = df_pl_test[x_features_cols]
		# y_test = df_pl_test[['points_next']]

		y_pred = knn.predict(X)

		print(classification_report(y_test, y_pred))

		#####################################################
		# preparando scaler para padronizar escala dos dados
		# usaremos o conjunto de treino
		min_max_scaler = preprocessing.MinMaxScaler()
		min_max_scaler.fit(X_train)

		# transformar os dados para a nova escala (treino e teste)
		X_train_norm = min_max_scaler.transform(X_train)
		X_test_norm = min_max_scaler.transform(X_test)

		# preparando o knn com voto ponderado
		knn_norm = KNeighborsClassifier(n_neighbors=7)

		# conjunto de treino
		#X_train = X_train_norm
		#y_train = df_train[['target']]
		knn_norm.fit(X_train_norm, y_train)

		# conjunto de teste
		#X_test = X_test_norm
		#y_test = df_test[['target']]

		y_pred = knn_norm.predict(X_test_norm)

		# comparando desempenho
		print(classification_report(y_test, y_pred))

		#####################################################
		scaler = StandardScaler()
		# Fit apenas no conjunto de treino
		scaler.fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		# Processando o conjunto de testes:
		# Note que não fazemos fit no teste
		X_test_scaled = scaler.transform(X_test)

		# treinando o knn
		knn_scaled = KNeighborsClassifier(n_neighbors=7)
		# model fit
		knn_scaled.fit(X_train_scaled, y_train)

		y_pred = knn_scaled.predict(X_test_scaled)

		# comparando desempenho
		print(classification_report(y_test, y_pred))
		#####################################################
		knn = KNeighborsClassifier(n_neighbors=7)
		bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=7), 
		                         max_features=0.5, max_samples=0.5, n_estimators=100)

		knn.fit(X_train, y_train)
		bagging.fit(X_train, y_train)

		knn_s_train = knn.score(X_train,y_train)
		knn_s_test = knn.score(X_test,y_test)
		print('KNN Score train.:', knn_s_train)
		print('KNN  Score test  :', knn_s_test)
		print()
		y_hat = knn.predict_proba(X_test)[:,1]
		print(y_hat)
		bag_s_train = bagging.score(X_train,y_train)
		bag_s_test = bagging.score(X_test,y_test)
		print('Bagging Score train.:', bag_s_train)
		print('Bagging Score test  :', bag_s_test)

		df_position['prediction']=0.0
		# return predictions to original dataframe
		for i, value in enumerate(list(y_hat.flatten())):
			print (df_position.iloc[i]['atleta_id'], value)
			df_position['prediction'].iloc[i] = value.astype('float32')

		df_position = df_position[df_position['rodada_id'] == self.curr_rodada_id]

		print (df_position.sort_values(by='prediction', ascending=False).head(20))
		return df_position


	def decision_trees_model(self):
		 # obter dataset balanceado
		df_position, players_dataset_sampled_data, players_dataset_sampled_targets = self.select_features_and_targets()
		# reinicializar X, y
		X = players_dataset_sampled_data
		y = players_dataset_sampled_targets
		# split em treino e teste
		X_train, X_test, y_train, y_test = train_test_split(players_dataset_sampled_data, players_dataset_sampled_targets, test_size=0.30,random_state=42)

		imp = SimpleImputer(strategy="mean")
		X_train_clean = imp.fit_transform(X_train)
		X_test_clean = imp.transform(X_test)


		estimator_dt = DecisionTreeClassifier(max_depth=4)

		estimator_dt.fit(X_train_clean, y_train)
		y_pred_dt = estimator_dt.predict(X_test_clean)

		estimator_rf = RandomForestClassifier(max_depth=4)

		estimator_rf.fit(X_train_clean, y_train)
		y_pred_rf = estimator_rf.predict(X_test_clean)


		estimator_xg = xgb.XGBClassifier(random_state=42)

		estimator_xg.fit(X_train_clean, y_train)
		y_pred_xg = estimator_xg.predict(X_test_clean)

		print('-------- Árvore de decisão -------------')
		print("%.2f" % accuracy_score(y_test, y_pred_dt))
		print("%.2f" % precision_score(y_test, y_pred_dt, average='micro'))
		print("%.2f" % recall_score(y_test, y_pred_dt, average='micro'))

		print('-------- Random Forest -------------')
		print("%.2f" % accuracy_score(y_test, y_pred_rf))
		print("%.2f" % precision_score(y_test, y_pred_rf, average='micro'))
		print("%.2f" % recall_score(y_test, y_pred_rf, average='micro'))

		print('-------- XGBoost -------------')
		print("%.2f" % accuracy_score(y_test, y_pred_xg))
		print("%.2f" % precision_score(y_test, y_pred_xg, average='micro'))
		print("%.2f" % recall_score(y_test, y_pred_xg, average='micro'))



	def predict_best(self, pos_df, pos=None, top=None, curr_rodada_id=None, strategy=None):
		
		from sklearn.linear_model import LinearRegression
		from sklearn.metrics import mean_squared_error
		
		# relevant features
		if pos=='ATA':
			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieo_gm_mean', 'iao_gm_mean',\
					'ippc_gm_mean', 'iro_gm','irp_gm', 'ranking_clube', 'fase', 'prob_vitoria', 'XP', 
					'media_movel', 'pontos_num']]
		
		elif pos=='MEI':
			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieo_gm_mean', 'iao_gm_mean',\
					'ippc_gm_mean', 'iro_gm', 'irp_gm', 'ranking_clube', 'fase', 'prob_vitoria', 'XP', 
					'media_movel','pontos_num']]
		
		elif pos=='LAT':
			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieo_gm_mean', 'ied_gm_mean',\
					'ippc_gm_mean', 'iro_gm', 'irp_gm', 'ranking_clube', 'fase', 'prob_vitoria', 'XP', 
					'media_movel', 'pontos_num']]
		
		elif pos=='ZAG':
			pos_pred = pos_df[['variacao_num', 'preco_open', 'ied_gm_mean',\
					'ippc_gm_mean', 'ranking_clube', 'irp_gm', 'fase', 'prob_vitoria', 'XP', \
					'media_movel', 'pontos_num']]
		
		elif pos == 'GOL':
			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieg_gm_mean', \
					'ippc_gm_mean', 'ranking_clube', 'irp_gm', 'fase', 'prob_vitoria', 'XP', 
					'media_movel', 'pontos_num']]

		elif pos == 'TEC':
			pos_pred = pos_df[['variacao_num', 'preco_open', 'ranking_clube', 'fase', 
								'prob_vitoria', 'XP', 'media_movel', 'pontos_num']]
		
		# set dependent variables (all but pontos_num)
		X = pos_pred.drop(axis=1, columns=['pontos_num'])
		# normalize data
		X = X.astype('float32') / 255.
		#print (X)
		# independent variable
		y = pos_pred['pontos_num']
		# normalize data
		y = y.astype('float32') / 255.
		#print(y)
		# create model
		
		model = LinearRegression()
		# fit
		model.fit(X, y)
		# predict
		y_hat = model.predict(X)
		#print (y_hat)
		# check for R
		r_sq = model.score(X, y)
		ic(r_sq)
		# mean squared error
		mse = mean_squared_error(y, y_hat)
		ic(mse)
		# initialize preiction column
		pos_df['prediction']=0.0
		# return predictions to original dataframe
		for i, value in enumerate(list(y_hat.flatten())):
			#print (pos_df.iloc[i]['atleta_id'], value)
			pos_df['prediction'].iloc[i] = value.astype('float32')

		pos_pred = pos_df		
		
		'''
		Prepare dataframe for being imported in app.py,
		so it has:
		
		1-  The same exact columns
		2-  Columns in the same order
		3 - The same restrictions,
		4 - With the correponding strategy

		'''
		# two possibe strategies
		if strategy == 'preco_open':
			valorizacao = Valorizacao(self.edicao_cartola)
			rodada_df = valorizacao.melhores_atletas_para_valorizar(df_agg=pos_pred, curr_rodada_id=curr_rodada_id)
			curr_pred = rodada_df.sort_values(by='prediction', ascending=False).head(top)
		else:
			rodada_df = pos_pred
			curr_pred = rodada_df[rodada_df['rodada_id']==curr_rodada_id]\
								.sort_values(by='prediction', ascending=False)

		################################################################################

		# set dummy values for odds coluns existent in app.py
		curr_pred['odds_home'] = None
		curr_pred['odds_away'] = None
		curr_pred['odds'] = None
		curr_pred['adversário'] = None
		curr_pred['status_pre'] = None
		curr_pred['XP_min_para_valorizacao'] = None
		curr_pred['media_movel_mandante'] = None
		curr_pred['media_movel_visitante'] = None
		curr_pred['media_movel'] = None
		curr_pred.rename(columns={'ranking_clube':'ranking_clube'},inplace=True)

		# set order os coluns as in app.py
		curr_pred = curr_pred[['atleta_id', 'apelido', 'clube', 'posicao', 'status_pre',
			'preco_open','pontos_num', 'variacao_num', 'ieo_gm_mean', 'iao_gm_mean', 
			'ied_gm_mean','ieg_gm_mean', 'ippc_gm_mean', 'iro_gm', 'irp_gm', 'XP_min_para_valorizacao', 
			'adversário', 'ranking_clube', 'fase', 'media_movel_mandante', 'media_movel_visitante',
			'media_movel', 'prob_vitoria_mandante', 'prob_vitoria_visitante', 
			'prob_vitoria','odds_home', 'odds_away', 'odds', 'mando', 'XP_mandante',
			'XP_visitante', 'XP']]

		# print full df
		print ('#################################')
		print (f'BEST {pos} PREDICTIONS FOR ROUND {curr_rodada_id+1}')

		# see df with full columns
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
			ic(curr_pred.head(top))

		#curr_pred = curr_pred.sort_values(by='prediction', ascending=False).reset_index(drop=True).copy()

		return curr_pred


if __name__ == "__main__":
	edicao_cartola = '2021'
	
	atletas = BinaryClassifier(edicao_cartola, 37)
	# df_position, X, y = atletas.select_features_and_targets()

	# print (df_position, X, y)
	#atletas.knn_model()

	atletas.best_model(rodada=37)











