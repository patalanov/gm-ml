# # data tools
# import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
# import numpy as np
# from icecream import ic
# from valorizacao import Valorizacao

# class LinearRegression:

# 	def __init__(self, edicao_cartola, curr_rodada_id):
# 		self.edicao_cartola = edicao_cartola
# 		self.curr_rodada_id = curr_rodada_id

# 	def get_dataset(self, curr_rodada_id):

# 		################################################################################################

# 		df_partidas = pd.DataFrame()
# 		# get dataset for all rounds, file by file and append to container
# 		for rodada in range(1,curr_rodada_id+1):
# 		    df_rodada = pd.read_csv(f'calculadora/data/model/dataset/{self.edicao_cartola}/R{rodada}.csv')
# 		    df_rodada['rodada_id'] = rodada
# 		    df_partidas = df_partidas.append(df_rodada,sort=False)
		    
# 		    print (f'UPLOADED ROUND {rodada}')
# 		################################################################################################

# 		# eliminate who did not play
# 		df_partidas = df_partidas[(df_partidas['variacao_num']!= 0.0) | (df_partidas['pontos_num']!= 0.0)]
# 		# fill zeros
# 		df_partidas = df_partidas.fillna(0)  
# 		# merge home factor into one XP only
# 		df_partidas['XP'] = df_partidas['XP_mandante'].add(df_partidas['XP_visitante'], fill_value=0)
# 		# merge home factor into one prob only
# 		df_partidas['prob_vitoria'] = df_partidas['prob_vitoria_mandante']\
# 								 .add(df_partidas['prob_vitoria_visitante'], fill_value=0)
# 		# get points achieved by player the following round in order tom compare variables
# 		df_partidas['points_next'] = df_partidas.groupby('atleta_id')['pontos_num'].shift(-1)
# 		# reset index
# 		df_partidas = df_partidas.reset_index(drop=True)
# 		# info
# 		#print (df_partidas.info())
# 		# ful sample
# 		# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# 		# 	print(df_partidas.sample(10))
		
# 		################################################################################################
		
# 		# extract only relevant features from dataframe
# 		df_features = df_partidas[['rodada_id', 
# 									'atleta_id',
# 									'apelido', 
# 									'clube',
# 									'posicao',
# 									'preco_open',
# 									'pontos_num', 
# 									'variacao_num', 
# 									'ieo_gm_mean', 
# 									'iao_gm_mean',
# 									'ied_gm_mean', 
# 									'ieg_gm_mean',
# 									'iro_gm', 
# 									'irp_gm', 
# 									'ippc_gm_mean', 
# 									'ranking_clube', 
# 									'fase', 
# 									'prob_vitoria_mandante',
# 									'prob_vitoria_visitante', 
# 									'prob_vitoria', 
# 									'mando', 
# 									'XP_mandante', 
# 									'XP_visitante', 
# 									'XP',
# 									'media_movel_mandante',
#                                     'media_movel_visitante',
#                                     'media_movel']].copy()

# 		################################################################################################
		
# 		# separate training data by position
# 		ATA_train = df_features[df_features['posicao']=='ATA'].reset_index(drop=True).copy()
# 		MEI_train = df_features[df_features['posicao']=='MEI'].reset_index(drop=True).copy()
# 		LAT_train = df_features[df_features['posicao']=='LAT'].reset_index(drop=True).copy()
# 		ZAG_train = df_features[df_features['posicao']=='ZAG'].reset_index(drop=True).copy()
# 		GOL_train = df_features[df_features['posicao']=='GOL'].reset_index(drop=True).copy()
# 		TEC_train = df_features[df_features['posicao']=='TEC'].reset_index(drop=True).copy()

# 		return ATA_train, MEI_train, LAT_train, ZAG_train, GOL_train, TEC_train

# 	################################################################################################

# 	def predict_best(self, pos_df, pos=None, top=None, curr_rodada_id=None, strategy=None):
		
# 		from sklearn.linear_model import LinearRegression
# 		from sklearn.metrics import mean_squared_error
		
# 		# relevant features
# 		if pos=='ATA':
# 			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieo_gm_mean', 'iao_gm_mean',\
# 					'ippc_gm_mean', 'iro_gm','irp_gm', 'ranking_clube', 'fase', 'prob_vitoria', 'XP', 
# 					'media_movel', 'pontos_num']]
		
# 		elif pos=='MEI':
# 			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieo_gm_mean', 'iao_gm_mean',\
# 					'ippc_gm_mean', 'iro_gm', 'irp_gm', 'ranking_clube', 'fase', 'prob_vitoria', 'XP', 
# 					'media_movel','pontos_num']]
		
# 		elif pos=='LAT':
# 			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieo_gm_mean', 'ied_gm_mean',\
# 					'ippc_gm_mean', 'iro_gm', 'irp_gm', 'ranking_clube', 'fase', 'prob_vitoria', 'XP', 
# 					'media_movel', 'pontos_num']]
		
# 		elif pos=='ZAG':
# 			pos_pred = pos_df[['variacao_num', 'preco_open', 'ied_gm_mean',\
# 					'ippc_gm_mean', 'ranking_clube', 'irp_gm', 'fase', 'prob_vitoria', 'XP', \
# 					'media_movel', 'pontos_num']]
		
# 		elif pos == 'GOL':
# 			pos_pred = pos_df[['variacao_num', 'preco_open', 'ieg_gm_mean', \
# 					'ippc_gm_mean', 'ranking_clube', 'irp_gm', 'fase', 'prob_vitoria', 'XP', 
# 					'media_movel', 'pontos_num']]

# 		elif pos == 'TEC':
# 			pos_pred = pos_df[['variacao_num', 'preco_open', 'ranking_clube', 'fase', 
# 								'prob_vitoria', 'XP', 'media_movel', 'pontos_num']]
		
# 		# set dependent variables (all but pontos_num)
# 		X = pos_pred.drop(axis=1, columns=['pontos_num'])
# 		# normalize data
# 		X = X.astype('float32') / 255.
# 		#print (X)
# 		# independent variable
# 		y = pos_pred['pontos_num']
# 		# normalize data
# 		y = y.astype('float32') / 255.
# 		#print(y)
# 		# create model
		
# 		model = LinearRegression()
# 		# fit
# 		model.fit(X, y)
# 		# predict
# 		y_hat = model.predict(X)
# 		#print (y_hat)
# 		# check for R
# 		r_sq = model.score(X, y)
# 		ic(r_sq)
# 		# mean squared error
# 		mse = mean_squared_error(y, y_hat)
# 		ic(mse)
# 		# initialize preiction column
# 		pos_df['prediction']=0.0
# 		# return predictions to original dataframe
# 		for i, value in enumerate(list(y_hat.flatten())):
# 			#print (pos_df.iloc[i]['atleta_id'], value)
# 			pos_df['prediction'].iloc[i] = value.astype('float32')

# 		pos_pred = pos_df		
		
# 		'''
# 		Prepare dataframe for being imported in app.py,
# 		so it has:
		
# 		1-  The same exact columns
# 		2-  Columns in the same order
# 		3 - The same restrictions,
# 		4 - With the correponding strategy

# 		'''
# 		# two possibe strategies
# 		if strategy == 'preco_open':
# 			valorizacao = Valorizacao(self.edicao_cartola)
# 			rodada_df = valorizacao.melhores_atletas_para_valorizar(df_agg=pos_pred, curr_rodada_id=curr_rodada_id)
# 			curr_pred = rodada_df.sort_values(by='prediction', ascending=False).head(top)
# 		else:
# 			rodada_df = pos_pred
# 			curr_pred = rodada_df[rodada_df['rodada_id']==curr_rodada_id]\
# 								.sort_values(by='prediction', ascending=False)

# 		################################################################################

# 		# set dummy values for odds coluns existent in app.py
# 		curr_pred['odds_home'] = None
# 		curr_pred['odds_away'] = None
# 		curr_pred['odds'] = None
# 		curr_pred['adversário'] = None
# 		curr_pred['status_pre'] = None
# 		curr_pred['XP_min_para_valorizacao'] = None
# 		curr_pred['media_movel_mandante'] = None
# 		curr_pred['media_movel_visitante'] = None
# 		curr_pred['media_movel'] = None
# 		curr_pred.rename(columns={'ranking_clube':'ranking_clube'},inplace=True)

# 		# set order os coluns as in app.py
# 		curr_pred = curr_pred[['atleta_id', 'apelido', 'clube', 'posicao', 'status_pre',
# 			'preco_open','pontos_num', 'variacao_num', 'ieo_gm_mean', 'iao_gm_mean', 
# 			'ied_gm_mean','ieg_gm_mean', 'ippc_gm_mean', 'iro_gm', 'irp_gm', 'XP_min_para_valorizacao', 
# 			'adversário', 'ranking_clube', 'fase', 'media_movel_mandante', 'media_movel_visitante',
# 			'media_movel', 'prob_vitoria_mandante', 'prob_vitoria_visitante', 
# 			'prob_vitoria','odds_home', 'odds_away', 'odds', 'mando', 'XP_mandante',
# 			'XP_visitante', 'XP']]

# 		# print full df
# 		print ('#################################')
# 		print (f'BEST {pos} PREDICTIONS FOR ROUND {curr_rodada_id+1}')

# 		# see df with full columns
# 		with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# 			ic(curr_pred.head(top))

# 		#curr_pred = curr_pred.sort_values(by='prediction', ascending=False).reset_index(drop=True).copy()

# 		return curr_pred

