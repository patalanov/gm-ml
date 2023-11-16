import pandas as pd
import numpy as np
from sources.apis import scoutservice as sct
from sources.apis import sde
from sources.apis import cartola as car
from utils import pivot_columns
import dataframes as dfutils

import json
from maths import coord as coord
from time import sleep
from random import randint



dictCoordenadas36 = {
    1:(20,50),
    2:(113,50),
    3:(206,50),
    4:(299,50),
    5:(392,50),
    6:(485,50),
    7:(20,183),
    8:(113,183),
    9:(206,183),
    10:(299,183),
    11:(392,183),
    12:(485,183),
    13:(20,316),
    14:(113,316),
    15:(206,316),
    16:(299,316),
    17:(392,316),
    18:(485,316),
    19:(20,449),
    20:(113,449),
    21:(206,449),
    22:(299,449),
    23:(392,449),
    24:(485,449),
    25:(20,582),
    26:(113,582),
    27:(206,582),
    28:(299,582),
    29:(392,582),
    30:(485,582),
    31:(20,715),
    32:(113,715),
    33:(206,715),
    34:(299,715),
    35:(392,715),
    36:(485,715),
}


def column_tempo(lance):
    if 'S' not in lance:
        lance = lance+'00'
    if 'M' in lance: 
        if len(lance.split('M')[0].split('T')[1])==1:
            tempo = '0'+lance.split('M')[0].split('T')[1]+':'+lance.split('M')[1].replace('S','')
        else:
            tempo = lance.split('M')[0].split('T')[1]+':'+lance.split('M')[1].replace('S','')
    else:
        tempo = lance.replace('PT','00:').replace('S','')
    
    if len(tempo.split(':')[1])==1:
        tempo = tempo.split(':')[0]+':0'+tempo.split(':')[1]
    return tempo


def lances_previos_scouts(database=None, index=None):
    '''
    Parâmetros Obrigatórios:
        database = base de dados;
        index = índice do lance
    Resposta: linha com lances anteriores
    '''
    df_lance_index = database[database['index']==index]
    index_list = [index]
    lance = database[database['index']==index-1][['index','Lance','Nome','clube_id','Codigo']]
    while lance['Lance'].iloc[0] == 'Passe' and any (lance['Nome'].iloc[0] in nome for nome in ['Completo','Decisivo']) and lance['clube_id'].iloc[0]==df_lance_index['clube_id'].iloc[0]:
        index_list.append(lance['index'].iloc[0])
        if database.index[database['index']==lance['index'].iloc[0]][0] > 0:
            lance = database[database['index']==lance['index'].iloc[0]-1][['index','Lance','Nome','clube_id']]
        elif database.index[database['index']==lance['index'].iloc[0]][0] == 0:
            break
    df_lance = database[database['index'].isin(index_list)]
    return df_lance


def consolidate_events(campeonato=None, edicao=None, rodada=None, scout=None, partida=False, global_dataset=False):
	'''
	Parâmetros Obrigatórios:
	campeonato: str da competição;
	edicao: ano da temporada no formato YYYY;
	rodada: rodada do campeonato
	scout: filtrar por scout
	partida: partida desejada para analise. se partida=false, se aplicara ao campeonato inteiro
	global_dataset: exporta datasset com nomemclatura de features padronizada mundo afora

	Resposta: Retorna a lista de eventos por partida, por atleta
	'''
	print('TEST NEW PIPE')
    
	print(sct.listar_campeonatos())

	# Selecionando jogos do campeonado Brasileiro 
	df_jogos = sct.listar_partidas(campeonato,edicao)
	if rodada: 
		df_jogos = df_jogos[df_jogos['Rodada']==rodada]

	df_jogos = df_jogos.dropna(subset=['Periodo']).sort_values(by='Rodada').reset_index(drop=True)

	df_jogos = dfutils.pivot_columns(df_jogos, ['Equipe1','Equipe2'],append=True)
	df_jogos = df_jogos[['Codigo', 'CodigoExterno','Rodada','DataRodada','GolsEquipe1', 'GolsEquipe2',
	            'Equipe1_Codigo', 'Equipe1_CodigoExterno','Equipe1_Nome',
	            'Equipe2_Codigo', 'Equipe2_CodigoExterno','Equipe2_Nome']]
	
	#print('DF JOGOS',df_jogos, df_jogos.info())

	# Inserindo a data de cada jogo
	df_jogos['Data'] = df_jogos['DataRodada'].apply(lambda x: x.split('T')[0])
	print('DF_JOGOS',df_jogos, df_jogos.info())


	############################################# INICIO COLETA NO WORKFLOW ######################################################
	
	# df_lances_scout_service = pd.DataFrame()
	# for jogo_id in df_jogos['CodigoExterno']:
	# 	print('JOGO_ID',jogo_id)
	# 	df_jogo    = df_jogos[df_jogos['CodigoExterno']==jogo_id]
	# 	print(df_jogo)
	# 	elenco_t1  = sde.get_elenco(df_jogo['Equipe1_CodigoExterno'].iloc[0],df_jogo['Data'].iloc[0])
	# 	elenco_t2  = sde.get_elenco(df_jogo['Equipe2_CodigoExterno'].iloc[0],df_jogo['Data'].iloc[0])
	# 	df_elencos = pd.concat([elenco_t1,elenco_t2])
	# 	print('ELENCOS',df_elencos)
	# 	lances         = sct.listar_lances_jogador(jogo_id)
	# 	print('LANCES',lances)
	# 	df         = pivot_columns(lances, ['Jogador','Partida'],append=True)
	# 	df.rename(columns={'Jogador_CodigoExterno':'atleta_id','Partida_CodigoExterno':'sde_id','PosicaoLance':'quadrante','TempoPartida':'tempo','Nome':'sequencia_lance'}, inplace=True)
	# 	df              = df.dropna(subset=['atleta_id'])
	# 	df['atleta_id'] = df['atleta_id'].astype('int')
	# 	df['lance']     = df['Codigo'].apply(lambda x: sct.get_lance(x).split(',')[0])
	# 	df['minutagem'] = df['Ocorrencia'].apply(lambda x: df_utils.minuto_ocorrencia(x))
	# 	df['equipe_id'] = df['atleta_id'].map(df_elencos.set_index('atleta_id')['equipe_id'])
	# 	df              = df[['atleta_id','equipe_id','sde_id','lance','sequencia_lance','quadrante','tempo','minutagem']]
	# 	data_columns    = [col for col in df.columns if col not in ['atleta_id', 'equipe_id','sde_id']]
	# 	df['data']      = json.dumps(df[data_columns].to_dict('records'))
	# 	df              = df.drop(columns=['lance', 'sequencia_lance','quadrante','tempo','minutagem'])
	# 	print(list(df['data']))
	# 	df_lances_scout_service = pd.concat([df_lances_scout_service, df], axis=0)
	
	############################################# FIM COLETA NO WORKFLOW ###############################################################
	
	df_database = pd.DataFrame()
	print('--------------------------')
	print('Loading: ',campeonato, edicao)

	for jogos_id_interno in df_jogos['Codigo']:

		try:
			print('--------------------------')
			print('Loading: ',jogos_id_interno)
			print('     Código Externo:', df_jogos[df_jogos['Codigo']==jogos_id_interno]['CodigoExterno'].unique()[0])
			print('     Carregando dados do jogo...')
			df_jogo = df_jogos[df_jogos['Codigo']==jogos_id_interno]
			print('DF JOGO',df_jogo, df_jogo.info())

			# Request dos elencos de um jogo
			elenco_t1 = sde.get_elenco(df_jogo['Equipe1_CodigoExterno'].iloc[0],df_jogo['Data'].iloc[0])
			elenco_t2 = sde.get_elenco(df_jogo['Equipe2_CodigoExterno'].iloc[0],df_jogo['Data'].iloc[0])
			df_elencos = pd.concat([elenco_t1,elenco_t2])
			#print('DF ELENCOS',df_elencos, df_elencos.info())

			# Lances de um jogo em específico
			print('     Carregando scouts do jogo...')
			lances_jogador = sct.listar_lances_jogador_partida(jogos_id_interno)
			#print('LANCES JOGADOR PRE PIVOT',lances_jogador.info())
			lances = dfutils.pivot_columns(lances_jogador, ['Jogador','Partida'],append=True)
			print('LANCES######################################################',lances, lances.info())

			# Inserindo informações dos atletas
			lances = lances.rename(columns={'Jogador_CodigoExterno':'atleta_id'})
			lances = lances.dropna(subset=['atleta_id'])
			lances['atleta_id'] = lances['atleta_id'].astype('int')

			lances = sde.atualiza_informacoes_atleta(lances)
			print('LANCES atualiza',lances, lances.info())

			lances['Lance'] = lances['Codigo'].apply(lambda x: sct.get_lance(x).split(',')[0])
			lances = lances[['Codigo', 'Lance', 'Nome', 'Ocorrencia', 'PosicaoLance', 'TempoPartida','TimestampStr',
			                  'atleta_id', 'apelido', 'sigla','Jogador_Posicao','Partida_CodigoExterno','CampoPosicaoX',
							  'CampoPosicaoY','ContraAtaque','Metros','TravePosicaoX','TravePosicaoY']]
			
			print('LANCES get lances',lances, lances.info())

			# Inserindo o tempo de cada ação do jogo
			lances['tempo'] = lances['Ocorrencia'].apply(lambda x: column_tempo(x))
			lances['tempo_2'] = lances['TimestampStr'].apply(lambda x: x.split('T')[1])

			# Inserindo clube de cada atleta
			lances['clube_id'] = lances['atleta_id'].map(df_elencos.set_index('atleta_id')['equipe_id'])

			# Convertendo defesas para bloqueios
			lances['Lance'][(lances['Lance']=='Goleiro') & (lances['sigla']!='GOL')]='Bloqueio'
			lances['Nome'][lances['Lance']=='Bloqueio']='Bloqueio'

			# Inserindo código do jogo
			lances['Codigo_SCT'] = lances['Codigo']
			lances['Partida_CodigoInterno'] = jogos_id_interno

			# Inserindo a rodada do jogo
			lances['Rodada'] = lances['Partida_CodigoInterno'].map(df_jogo.set_index('Codigo')['Rodada'])

			lances['Torneio'] = campeonato
			lances['Edicao'] = edicao

			print('LANCES pos EDICAO',lances, lances.info())

			#### Mapeando confronto ####
			if campeonato == 'Paulista':
				df_jogos_sde = sde.get_jogos_sde(edicao,regional=True)
			else:
				df_jogos_sde = sde.get_jogos_sde(edicao)

			print('LANCES pos GET JOGOS SDE',df_jogos_sde, df_jogos_sde.info())

			df_partidas_rodada = df_jogos_sde[df_jogos_sde['rodada']==df_jogo['Rodada'].iloc[0]]

			games = df_partidas_rodada[['equipe_mandante_id', 'equipe_visitante_id']].copy()

			jogos_rodada = dict(zip(games.equipe_mandante_id, games.equipe_visitante_id))

			inv_jogos = {v: k for k, v in jogos_rodada.items()}  # Inverte os times para mapeamento futuro
			confrontos = {**jogos_rodada, **inv_jogos}  # Confrontos da rodada (mando e mando invertido)

			lances['oponente_id'] = lances['clube_id'].map(confrontos)
			# mapear mando de campo
			lances['home_dummy'] = lances['clube_id'].apply(lambda x: 1 if x in list(games['equipe_mandante_id'].values) else 0)

			print('LANCES pos MAPEAMENTO de confrontos',lances, lances.info())

			# # Inserindo coordenadas x e y do campo
			# lances['coordenadas'] = lances['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))
			# x=[]
			# y=[]
			# #print ('LANCES',lances)
			# for i, row in lances.iterrows():
			# 	if row['PosicaoLance']==-1:
			# 		x.append(-1)
			# 		y.append(-1)
			# 	elif row['PosicaoLance']>=36:
			# 		x.append(-1)
			# 		y.append(-1)
			# 	else:
			# 		x.append(round(row['coordenadas'][0]+randint(10, 100),2))
			# 		y.append(round(row['coordenadas'][1]+randint(0, 130),2))

			# lances['coord_x'] = x
			# lances['coord_y'] = y

			# # Check if was penalty
			# list_penalty = [60,61,62,63]
			# lances['coord_x'][lances['Codigo'].isin(list_penalty)] = 300
			# lances['coord_y'][lances['Codigo'].isin(list_penalty)] = 777

			#Concatenando a database final
			df_database = df_database.append(lances).reset_index(drop=True)

			print('DATABASE',df_database, df_database.info())

			print()
			print('----------------------------------')
			print('Salvando database - ScoutService')
			print(df_database.info())
			print(df_database[df_database['Lance']=='Finalização'])

			print('Done')
		except Exception as e:
			print(e)
	return df_database
	#return df_lances_scout_service


def last_round(campeonato,edicao):
	# Selecionando jogos do campeonado Brasileiro 
	df_jogos = sct.listar_partidas(campeonato,edicao)
	print('Ultima rodada: ', df_jogos['Rodada'].max())
	return df_jogos['Rodada'].max()


def get_rodada():
    '''
    Rodada Atual
    '''
    res = car.consulta_url_cartola('/mercado/status')
    return res['rodada_atual']


def stats_atletas(torneios=None,rodada=None,edicao=None,recorte=False):
	#df_torneios = pd.DataFrame()
	rodada = get_rodada()
	print ('Ultima rodada:',get_rodada())
	print ('#################')
	#curr_result = consolidate_events(campeonato='Brasileiro', edicao=2023).reset_index()

	try:
		df_torneios_curr = pd.read_csv(f'database/{edicao}/scout_service/events/Eventos_All.gz',compression='gzip', low_memory=False)
		print ('Fetching dataframe of all current tournaments from disk...')
		print ('##########################################################')
		print('\n>>> Tamanho do dataframe completo inicial em memória')
		print(df_torneios_curr.memory_usage())
	except:
		df_torneios_curr = pd.DataFrame()
		print ('Creating new container for tournaments data...')
		print ('##############################################')
	for torneio in torneios:
		print ('Torneio',torneio)
		print ('########################')
		for edicao in range(edicao,edicao+1):
			print ('Edição',edicao)
			print ('###########')
			try:
				df_edicao_curr = pd.read_csv(f'database/{edicao}/scout_service/events/Eventos_Torneio_{torneio}_Edicao_{edicao}.gz',compression='gzip', low_memory=False)
				print ('Fetching dataframe of current edition from disk...')
				print ('#######################################################')
				print('\n>>> Tamanho do dataframe da edição inicial em memória')
				print(df_edicao_curr.memory_usage())
			except:
				df_edicao_curr = pd.DataFrame()
				print ('Creating new container for edition data...')
				print ('##########################################')
			try:
				curr_result = consolidate_events(campeonato=torneio, edicao=edicao).reset_index()
				#print(curr_result)
				
				df_edicao_curr = df_edicao_curr.append(curr_result,ignore_index=True,sort=False).reset_index(drop=True).copy()					
				print ('Appended result to current edition...')
				print ('#####################################')
				print('\n>>> Tamanho do dataframe da edição apendado em memória')
				print(df_edicao_curr.memory_usage())
				#df_edicao_temp = df_edicao_curr.append(curr_result,ignore_index=True,sort=False).reset_index(drop=True).copy()					
				# concatenate previous edicao with new entries and drop duplicates
				#df_edicao = pd.concat([df_edicao_curr,df_edicao_temp]).drop_duplicates(keep='last').reset_index(drop=True)				
				print ('Dropping eventual duplicate results...')
				print ('######################################')
				df_edicao_curr = df_edicao_curr.drop_duplicates(keep='last').reset_index(drop=True)
				# copiar df
				df_edicao = df_edicao_curr.copy()
				print('\n>>> Tamanho do dataframe da edição purgado em memória')
				print(df_edicao.memory_usage())
				print (df_edicao.info(), df_edicao.head(50))

				print ('Updating dataframe of current edition to disk...')
				print ('################################################')

				df_edicao.to_csv(f'database/{edicao}/scout_service/events/Eventos_Torneio_{torneio}_Edicao_{edicao}.gz',index=False, compression='gzip')
				#df_torneios_temp = df_torneios_curr.append(df_edicao,ignore_index=True,sort=False).reset_index(drop=True).copy()
				df_torneios_curr = df_torneios_curr.append(df_edicao,ignore_index=True,sort=False).reset_index(drop=True).copy()
				print('\n>>> Tamanho do dataframe completo apendado em memória')
				print(df_torneios_curr.memory_usage())
				sleep(1)

			except Exception as e:
				print(f'Não foi possível carregar os dados de {torneio}, {edicao}',e)
				pass

	print ('Dropping duplicate results from all data...')
	print ('###########################################')
	df_torneios_curr = df_torneios_curr.drop_duplicates(keep='last').reset_index(drop=True)
	# concatenate old torneio database with current entries 0and drop duplicates
	#df_torneios = pd.concat([df_torneios_curr,df_torneios_temp]).drop_duplicates(keep='last').reset_index(drop=True)
	#df_torneios = df_torneios_curr.copy()
	print('\n>>> Tamanho do dataframe completo purgado em memória')
	print(df_torneios_curr.memory_usage())

	print ('Updating dataframe of all tournaments to disk...')
	print ('################################################')

	df_torneios_curr.to_csv(f'database/{edicao}/scout_service/events/Eventos_All.gz',index=False, compression='gzip')



if __name__ == "__main__":
	
	# torneios = ['AmistososSelecao', 'Baiano', 'Brasileiro', 'BrasileiroB', 
	# 			'BrasileiroFeminino', 'Carioca', 'CopaAmerica', 'CopaBrasil', 'CopaNordeste', 'CopaSPJunior', 
	# 			'EliminatoriasCopaMundo', 'EuroCopa', 'Gaucho', 'Libertadores', 'Mineiro', 'OlimpiadasFutebolFeminino', 
	# 			'OlimpiadasFutebolMasculino', 'Paulista', 'Pernambucano', 'PreLibertadores', 'RecopaSulAmericana', 
	# 			'SulAmericana', 'SuperCopaBrasil']

	rodada = get_rodada()
	print (get_rodada())

	# rodar sempre todos os torneios em andamento na temporada
	#torneios = ['Brasileiro','BrasileiroB','CopaBrasil','Libertadores','SulAmericana']
	torneios = ['Brasileiro']
	#stats_atletas(torneios,rodada=rodada,recorte=False)
	# rodar apenas a atualizacao do brasileiro, na base
	stats_atletas(torneios, edicao=2023)

