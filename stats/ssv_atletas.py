import pandas as pd
from pandas.errors import ParserError
import numpy as np
import os
from sources.apis import scoutservice as sct
from sources.apis import sde
from sources.apis import cartola as car
from utils import pivot_columns
import dataframes as dfutils
from datetime import datetime


import json
from maths import coord as coord
from time import sleep
from random import randint




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



def mapear_confrontos_sde(campeonato, edicao, df_jogo, lances):
	#### Mapeando confronto ####
	if campeonato == 'Paulista':
		df_jogos_sde = sde.get_jogos_sde(edicao,regional=True)
	else:
		df_jogos_sde = sde.get_jogos_sde(edicao)

	#print('LANCES pos GET JOGOS SDE',df_jogos_sde, df_jogos_sde.info())

	df_partidas_rodada = df_jogos_sde[df_jogos_sde['rodada']==df_jogo['Rodada'].iloc[0]]

	games = df_partidas_rodada[['equipe_mandante_id', 'equipe_visitante_id']].copy()

	jogos_rodada = dict(zip(games.equipe_mandante_id, games.equipe_visitante_id))

	inv_jogos = {v: k for k, v in jogos_rodada.items()}  # Inverte os times para mapeamento futuro
	confrontos = {**jogos_rodada, **inv_jogos}  # Confrontos da rodada (mando e mando invertido)

	lances['oponente_id'] = lances['clube_id'].map(confrontos)
	# mapear mando de campo
	lances['home_dummy'] = lances['clube_id'].apply(lambda x: 1 if x in list(games['equipe_mandante_id'].values) else 0)
	return lances


def consolidate_events(campeonato=None, edicao=None, start_date=None, rodada=None, scout=None, partida=False, global_dataset=False):
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
	#print(sct.listar_campeonatos())

	# Selecionando jogos do campeonado Brasileiro 
	df_jogos = sct.listar_partidas(campeonato,edicao)
	#print(df_jogos.info())
     
	 # Always create the 'Data' column
	df_jogos['Data'] = df_jogos['DataRodada'].apply(lambda x: x.split('T')[0])
    
	# Sorting by 'Data' in ascending order
	df_jogos = df_jogos.sort_values(by='Data')
	# Filter matches based on the start date if available
	print(f"Start date for filtering: {start_date}")
	#print("Data in df_jogos after sorting:", df_jogos['Data'].tolist())
      
	if start_date:
		print(f"Filtering data from start date: {start_date}")
		df_jogos = df_jogos[df_jogos['Data'] >= start_date]
		print(df_jogos.info())
	else:
		print("No start date provided, including all data.")
          	
	print('JSON normalizing df_jogos')
	df_jogos = pd.json_normalize(df_jogos.to_dict('records'), sep='_')
	df_jogos = df_jogos[['DataRodada', 'Data','Codigo', 'CodigoExterno', 'Rodada', 'GolsEquipe1', 'GolsEquipe2',
                    'Equipe1_Codigo', 'Equipe1_CodigoExterno', 'Equipe1_Nome',
                    'Equipe2_Codigo', 'Equipe2_CodigoExterno', 'Equipe2_Nome']]
	
	
	#df_database = pd.DataFrame()
	print('--------------------------')
	print('Loading: ',campeonato, edicao)

	for jogos_id_interno in df_jogos['Codigo']:

		try:
			print('--------------------------')
			print('Loading: ',jogos_id_interno)
			print('     Código Externo:', df_jogos[df_jogos['Codigo']==jogos_id_interno]['CodigoExterno'].unique()[0])
			print('     Carregando dados do jogo...')
			df_jogo = df_jogos[df_jogos['Codigo']==jogos_id_interno]
            
			#print('DF JOGO',df_jogo, df_jogo.info())

			# Request dos elencos de um jogo
			elenco_t1 = sde.get_elenco(df_jogo['Equipe1_CodigoExterno'].iloc[0], df_jogo['Data'].iloc[0])
			elenco_t2 = sde.get_elenco(df_jogo['Equipe2_CodigoExterno'].iloc[0], df_jogo['Data'].iloc[0])

			df_elencos = pd.concat([elenco_t1,elenco_t2])

			# Lances de um jogo em específico
			print('     Carregando scouts do jogo...')
			lances_jogador = sct.listar_lances_jogador_partida(jogos_id_interno)
			#print('LANCES JOGADOR PRE PIVOT',lances_jogador.info())
			lances = dfutils.pivot_columns(lances_jogador, ['Jogador','Partida'],append=True)
			
			# Include 'Data' column from df_jogo into lances
			lances['Data'] = df_jogo['Data'].iloc[0]
			lances['DataRodada'] = df_jogo['DataRodada'].iloc[0]

			# Inserindo informações dos atletas
			lances = lances.rename(columns={'Jogador_CodigoExterno':'atleta_id'})
			lances = lances.dropna(subset=['atleta_id'])
			lances['atleta_id'] = lances['atleta_id'].astype('int')
			
			lances = sde.atualiza_informacoes_atleta(lances)
			#print('LANCES atualiza',lances, lances.info())

			lances['Lance'] = lances['Codigo'].apply(lambda x: sct.get_lance(x).split(',')[0])
			
			# Final selection of columns for lances
			selected_columns = ['Data', 'DataRodada', 'Codigo', 'Lance', 'Nome', 'Ocorrencia', 'PosicaoLance', 'TempoPartida',
                    'TimestampStr', 'atleta_id', 'apelido', 'sigla', 'Jogador_Posicao',
                    'Partida_CodigoExterno', 'CampoPosicaoX', 'CampoPosicaoY', 'ContraAtaque',
                    'Metros', 'TravePosicaoX', 'TravePosicaoY']
			
			lances = lances[selected_columns]

			# Inserindo o tempo de cada ação do jogo
			lances['tempo'] = lances['Ocorrencia'].apply(lambda x: column_tempo(x))
			lances['tempo_2'] = lances['TimestampStr'].apply(lambda x: x.split('T')[1])

			# Inserindo clube de cada atleta
			lances['clube_id'] = lances['atleta_id'].map(df_elencos.set_index('atleta_id')['equipe_id'])

			# # Convertendo defesas para bloqueios
			lances['Lance'][(lances['Lance']=='Goleiro') & (lances['sigla']!='GOL')]='Bloqueio'
			lances['Nome'][lances['Lance']=='Bloqueio']='Bloqueio'

			# Inserindo código do jogo
			lances['Codigo_SCT'] = lances['Codigo']
			lances['Partida_CodigoInterno'] = jogos_id_interno

			# Inserindo a rodada do jogo
			lances['Rodada'] = lances['Partida_CodigoInterno'].map(df_jogo.set_index('Codigo')['Rodada'])

			lances['Torneio'] = campeonato
			lances['Edicao'] = edicao
			
			lances = mapear_confrontos_sde(campeonato, edicao, df_jogo, lances)
			
			print('LANCES pos MAPEAMENTO de confrontos',lances, lances.info())

			yield lances

		except Exception as e:
			print(e)



def get_latest_date_from_existing_data(torneio, edicao):
    try:    
        df_existing = pd.read_csv(f'database/2023/scout_service/events/Eventos_All.gz', 
                                  compression='gzip')

        # Explicitly check if 'Data' column exists in df_existing
        if 'Data' in df_existing.columns and not df_existing.empty:
            df_tournament = df_existing[df_existing['Torneio'] == torneio]
            df_tournament_sorted = df_tournament.sort_values(by='Data', ascending=False)

            latest_date = df_tournament_sorted['Data'].iloc[0] if not df_tournament_sorted.empty else None
            latest_rodada = df_tournament_sorted['Rodada'].iloc[0] if not df_tournament_sorted.empty else None

            print(f"Latest date for {torneio} in {edicao}: {latest_date}")
            print(f"Corresponding 'Rodada' for the latest date: {latest_rodada}")
        else:
            print(f"No 'Data' column or empty DataFrame found for {torneio} in {edicao}")
            return None

    except FileNotFoundError:
        print(f"No file found for {torneio} in {edicao}")
        return None





    
def stats_atletas(torneios, edicao):
    for torneio in torneios:
        # Fetch the latest date from existing data
        latest_date_saved = get_latest_date_from_existing_data(torneio, edicao)

        # Fetch the latest date available from the API
        df_jogos_api = sct.listar_partidas(torneio, edicao)
        if df_jogos_api.empty:
            print(f"O DataFrame para {torneio} em {edicao} está vazio. Pulando para o próximo torneio.")
            continue  # Pular para o próximo torneio se o DataFrame estiver vazio
        df_jogos_api['Data'] = df_jogos_api['DataRodada'].apply(lambda x: x.split('T')[0])
        latest_date_api = df_jogos_api['Data'].max()

        # Compare the dates to decide whether to fetch new data
        if latest_date_saved is None or latest_date_saved < latest_date_api:
            for lances in consolidate_events(torneio, edicao, latest_date_saved):
                # File paths for saving
                file_path_tournament = f'database/2023/scout_service/events/Eventos_Torneio_{torneio}.gz'
                file_path_all = f'database/2023/scout_service/events/Eventos_All.gz'
                # Save the data
                lances.to_csv(file_path_tournament, mode='a', header=not os.path.exists(file_path_tournament), index=False, compression='gzip')
                lances.to_csv(file_path_all, mode='a', header=not os.path.exists(file_path_all), index=False, compression='gzip')
        else:
            print(f"All data for {torneio} in {edicao} is already up-to-date.")



	