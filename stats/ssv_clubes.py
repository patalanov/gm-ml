# import numpy as np
# import pandas as pd
# import sys 
# import pprint 
# sys.path.append("../../..")
# from gatomestre_gm_vitor.extract.apis import scoutservice as sct
# from gatomestre_gm_vitor.utils import dataframes as dfutils
# from time import sleep

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


# torneios = sct.listar_campeonatos()
# stacked = np.array(torneios)
# torneios = [item for sublist in stacked for item in sublist]
# print(torneios)
def listar_lances_local(campeonato, ano, partida):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;
        codPartida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
    Resposta: Retorna o quantitativo de lances executados pelas equipes em uma determinada partida.
    '''
    lances = sct.request_scout(f'/ListarLances?campeonato={campeonato}&ano={ano}&codPartida={partida}')
    df = pd.DataFrame.from_dict(lances['ListarLancesResult']['Partidas'], orient='columns')
    df_lances_consolidado = pd.DataFrame(lances['ListarLancesResult']['Partidas'][0]['Lances'])
    return df, df_lances_consolidado

def get_lances_consolidados(campeonato, ano, partida): 
	df_lances, df_lances_consolidados  = listar_lances_local(campeonato,ano,partida)
	print(df_lances.info(),df_lances_consolidados.info())
	df_lances = dfutils.pivot_columns(df_lances, ['Equipe1','Equipe2','Cronometragem'],append=True)
	#print(df_lances.info())
	df_lances_consolidados['Equipe1'] = df_lances['Equipe1_CodigoExterno'].iloc[0]
	df_lances_consolidados['Equipe2'] = df_lances['Equipe2_CodigoExterno'].iloc[0]
	df_lances_consolidados['CodigoExterno'] = df_lances['CodigoExterno'].iloc[0]
	df_lances_consolidados['Codigo'] = partida
	return df_lances_consolidados



def stats_confrontos(torneios,edicao):
	df_torneios = pd.DataFrame()
	for torneio in torneios:
		for edicao in range(edicao-3,edicao):
			df_edicao = pd.DataFrame()
			try:
				response = sct.listar_partidas(torneio, edicao)
				print (response)
				partidas = list(response['Codigo'].values)
				for partida in partidas:
					# obeter posse de bola
					curr_result = get_lances_consolidados(torneio, edicao, partida)
					#print('CURR RESULT',curr_result,curr_result.info())
					# colunas relevantes
					curr_result = curr_result[['Equipe1', 'Equipe2', 'Nome', 'QuantidadeEquipe1', 'QuantidadeEquipe2', 'Codigo', 'CodigoExterno']]
					# formatar estatisticas com underscore entre strings
					curr_result['Nome'] = curr_result['Nome'].apply(lambda x :x.replace(' ', '_'))
					#print('CURR RESULT NOME',curr_result)
					# pivotar valores de 'Nome' da estatistica de cada time numa partida para uma uma coluna cada
					#curr_result = curr_result.pivot(['Equipe1','Equipe2','jogo_id'],'Nome')
					#print(curr_result.duplicated(['Equipe1','Equipe2','jogo_id']).any())

					curr_result = curr_result.pivot_table(index=['Equipe1', 'Equipe2', 'Codigo','CodigoExterno'], columns='Nome')
					#print('CURR RESULT PIVOT',curr_result)
					# formatar colunas com underscore
					curr_result.columns = curr_result.columns.map('{0[1]}_{0[0]}'.format)
					# resetar indice
					curr_result.reset_index(inplace=True)
					# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):  # more options can be specified also
					#print(curr_result.head(5),curr_result.info())
					df_edicao = df_edicao.append(curr_result,ignore_index=True,sort=False).reset_index(drop=True).copy()
					print('DF_EDICAO',df_edicao)
					#sleep(1)
				
				df_edicao['Torneio']=torneio
				df_edicao['Edicao']=edicao

				#df_edicao = df_edicao.drop_duplicates(keep='last').reset_index(drop=True)
				print (df_edicao['Edicao'])
				print (df_edicao.info(), df_edicao.head(50))
				df_torneios = df_torneios.append(df_edicao,ignore_index=True,sort=False).reset_index(drop=True).copy()
				df_edicao.to_csv(f'database/{edicao}/scout_service/scouts_equipes/Stats_Torneio{torneio}_Edicao_{edicao}.csv')
			
			except Exception as e:
				print(f'Não foi possível carregar os dados de {torneio}, {edicao}',repr(e))
	df_torneios.to_csv(f'database/{edicao}/scout_service/scouts_equipes/Stats_All.csv')
	return df_torneios

if __name__ == "__main__":

	torneios = ['AmistososClubes', 'AmistososSelecao', 'Baiano', 'Brasileiro', 'BrasileiroB', 
				'BrasileiroFeminino', 'Carioca', 'CopaAmerica', 'CopaBrasil', 'CopaNordeste', 'CopaSPJunior', 
				'EliminatoriasCopaMundo', 'EuroCopa', 'Gaucho', 'Libertadores', 'Mineiro', 'OlimpiadasFutebolFeminino', 
				'OlimpiadasFutebolMasculino', 'Paulista', 'Pernambucano', 'PreLibertadores', 'RecopaSulAmericana', 
				'SulAmericana', 'SuperCopaBrasil']

	stats_confrontos(torneios)






