######################################################################
#         Desenvolvido aqui:                                         #
#        _____       _          __  __           _                   #
#       / ____|     | |        |  \/  |         | |                  #
#      | |  __  __ _| |_ ___   | \  / | ___  ___| |_ _ __ ___        #
#      | | |_ |/ _` | __/ _ \  | |\/| |/ _ \/ __| __| '__/ _ \       #
#      | |__| | (_| | || (_) | | |  | |  __/\__ \ |_| | |  __/       #
#       \_____|\__,_|\__\___/  |_|  |_|\___||___/\__|_|  \___|       #
#                                                                    #
#         Anderson Cordeiro || Bruno Bedo || Vitor Patalano          #
#                                                                    #
######################################################################

# Biblioteca de acesso ao Cartola
# Globo @ Cartola FC @ Gato Mestre
# Versão 1.0

# STABLE

from decouple import config
#from gatomestre.utils import dataframes as dfutils
import pandas as pd
import requests
import json
from decouple import config



CARTOLA_URL = config('CARTOLA_URL')


# CARTOLA_URL = config('CARTOLA_URL')
#CARTOLA_URL = 'https://api.cartolafc.globo.com'
dictPosicoes = {
    1: 'GOL',
    2: 'LAT',
    3: 'ZAG',
    4: 'MEI',
    5: 'ATA',
    6: 'TEC'
}


def consulta_url_cartola(url):
    res = requests.get(CARTOLA_URL + url)
    res = json.loads(str(res.text))
    return res


# TODO - Remover essa funçao e utilizar de dfutils (testar)

def pivot_column(df, coluna):
    try:
        df[coluna] = df[coluna].apply(lambda x: str(x).replace('\'','"').replace('None', '{}'))
    except:
        print(f"Erro ao gerar pivot table. Coluna '{coluna}' existe?")
        return df
    df_temp = df[coluna].apply(json.loads).apply(pd.Series)
    df = pd.concat([df, df_temp],axis=1)
    return df.drop(coluna, axis=1)


### Endpoints API Cartola
def get_status():
    '''
    Status do mercado
    '''
    res = consulta_url_cartola('/mercado/status')
    return res['status_mercado']


def get_esquemas():
    '''
    Lista os esquemas táticos (4-3-3, 3-4-3, ...)
    '''
    res = consulta_url_cartola('/esquemas')
    return pd.DataFrame.from_dict(res)


def get_rodadas():
    '''
    Lista todas as rodadas do campeonato
    '''
    res = consulta_url_cartola('/rodadas')
    return pd.DataFrame.from_dict(res)


def get_rodada():
    '''
    Rodada Atual
    '''
    res = consulta_url_cartola('/mercado/status')
    return res['rodada_atual']


def get_mais_escalados():
    '''
    Lista dos jogadores mais escalados
    '''
    res = consulta_url_cartola('/mercado/destaques')
    return pd.DataFrame.from_dict(res)


def get_partidas():
    '''
    Próximas partidas do campeonato
    '''
    res = consulta_url_cartola(f'/partidas')
    return pd.DataFrame.from_dict(res['partidas'])


def get_partidas_rodada(id_rodada):
    '''
    Partidas da rodada
    '''
    res = consulta_url_cartola(f'/partidas/{id_rodada}')
    return pd.DataFrame.from_dict(res['partidas'])['partida_id']


def get_clubes():
    '''
    Lista de clubes
    '''
    res = consulta_url_cartola('/clubes')
    return pd.DataFrame.from_dict(res).T


def get_mercado():
    '''
    Lista de todos os jogadores
    (retorna todas as informações)
    '''
    res = consulta_url_cartola('/atletas/mercado')
    atletas = pd.DataFrame.from_dict(res['atletas'])
    return atletas


def get_status_id():
    '''
    Retorna id com o id de cada status
    (retorna todas as informações)
    '''
    res = consulta_url_cartola('/atletas/mercado')    
    status = pd.DataFrame.from_dict(res['status']).T.reset_index(drop=True)
    return status



def get_pontuados(id_rodada):
    '''
    Pontuação da rodada em andamento
    '''
    res = consulta_url_cartola(f'/atletas/pontuados/{id_rodada}')
    df = pd.DataFrame.from_dict(res['atletas']).T
    df['rodada_id'] = id_rodada
    return df


def get_time_destaque():
    '''
    Time que mais pontuou na rodada anterior
    '''
    res = consulta_url_cartola('/pos-rodada/destaques')
    return pd.DataFrame.from_dict(res)['mito_rodada']


def get_status_id():
    res = consulta_url_cartola('/atletas/status')
    status = pd.DataFrame.from_dict(res).T
    return status



def get_atletas_detalhes(atleta_id):
    res = consulta_url_cartola(f'/atletas/detalhes/{atleta_id}')
    res = pd.DataFrame.from_dict(res)
    return res
    
