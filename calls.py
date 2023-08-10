import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import requests
import json
import pprint
from icecream import ic
import dpath.util
from decouple import config

token = config('SDE_TOKEN1')
CARTOLA_URL = config('CARTOLA_URL')
SDE_URL = config('SDE_URL')

# util
def consulta_url_sde(url):
    headers = {'token': token}
    SDE_URL = config('SDE_URL')
    res = requests.get(SDE_URL + url, headers=headers)
    res = json.loads(str(res.text))
    return res

# util
def _request_sde(url):
    res = consulta_url_sde(url)
    result = res
    try:
        while res['paginacao'].get('proximo'):
            res = consulta_url_sde(res['paginacao'].get('proximo'))
            dpath.util.merge(result, res)
    except Exception as e:
        pass
    return result

### [In] Equipes
def get_equipes_sde(edicao):
    result = _request_sde(f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}/equipes')
    equipes = result['resultados']['equipes']
    #pprint.pprint(equipes)
    dict_equipes_ids = {equipe['equipe_id']: equipe for equipe in equipes}

    clubes=[]
    escudos=[]
    _ids=[]
 
    for equipe in equipes:
        clube = equipe['nome_popular']
        _id = equipe['equipe_id']
        escudo = equipe['escudos']['svg']
        clubes.append(clube)
        _ids.append(_id)
        escudos.append(escudo)

    # make dictionary of teams and ids
    clubes_e_ids = dict(zip(clubes, _ids))
    # account for name inconsistency
    #clubes_e_ids['Athlético-PR'] = clubes_e_ids.pop('Athletico-PR')
    # make dictionary of teams and badges
    clubes_e_escudos = dict(zip(clubes,escudos))

    return clubes, clubes_e_ids, dict_equipes_ids


def get_elenco_sde(edicao, equipes):
    """
    Busca todos os atletas e técnicos das equipes no SDE

    Parameters:
        edicao (str): Ano da edição (ex: '2020')
        equipes (list): Vetor com os ids das equipes do SDE participantes
    Returns:
        Objeto com os dados de todos os atletas e técnicos `{atleta_id: dadosAtleta}` `{tecnico_id: dadosTecnico}`
    """
    dados_edicao = _request_sde(f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}')
    data_fim_campeonato = dados_edicao['resultados']['data_fim']

    atletas = {}
    tecnicos = {}
    hash_atletas_equipes = {}  # {'atleta_id': 'equipe_id'}
    for equipe_id in equipes:
        result = _request_sde(f'/equipes/{equipe_id}/elenco?data={data_fim_campeonato}')
        atletas = {**atletas, **result['referencias']['atletas']}
        tecnicos = {**tecnicos, **result['referencias']['tecnicos']}
        elenco = {atleta_id: str(equipe_id) for atleta_id in result['referencias']['atletas'].keys()}
        hash_atletas_equipes = {**hash_atletas_equipes, **elenco}

    return atletas, tecnicos, hash_atletas_equipes  # TODO remover hash_atletas_equipes

### [In] Atletas
def get_atletas_sde(edicao, equipes):
    """
    Busca todos os atletas das equipes no SDE

    Parameters:
        equipes (list): Vetor com os ids das equipes do SDE participantes
    Returns:
        Objeto com os dados de todos os atletas `{atleta_id: dadosAtleta}`
    """
    atletas = {}
    for equipe_id in equipes:
        result = _request_sde(f'/equipes/{equipe_id}/elenco')
        atletas = {**atletas, **result['referencias']['atletas']}

    return atletas


def get_tecnico_sde(tecnico_id):
    """
    Busca os dados de um técnico no SDE

    Parameters:
        tecnico_id (int): ID do técnico no SDE
    Returns:
        Objeto com os dados do técnico
    """
    dados_tecnico = _request_sde(f'/tecnicos/{tecnico_id}')
    dados_tecnico = dados_tecnico.get('resultados')

    return dados_tecnico

    

def get_jogos_sde(edicao, R, times_dict):

    results = _request_sde(f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}/jogos')
    partidas = pd.DataFrame(results['resultados']['jogos'])

    rodada =  partidas[partidas['rodada']==R+1].reset_index()
    # Inverter times e ids para mapeamento futuro
    times_dict_r = {v:k for k, v in times_dict.items()}

    games = rodada[['equipe_mandante_id','equipe_visitante_id']].copy()
    games['mandante'] = games['equipe_mandante_id'].map(times_dict_r)
    games['visitante'] = games['equipe_visitante_id'].map(times_dict_r)
    games = games[['mandante', 'visitante']].copy()

    # Jogos da rodada
    jogos = dict(zip(games.mandante, games.visitante))
    # Inverter jogos para mapeamento futuro
    inv_jogos = {v: k for k, v in jogos.items()}
    # Confrontos da rodada (mando e mando invertido)
    confrontos = {**jogos, **inv_jogos}

    jogos_que_faltam =  partidas[partidas['rodada']>R].reset_index()
    next_games = jogos_que_faltam[['equipe_mandante_id','equipe_visitante_id']].copy()
    next_games['mandante'] = next_games['equipe_mandante_id'].map(times_dict_r)
    next_games['visitante'] = next_games['equipe_visitante_id'].map(times_dict_r)
    next_games = next_games[['mandante', 'visitante']].copy()
    # proximos jogos em casa e fora
    home_ahead = list(next_games['mandante'].values)
    away_ahead = list(next_games['visitante'].values)
    # zip
    match_ups_ahead = (list(zip(home_ahead, away_ahead)))

    return jogos, confrontos, match_ups_ahead



### [In] Partidas Edição
def partidas_cartola(clubes=None):

    #   Last team results
    fase_dict = {'v':3, 'e':1, 'd':0}

    #   Inverter ordem key, value
    times_dict_r = {v:k for k, v in clubes.items()}

    # In[Loading Partidas]

    url_partidas = CARTOLA_URL+"/partidas/"

    partidas = requests.get(url_partidas).json()
    df_partidas = json_normalize(partidas['partidas'])

    # get esencial columns
    df_partidas = df_partidas[['clube_casa_id', 'clube_casa_posicao','clube_visitante_id',\
                'clube_visitante_posicao', 'aproveitamento_mandante', 'aproveitamento_visitante']].copy()

    # convert strings 'v', 'd', 'e' into points for the last 5 matches and sum them all up
    df_partidas['aproveitamento_mandante'] = df_partidas['aproveitamento_mandante']\
                            .apply(lambda values: sum(fase_dict.get(v, np.nan) for v in values))
    # convert to percentage
    df_partidas['aproveitamento_mandante'] = round((df_partidas['aproveitamento_mandante']/15)*100, 2)
    # convert strings 'v', 'd', 'e' into points for the last 5 matches and sum them all up
    df_partidas['aproveitamento_visitante'] = df_partidas['aproveitamento_visitante']\
                            .apply(lambda values: sum(fase_dict.get(v, np.nan) for v in values))
    # convert to percentage
    df_partidas['aproveitamento_visitante'] = round((df_partidas['aproveitamento_visitante']/15)*100, 2)

    # mandantes
    df_mandantes = df_partidas[['clube_casa_id', 'clube_casa_posicao','aproveitamento_mandante']].copy()
    df_mandantes.rename(columns={'clube_casa_id':'clube',
                                 'clube_casa_posicao':'ranking_clube',
                                 'aproveitamento_mandante':'fase'}, inplace=True)
    # visitantes
    df_visitantes = df_partidas[['clube_visitante_id', 'clube_visitante_posicao','aproveitamento_visitante']].copy()
    df_visitantes.rename(columns={'clube_visitante_id':'clube',
                                 'clube_visitante_posicao':'ranking_clube',
                                 'aproveitamento_visitante':'fase'}, inplace=True)
    # join
    dfs = [df_mandantes, df_visitantes]
    # concat
    df_partidas = pd.concat(dfs).reset_index(drop=True).copy()
    # from id to clube name
    df_partidas['clube'] = df_partidas['clube'].map(times_dict_r)


    print ('##################################################################')
    print ('Teams Moment As Per The Official Tournament Table (last 5 macthes)')
    print ('##################################################################')
    ic (df_partidas.sort_values(by='fase', ascending=False).reset_index(drop=True).copy())

    return df_partidas.sort_values(by='fase', ascending=False).reset_index(drop=True).copy()

def partidas_validas():
    # In[Loading Partidas]

    url_partidas = CARTOLA_URL+"/partidas/"

    partidas = requests.get(url_partidas).json()
    df_partidas = json_normalize(partidas['partidas'])
    df_partidas.rename(columns={'clube_casa_id':'clube_id',
                                'clube_visitante_id':'adversario_id',
                                'partida_id':'confronto_id'}, inplace=True)
    print (df_partidas[['confronto_id', 'valida', 'clube_id', 'adversario_id']])
    return df_partidas[['confronto_id', 'valida', 'clube_id', 'adversario_id']]


### In[Loading current status]
def mercado_cartola():

    link_athletes = 'https://api.cartolafc.globo.com'+'/atletas/mercado'
    # call api
    resp_cartola = requests.get(link_athletes)
    # to json
    df_current_cartola = resp_cartola.json()
    # to df
    df_current = pd.DataFrame(df_current_cartola['atletas'])
    # get current status
    status_df = pd.DataFrame(df_current_cartola['status']).T.reset_index(drop=True).copy()
    # make Ids of type int
    status_df['id'] = status_df['id'].map(lambda x: int(x))

    df_current['status'] = df_current['status_id'].map(status_df.set_index('id')['nome'])

    return df_current

# ###################################################################################################################
# df_partidas, df_current = callApiCartola()
# ###################################################################################################################

### In[Atletas sendo mais escalados no fantasy]
def mais_escalados_cartola(reservas=False):
    if reservas:
        link_athletes = CARTOLA_URL+'/mercado/destaques/reservas'
    else:
        link_athletes = CARTOLA_URL+'/mercado/destaques/titulares'

    try:
        resp_cartola = requests.get(link_athletes)
        destaques = resp_cartola.json()
        # turn json into dataframe
        df = pd.DataFrame.from_dict(destaques, orient='columns')
        # turn one column into series
        athlete_dicts = df.Atleta.apply(pd.Series)
        # turn that flattened list of dicts into a dataframe
        df_mais_escalados = pd.DataFrame(athlete_dicts)
        # relevant columns
        df_mais_escalados = df_mais_escalados[['atleta_id', 'apelido']]
        # creater anking
        df_mais_escalados['ranking'] = range(1,len(df_mais_escalados.index)+1)
    except:
        df_mais_escalados = pd.DataFrame()
        df_mais_escalados['ranking'] = np.nan

    print('#######################################')    
    print('Most Selected Players In The Market Now')
    print('############################################')    
    ic(df_mais_escalados)
    
    return df_mais_escalados







