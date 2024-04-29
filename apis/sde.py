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

# Biblioteca de acesso ao SDE
# Globo @ Cartola FC @ Gato Mestre
# Versão 1.0



########## EM CONSTRUÇÃO


import requests
import json
import numpy as np
import pandas as pd
import dpath.util as dpu
from decouple import config
import time

SDE_TOKEN = config('SDE_TOKEN1')
SDE_URL   = config('SDE_URL')
DEBUG     = config('DEBUG_SDE')

SCOUT_LIST = [ 'FC', 'FR', 'PE', 'RB', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 
               'ZF', 'ZG', 'ZT', 'ZB', 'ZD', 'CA', 'CV', 'IM', 'DD', 'DT', 
               'DE', 'GS', 'GC', 'UP', 'PD', 'TC', 'TD', 'TF', 'TT', 'EF' ]


dictSlugs = {
                2010:'campeonato-brasileiro-2010',
                2011:'brasileirao2011', 
                2012:'brasileirao2012', 
                2013:'brasileirao2013',
                2014:'brasileirao2014', 
                2015:'brasileirao-2015', 
                2016:'campeonato-brasileiro-2016', 
                2017:'campeonato-brasileiro-2017',
                2018:'campeonato-brasileiro-2018', 
                2019:'campeonato-brasileiro-2019', 
                2020:'campeonato-brasileiro-2020',
                2021:'campeonato-brasileiro-2021', 
                2022:'campeonato-brasileiro-2022',
            }


def handle_error(type, code, url, label='error'):
    return {label: type, 'status-code': code, 'url': url}


def consulta_url_sde(url):
    headers = {'token': SDE_TOKEN}
    if '?' not in url:
        url = url + '?por_pagina=200'
        
    log = {}
    res = None
    
    try:
        req = requests.get(SDE_URL + url, headers=headers)
        res = json.loads(str(req.text))
        log = handle_error("Success", 200, url,'OK')
    except requests.exceptions.HTTPError as http_error:
        log = handle_error(f"Http Error: {http_error}", req.status_code, url)
    except requests.exceptions.ConnectionError as connection_error:
        log = handle_error(f"Error Connecting:{connection_error}", req.status_code, url)
    except requests.exceptions.Timeout as timeout_error:
        log = handle_error(f"Timeout Error:{timeout_error}", req.status_code, url)
    except requests.exceptions.RequestException as general_error:
        log = handle_error(f"Oooops :{general_error}", req.status_code, url)
    
    if eval(DEBUG):
        print(log)
    #time.sleep(20)
    return res


#def consulta_url_sde(url):
#    headers = {'token': SDE_TOKEN}
#    SDE_URL = config('SDE_URL')
#    res = requests.get(SDE_URL + url, headers=headers)
#    #print('STATUS CODE em call.py: ',res.status_code)
#    res = json.loads(str(res.text))
#    return res


def _request_sde(url, paginacao=True):
    res = consulta_url_sde(url)
    result = res
    if paginacao:
        if result:
            while res['paginacao']['proximo']:
                res = consulta_url_sde(res['paginacao']['proximo'])
                dpu.merge(result, res)
    return result



def pivot_column(df, coluna):
    try:
        df[coluna] = df[coluna].apply(lambda x: str(x).replace('\'','"').replace('None', '{}'))
    except:
        return df
    df_temp = df[coluna].apply(json.loads).apply(pd.Series)
    df = pd.concat([df, df_temp],axis=1)
    return df.drop(coluna, axis=1)



# Retorna informações do jogo
def get_ficha_jogo(jogo_id):
    url = f'''/jogos/{jogo_id}'''
    response = _request_sde(url, False)
    campeonato = pd.DataFrame.from_dict(response['referencias']['campeonatos']).T
    df = pd.DataFrame.from_dict(response['resultados'], orient='index').T
    df['campeonato_id'] = campeonato['campeonato_id'].iloc[0]
    return df



# Retorna estatisticas do jogo
def get_estatisticas_jogo(jogo_id):
    url = f'''/jogos/{jogo_id}/estatisticas/equipes'''
    response = _request_sde(url, False)
    df = pd.DataFrame.from_dict(response['resultados']['estatisticas_equipes'])
    return df



# Retorna estatisticas dos clubes
def get_jogos_data(data):
    url = f'''/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/data/{data}/jogos'''
    response = _request_sde(url, False)
    df = pd.DataFrame.from_dict(response['resultados'])
    return df



# Retorna estatisticas dos clubes
def get_estatisticas_edicao(edicao):
    url = f'''/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/{dictSlugs.get(edicao, 2020)}/estatisticas/equipes'''
    response = _request_sde(url, False)
    df = pd.DataFrame.from_dict(response['resultados']['estatisticas_equipes'])
    return df



# Retorna informações dos atletas
def get_atletas_sde(atleta_id):
    url = f'/atletas/{atleta_id}'
    try:
        response = _request_sde(url, False)
        df = pd.DataFrame.from_dict(response).T
        df = pivot_column(df, 'posicao')
    except:
        url = f'/tecnicos/{atleta_id}'
        response = _request_sde(url, False)
        try:
            df = pd.DataFrame.from_dict(response).T
            df['macro'] = 'TEC'
            df['atleta_id'] = df['tecnico_id']
        except:
            df = pd.DataFrame()
    return df



def get_elenco(equipe_id, data=None):
    '''
    Parâmetros obrigatórios:
        clube_id: id do clube de acordo com o SDE;
    Parâmetros Opcionais:
        data = Uma data no formato: yyyy-mm-dd.
      Resposta: Retorna o json do endpoint
    '''

    url = f"/equipes/{equipe_id}/elenco"
    if data:
        url += f"?data={data}"

    try:
        response = _request_sde(url, paginacao=False)
    except:
        print('Elenco não encontrado')
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(response['resultados']['vinculos']['atletas'])
    return df



def get_dicts_elenco_sde(edicao, equipes):
    """
    Busca todos os atletas e técnicos das equipes no SDE

    Parameters:
        edicao (str): Ano da edição (ex: '2020')
        equipes (list): Vetor com os ids das equipes do SDE participantes
    Returns:
        Objeto com os dados de todos os atletas e técnicos `{atleta_id: dadosAtleta}` `{tecnico_id: dadosTecnico}`
    """
    dados_edicao = _request_sde(f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}', paginacao=False)
    data_fim_campeonato = dados_edicao['resultados']['data_fim']
    atletas = {}
    tecnicos = {}
    hash_atletas_equipes = {}  # {'atleta_id': 'equipe_id'}
    for equipe_id in equipes:
        result               = _request_sde(f'/equipes/{equipe_id}/elenco?data={data_fim_campeonato}', paginacao=False)
        atletas              = {**atletas, **result['referencias']['atletas']}
        tecnicos             = {**tecnicos, **result['referencias']['tecnicos']}
        elenco               = {atleta_id: str(equipe_id) for atleta_id in result['referencias']['atletas'].keys()}
        hash_atletas_equipes = {**hash_atletas_equipes, **elenco}
    return atletas, tecnicos, hash_atletas_equipes



# Retorna informações dos atletas
def get_equipes_sde(equipe_id):
    url = f'/equipes/{equipe_id}'
    response = _request_sde(url, False)
    df = pd.DataFrame.from_dict(response).T
    return df[:1]



def get_equipes_sde_v2(edicao):
    result = _request_sde(f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}/equipes', paginacao=False)
    equipes = result['resultados']['equipes']
    dict_equipes_ids = {equipe['equipe_id']: equipe for equipe in equipes}
    return dict_equipes_ids


def get_elenco_sde_v2(edicao, equipes):
    """
    Busca todos os atletas e técnicos das equipes no SDE

    Parameters:
        edicao (str): Ano da edição (ex: '2020')
        equipes (list): Vetor com os ids das equipes do SDE participantes
    Returns:
        Objeto com os dados de todos os atletas e técnicos `{atleta_id: dadosAtleta}` `{tecnico_id: dadosTecnico}`
    """
    dados_edicao = _request_sde(f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}', paginacao=False)
    data_fim_campeonato = dados_edicao['resultados']['data_fim']

    atletas = {}
    tecnicos = {}
    hash_atletas_equipes = {}  # {'atleta_id': 'equipe_id'}
    for equipe_id in equipes:
        result = _request_sde(f'/equipes/{equipe_id}/elenco?data={data_fim_campeonato}', paginacao=False)
        atletas = {**atletas, **result['referencias']['atletas']}
        tecnicos = {**tecnicos, **result['referencias']['tecnicos']}
        elenco = {atleta_id: str(equipe_id) for atleta_id in result['referencias']['atletas'].keys()}
        hash_atletas_equipes = {**hash_atletas_equipes, **elenco}

    return atletas, tecnicos, hash_atletas_equipes  # TODO remover hash_atletas_equipes






def get_dict_equipes(temporada):
    url = f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{temporada}/equipes'
    try:
        response = _request_sde(url, False)
        return response['resultados']['equipes']
    except:
        return {}



def get_arbitros(jogo_id):
    url = f'''/jogos/{jogo_id}'''
    try:
        response = _request_sde(url, False)
        df = pd.DataFrame.from_dict(response['resultados']['arbitragem'])
    except:
        df = pd.DataFrame()
    return df



def get_arbitro(arbitro_id):
    url = f'''/arbitros/{arbitro_id}'''
    try:
        response = _request_sde(url, False)
        df = pd.DataFrame.from_dict(response['resultados'])
    except:
        df = pd.DataFrame()
    return df



def get_sede(sede_id):
    url = f'''/sedes/{sede_id}'''
    try:
        response = _request_sde(url, False)
    except:
        return pd.DataFrame()

    try:    
        df = pd.DataFrame.from_dict(response['resultados'])
        return df[:1]
    except:
        return pd.DataFrame()



def get_scouts_jogos_sde(id_jogo):
    url = f'/jogos/{id_jogo}/scouts'
    response = _request_sde(url, False)
    try:
        df = pd.DataFrame.from_dict(response['resultados']['scouts'])
    except:
        df = pd.DataFrame()
    return df 



def get_estatisticas_atletas_sde(id_jogo):
    scouts = ','.join(SCOUT_LIST)
    url = f'/jogos/{id_jogo}/estatisticas/atletas?scout_tipo_siglas={scouts}'
    response = _request_sde(url, False)
    return response
    try:
        df = pd.DataFrame.from_dict(response['resultados']['estatisticas_atletas'])
    except:
        df = pd.DataFrame()
    return df 



def get_estatisticas_equipes_sde(id_jogo):
    scouts = ','.join(SCOUT_LIST)
    url = f'/jogos/{id_jogo}/estatisticas/equipes?scout_tipo_siglas={scouts}'
    response = _request_sde(url, False)
    try:
        df = pd.DataFrame.from_dict(response['resultados']['estatisticas_equipes'])
    except:
        df = pd.DataFrame()
    return df 



def get_momento_scout(edicao, scout):
    url = f'''/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/{dictSlugs.get(edicao, 2020)}/scout_tipos/{scout}/detalhado'''
    try:
        response = _request_sde(url)
    except:
        print('Scout não encontrado')
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(response['resultados']['atletas'])
    dfscouts_completo = pd.DataFrame()
    try:
        for scout_list in df['scouts']:
            for scout in scout_list:
                data = {'atleta_id': scout['atleta_id'], 'equipe_id': scout['equipe_id'],
                        'jogo_id': scout['jogo_id'],
                        'scout': scout['scout_tipo']['sigla'], 'momento': scout['momento'],
                        'periodo': scout['periodo']}
                dfscouts_atleta = pd.DataFrame(data, index=[0])
                print(dfscouts_atleta)
                dfscouts_completo = pd.concat([dfscouts_atleta, dfscouts_completo], axis=0)
    except:
        print('sem scouts')
    return dfscouts_completo



def get_edicoes_campeonatos():
    url = '/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes'
    try:
        response = _request_sde(url, False)
    except:
        print('Edição não encontrada')
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(response['resultados']['edicoes'])
    return df



def get_classificacao(edicao, rodada):
    url = f'''/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/{dictSlugs.get(edicao, 2020)}/classificacao?rodada={rodada}'''
    try:
        response = _request_sde(url, False)
    except:
        print('Classificacao não encontrada')
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(response['resultados']['classificacoes'][0]['classificacao'])
    df['edidao'] = edicao
    df['rodada'] = rodada
    return df



def get_jogos_sde(edicao):
    url = f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/{dictSlugs.get(edicao, 2021)}/jogos'
    response = _request_sde(url)
    df = pd.DataFrame.from_dict(response['resultados']['jogos'])  
    df['ano'] = edicao
    return df 



def atualiza_informacoes_atleta(df):
    df_atletas = pd.DataFrame()
    for atleta_id in df['atleta_id'].unique():
        result = get_atletas_sde(int(atleta_id))
        df_atletas = pd.concat([df_atletas, result])
    resultado = df.merge(df_atletas, how='left', on='atleta_id')
    return resultado.fillna(0)



def atualiza_informacoes_clube(df):
    df_clubes = pd.DataFrame()
    for clube_id in df['clube_id'].unique():
        result = get_equipes_sde(int(clube_id))
        df_clubes = pd.concat([df_clubes, result])
    resultado = df.merge(df_clubes, how='left', left_on='clube_id', right_on='equipe_id')
    return resultado



def get_atletas_vinculos(atleta_id,aberto=None):
    '''
    Serviço que retorna todos os vínculos de um atleta, com as referências das equipes vinculadas.
    Se True traz apenas status/vinculos atuais, se False traz todos.
    atletas/63007/vinculos/?aberto=True
    '''
    if aberto:
        url = f'/atletas/{atleta_id}/vinculos/?aberto={aberto}'
    else:
        url = f'/atletas/{atleta_id}/vinculos'
    response = _request_sde(url,paginacao=False)
    df = pd.DataFrame(response['resultados'])  
    df['atleta_id'] = atleta_id
    return df



def get_jogos_equipe_ano_mes(equipe_slug, data): 
    '''
    Parameters
    ----------
    equipe_slug : slug
        slug de cada equipe, segundo SDE.
    data : string
        data no formato: AAAA-MM.

    Returns
    -------
    df : dataframe com confrontos no mês escolhido.

    '''
    try:
        url = f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/equipe/{equipe_slug}/data/{data}/jogos'
        response = _request_sde(url, False)
        df = pd.DataFrame.from_dict(response['resultados']['jogos'])
        return df
    except:
        return None



def get_jogos_equipe_ano_mes(equipe_slug, data): 
    '''
    Parameters
    ----------
    equipe_slug : slug
        slug de cada equipe, segundo SDE.
    
    data : string
        data no formato: YYYY-MM.

    Returns
    -------
    df : dataframe com confrontos no mês.

    '''
    try:
        url = f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/equipe/{equipe_slug}/data/{data}/jogos'
        response = _request_sde(url, False)
        fase = pd.DataFrame.from_dict(response['referencias']['fases']).T
        edicao = pd.DataFrame.from_dict(response['referencias']['edicoes']).T
        fase['campeonato_id'] = fase['edicao_id'].map(edicao.set_index('edicao_id')['campeonato_id'])
        df = pd.DataFrame.from_dict(response['resultados']['jogos'])
        df['campeonato_id'] = df['fase_id'].map(fase.set_index('fase_id')['campeonato_id']).astype('int')
        return df
    except:
        return None



def get_campeonatos(esporte_slug='futebol', modalidade_slug='futebol_de_campo',categoria_slug='profissional'):
    '''
    O método retorna os campeonatos de um determinado esporte, modalidade e categoria.
    
    Parameters
    ----------
    esporte_slug : string
        slug do esporte, segundo SDE.

    modalidade_slug : string
        slug da modalidade, segundo SDE.
    
    categoria_slug : string
        slug da categoria, segundo SDE.
    
    Returns
    -------
    df : dataframe com todos os campeonatos de acordo com os parâmetros escolhidos.
    '''
    url = f'/esportes/{esporte_slug}/modalidades/{modalidade_slug}/categorias/{categoria_slug}/campeonatos'
    response = _request_sde(url)
    df = pd.DataFrame.from_dict(response['resultados']['campeonatos'])  
    return df