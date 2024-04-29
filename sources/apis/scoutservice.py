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

# Biblioteca de acesso ao Scout Service
# Globo @ Cartola FC @ Gato Mestre
# Versão 1.0
# Help Page: https://scoutservice.apps.g.globo/ScoutJson.svc/help

# STABLE

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import requests
import json
import math
from decouple import config



SCOUT_SERVICE_URL = config('SCOUT_SERVICE_URL')
SCOUT_SERVICE_USER = config('SCOUT_SERVICE_USER')
SCOUT_SERVICE_PASSWORD = config('SCOUT_SERVICE_PASSWORD')



dictLances = {

    'd': {

        ## Finalização
        1:  'Finalização,Fora da Área,Bloqueado',
        10: 'Finalização,Dentro Área,Defendido',
        11: 'Finalização,Dentro Área,Gol',
        12: 'Finalização,Dentro Área,Trave',
        13: 'Finalização,Fora da Área,Defendido',
        20: 'Finalização,Dentro Área,Fora',
        21: 'Finalização,Dentro Área,Bloqueado',
        22: 'Finalização,Fora da Área,Trave',
        23: 'Finalização,Fora da Área,Gol',
        24: 'Finalização,Fora da Área,Para Fora',
        45: 'Finalização,Pênalti Decisivo',
        89: 'Finalização,Finalização,Certa',
        90: 'Finalização,Finalização,Errada',

        ## Finalização de Cabeça
        3:  'Finalização Cabeça,Dentro Peq.Área,Defendido',
        4:  'Finalização Cabeça,Dentro Peq.Área,Bloqueado',
        5:  'Finalização Cabeça,Dentro Peq.Área,Trave',
        6:  'Finalização Cabeça,Dentro Peq.Área,Gol',
        7:  'Finalização Cabeça,Dentro Peq.Área,Fora',
        8:  'Finalização Cabeça,Grande Área,Defendido',
        9:  'Finalização Cabeça,Grande Área,Trave',
        17: 'Finalização Cabeça,Grande Área,Gol',
        18: 'Finalização Cabeça,Grande Área,Para Fora',
        19: 'Finalização Cabeça,Grande Área,Bloqueado',
        77: 'Finalização Cabeça,Cabeceio,Indefinido',

        ## Passe
        14: 'Passe,Decisivo',
        25: 'Passe,Incompleto',
        74: 'Passe,Completo',

        ## Goleiro
        27: 'Goleiro,Reposição Errada',
        28: 'Goleiro,Saída Errada',
        29: 'Goleiro,Saída Certa',
        32: 'Goleiro,Defesa Difícil',
        47: 'Goleiro,Defesa Normal',
        49: 'Goleiro,Reposição Certa',
        65: 'Goleiro,Gol Sofrido',
        66: 'Goleiro,Defesa de Pênalti',

        ## Desarme
        30: 'Desarme,Desarme',
        75: 'Desarme,Desarme Incompleto',

        ## Drible
        94: 'Drible,Drible Certo',
        95: 'Drible,Drible Errado',

        ## Falta
        55: 'Falta,Gol',
        56: 'Falta,Fora',
        57: 'Falta,Defendida',
        58: 'Falta,Trave',
        59: 'Falta,Bloqueada',

        ## Penalti
        60: 'Pênalti,Gol',
        61: 'Pênalti,Fora',
        62: 'Pênalti,Defendida',
        63: 'Pênalti,Trave',

        ## Equipe
        40: 'Equipe,Escanteio a Favor',
        76: 'Equipe,Chance real',

        ## Arbitragem
        33: 'Arbitragem,Falta Cometida',
        34: 'Arbitragem,Falta Recebida',
        35: 'Arbitragem,Impedimento',
        37: 'Arbitragem,Cartão Vermelho',
        46: 'Arbitragem,Cartão Amarelo',
        71: 'Arbitragem,Cartão Amarelo + Vermelho',
        92: 'Arbitragem,Pênalti Recebido',
        93: 'Arbitragem,Pênalti Cometido',

        ## Genérico
        38: 'Ações Genéricas,Gol Contra',
        43: 'Ações Genéricas,Bola Levantada',
        64: 'Ações Genéricas,Gol Indefinido'
     },

    'a':{
        
        # Criar dicionario de scouts agrupados

    }
}


def get_lance(id, tipo='d'):
    '''
    Parâmetros:
        id: ID do lance;
        tipo:
        - d: detalhado
        - a: agrupado

      Resposta: Retorna a descrição do lance
    '''
    lance = dictLances[tipo].get(id, 'Não Encontrado')
    return lance


def request_scout(url):
    '''
    Parâmetros:
        url: endpoint do scout service;

      Resposta: Retorna o json do endpoint
    '''
    user =  'CARTOLA'
    password =  'C2wFRuvj6ldguiS5lrXDDUgmz$q^xq'
    url = SCOUT_SERVICE_URL + url
    resp = requests.get(url, auth=(SCOUT_SERVICE_USER, SCOUT_SERVICE_PASSWORD)).content
    resp = json.loads(resp)
    return resp


def listar_biografia_jogador(id):
    '''
    Parâmetros Obrigatórios:
        idSde: ID SDE de um determinado jogador;

    Resposta: Retorna informações de um jogador.
    '''
    jogador = request_scout(f'/ListarBiografiaJogador/{id}')
    return jogador


def listar_biografia_jogadores(ids):
    '''
    Parâmetros Obrigatórios:
        idsSde: ID SDE de um ou mais jogadores, separados por vírgula;

    Resposta: Retorna informações dos jogadores informados.
    '''
    jogadores = request_scout(f'/ListarBiografiaJogadores/{ids}')
    return jogadores


def listar_biografia_tecnico(id):
    '''
    Parâmetros Obrigatórios:
        idSde: ID SDE de um determinado técnico;

    Resposta: Retorna informações de um técnico.
    '''
    tecnico = request_scout(f'/ListarBiografiaTecnico/{id}')
    return tecnico


def listar_biografia_tecnicos(ids):
    '''
    Parâmetros Obrigatórios:
        idsSde: ID SDE de um ou mais técnicos, separados por vírgula;

    Resposta: Retorna informações dos técnicos informados.
    '''
    tecnicos = request_scout(f'/ListarBiografiaTecnicos/{ids}')
    return tecnicos


def listar_campeonatos():
    '''
        Parâmetros: Não há;
        Resposta: Retorne a lista de competições que o Grupo Globo transmite e que há estatiscas de partidas.
        Descrição: Competições identificadas pela Sigla, que devem ser usadas como parâmetro em outros
        métodos do ScoutService;
    '''
    campeonatos = request_scout('/ListarCampeonatos')
    df = pd.DataFrame.from_dict(campeonatos, orient='columns')
    return df


def listar_campeonatos_ativos_footstats():
    campeonatos = request_scout('/ListarCampeonatosAtivosFootstats')
    return campeonatos


def listar_lances_jogador(partida='', tempo1='', minuto1='', tempo2='', minuto2=''):
    '''
    Parâmetros Obrigatórios:
        partida: ID SDE de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (CodigoExterno).
     Parâmetros Opcionais:
         tempo1 e tempo2: período da partida, aceitando os seguintes valores:
         - 1 para o Primeiro Tempo;
         - 2 para o Segundo Tempo;
         - 3 para o Primeiro Tempo da Prorrogação;
         - 4 para o Segundo Tempo da Prorrogação;
         - 5 para a Disputa de Pênaltis.
        minuto1 e minuto2: Tempo da partida em minutos e segundos a ser pesquisado no
        intervalo informado.

    Resposta: Retorna a lista de lances (ações) dos jogadores em uma partida específica.
    '''
    lances = request_scout(f'/ListarLancesJogador?partida={partida}&tempo1={tempo1}&minuto1={minuto1}&tempo2={tempo2}&minuto2={minuto2}')
    print('LANCES',lances)
    return lances


def listar_status_jogadores_assinatura(campeonato, ano, partida):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;
        codPartida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).

    Resposta: Retorna o quantitativo de lances executados pelos jogadores das equipes em uma
    determinada partida.
    '''
    lances = request_scout(f'/ListarStatusJogadoresAssinatura?campeonato={campeonato}&ano={ano}&codPartida={partida}')
    return lances


def listar_lances(campeonato, ano, partida):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;
        codPartida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
    Resposta: Retorna o quantitativo de lances executados pelas equipes em uma determinada partida.
    '''
    lances = request_scout(f'/ListarLances?campeonato={campeonato}&ano={ano}&codPartida={partida}')
    df = pd.DataFrame.from_dict(lances['ListarLancesResult']['Partidas'], orient='columns')
    return df


def listar_lances_jogador_partida(partida, tempo1='', minuto1='', tempo2='', minuto2=''):
    '''
    Parâmetros Obrigatórios:
        partida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
     Parâmetros Opcionais:
         tempo1 e tempo2: período da partida, aceitando os seguintes valores:
         - 1 para o Primeiro Tempo;
         - 2 para o Segundo Tempo;
         - 3 para o Primeiro Tempo da Prorrogação;
         - 4 para o Segundo Tempo da Prorrogação;
         - 5 para a Disputa de Pênaltis.
        minuto1 e minuto2: Tempo da partida em minutos e segundos a ser pesquisado no
        intervalo informado.

    Resposta: Retorna a lista de lances (ações) dos jogadores em uma partida específica.
    '''
    lances = request_scout(f'/ListarLancesJogadorPartida?partida={partida}&tempo1={tempo1}&minuto1={minuto1}&tempo2={tempo2}&minuto2={minuto2}')
    #print('FUNCIONA',lances)

    df = pd.DataFrame.from_dict(lances['ListarLancesJogadorPartidaResult']['Lances'], orient='columns')
    return df


def listar_partidas(campeonato, ano):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;

    Resposta: Retorna todas as partidas cadastradas para a competição/temporada informada.
    Descrição: Cada partida tem um id único (Codigo) e id SDE (CodigoExterno) e que podem ser usados
    nos métodos para a busca de informações das partidas como Escalação, Lance a Lance e outros.
    '''
    partidas = request_scout(f'/ListarPartidas?campeonato={campeonato}&ano={ano}')
    df = pd.DataFrame.from_dict(partidas['ListarPartidasResult'], orient='columns')
    return df


def obter_partidas(campeonato, ano):
    partidas = request_scout(f'/ObterPartidas?campeonato={campeonato}&ano={ano}')
    return partidas


def obter_elenco_equipes_campeonato(campeonato, ano, equipe):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;
        equipe: código interno de uma ou mais equipes, informados separados por vírgula.

    Resposta: Retorna a lista de jogadores para uma ou mais equipes de uma competição/temporada
    informada.
    '''
    elenco = request_scout(f'/ObterElencoEquipesCampeonato?campeonato={campeonato}&ano={ano}&equipe={equipe}')
    return elenco


def obter_fases_campeonato(campeonato, ano):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;

    Resposta: Retorna todas as fases de uma competição/temporada informada.
    Descrição: Cada fase tem um id único (Codigo).
    '''
    fases = request_scout(f'/ObterFasesCampeonato?campeonato={campeonato}&ano={ano}')
    return fases


def obter_jogos_dia_campeonato(campeonato, ano, data):
    '''
    Parâmetros Obrigatórios:
        campeonato: Sigla da competição buscada no ListarCampeonatos;
        ano: ano da temporada no formato YYYY;
        data: data da partida no formato YYYY-MM-DD.

    Resposta: Retorna todas as partidas que acontecerão em uma competição/temporada na data
    informada.
    Descrição: Cada partida tem um id único (Codigo) e id SDE (CodigoExterno) e que podem ser usados
    nos métodos para a busca de informações das partidas como Escalação, Lance a Lance e outros.
    '''
    jogos = request_scout(f'/ObterJogosDiaCampeonato?campeonato={campeonato}&ano={ano}&data={data}')
    return jogos


def obter_partida_escalacao_arbitragem(cod_partida, escalacao, arbitro):
    '''
    Parâmetros Obrigatórios:
        obterEscalacao: parâmetro booleano, se deseja listar ou não os jogadores da partida;
        obterArbitro: parâmetro booleano, se deseja listar a equipe de arbitragem da partida;
        codPartida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).

    Resposta: Retorna a lista de jogadores titulares e reservas das equipes competição/temporada
    informada. Retorna também a equipe de Arbitragem, Resultado, Estádio e Status da Partida.
    '''
    arbitragem = request_scout(f'/ObterPartidaEscalacaoArbitragem?codPartida={cod_partida}&obterEscalacao={escalacao}&obterArbitro={arbitro}')
    return arbitragem


def obter_mapa_calor_partida_equipe(cod_partida, cod_equipe='', tempo=''):
    '''
    Parâmetros Obrigatórios:
        codPartida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
    Parâmetros Opcionais:
        codEquipe: código interno de uma determinada equipe da partida informada;
        tempoPartida: período da Partida em que se deseja retornar o mapa de calor, aceitando os
        seguintes valores:
         - 1 para o Primeiro Tempo;
         - 2 para o Segundo Tempo;
         - 3 para o Primeiro Tempo da Prorrogação;
         - 4 para o Segundo Tempo da Prorrogação;

    Resposta: Retorna o quantitativo de lances executados por cada time em cada quadrante do campo.
    '''
    mapa = request_scout(f'/ObterMapaCalorPartidaEquipe?codPartida={cod_partida}&codEquipe={cod_equipe}&tempoPartida={tempo}')
    df = pd.DataFrame.from_dict(mapa['ObterMapaCalorPartidaEquipeResult'], orient='columns')
    return df


def obter_mapa_calor_partida_jogador(cod_partida, cod_equipe='', tempo=''):
    '''
    Parâmetros Obrigatórios:
        codPartida: código interno de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
    Parâmetros Opcionais:
        codEquipe: código interno de uma determinada equipe da partida informada;
        tempoPartida: período da Partida em que se deseja retornar o mapa de calor, aceitando os
        seguintes valores:
         - 1 para o Primeiro Tempo;
         - 2 para o Segundo Tempo;
         - 3 para o Primeiro Tempo da Prorrogação;
         - 4 para o Segundo Tempo da Prorrogação;

    Resposta: Retorna o quantitativo de lances executados em cada quadrante do campo por cada
    jogador dos times em uma partida.
    '''
    mapa = request_scout(f'/ObterMapaCalorPartidaJogador?codPartida={cod_partida}&codEquipe={cod_equipe}&tempoPartida={tempo}')
    df = pd.DataFrame.from_dict(mapa['ObterMapaCalorPartidaJogadorResult'], orient='columns')
    return df


def obter_mapa_calor_partida_equipe_sde(cod_partida, cod_equipe='', tempo=''):
    '''
    Parâmetros Obrigatórios:
        codPartida: ID SDE de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
    Parâmetros Opcionais:
        codEquipe: ID SDE de uma determinada equipe da partida informada;
        tempoPartida: período da Partida em que se deseja retornar o mapa de calor, aceitando os
        seguintes valores:
         - 1 para o Primeiro Tempo;
         - 2 para o Segundo Tempo;
         - 3 para o Primeiro Tempo da Prorrogação;
         - 4 para o Segundo Tempo da Prorrogação;

    Resposta: Retorna o quantitativo de lances executados por cada time em cada quadrante do campo.
    '''
    mapa = request_scout(f'/ObterMapaCalorPartidaEquipeSDE?codPartida={cod_partida}&codEquipe={cod_equipe}&tempoPartida={tempo}')
    return mapa


def obter_mapa_calor_partida_jogador_sde(cod_partida, cod_equipe='', tempo=''):
    '''
    Parâmetros Obrigatórios:
        codPartida: ID SDE de uma partida, buscada dos métodos ListarPartidas ou
        ObterPartidasDiaCampeonato (Codigo).
    Parâmetros Opcionais:
        codEquipe: ID SDE de uma determinada equipe da partida informada;
        tempoPartida: período da Partida em que se deseja retornar o mapa de calor, aceitando os
        seguintes valores:
         - 1 para o Primeiro Tempo;
         - 2 para o Segundo Tempo;
         - 3 para o Primeiro Tempo da Prorrogação;
         - 4 para o Segundo Tempo da Prorrogação;

    Resposta: Retorna o quantitativo de lances executados em cada quadrante do campo por cada
    jogador dos times em uma partida.
    '''
    mapa = request_scout(f'/ObterMapaCalorPartidaJogadorSDE?codPartida={cod_partida}&codEquipe={cod_equipe}&tempoPartida={tempo}')
    return mapa


def get_coordenadas(posicao, xbin, ybin, xsize, ysize, width, height):
    x = math.ceil(posicao/xsize)-(xbin/100)
    y = math.ceil(((posicao/ysize)-x)*ysize)
    x = x * xbin
    y = (y * ybin)-ybin
    return (x,y)


def campinho(df=None, tipo='', quadrantes=[6,6]):
    # Configurações do campo
    height = 600
    width  = 400

    xbins = height/quadrantes[0]
    ybins = width/quadrantes[1]
    img = plt.imread("gatomestre/apis/campinho.jpg")

    # Data
    lances = df['PosicaoLance'].apply(lambda x: get_coordenadas(x, xbins, ybins, quadrantes[0], quadrantes[1], width, height))

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.ylim([width,0])
    plt.xlim([0,height])

    for lance in lances:
        ax.add_artist(Rectangle(xy=lance,
                      color='firebrick',
                      width=xbins, alpha=0.05, height=ybins))


    plt.show()

