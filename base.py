import requests
import json
import pandas as pd
import numpy as np
from utils import exportar_dataframe
from atualizar_redis import AtualizarRedis
import dpath.util as dpu
from decouple import config
import gatomestre_sde as sde
import gatomestre_utils as dfutils
from stats.ssv_atletas import stats_atletas
from stats.ssv_clubes import stats_confrontos


APP = config('APP')
DEBUG = config('DEBUG')
token = config('SDE_TOKEN1')
CARTOLA_URL = config('CARTOLA_URL')
SDE_URL = config('SDE_URL')

if APP == 'Calculadora':
    PATH_APP = ''
else:
    PATH_APP = 'calculadora-gato-mestre-backend/'


class Base:
    def __init__(self, edicao_cartola):
        self.edicao_cartola = edicao_cartola
        self.scout_list = ['FF', 'FS', 'G', 'A', 'PS', 'PC', 'SG', 'DE', 'DP', 'DS', 'FC', 'GC', 'GS', 'FD', 'CA', 'FT', 'I', 'PI', 'PP', 'CV']



    def consulta_url_sde(self, url):
        headers = {'token': token}
        res = requests.get(SDE_URL + url, headers=headers)
        res = json.loads(str(res.text))
        return res  

 
       
    def _request_sde(self, url, paginacao=True):
        res = self.consulta_url_sde(url)
        result = res
        if paginacao:
            if result:
                while res['paginacao']['proximo']:
                    res = self.consulta_url_sde(res['paginacao']['proximo'])
                    dpu.merge(result, res)
        return result


    def pivot_scouts(self, df):
        print(f'Cartola [{self.edicao_cartola}]: Pivoting Scouts...')
        df['scout'] = df['scout'].apply(lambda
                                            x: (str(x).replace('\'', '"').replace('None', '{}'))
                                            if len(x) > 0
                                            else str(dict.fromkeys(self.scout_list, 0)).replace('\'', '"')
                                        )  # Caso não existam scouts, retorna um dicionario com 0s
        df_scout = df['scout'].apply(json.loads).apply(pd.Series) 
        df = pd.concat([df, df_scout], axis=1).fillna(0)
        return df



    def mercado_cartola(self):
        print('Puxando dados do Cartola...')
        
        ## Apagar depois 
        link_athletes = CARTOLA_URL + '/atletas/mercado'
        # call api
        resp_cartola = requests.get(link_athletes)
        # to json
        df_current_cartola = resp_cartola.json()
        # to df
        df_current = pd.DataFrame(df_current_cartola['atletas'])
        atletas = self.pivot_scouts(df_current)
        atletas.drop(columns=['nome', 'slug', 'apelido', 'apelido_abreviado', 'foto','scout'], inplace=True)
        rodada_atual = atletas['rodada_id'].max()
        return atletas, df_current_cartola['clubes'], df_current_cartola['posicoes'], df_current_cartola['status'], rodada_atual



    def get_jogos_sde(self, edicao):
        url = f'/esportes/futebol/modalidades/futebol_de_campo/categorias/profissional/campeonatos/campeonato-brasileiro/edicoes/campeonato-brasileiro-{edicao}/jogos'
        response = self._request_sde(url)
        df = pd.DataFrame.from_dict(response['resultados']['jogos'])
        return df



    def get_atletas_sde(self, atleta_id):
        url = f'/atletas/{atleta_id}'
        try:
            response = sde._request_sde(url, False)
            df = pd.DataFrame.from_dict(response).T

        except:
            url = f'/tecnicos/{atleta_id}'
            try:
                response = sde._request_sde(url, False)
                df = pd.DataFrame.from_dict(response).T
                df['atleta_id'] = df['tecnico_id']
            except:
                return None
        df = sde.pivot_column(df, 'posicao')
        return df



    def get_scouts(self, df_atletas, df_jogos, dict_clubes, rodada):
        print('Organizando scouts...')
        df_jogos = df_jogos[df_jogos['rodada']==rodada]      
        df_completo = pd.DataFrame()   
        for lado in ['mandante', 'visitante']:
            df = pd.merge(df_atletas, df_jogos, left_on='clube_id', right_on=f'equipe_{lado}_id')
            df['home_dummy'] = 1 if lado == 'mandante' else 0
            df = df[['atleta_id', 'rodada_id', 'clube_id', 'posicao_id', 'status_id',
                                       'pontos_num', 'preco_num', 'variacao_num', 'media_num', 'jogos_num',
                                       'minimo_para_valorizar', 'FF', 'FS', 'G', 'A', 'PS', 'PC', 'SG', 'DE',
                                       'DP', 'DS', 'FC', 'GC', 'GS', 'FD', 'CA', 'FT', 'I', 'PI', 'PP', 'CV',
                                       'escalacao_mandante_id', 'equipe_mandante_id', 'vencedor_jogo',
                                       'suspenso', 'rodada', 'wo', 'hora_realizacao', 'escalacao_visitante_id',
                                       'placar_oficial_visitante', 'equipe_visitante_id',
                                       'placar_penaltis_visitante', 'decisivo', 'jogo_id',
                                       'placar_penaltis_mandante', 'cancelado', 'sede_id',
                                       'placar_oficial_mandante', 'data_realizacao', 'fase_id', 'home_dummy']]
            df['adversario_id'] =  df['equipe_visitante_id'] if lado == 'mandante' else df['equipe_mandante_id']
            df['opponent'] =  df['adversario_id']
            df['clube'] =  df['equipe_visitante_id'] if lado == 'visitante' else df['equipe_mandante_id']
            df.rename(columns={ 
                                'placar_oficial_mandante':'team_goals' if lado == 'mandante' else 'opp_goals',
                                'placar_oficial_visitante':'opp_goals' if lado == 'mandante' else 'team_goals',
                                'jogo_id':'match_id',
                                'homme_dummy': 'mandante'
                                }, inplace=True)
            df_completo = pd.concat([df_completo, df], axis=0)
        return df_completo



    def atualiza_informacoes_atleta(self, df):
        df_atletas = pd.DataFrame()
        for atleta_id in df['atleta_id'].unique():
            result = self.get_atletas_sde(int(atleta_id))
            df_atletas = pd.concat([df_atletas, result]).reset_index(drop=True)
        resultado = df.merge(df_atletas, how='left', on='atleta_id')
        return resultado



    def formatar_colunas(self, df, dict_clubes, dict_posicoes, dict_status):
        print('Formatting columns...')
        df['ano'] = self.edicao_cartola
        df['preco_open'] = df['preco_num'] - df['variacao_num']
        df['posicao'] = df['posicao_id'].apply(lambda x: dict_posicoes[str(x)]['nome'])    
        df['status_pre'] = df['status_id'].apply(lambda x: dict_status[str(x)]['nome'])  
        lados = ['clube', 'adversario_id']
        for lado in lados:
            clube = df.rename(columns={'clube_id': 'old_club_id', lado:'clube_id'})
            df = sde.atualiza_informacoes_clube(clube)
        df.rename(columns=lambda x: x.replace('_y', '_adversario').replace('_x', '_clube'), inplace=True)
        df = df[df['posicao_id'].isin(np.arange(1,7))]
        df = self.atualiza_informacoes_atleta(df)
        df.drop('clube_id', axis=1, inplace=True)
        df.rename(columns={'old_club_id':'clube_id', 
                           'nome_popular_adversario':'adversario_nome',
                           'nome_popular_clube':'clube'}, 
                  inplace=True)

        # Inserindo a informação se o atleta jogou ou não
        pontuados = cartola.get_pontuados(df['rodada_id'].max())
        df['entrou_em_campo'] = df['atleta_id'].map(pontuados.set_index('atleta_id')['entrou_em_campo']).fillna(False)
        df['exibir_atleta'] = df.apply(lambda x: True if ((x['pontos_num']!=0) or (x['variacao_num']!=0) and (x['entrou_em_campo']==True) ) else False, axis=1)
        
        df = df[['nome', 'slug', 'apelido', 'atleta_id', 'rodada_id', 'clube_id', 'posicao_id',
                        'status_id', 'pontos_num','preco_num','variacao_num', 'media_num', 'jogos_num',
                        'clube', 'posicao', 'status_pre', 'FF', 'FS', 'G', 'A', 'SG', 'DE', 'DS', 'DP', 'FC',
                        'GC', 'GS', 'FD', 'CA', 'FT', 'I', 'PP', 'CV', 'PC', 'PI', 'PS', 'preco_open', 'ano', 
                        'home_dummy', 'opponent', 'team_goals', 'opp_goals', 'match_id', 'adversario_nome','entrou_em_campo','exibir_atleta']]
        
        df = df.loc[:,~df.columns.duplicated()]        
        return df.sort_values(by='rodada_id')



    def get_escalacao(self, jogo_id):
        all_fixtures = pd.DataFrame()
        
        url = f'''/jogos/{jogo_id}'''
        response = sde._request_sde(url, False)
        
        try:
            escalacao_id = response['referencias']['escalacao']
        except:
            escalacao_id = {}
            
        for id_ in list(escalacao_id.keys()):
            curr_titulares = pd.DataFrame(response['referencias']['escalacao'][id_]['titulares']).copy()
            curr_reservas = pd.DataFrame(response['referencias']['escalacao'][id_]['reservas']).copy()
            
            curr_titulares['status_inicial'] = 'titular'
            curr_reservas['status_inicial'] = 'reserva'
            
            all_fixtures = all_fixtures.append(curr_titulares,ignore_index=True,sort=False).reset_index(drop=True).copy()
            all_fixtures = all_fixtures.append(curr_reservas,ignore_index=True,sort=False).reset_index(drop=True).copy()
            
        
        df_ficha_jogo = sde.get_ficha_jogo(jogo_id)
        
        df_equipe = pd.DataFrame(df_ficha_jogo[['escalacao_mandante_id','equipe_mandante_id']].values)
        df_equipe = df_equipe.append(pd.DataFrame(df_ficha_jogo[['escalacao_visitante_id','equipe_visitante_id']].values))
        df_equipe.columns = ['escalacao_id','equipe_id']; 
        
        all_fixtures['jogo_id'] = jogo_id
        all_fixtures = all_fixtures.rename(columns={'atleta_id':'atleta_id_comecou'})
        try:
            all_fixtures['equipe_id'] = all_fixtures['escalacaoequipe_id'].map(df_equipe.set_index('escalacao_id')['equipe_id'])
        except:
            all_fixtures['equipe_id'] = 0
            
        return all_fixtures



    def convert_time_to_seg(self, time):
        ftr = [60,1]
        time = time.split(':')[0:2]
        time = ':'.join(time)
        time_sef = sum([a*b for a,b in zip(ftr, map(int,time.split(':')))])
        return time_sef 



    def get_duracao_partida_sde(self, jogo_id=None, unidade='seg'): 
        lances = sde.get_scouts_jogos_sde(jogo_id)
        if len(lances) == 0:
            return 0, 0
        lances['tempo'] = lances['momento'] = lances['momento'].fillna('00:00')
        lances['momento'] = lances['momento'].apply(lambda x: self.convert_time_to_seg(x))
        
        # Primeiro tempo
        df_1t = lances[lances['periodo']=='1tr'].copy()
        duracao_t1 = df_1t['momento'].max()
        
        #Segundo tempo 
        df_2t = lances[lances['periodo']=='2tr'].copy()
        duracao_t2 = df_2t['momento'].max()
        
            
        if unidade == 'segundos' or unidade == 'seg':
            None
        
        elif  unidade == 'minutos' or unidade == 'min': 
            duracao_t1 = duracao_t1/60
            duracao_t2 = duracao_t2/60 
        
        else:
            duracao_t1 = 45
            duracao_t2 = 45
            
        return round(duracao_t1), round(duracao_t2)



    def minutos_jogados(self, jogo_id):
        duracao_t1, duracao_t2 = self.get_duracao_partida_sde(jogo_id,'min')
        tempo_total_jogo = duracao_t1 + duracao_t2
           
        df = self.get_escalacao(jogo_id)
        # print(df)
        df = dfutils.pivot_columns(df, ['substituido'])
        df = df.rename(columns={'atleta_id':'atleta_id_entrou',
                                'atleta_id_comecou':'atleta_id'})
        df['tempo_atleta_jogou'] = np.nan
        if tempo_total_jogo == 0:
            return df.fillna(0), duracao_t1, duracao_t2
        
        df_atletas_subtituidos = df.dropna(subset=['periodo'])
        
        for date, row in df_atletas_subtituidos.iterrows():
            momento = row.momento
            
            if momento == '':
                momento = '00:00'        
            
            tempo_seg = self.convert_time_to_seg(momento)
            tempo_min = round(tempo_seg/60)
            
            atleta_entrou = int(row.atleta_id_entrou)
    
            atleta_saiu = int(row.atleta_id)
            
            if row.periodo == 'itr': 
                tempo_atleta_saiu = duracao_t1
                tempo_atleta_entrou = duracao_t2
                
            elif row.periodo == '1tr':
                tempo_atleta_saiu = tempo_min
                tempo_atleta_entrou = (duracao_t1) - tempo_min + (duracao_t2)
                
            elif row.periodo == '2tr':
                tempo_atleta_saiu = tempo_min + duracao_t1
                tempo_atleta_entrou = (duracao_t2) - tempo_min
                
            # Inserindo o tempo jogado. 
            # Atleta que entrou no jogo
            df.loc[df['atleta_id'] == atleta_entrou, 'tempo_atleta_jogou'] = tempo_atleta_entrou
            # Atleta que foi substituido
            df.loc[df['atleta_id'] == atleta_saiu, 'tempo_atleta_jogou'] = tempo_atleta_saiu
        
        df['tempo_atleta_jogou'] = df.apply(lambda x: tempo_total_jogo if x['status_inicial'] == 'titular' and pd.isna(x['tempo_atleta_jogou']) else x['tempo_atleta_jogou'], axis=1)
        
        df['tempo_atleta_jogou'] =  df['tempo_atleta_jogou'].fillna(0)
        # df['tempo_atleta_jogou'] = df['tempo_atleta_jogou']
        return df, duracao_t1, duracao_t2


    def inserir_minutagem(self, database): 
        '''
        Inserir a minutage cada atleta em cada jogo
        '''
        print('Carregando minutos jogados de cada atletas...')
        df = pd.DataFrame()
        for i in database['match_id'].unique():
            try: 
                df_jogo = database[database['match_id']==i].copy()
                df_minutagem, duracao_t1, duracao_t2 = self.minutos_jogados(int(i))
                duracao_jogo = duracao_t1 + duracao_t2
                df_jogo['minutos_jogados'] = df_jogo['atleta_id'].map(df_minutagem.set_index('atleta_id')['tempo_atleta_jogou']).fillna(0)
                df_jogo['status_inicial'] = df_jogo['atleta_id'].map(df_minutagem.set_index('atleta_id')['status_inicial']).fillna(0)
                df_jogo['minutos_jogados'] = df_jogo.apply(lambda x: duracao_jogo if x['posicao_id'] == 6 else x['minutos_jogados'], axis=1)
                df_jogo['duracao_1t'] = duracao_t1
                df_jogo['duracao_2t'] = duracao_t2
                df_jogo['duracao_partida'] = duracao_jogo
                df_jogo['pontos_por_minuto'] = df_jogo['pontos_num']/df_jogo['minutos_jogados']
            except: 
                print(f'Erro ao pegar a minutagem do jogo {i}')
                None
            df = df.append(df_jogo).sort_values(by='rodada_id').reset_index(drop=True)

       
        return df.fillna(0)



    def atualizar_base_cartola(self, df, rodada):
        print('Saving Results...')
        if rodada > 1:
            try:
                # Geral 
                df_cartola = pd.read_csv(f'{PATH_APP}database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}', compression='gzip', low_memory=False)
                df_cartola = df_cartola[df_cartola['rodada_id']<rodada]
                df_concat = pd.concat([df_cartola, df]).sort_values(by='rodada_id').drop_duplicates(subset=['match_id','atleta_id'],keep='last').reset_index(drop=True)
                
                # Individual
                df_cartola_individual = pd.read_csv(f'{PATH_APP}database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}_Individual', compression='gzip', low_memory=False)
                df_cartola_individual = df_cartola_individual[df_cartola_individual['rodada_id']<rodada]
                df_cartola_individual = pd.concat([df_cartola_individual, df]).sort_values(by='rodada_id').drop_duplicates(subset=['match_id','atleta_id'],keep='last').reset_index(drop=True)
                
            except:
                print('Não encontrou o arquivo base')
            
            # Salvando Cartola_2022
            df_concat.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}',index=False, compression='gzip')
            df_concat.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/backup/Cartola_{self.edicao_cartola}_R1a{rodada}',index=False, compression='gzip')
            
            
            print('Finalizando base de dados...')
            df_pontuados_rodada = cartola.get_pontuados(rodada)
            df_pontuados_rodada = dfutils.pivot_columns(df_pontuados_rodada, ['scout']).fillna(0)
            
            # Salvnado Cartola_2022_Individual
            scout_list = ['A', 'CA', 'CV', 'DE', 'DP', 'DS', 'FC', 'FD', 'FF', 'FS', 'FT', 'G', 'GC', 'GS', 'I', 'PC', 'PI', 'PP', 'PS', 'SG']
            df = df_cartola_individual[df_cartola_individual['rodada_id']==rodada].copy()
            df['entrou_em_campo'] = df['atleta_id'].map(df_pontuados_rodada.set_index('atleta_id')['entrou_em_campo']).fillna(False)
            
            df_cartola_individual = df_cartola_individual[df_cartola_individual['rodada_id']<rodada]
            for scout in scout_list:
                df[scout] = np.nan
                try: 
                    df[scout] = df['atleta_id'].map(df_pontuados_rodada.set_index('atleta_id')[scout])
                except:
                    print(f'Na rodada {rodada} não houve {scout}')
                    df[scout] = 0
            df_cartola_individual = df_cartola_individual.append(df).sort_values(by='rodada_id').reset_index(drop=True).fillna(0)
            df_cartola_individual = df_cartola_individual.astype({'atleta_id': 'int32', 'rodada_id': 'int32', 'clube_id': 'int32', 'posicao_id': 'int32', 'status_id': 'int32',
                                                                  'jogos_num': 'int32', 'home_dummy': 'int32', 'opponent': 'int32', 'match_id': 'int32'})
                    
           
            # Salvando scouts individuais
            df_cartola_individual.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}_Individual',index=False, compression='gzip')
            
            # Salvando Backup
            df_cartola_individual.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/backup/Cartola_{self.edicao_cartola}_Individual_R1a{rodada}',index=False, compression='gzip')
            
            print(df, df.info(verbose=True))
            # print(df_cartola_individual, df_cartola_individual.info(verbose=True))
        else:    
            # Salvando Scouts Somados
            df.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}',index=False, compression='gzip')
            df.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}_Individual',index=False, compression='gzip')
             # Salvando Backup
            df.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/backup/Cartola_{self.edicao_cartola}_R1a{rodada}',index=False, compression='gzip')
            df.to_csv(f'{PATH_APP}database/{self.edicao_cartola}/backup/Cartola_{self.edicao_cartola}_Individual_R1a{rodada}',index=False, compression='gzip')
            
            print(df, df.info(verbose=True))
        
        # print(df_cartola_individual, df_cartola_individual.info(verbose=True))
        return df_cartola_individual



    def atualizar_base_historica(self, ultima_edicao=2020):
        """
        Gerar base do Cartola Historica Atualizada

        Parameters
        ----------
        rodada: ultima edicao a ser adicionada a base historica completa

        Returns
        -------
        df: pd.DataFrame
        Dataframe com a base historica atualizada
        """
        cartola_all_previous = pd.read_csv(f'{PATH_APP}database/All/Cartola_All.gz', compression='gzip', low_memory=False)

        df_cartola_update = pd.read_csv(f'{PATH_APP}database/{ultima_edicao}/Cartola_{ultima_edicao}_Individual', compression='gzip', low_memory=False)
        
        df_to_append = df_cartola_update[['match_id','rodada_id','clube_id','clube','opponent','adversario_nome','home_dummy',
                                'team_goals', 'opp_goals','atleta_id','apelido','posicao','pontos_num',
                                'preco_num', 'variacao_num','media_num', 'jogos_num','status','ano']].copy()

        df_to_append['ano'] = df_to_append['ano'].mask(df_to_append['ano']<=0,int(ultima_edicao))

        df_to_append = df_to_append.rename(columns={'match_id':'jogo_id',
                                                    'clube_id':'time_id',
                                                    'clube':'time_nome',
                                                    'opponent':'adversario_id',
                                                    'home_dummy':'mandante',
                                                    'team_goals':'gols_time',
                                                    'opp_goals':'gols_adv',
                                                    'apelido':'atleta_nome'})

        df = pd.concat([cartola_all_previous, df_to_append])
                
        df = df.astype({'time_id': 'int32', 'ano': 'int32'})
        df.to_csv(f'{PATH_APP}database/All/Cartola_All.gz',index=False, compression='gzip')
        df.to_csv(f'{PATH_APP}database/All/Cartola_All.csv')
        print (df, df.info(verbose=True))

        return df



    def run(self, export_format=None, vault=None):
        """
        Gerar base do Cartola Atualizada

        Parameters
        ----------
        rodada: ultima rodada jogada e atualizada no API Cartola
        export_format: excel, csv or json
        vault: setar para True quando for subir no Vault

        Returns
        -------
        df: pd.DataFrame
        Dataframe a base atualizada
        """
        df_atletas, dict_clubes, dict_posicoes, dict_status, rodada_atual = self.mercado_cartola()
        df_jogos = self.get_jogos_sde(self.edicao_cartola)
        df_scouts = self.get_scouts(df_atletas, df_jogos, dict_clubes, rodada_atual)
        df_scouts = self.formatar_colunas(df_scouts, dict_clubes, dict_posicoes, dict_status)
        df_scouts = self.inserir_minutagem(df_scouts)
        df_individual = self.atualizar_base_cartola(df_scouts, rodada_atual)

        # Nomeia e exporta o arquivo para o formato desejado
        if export_format:
            conteudo = 'base'
            nome_arquivo = 'Base_Rodada_{}'.format(rodada)
            exportar_dataframe(df_individual, conteudo, nome_arquivo, export_format, vault)

        return rodada_atual, df_individual



    def run_scout_service(self, export_format=None, vault=None, edicao=2023, teams=False):
        """
        Gerar base de eventos dos atletas, conforme API Scout Service

        Parameters
        ----------
        export_format: excel, csv or json
        vault: setar para True quando for subir no Vault

        Returns
        -------
        df: pd.DataFrame
        Dataframe a base atualizada
        """
        


        # torneios = ['Libertadores',  'Brasileiro', 'BrasileiroB', 'CopaBrasil', 'CopaNordeste', 'CopaSPJunior', 'Carioca','Gaucho', 'Mineiro', 'Baiano', 'Paulista', 'Pernambucano',
        #             'PreLibertadores', 'RecopaSulAmericana', 'SulAmericana', 'SuperCopaBrasil']        
    
        torneios = ['Libertadores','CopaBrasil','Brasileiro']

        if teams:
            torneios = ['Brasileiro']
            df_eventos = stats_confrontos(torneios,edicao=edicao)
        else:
            print(f'Atualizando base Scout Service para a temporada {edicao}...')
            print ('###########################################################')
            print(f'Torneios sendo atualizados: {torneios}...')
            print ('#####################################################################################################')

            df_eventos = stats_atletas(torneios=torneios,edicao=edicao)

        return df_eventos

        
if __name__ == "__main__":
    atualizador = AtualizarRedis('2023')
    base = Base('2023')
    # rodada, df_individual = base.run(vault=True)
    df_eventos = base.run_scout_service(edicao=2023)
    #df_eventos = base.run_scout_service(edicao=2023,teams=True)

    #atualizador.atualizar_base(df_individual, rodada)

    
