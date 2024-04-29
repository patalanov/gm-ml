import requests
import json
import numpy as np
import pandas as pd
import warnings
import dpath.util as dpu
from decouple import config
from sqlalchemy import create_engine, delete
import os

import apis.sde as sde
import apis.scoutservice as sct
import apis.cartola as cartola
import utils.dataframes as df_utils

class DatabaseGM:
  
  
    def __init__(self) -> None:
        STR = config('DBAAS_MYSQL_ENDPOINT')
        CREDENCIAIS_DB = STR.split('//')[1]
        SPLIT_CREDENCIAIS = CREDENCIAIS_DB.split(':')
        USER_DB = SPLIT_CREDENCIAIS[0]
        PASS_DB = (SPLIT_CREDENCIAIS[1]).split('@')[0]
        HOST_DB = (SPLIT_CREDENCIAIS[1]).split('@')[1]
        NAME_DB = STR.split(':3306/')[1]

        self.ENGINE   = 'mysql' #config('GM_DATABASE_ENGI')
        if self.ENGINE == 'mysql':
            self.engine = create_engine(f"""mysql+pymysql://{USER_DB}:{PASS_DB}@{HOST_DB}:3306/{NAME_DB}""")
        else:
            self.engine = create_engine("sqlite:///gm_database_backup.db")

    
    
    def drop_table(self, table):
        con = self.engine.connect()
        trans = con.begin()
        con.execute('SET FOREIGN_KEY_CHECKS = 0;')
        q = f"DROP TABLE {table};" 
        con.execute(q)
        con.execute('SET FOREIGN_KEY_CHECKS = 1;')
        trans.commit()
               
    
    def test_connection(self):
        try:
            self.engine.execute("SELECT 1")
            print('Connected')
        except Exception as e:
            print('Error: ', e)
            
        
    def perform_query(self, query):
        result = pd.read_sql(query, self.engine)
        print(result)
    
         
    def identify_news(self, entidade, lst, key='sde_id'):
        '''
            Recebe uma lista de ids e verifica quais ainda não
            estão no banco de dados, na tabela indicada.
        '''
        sql          = f'SELECT {key} FROM core_{entidade}'
        database_ids = list(pd.read_sql(sql, self.engine)[key])
        new_ids      = list(filter(lambda x: x not in database_ids, lst))
        print(f'[{entidade.upper()}] {len(new_ids)} registros novos encontrados')
        return new_ids
    
    
    
    def get_query_scouts(self):
        AGGS   = ['SUM', 'AVG']
        SCOUTS = ['G', 'A', 'FT', 'FD', 'FF', 'FS', 'PS', 'PP', 'I',
                    'DP', 'SG', 'DE', 'DS', 'GC', 'CV', 'CA', 'GS', 'FC', 'PC']
        lista = []
        for scout in SCOUTS:
            for agg in AGGS:
                lista.append(f"{agg}({scout}) as {scout}_{agg}")
        query_scouts = ", ".join(lista)
        return query_scouts
    
    
    
    def get_atletas(self):
        #query_scouts = self.get_query_scouts()
        query = f"""SELECT * FROM core_atletas"""
        df_atletas = pd.read_sql(query, self.engine)
        return df_atletas  
    
    
    
    def get_scouts(self, rodada, temporada, mando='geral'):
        query_scouts = self.get_query_scouts()
        query = f"""SELECT atleta_id, {query_scouts} FROM core_scoutsfantasy WHERE rodada_id <={rodada} AND temporada = {temporada} GROUP BY atleta_id"""
        if mando == 'mandante':
            query = query.replace('AND', 'AND home_dummy = 1 AND')
        if mando == 'visitante':
            query = query.replace('AND', 'AND home_dummy = 0 AND')
        df_scouts = pd.read_sql(query, self.engine)
        return df_scouts    
            
    
    def drop_old(self, entidade, key, list):
        tuple_ids = tuple(list)
        print('Droping', tuple_ids)
        q = f"DELETE FROM core_{entidade} WHERE {key} IN {tuple_ids}" 
        self.engine.execute(q)
    
    
    
    def get_ids(self, entidade):
        sql          = f'SELECT sde_id FROM core_{entidade}'
        database_ids = list(pd.read_sql(sql, self.engine)['sde_id'])
        return database_ids
    
    
    
    
    def clear(self, table=None):
        if table == 'all':
            tables_list = [
                'core_mercadofantasy',
                'core_scoutsfantasy',
                'core_atletas',
                'core_arbitros',
                'core_arbitragem',
                'core_jogos',
                'core_equipes',
                'core_scoutssde',
                'core_sedes'
            ]
        else:
            tables_list = [table]
              
        for table in tables_list:
            q = f'DELETE FROM {table}'
            self.engine.execute(q)
            
            
            
            
    def get_sde_id(self, rodada, temporada, equipe_id):
        '''
            Recebe uma rodada e um clube id e retornar o id sde do jogo 
        '''
        sql  = f'SELECT sde_id FROM core_jogos WHERE rodada = {rodada} AND temporada = {temporada} AND (equipe_mandante_id = {equipe_id} OR equipe_visitante_id = {equipe_id})'
        try:
            database_id  = int(pd.read_sql(sql, self.engine)['sde_id'].iloc[0])
        except Exception as e:
            print(e)
            database_id  = 0
        return database_id
    
    
    
    
    def get_dados_rodada(self, rodada, temporada):
        print("Retornando dados da rodada")
        SQL = f"""
                SELECT 
                    jogos.data_realizacao as data_realizacao, 
                    mercado.rodada as rodada_id, 
                    mercado.atleta_id, 
                    mercado.equipe_id,
                    atletas.nome,
                    atletas.nome as apelido, 
                    atletas.pos_macro, 
                    atletas.sde_slug as slug, 
                    atletas.equipe_id as clube_id,
                    jogos.equipe_mandante_id, 
                    jogos.equipe_visitante_id, 
                    jogos.sde_id as match_id,
                    scoutssde.data as data_sde, 
                    equipes.nome_popular as clube,
                    IF(mercado.equipe_id = jogos.equipe_mandante_id, jogos.equipe_visitante_id, jogos.equipe_mandante_id) as opponent, 
                    IF(mercado.equipe_id = jogos.equipe_mandante_id, jogos.placar_oficial_mandante, jogos.placar_oficial_visitante) as team_goals, 
                    IF(mercado.equipe_id = jogos.equipe_mandante_id, jogos.placar_oficial_visitante, jogos.placar_oficial_mandante) as opp_goals,
                    IF(mercado.equipe_id = jogos.equipe_mandante_id, 1, 0) as home_dummy,
                    IF(mercado.equipe_id = jogos.equipe_mandante_id, visitantes.nome_popular, mandantes.nome_popular) as adversario_nome,
                    fantasy.pontos_num,
                    fantasy.data as data_cartola,
                    mercado.preco_num, 
                    mercado.preco_open,
                    mercado.variacao_num,
                    mercado.jogos_num,
                    mercado.media_num,
                    mercado.status_id,
                    {temporada} as ano,
                    1 as entrou_em_campo,
                    mercado.posicao_id
                FROM core_jogos jogos
                LEFT JOIN core_equipes mandantes 
                    ON jogos.equipe_mandante_id = mandantes.sde_id
                LEFT JOIN core_equipes visitantes 
                    ON jogos.equipe_visitante_id = visitantes.sde_id 
                LEFT JOIN core_atletas atletas
                    ON mandantes.sde_id = atletas.equipe_id OR visitantes.sde_id = atletas.equipe_id
                LEFT JOIN core_mercadofantasy mercado 
                    ON jogos.sde_id = mercado.sde_id AND mercado.atleta_id = atletas.sde_id
                LEFT JOIN core_scoutssde scoutssde
                    ON jogos.sde_id = scoutssde.sde_id AND atletas.sde_id = scoutssde.atleta_id
                LEFT JOIN core_scoutsfantasy fantasy
                    ON jogos.sde_id = fantasy.sde_id AND atletas.sde_id = fantasy.atleta_id
                LEFT JOIN core_equipes equipes 
                    ON atletas.equipe_id = equipes.sde_id
                WHERE jogos.temporada = {temporada} AND jogos.rodada <= {rodada} AND jogos.data_realizacao is not null
                """
        if self.ENGINE == 'sqlite':
            SQL = SQL.replace("IF(", "IIF(")
        df_cartola = pd.read_sql(SQL, self.engine)
        return df_cartola.dropna(subset=['data_cartola'])