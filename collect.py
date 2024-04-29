import requests
import json
import numpy as np
import pandas as pd
import warnings
import dpath.util as dpu
from decouple import config
from sqlalchemy import create_engine
from tqdm import tqdm

from database import DatabaseGM
from pred import odds
import apis.sde as sde
import apis.scoutservice as sct
import apis.cartola as cartola
import utils.dataframes as df_utils

class CollectGM:
    
    def __init__(self) -> None:
        self.database  = DatabaseGM()
        self.df_jogos  = []
        self.equipes   = []
        self.sedes     = []
        self.ids_jogos = []
        self.atletas   = {}
        self.arbitros  = []
        self.proximos_jogos = []
        
    
    def undo_collect(self, table):
        self.database.clear(table)
     
    
    
    def collect_lances_scoutservice(self):
        jogos_sem_lances        = self.database.identify_news('lancesscoutservice', self.ids_jogos)
        df_lances_scout_service = pd.DataFrame()
        for jogo_id in jogos_sem_lances:
            df_jogo    =  self.jogos[ self.jogos['jogo_id']==jogo_id]
            elenco_t1  = sde.get_elenco(df_jogo['equipe_mandante_id'].iloc[0],df_jogo['data_realizacao'].iloc[0])
            elenco_t2  = sde.get_elenco(df_jogo['equipe_visitante_id'].iloc[0],df_jogo['data_realizacao'].iloc[0])
            df_elencos = pd.concat([elenco_t1,elenco_t2])
            df         = sct.listar_lances_jogador(jogo_id)
            df         = df_utils.pivot_columns(df, ['Jogador','Partida'],append=True)
            df.rename(columns={'Jogador_CodigoExterno':'atleta_id','Partida_CodigoExterno':'sde_id','PosicaoLance':'quadrante','TempoPartida':'tempo','Nome':'sequencia_lance'}, inplace=True)
            df              = df.dropna(subset=['atleta_id'])
            df['atleta_id'] = df['atleta_id'].astype('int')
            df['lance']     = df['Codigo'].apply(lambda x: sct.get_lance(x).split(',')[0])
            df['minutagem'] = df['Ocorrencia'].apply(lambda x: df_utils.minuto_ocorrencia(x))
            df['equipe_id'] = df['atleta_id'].map(df_elencos.set_index('atleta_id')['equipe_id'])
            df              = df[['atleta_id','equipe_id','sde_id','lance','sequencia_lance','quadrante','tempo','minutagem']]
            data_columns    = [col for col in df.columns if col not in ['atleta_id', 'equipe_id','sde_id']]
            df['data']      = json.dumps(df[data_columns].to_dict('records'))
            df              = df.drop(columns=['lance', 'sequencia_lance','quadrante','tempo','minutagem'])
            df_lances_scout_service = pd.concat([df_lances_scout_service, df], axis=0)
        return df_lances_scout_service
   
