import requests
import json
import numpy as np
import pandas as pd
import warnings
import dpath.util as dpu
from decouple import config
from sqlalchemy import create_engine

from database import DatabaseGM
import apis.sde as sde
import apis.scoutservice as sct
import apis.cartola as cartola
import utils.dataframes as df_utils

class IngestGM:
    
    def __init__(self) -> None:
        self.database  = DatabaseGM()
                
                
    def ingest_lances_scoutservice(self, df):
        if len(df) > 0:
            df[[
                'equipe_id',
                'atleta_id',
                'data',
                'sde_id'
                ]].to_sql('core_lancesscoutservice', 
                          con=self.database.engine, 
                          if_exists='append', 
                          index=False)
    
    
    