# import sys 
# sys.path.append("../")
# In[import libs]
import pandas as pd
import numpy as np
import os
import json
import pickle
import math
import time

import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from matplotlib.patches import Arc
from random import randint

from sklearn import metrics 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sources.apis.utils import sct as sct
from maths.pitch import dictCoordenadas36, draw_pitch, set_xy_coordinates
from maths.trig import calculate_distance_angles
from prettytable import PrettyTable

from calls import get_equipes_sde, get_jogos_sde
import gatomestre_sde as sde


# limpar base de dados wue contenham atltas com '0'
def atualiza_informacoes_atleta(df):
    df_atletas = pd.DataFrame()
    for atleta_id in df['atleta_id'].unique():
        result = sde.get_atletas_sde(int(atleta_id))
        print(result)
        df_atletas = pd.concat([df_atletas, result]).reset_index(drop=True)
    resultado = df.merge(df_atletas, how='left', on='atleta_id')
    return resultado



class xG:

        def __init__(self, edicao_cartola):       
                self.edicao_cartola = edicao_cartola
                self.file_cartola = f'database/{self.edicao_cartola}/Cartola_{self.edicao_cartola}_Individual'
                self.cartola = pd.read_csv(self.file_cartola, compression='gzip', low_memory=False)        

                self.rodada = self.cartola['rodada_id'].max()
                #print ('RODADA cartola', self.rodada)
                self.clubes, self.clubes_e_ids, self.equipes_sde = get_equipes_sde(self.edicao_cartola)
                self.times_dict_r = {v:k for k, v in self.clubes_e_ids.items()}
                self.clubes_ids = list(self.clubes_e_ids.values())

                # all history events
                self.file_scout_service = f'database/{self.edicao_cartola}/scout_service/events/Eventos_All.gz'                
                self.eventos = pd.read_csv(self.file_scout_service, compression='gzip', low_memory=False)        
                # lidar com Nan
                self.eventos = self.eventos.fillna(0)
                print(self.eventos.info())
                # limpar colunas
                del self.eventos['index']
                #del raw['Unnamed: 0']
                # dados que precisam ser type <int>
                self.eventos = self.eventos.astype({'Edicao':'int32', 'PosicaoLance': 'int32', 'TempoPartida': 'int32', 'Jogador_Posicao': 'int32', 'atleta_id': 'int32', 'Rodada': 'int32','Codigo':'int32', 'clube_id':'int32', 'oponente_id':'int32', 'home_dummy': 'int32'})
                # eliminar lances inexistentes no dataframe
                self.eventos = self.eventos[self.eventos['PosicaoLance']!=-1]
                print ('Loaded events from updated Scout Service databse...')
                print ('###################################################')
                print (self.eventos)

                # padronizar colunas de acordo com pipeline cartola FC
                self.eventos = self.eventos.rename(columns={'Jogador_Posicao':'posicao_id',
                                         'Rodada':'rodada_id',
                                         'oponente_id':'adversario_id',
                                         'Partida_CodigoExterno': 'match_id',
                                         'Partida_CodigoInterno':'match_id_sct'})

                # remover duplicados
                self.eventos = self.eventos.drop_duplicates(keep='last').reset_index(drop=True)

                print(self.eventos.info())

                # transformar coordenadas centralizadas nos 36 quadrantes em coordenadas (x,y) com variância randomizada
                # filtragem por finalizações
                temp_filename = f'database/{self.edicao_cartola}/scout_service/finalizacoes/Finalizacoes_R{self.rodada}.csv'
                if os.path.exists(temp_filename):
                        self.df_finalizacoes = pd.read_csv(temp_filename,index_col=0)
                else:
                        print ('Setting up coordinates for pitch events...')
                        print ('##########################################')
                        self.df_finalizacoes = set_xy_coordinates(self.eventos)
                        self.df_finalizacoes.to_csv(f'database/{self.edicao_cartola}/scout_service/finalizacoes/Finalizacoes_R{self.rodada}.csv')

                # df_finalizacoes = pd.read_csv('database/scout_service/finalizacoes/Finalizacoes.csv')
                print('RODADA Scout Service', self.df_finalizacoes['rodada_id'].max())
                print('df_finalizacoes',self.df_finalizacoes, self.df_finalizacoes.info())
                # print('TORNEIOS',set(self.df_finalizacoes['Torneio']))
                # print('EDICAO',set(self.df_finalizacoes['Edicao']))
                # print('EDICAO CARTOLA', self.edicao_cartola)


        def gerar_metricas_xG(self,torneio='Brasileiro',edicao=2023):

                df_geral = self.logistic_regression_xG(df=self.df_finalizacoes)
                print('df_geral',df_geral, df_geral.info())

                # # filtrar apenas lances do campeonato brasileiro
                # df_brasileiro = df_geral.loc[(self.df_finalizacoes['Torneio']=='Brasileiro')&(self.df_finalizacoes['Edicao']==self.edicao_cartola)]
                df_brasileiro = df_geral.loc[(self.df_finalizacoes['Torneio'] == torneio) & (self.df_finalizacoes['Edicao'] == edicao)]
                print('df_brasileiro',df_brasileiro, df_brasileiro.info())

                predictions_filename = f'temp/csv/xG_preds_R{self.rodada}.csv'
                temp_filename = f'temp/csv/clubes_xg_R{self.rodada}.csv'
                
                if os.path.exists(temp_filename):
                        print('---------------------------------------------------------------')
                        print('Métricas de xG já foram calculadas para a rodada. Recuperando..')
                        print('---------------------------------------------------------------')
                        clubes_xg = pd.read_csv(f'temp/csv/clubes_xg_R{self.rodada}.csv',index_col=0)
                        tabela_xp = pd.read_csv(f'temp/csv/tabela_xg_R{self.rodada}.csv',index_col=0)
                        clubes_xg_mando = pd.read_csv(f'temp/csv/clubes_xg_mando_R{self.rodada}.csv',index_col=0)
                        clubes_xg_diff = pd.read_csv(f'temp/csv/clubes_xg_diff_R{self.rodada}.csv',index_col=0)
                        clubes_xg_saldo = pd.read_csv(f'temp/csv/clubes_xg_saldo_R{self.rodada}.csv',index_col=0)
                        clubes_xG_probs = pd.read_csv(f'temp/csv/clubes_xG_probs_R{self.rodada}.csv',index_col=0)
                        clubes_xg_defense = pd.read_csv(f'temp/csv/clubes_xg_defense_R{self.rodada}.csv',index_col=0)
                        clubes_goals_mando = pd.read_csv(f'temp/csv/clubes_goals_mando_R{self.rodada}.csv',index_col=0)
                        clubes_gols_diff = pd.read_csv(f'temp/csv/clubes_gols_diff_R{self.rodada}.csv',index_col=0)
                        clubes_xG_vs_G_composite = pd.read_csv(f'temp/csv/clubes_xG_vs_G_composite_R{self.rodada}.csv')
                        clubes_xg_potencial = pd.read_csv(f'temp/csv/clubes_xg_potencial_R{self.rodada}.csv',index_col=0)
                        atletas_xg = pd.read_csv(f'temp/csv/atletas_xg_R{self.rodada}.csv',index_col=0)
                        atletas_xg_detalhado = pd.read_csv(f'temp/csv/atletas_xg_detalhado_R{self.rodada}.csv',index_col=0)
                        atletas_xg_per_match = pd.read_csv(f'temp/csv/atletas_xg_per_match_R{self.rodada}.csv',index_col=0)
                        atletas_xg_eficiencia = pd.read_csv(f'temp/csv/atletas_xg_eficiencia_R{self.rodada}.csv',index_col=0)
                        atletas_xg_potencial = pd.read_csv(f'temp/csv/atletas_xg_potencial_R{self.rodada}.csv',index_col=0)

                else:
                        # check for model predictions
                        if os.path.exists(predictions_filename):
                                print('---------------------------------------------')
                                print('Predições de xG já foram feitas para a rodada')
                                print('---------------------------------------------')
                                df_geral = pd.read_csv(f'temp/csv/xG_preds_R{self.rodada}.csv',index_col=0)
                        else:
                                # run predictions
                                print('----------------------------------------------------')
                                print('Criando modelo e fazendo cálculo de xG para a rodada')
                                print('----------------------------------------------------')
                                df_geral = self.logistic_regression_xG(df=self.df_finalizacoes)
                                df_geral.to_csv(f'temp/csv/xG_preds_R{self.rodada}.csv')

                        ### TODO: na primeira rodada buscar torneio Brasileiro e BrasileiroB da edicao passada
                        # filtrar apenas lances do campeonato brasileiro
                        df_brasileiro = df_geral.loc[(self.df_finalizacoes['Torneio']=='Brasileiro')&(self.df_finalizacoes['Edicao']==int(self.edicao_cartola))]
                        # clean database from errors
                        print('------------------------------------------------')
                        print ('Cleaning database from string errors, 0s etc...')
                        print('------------------------------------------------')
                        #df_brasileiro = atualiza_informacoes_atleta(df_brasileiro)
                        #renomear algumas colunas relevantes
                        df_brasileiro = df_brasileiro.rename(columns={'apelido_x':'apelido',
                                                                     'sigla_x':'sigla',
                                                                     'altura_x':'altura',
                                                                     'peso_x':'peso',
                                                                     'data_nascimento_x':'data_nascimento',
                                                                     'nome_x':'nome',
                                                                     'slug_x':'slug',
                                                                     'macro_x':'macro_x',
                                                                     'macro_posicao_x':'macro_posicao',
                                                                     'twitter_x':'twitter'})
                        print('DF_BRASILEIRO',df_brasileiro,df_brasileiro.info())
                        self.df_brasileiro = df_brasileiro.copy()

                        print('------------------------------------------------')
                        print('Calculando todas as métricas de xG para a rodada')
                        print('------------------------------------------------')
                        clubes_xg = self.xG_clubes(df_brasileiro)
                        tabela_xp = self.tabelaxP(df_brasileiro)
                        clubes_xg_mando = self.xG_clubes_mando(df_brasileiro)
                        clubes_xg_saldo = self.xG_clubes_saldo(df_brasileiro)
                        clubes_xg_diff = self.xG_clubes_mando_diff(clubes_xg_mando)
                        clubes_xG_probs = self.xG_clubes_probs(clubes_xg_diff, self.edicao_cartola, self.rodada)
                        clubes_xg_defense = self.xG_clubes_defensivo(clubes_xg_diff)
                        clubes_goals_mando = self.G_clubes_mando(df_brasileiro)
                        clubes_gols_diff = self.G_clubes_mando_diff(clubes_goals_mando)
                        clubes_xG_vs_G_composite = self.xG_vs_G_composit(clubes_xg_diff,clubes_gols_diff)
                        clubes_xg_potencial = self.xG_clubes_potencial(clubes_xg_diff,clubes_gols_diff)
                        atletas_xg = self.xG_atletas(df_brasileiro)
                        atletas_xg_detalhado = self.xG_atletas_detalhado(df_brasileiro)
                        atletas_xg_per_match = self.xG_atletas_per_match(df_brasileiro,atletas_xg_detalhado)
                        atletas_xg_eficiencia = self.xG_atletas_eficiencia(atletas_xg_detalhado)
                        atletas_xg_potencial = self.xG_atletas_potencial(atletas_xg_detalhado)
                        print('------------------------------------------')
                        print('Salvando todas as métricas de xG da rodada')
                        print('------------------------------------------')
                        # Salvando arquivos temporarios
                        clubes_xg.to_csv(f'temp/csv/clubes_xg_R{self.rodada}.csv')
                        tabela_xp.to_csv(f'temp/csv/tabela_xg_R{self.rodada}.csv')
                        clubes_xg_mando.to_csv(f'temp/csv/clubes_xg_mando_R{self.rodada}.csv')
                        clubes_xg_saldo.to_csv(f'temp/csv/clubes_xg_saldo_R{self.rodada}.csv')
                        clubes_xg_diff.to_csv(f'temp/csv/clubes_xg_diff_R{self.rodada}.csv')
                        clubes_xG_probs.to_csv(f'temp/csv/clubes_xG_probs_R{self.rodada}.csv')
                        clubes_xg_defense.to_csv(f'temp/csv/clubes_xg_defense_R{self.rodada}.csv')
                        clubes_goals_mando.to_csv(f'temp/csv/clubes_goals_mando_R{self.rodada}.csv')
                        clubes_gols_diff.to_csv(f'temp/csv/clubes_gols_diff_R{self.rodada}.csv')
                        clubes_xG_vs_G_composite.to_csv(f'temp/csv/clubes_xG_vs_G_composite_R{self.rodada}.csv')
                        clubes_xg_potencial.to_csv(f'temp/csv/clubes_xg_potencial_R{self.rodada}.csv')
                        atletas_xg.to_csv(f'temp/csv/atletas_xg_R{self.rodada}.csv')
                        atletas_xg_detalhado.to_csv(f'temp/csv/atletas_xg_detalhado_R{self.rodada}.csv')
                        atletas_xg_per_match.to_csv(f'temp/csv/atletas_xg_per_match_R{self.rodada}.csv')
                        atletas_xg_eficiencia.to_csv(f'temp/csv/atletas_xg_eficiencia_R{self.rodada}.csv')
                        atletas_xg_potencial.to_csv(f'temp/csv/atletas_xg_potencial_R{self.rodada}.csv')

                return clubes_xg, tabela_xp, clubes_xg_mando, clubes_xg_saldo, clubes_xg_diff, clubes_xG_probs, clubes_xg_defense, clubes_goals_mando, \
                        clubes_gols_diff, clubes_xG_vs_G_composite, clubes_xg_potencial, atletas_xg, atletas_xg_detalhado, \
                        atletas_xg_per_match, atletas_xg_eficiencia, atletas_xg_potencial

        # modelo de regressão logística
        def logistic_regression_xG(self, df=None, torneio=None, edicao=None, filtrar=False):
                if filtrar:
                        df = df.loc[(df['Torneio']==torneio)&(df['Edicao']==edicao)]

                # inicializamos o dataset
                dataset = df[['Goal','Distance','Angle Radians','header']].copy()
                # split em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Goal',axis=1), 
                                                                dataset['Goal'], test_size=0.20, 
                                                                random_state=10)
                # treinar modelo
                logistic_model = LogisticRegression(fit_intercept=False,solver='lbfgs')

                # persistir modelo
                with open(f"gmodels/pickle/xG/R{self.rodada}Logistic_Regressoin_solver='lbfgs'.pkl", 'wb') as f:
                        pickle.dump(logistic_model, f)

                # # padronizar os dados
                # scaler = StandardScaler()

                # # Fit apenas no conjunto de treino:
                # scaler.fit(X_train)

                # # Vamos escalar tanto os dados de treino quanto de teste.
                # X_treino_escalado = scaler.transform(X_train)
                # X_teste_escalado = scaler.transform(X_test)

                # Fit do modelo nos dados de treino:         
                #logistic_model.fit(X_treino_escalado, y_train)
                logistic_model.fit(X_train, y_train)


                # calcular probabilidades
                y_pred = logistic_model.predict_proba(X_test)[:,1]
                # para obter probabilidades de todas as linhas do dataset
                y_hat = logistic_model.predict_proba(dataset.drop('Goal',axis=1))[:,1]
                y_hat.shape
                ## VETOR xG - adicionar y_probs como uma coluna ao dataframe original
                df['xG'] = y_hat
                return df


        ########################################### CLUBES #############################################

        def xG_clubes(self,df):
                # xG total
                df_clubes_xg_total = df.groupby(['clube_id'])['xG'].agg([('xG_total','sum')]).reset_index()
                # soma de xG por partida
                df_clubes_xg_match = df.groupby(['clube_id','rodada_id'])['xG'].agg([('xG_sum','sum')]).reset_index()
                # media de xG em todas as partidas
                df_clubes_xg_mean = df_clubes_xg_match.groupby('clube_id')['xG_sum'].agg([('xG_mean',np.mean)]).reset_index()
                # merge
                df_clubes_xg = df_clubes_xg_total.merge(df_clubes_xg_mean, on='clube_id')
                # equipes por strings
                df_clubes_xg['clube'] = df_clubes_xg['clube_id'].map(self.times_dict_r)
                # sort por média xG
                df_clubes_xg = df_clubes_xg.sort_values(by='xG_total',ascending=False).reset_index(drop=True)[['clube', 'xG_mean', 'xG_total']]

                print(df_clubes_xg)
                return df_clubes_xg.dropna()


        #df_clubes_xg = xG_clubes(df_brasileiro)

        def confrontos_xG(self, df_brasileiro):

                times_dict = {'Flamengo':262, 'Botafogo': 263, 'Corinthians': 264, 'Cuiabá':1371, 'Fluminense':266,
                'Palmeiras':275, 'São Paulo': 276, 'Santos': 277, 'Bragantino': 280, 'Atlético-MG': 282,
                'Vasco':267, 'Internacional':285, 'Goiás':290, 'Bahia':265, 'Athletico-PR':293, 'Coritiba':294,
                'Cruzeiro':283, 'Fortaleza':356, 'Grêmio':284,'América-MG':327}

                times_dict_r = {v:k for k, v in times_dict.items()}

                confrontos_xg = df_brasileiro.groupby(['clube_id', 'adversario_id'])['xG'].agg([('xG_sum','sum')]).reset_index()

                df_confrontos_xg = confrontos_xg.merge(
                                        confrontos_xg.set_index(['adversario_id','clube_id']),
                                        left_on=['clube_id','adversario_id'],
                                        right_index=True,
                                        suffixes=('_clube','_adversario'),
                                        how='left'
                                    )
                df_confrontos_xg['clube'] = df_confrontos_xg['clube_id'].map(times_dict_r)
                df_confrontos_xg['adversario'] = df_confrontos_xg['adversario_id'].map(times_dict_r)
                return df_confrontos_xg

        # confrontos_xG(df_brasileiro)

        def xP(self, df, table=False):
                # Poisson with MonteCarlo Simulation
                all_team = []
                all_adversary=[]
                all_draws=[]
                all_team_xP = []
                all_adversary_xP = []
                num_simulations = 100000

                for i,row in df.iterrows():
                        clube = df.iloc[i]['clube']
                        if table:
                                print("* Game #", i+1, "*")
                                print("* Team:", df.iloc[i]['clube'])
                                print("* Adversary:", df.iloc[i]['adversario'])
                                print("* Team xG:", df.iloc[i]['xG_sum_clube'])
                                print("* Adversary xG:", df.iloc[i]['xG_sum_adversario'])
                        input_team = df.iloc[i]['clube_id']
                        input_adversary = df.iloc[i]['adversario_id']
                        input_team_xg = df.iloc[i]['xG_sum_clube']
                        input_adversary_xg = df.iloc[i]['xG_sum_adversario']

                        #print the simulation table and run simulations
                        # print ("********************")
                        # print ("*                  *")
                        # print ("* SIMULATION TABLE *")
                        # print ("*                  *")
                        # print ("********************")
                        count_team_wins = 0
                        count_team_loss = 0
                        count_adversary_wins = 0
                        count_adversary_loss = 0
                        count_draws = 0

                        score_mat = []
                        tot_sim_time = 0
                        sim_table = PrettyTable(["SIMULATION #", "SIMULATION TIME (s)", input_team, input_adversary, "TEAM WIN", "ADVERSARY WIN", "DRAW", "SCORE MARGIN"])
                        for i in range(num_simulations):
                                #get simulation start time
                                start_time = time.time()
                                #run the sim - generate a random Poisson distribution
                                target_team_goals_scored = np.random.poisson(input_team_xg)
                                target_adversary_goals_scored = np.random.poisson(input_adversary_xg)
                                team_win = 0
                                adversary_win = 0
                                draw = 0
                                margin = 0
                                # if more goals for home team => home team wins
                                if target_team_goals_scored > target_adversary_goals_scored:
                                        count_team_wins += 1
                                        count_adversary_loss += 1
                                        team_win = 1
                                        margin = target_team_goals_scored - target_adversary_goals_scored
                                # if more goals for away team => away team wins
                                elif target_team_goals_scored < target_adversary_goals_scored:
                                        count_adversary_wins += 1
                                        count_team_loss += 1
                                        adversary_win = 1
                                        margin = target_adversary_goals_scored - target_team_goals_scored
                                # drwas
                                elif target_team_goals_scored == target_adversary_goals_scored:
                                        draw = 1
                                        count_draws += 1
                                        margin = target_adversary_goals_scored - target_team_goals_scored

                                # add score to score matrix
                                score_mat.append((target_team_goals_scored, target_adversary_goals_scored))
                                #get end time
                                end_time = time.time()
                                #add the time to the total simulation time
                                tot_sim_time += round((end_time - start_time),5)
                                #add the info to the simulation table
                                sim_table.add_row([i+1, round((end_time - start_time),5), target_team_goals_scored, target_adversary_goals_scored, team_win, adversary_win, draw, margin])
                                #print(sim_table)

                        # calculate probabilities to win/lose/draw
                        team_win_probability = round((count_team_wins/num_simulations * 100),2)
                        all_team.append(team_win_probability)

                        adversary_win_probability = round((count_adversary_wins/num_simulations * 100),2)
                        all_adversary.append(adversary_win_probability)

                        draw_probability = round((count_draws/num_simulations * 100),2)
                        all_draws.append(draw_probability)

                        # calculate xP over expected value
                        team_xP = round((team_win_probability / 100) * 3.0 + (draw_probability / 100) * 1.0 + (adversary_win_probability / 100) * 0.0, 2)
                        all_team_xP.append(team_xP)

                        adversary_xP = round((adversary_win_probability / 100) * 3.0 + (draw_probability / 100) * 1.0 + (team_win_probability / 100) * 0.0, 2)
                        all_adversary_xP.append(adversary_xP)

                        # print the simulation statistics
                        # print ("*************")
                        # print ("*           *")
                        # print ("* SIM STATS *")
                        # print ("*           *")
                        # print ("*************")
                        sim_table_stats = PrettyTable(["Total # of sims", "Total time (s) for sims", "TEAM WINS", "ADVERSARY WINS", "DRAWS"])
                        sim_table_stats.add_row([num_simulations, round(tot_sim_time,3), count_team_wins, count_adversary_wins, count_draws])
                        sim_table_stats.add_row(["-", "-", str(team_win_probability)+"%", str(adversary_win_probability)+"%", str(draw_probability)])
                        if table:
                                print(sim_table_stats)


                df['percent_clube'] = all_team
                df['percent_adversario'] = all_adversary
                df['percent_empate'] = all_draws

                df['xP_clube'] = all_team_xP
                df['xP_adversario'] = all_adversary_xP

                return clube, df['xP_clube'].sum()


        def tabelaxP(self, df_brasileiro):
                df_confrontos_xg = self.confrontos_xG(df_brasileiro)
                # copy df
                df = df_confrontos_xg.copy()
                # container
                all_xP = []
                teams = list(df_confrontos_xg.dropna()['clube'].unique())
                # calculate match ups for every team separetly
                for team in teams:
                        df_team = df[df['clube']==team].reset_index(drop=True)
                        clube, xp = self.xP(df_team,table=False)
                        all_xP.append(tuple([clube, xp]))
                # frame from list of perfoamances
                df_tabela = pd.DataFrame(all_xP, columns=['clube', 'xP'])
                #df_tabela.sort_values(by='xP', ascending=False).reset_index(drop=True)
                df_tabela['ranking_xp'] = df_tabela['xP'].rank(ascending=False)
                # rank as integer
                df_tabela = df_tabela.astype({'ranking_xp': int})
                df_tabela = df_tabela.sort_values(by='ranking_xp', ascending=True).reset_index(drop=True)

                print('Tabela do Brasileiro segundo a expectativa de pontos (xP)')
                print('---------------------------------------------------------')
                print(df_tabela)

                return df_tabela

        #df_tabela_df = tabelaxP(df_brasileiro)


        # método para calculo de Gols e xG conquistados e cedidos ao longo do certame
        def goals_expected_and_allowed(self, df, metric=None):

                """
                Calcular as médias de xG cedidos e xGA, por clube
                
                G - Goals
                xG - expected goals
                xGA - expected goals allowed

                Returns
                -------
                df: pd.DataFrame
                Dataframe com médias de pontos cedidos e pontos conquistados
                """
                # agrupar adversarios por mando, target -> pontos e agregar soma de todos os atletas por time
                df_mando = pd.pivot_table(df, values=[metric],
                                             index=['clube_id', 'adversario_id','home_dummy'],
                                             aggfunc=[np.sum]).sort_values(by='clube_id',ascending=True)

                # unstack para dataframe normal sem levels
                df_mando = df_mando.reset_index(drop=False).T.reset_index(drop=True).T.copy()

                # renomear
                df_mando.rename(columns={0:'clube_id',
                                        1:'opponent',
                                        2:'home_dummy',
                                        3:metric},
                                        inplace=True)

                # ler o dataframe acima segundo a seguinte lógica

                '''

                          'clube'            'home_dummy'            'opponent'                 'xG' ou 'G'

                conquistados como mandante         1              cedidos como visitantes            x
                conquistados como visitante        0              cedidos como mandantes             y

                '''
                #########################################################################################################
                conquistados_como_mandante = df_mando[df_mando['home_dummy']==1]

                # convert column pontos_num (object) to numeric
                conquistados_como_mandante = conquistados_como_mandante.astype({metric: float})
                # agrupar por clube e rankear as médias

                conquistados_como_mandante = conquistados_como_mandante.groupby('clube_id')\
                                                [metric].mean().sort_values(ascending=False).reset_index()
                # ranquear
                conquistados_como_mandante['ranking'] = conquistados_como_mandante[metric].rank(ascending=False)
                
                # ranking to int
                conquistados_como_mandante = conquistados_como_mandante.astype({'ranking': int})

                conquistados_como_mandante = conquistados_como_mandante.astype({'clube_id': int})
                
                conquistados_como_mandante.rename(columns={metric:f'{metric}_casa'}, inplace=True)

                #print ('Média de xG Conquistados como Mandante', conquistados_como_mandante)
                # #########################################################################################################
                conquistados_como_visitante = df_mando[df_mando['home_dummy']==0]

                # convert column pontos_num (object) to numeric
                conquistados_como_visitante = conquistados_como_visitante.astype({metric: float})
          
                # agrupar por clube e rankear as médias
                conquistados_como_visitante = conquistados_como_visitante.groupby('clube_id')\
                                                [metric].mean().sort_values(ascending=False).reset_index()
                # ranquear
                conquistados_como_visitante['ranking'] = conquistados_como_visitante[metric].rank(ascending=False)
                # ranking to int
                conquistados_como_visitante = conquistados_como_visitante.astype({'ranking': int})

                conquistados_como_visitante = conquistados_como_visitante.astype({'clube_id': int})
                
                conquistados_como_visitante.rename(columns={metric:f'{metric}_fora'}, inplace=True)


                #print ('Média de xG Conquistados como Visitante', conquistados_como_visitante)
                # #########################################################################################################
                cedidos_como_visitante = df_mando[df_mando['home_dummy']==1]

                # convert column pontos_num (object) to numeric
                cedidos_como_visitante = cedidos_como_visitante.astype({metric: float})
         
                # agrupar por clube e rankear as médias
                cedidos_como_visitante = cedidos_como_visitante.groupby('opponent')\
                                                [metric].mean().sort_values(ascending=False).reset_index()
                # ranquear
                cedidos_como_visitante['ranking'] = cedidos_como_visitante[metric].rank(ascending=False)
                # ranking to int
                cedidos_como_visitante = cedidos_como_visitante.astype({'ranking': int})

                cedidos_como_visitante.rename(columns={metric:f'{metric}A_fora', 'opponent':'clube_id'}, inplace=True)

                cedidos_como_visitante = cedidos_como_visitante.astype({'clube_id': int})

                #print ('Média de xGA como Visitante', cedidos_como_visitante)
                # #########################################################################################################
                cedidos_como_mandante = df_mando[df_mando['home_dummy']==0]
                # convert column pontos_num (object) to numeric
                cedidos_como_mandante = cedidos_como_mandante.astype({metric: float})

                # agrupar por clube e rankear as médias
                cedidos_como_mandante = cedidos_como_mandante.groupby('opponent')\
                                                [metric].mean().sort_values(ascending=False).reset_index()
                # ranquear
                cedidos_como_mandante['ranking'] = cedidos_como_mandante[metric].rank(ascending=False)
                # ranking to int
                cedidos_como_mandante = cedidos_como_mandante.astype({'ranking': int})

                cedidos_como_mandante.rename(columns={metric:f'{metric}A_casa', 'opponent':'clube_id'}, inplace=True)
                
                cedidos_como_mandante = cedidos_como_mandante.astype({'clube_id': int})

                #print ('Média de xGA como Mandante', cedidos_como_mandante)

                return conquistados_como_mandante, cedidos_como_mandante, conquistados_como_visitante, cedidos_como_visitante

        # método para calculo de Gols e xG conquistados e cedidos ao longo do certame
        def goals_expected_and_allowed_total(self, df, metric=None):

                """
                Calcular as médias de xG cedidos e xGA, por clube
                
                G - Goals
                xG - expected goals
                xGA - expected goals allowed

                Returns
                -------
                df: pd.DataFrame
                Dataframe com médias de pontos cedidos e pontos conquistados
                """
                # agrupar adversarios por mando, target -> pontos e agregar soma de todos os atletas por time
                df_mando = pd.pivot_table(df, values=[metric],
                                             index=['clube_id', 'adversario_id','home_dummy'],
                                             aggfunc=[np.sum]).sort_values(by='clube_id',ascending=True)

                # unstack para dataframe normal sem levels
                df_mando = df_mando.reset_index(drop=False).T.reset_index(drop=True).T.copy()

                # renomear
                df_mando.rename(columns={0:'clube_id',
                                        1:'opponent',
                                        2:'home_dummy',
                                        3:metric},
                                        inplace=True)

                # ler o dataframe acima segundo a seguinte lógica

                '''

                          'clube'            'home_dummy'            'opponent'                 'xG' ou 'G'

                conquistados como mandante         1              cedidos como visitantes            x
                conquistados como visitante        0              cedidos como mandantes             y

                '''
                #########################################################################################################
                conquistados_como_mandante = df_mando[df_mando['home_dummy']==1]

                # convert column pontos_num (object) to numeric
                conquistados_como_mandante = conquistados_como_mandante.astype({metric: float})
                # agrupar por clube e rankear as médias

                conquistados_como_mandante = conquistados_como_mandante.groupby('clube_id')\
                                                [metric].sum().sort_values(ascending=False).reset_index()
                # ranquear
                conquistados_como_mandante['ranking'] = conquistados_como_mandante[metric].rank(ascending=False)
                
                # ranking to int
                conquistados_como_mandante = conquistados_como_mandante.astype({'ranking': int})

                conquistados_como_mandante = conquistados_como_mandante.astype({'clube_id': int})
                
                conquistados_como_mandante.rename(columns={metric:f'{metric}_casa'}, inplace=True)

                #print ('Média de xG Conquistados como Mandante', conquistados_como_mandante)
                # #########################################################################################################
                conquistados_como_visitante = df_mando[df_mando['home_dummy']==0]

                # convert column pontos_num (object) to numeric
                conquistados_como_visitante = conquistados_como_visitante.astype({metric: float})
          
                # agrupar por clube e rankear as médias
                conquistados_como_visitante = conquistados_como_visitante.groupby('clube_id')\
                                                [metric].sum().sort_values(ascending=False).reset_index()
                # ranquear
                conquistados_como_visitante['ranking'] = conquistados_como_visitante[metric].rank(ascending=False)
                # ranking to int
                conquistados_como_visitante = conquistados_como_visitante.astype({'ranking': int})

                conquistados_como_visitante = conquistados_como_visitante.astype({'clube_id': int})
                
                conquistados_como_visitante.rename(columns={metric:f'{metric}_fora'}, inplace=True)


                #print ('Média de xG Conquistados como Visitante', conquistados_como_visitante)
                # #########################################################################################################
                cedidos_como_visitante = df_mando[df_mando['home_dummy']==1]

                # convert column pontos_num (object) to numeric
                cedidos_como_visitante = cedidos_como_visitante.astype({metric: float})
         
                # agrupar por clube e rankear as médias
                cedidos_como_visitante = cedidos_como_visitante.groupby('opponent')\
                                                [metric].sum().sort_values(ascending=False).reset_index()
                # ranquear
                cedidos_como_visitante['ranking'] = cedidos_como_visitante[metric].rank(ascending=False)
                # ranking to int
                cedidos_como_visitante = cedidos_como_visitante.astype({'ranking': int})

                cedidos_como_visitante.rename(columns={metric:f'{metric}A_fora', 'opponent':'clube_id'}, inplace=True)

                cedidos_como_visitante = cedidos_como_visitante.astype({'clube_id': int})

                #print ('Média de xGA como Visitante', cedidos_como_visitante)
                # #########################################################################################################
                cedidos_como_mandante = df_mando[df_mando['home_dummy']==0]
                # convert column pontos_num (object) to numeric
                cedidos_como_mandante = cedidos_como_mandante.astype({metric: float})

                # agrupar por clube e rankear as médias
                cedidos_como_mandante = cedidos_como_mandante.groupby('opponent')\
                                                [metric].sum().sort_values(ascending=False).reset_index()
                # ranquear
                cedidos_como_mandante['ranking'] = cedidos_como_mandante[metric].rank(ascending=False)
                # ranking to int
                cedidos_como_mandante = cedidos_como_mandante.astype({'ranking': int})

                cedidos_como_mandante.rename(columns={metric:f'{metric}A_casa', 'opponent':'clube_id'}, inplace=True)
                
                cedidos_como_mandante = cedidos_como_mandante.astype({'clube_id': int})

                #print ('Média de xGA como Mandante', cedidos_como_mandante)

                return conquistados_como_mandante, cedidos_como_mandante, conquistados_como_visitante, cedidos_como_visitante


        def xG_clubes_mando(self, df):
                # obter dataframes desejados para xG e xGA, por mando
                xG_mandante, xGA_mandante, xG_visitante, xGA_visitante = self.goals_expected_and_allowed(df, metric='xG')
                # aglutinar dataframes
                frames = [xG_mandante, xG_visitante, xGA_mandante, xGA_visitante]
                # concatená-lo num único dataframe
                df_xg_mando = pd.concat(frames)
                # agrupar por equipe
                df_xg_mando = df_xg_mando.groupby('clube_id')[['xG_casa','xGA_casa','xG_fora', 'xGA_fora']].first().reset_index()
                # trazer nomes equipes
                df_xg_mando['clube'] = df_xg_mando['clube_id'].map(self.times_dict_r)
                # ordem des ejada das coluas
                df_xg_mando = df_xg_mando[['clube', 'xG_casa', 'xGA_casa', 'xG_fora', 'xGA_fora']]
                print(df_xg_mando)
                return df_xg_mando.dropna()

        #df_xg_mando = xG_clubes_mando(df_brasileiro)


        def xG_clubes_saldo(self, df):
                # obter dataframes desejados para xG e xGA, por mando
                xG_mandante_total, xGA_mandante_total, xG_visitante_total, xGA_visitante_total = self.goals_expected_and_allowed_total(df, metric='xG')
                # aglutinar dataframes
                frames_total = [xG_mandante_total, xGA_mandante_total, xG_visitante_total, xGA_visitante_total]
                # concatená-lo num único dataframe
                df_xg_mando_total = pd.concat(frames_total)
                # agrupar por equipe
                df_xg_mando_total = df_xg_mando_total.groupby('clube_id')[['xG_casa','xGA_casa','xG_fora', 'xGA_fora']].first().reset_index()
                # trazer nomes equipes
                df_xg_mando_total['clube'] = df_xg_mando_total['clube_id'].map(self.times_dict_r)
                # ordem des ejada das coluas
                df_xg_mando_total = df_xg_mando_total[['clube', 'xG_casa', 'xGA_casa', 'xG_fora', 'xGA_fora']]
                # obter totais
                df_xg_mando_total['xG_total'] = df_xg_mando_total['xG_casa'] + df_xg_mando_total['xG_fora']
                df_xg_mando_total['xGA_total'] = df_xg_mando_total['xGA_casa'] + df_xg_mando_total['xGA_fora']
                df_xg_mando_total['xG_saldo'] = df_xg_mando_total['xG_total'] - df_xg_mando_total['xGA_total']
                # saldo
                df_xg_saldo = df_xg_mando_total.sort_values(by='xG_saldo',ascending=False).reset_index(drop=True)[['clube', 'xG_total', 'xGA_total','xG_saldo']].dropna()
                print(df_xg_saldo)
                return df_xg_saldo.dropna()


        def xG_clubes_mando_diff(self, df_xg_mando):
                df_xg_diff= df_xg_mando.copy()

                df_xg_diff['xG_geral'] = (df_xg_diff['xG_casa'] + df_xg_diff['xG_fora'])/2
                df_xg_diff['xGA_geral'] = (df_xg_diff['xGA_casa'] + df_xg_diff['xGA_fora'])/2
                df_xg_diff['xG_diff_casa'] = df_xg_diff['xG_casa'] - df_xg_diff['xGA_casa']
                df_xg_diff['xG_diff_fora'] = df_xg_diff['xG_fora'] - df_xg_diff['xGA_fora']
                df_xg_diff['xG_diff_geral'] = df_xg_diff['xG_geral'] - df_xg_diff['xGA_geral']

                df_xg_diff = df_xg_diff.sort_values(by='xG_diff_geral',ascending=False).reset_index(drop=True)
                print(df_xg_diff)
                return df_xg_diff.dropna()


        #df_xg_diff = xG_clubes_mando_diff(df_xg_mando)


        def xG_clubes_probs(self, df_xg_diff, edicao, rodada):
                times_dict = {'Flamengo':262, 'Botafogo': 263, 'Corinthians': 264, 'Cuiabá':1371, 'Fluminense':266,
                                'Palmeiras':275, 'São Paulo': 276, 'Santos': 277, 'Bragantino': 280, 'Atlético-MG': 282,
                                'Vasco':267, 'Internacional':285, 'Goiás':290, 'Bahia':265, 'Athletico-PR':293, 'Coritiba':294,
                                'Cruzeiro':283, 'Fortaleza':356, 'Grêmio':284,'América-MG':327}

                times_dict_r = {v:k for k, v in times_dict.items()}

                jogos, _, _ = get_jogos_sde(edicao,rodada,times_dict)
                print('Jogos da próxima rodada',jogos)

                #df_xg_diff['clube'] = df_xg_diff['clube'].map(times_dict)
                # container for all simulations
                df_xG_probs = pd.DataFrame()
                for m,v in jogos.items():
                        # m = times_dict.get(m)
                        # v = times_dict.get(v)
                        print (m,v)
                        xG_casa = df_xg_diff[df_xg_diff['clube']==m]['xG_casa'].iloc[0]
                        xGA_casa = df_xg_diff[df_xg_diff['clube']==m]['xGA_casa'].iloc[0]
                        xG_diff_casa = df_xg_diff[df_xg_diff['clube']==m]['xG_diff_casa'].iloc[0]
                        xG_fora = df_xg_diff[df_xg_diff['clube']==v]['xG_fora'].iloc[0]
                        xGA_fora = df_xg_diff[df_xg_diff['clube']==v]['xGA_fora'].iloc[0]
                        xG_diff_fora = df_xg_diff[df_xg_diff['clube']==v]['xG_diff_fora'].iloc[0]

                        df_xG_probs = df_xG_probs.append({'rodada_id': rodada+1,
                                                            'mandante':m, 
                                                            'visitante':v, 
                                                            'xG_mandante':(xG_casa+xGA_fora)/2,
                                                            'xG_visitante':(xG_fora+xG_casa)/2},
                                                            ignore_index=True)
                # Poisson with MonteCarlo Simulation
                all_homes = []
                all_aways=[]
                all_draws=[]
                all_homes_sg=[]
                all_aways_sg=[]
                all_draws_sg=[]
                num_simulations = 20000

                for i,row in df_xG_probs.iterrows():
                        print("* Game #", i+1, "*")
                        print("* Home team:", df_xG_probs.iloc[i]['mandante'])
                        print("* Away team:", df_xG_probs.iloc[i]['visitante'])
                        print("* Home team xG:", df_xG_probs.iloc[i]['xG_mandante'])
                        print("* Away team xG:", df_xG_probs.iloc[i]['xG_visitante'])
                        input_home_team = df_xG_probs.iloc[i]['mandante']
                        input_home_team_xg = df_xG_probs.iloc[i]['xG_mandante']
                        input_away_team = df_xG_probs.iloc[i]['visitante']
                        input_away_team_xg = df_xG_probs.iloc[i]['xG_visitante']
                        #print the simulation table and run simulations
                        # print ("********************")
                        # print ("*                  *")
                        # print ("* SIMULATION TABLE *")
                        # print ("*                  *")
                        # print ("********************")
                        count_home_wins = 0
                        count_home_loss = 0
                        count_away_wins = 0
                        count_away_loss = 0
                        count_draws = 0
                        count_home_sg = 0
                        count_away_sg = 0
                        count_draw_sg = 0

                        score_mat = []
                        tot_sim_time = 0
                        sim_table = PrettyTable(["SIMULATION #", "SIMULATION TIME (s)", input_home_team, input_away_team, "HOME WIN", "AWAY WIN", "DRAW", "SCORE MARGIN", "SG"])
                        for i in range(num_simulations):
                            #get simulation start time
                            start_time = time.time()
                            #run the sim - generate a random Poisson distribution
                            target_home_goals_scored = np.random.poisson(input_home_team_xg)
                            target_away_goals_scored = np.random.poisson(input_away_team_xg)
                            home_win = 0
                            away_win = 0
                            draw = 0
                            margin = 0
                            sg = 0
                            # if more goals for home team => home team wins
                            if target_home_goals_scored > target_away_goals_scored:
                                count_home_wins += 1
                                count_away_loss += 1
                                home_win = 1
                                margin = target_home_goals_scored - target_away_goals_scored
                                # check if sg
                                if target_away_goals_scored == 0:
                                    sg = 1
                                    count_home_sg += 1
                            # if more goals for away team => away team wins
                            elif target_home_goals_scored < target_away_goals_scored:
                                count_away_wins += 1
                                count_home_loss += 1
                                away_win = 1
                                margin = target_away_goals_scored - target_home_goals_scored
                                # check if sg
                                if target_home_goals_scored == 0:
                                    sg = 1
                                    count_away_sg += 1
                            elif target_home_goals_scored == target_away_goals_scored:
                                draw = 1
                                count_draws += 1
                                margin = target_away_goals_scored - target_home_goals_scored
                                # check if sg
                                if target_home_goals_scored == 0 and target_away_goals_scored == 0:
                                    sg = 1
                                    count_draw_sg += 1
                            # add score to score matrix
                            score_mat.append((target_home_goals_scored, target_away_goals_scored))
                            #get end time
                            end_time = time.time()
                            #add the time to the total simulation time
                            tot_sim_time += round((end_time - start_time),5)
                            #add the info to the simulation table
                            sim_table.add_row([i+1, round((end_time - start_time),5), target_home_goals_scored, target_away_goals_scored, home_win, away_win, draw, margin, sg])
                        #print(sim_table)

                        # calculate probabilities to win/lose/draw
                        home_win_probability = round((count_home_wins/num_simulations * 100),2)
                        all_homes.append(home_win_probability)
                        
                        away_win_probability = round((count_away_wins/num_simulations * 100),2)
                        all_aways.append(away_win_probability)
                        
                        draw_probability = round((count_draws/num_simulations * 100),2)
                        all_draws.append(draw_probability)
                        
                        home_sg_probability = round((count_home_sg/num_simulations * 100),2)
                        all_homes_sg.append(home_sg_probability)
                        
                        away_sg_probability = round((count_away_sg/num_simulations * 100),2)
                        all_aways_sg.append(away_sg_probability)
                        
                        draw_sg_probability = round((count_draw_sg/num_simulations * 100),2)
                        all_draws_sg.append(draw_sg_probability)


                        # print the simulation statistics
                        # print ("*************")
                        # print ("*           *")
                        # print ("* SIM STATS *")
                        # print ("*           *")
                        # print ("*************")
                        sim_table_stats = PrettyTable(["Total # of sims", "Total time (s) for sims", "HOME WINS", "AWAY WINS", "DRAWS", "HOME SG", "AWAY SG", "DRAW SG"])
                        sim_table_stats.add_row([num_simulations, round(tot_sim_time,3), count_home_wins, count_away_wins, count_draws, count_home_sg, count_away_sg, count_draw_sg])
                        sim_table_stats.add_row(["-", "-", str(home_win_probability)+"%", str(away_win_probability)+"%", str(draw_probability)+"%", str(home_sg_probability)+"%", str(away_sg_probability)+"%", str(draw_sg_probability)+"%"])
                        print(sim_table_stats)
                        

                df_xG_probs['percent_mandante'] = all_homes
                df_xG_probs['percent_visitante'] = all_aways
                df_xG_probs['percent_empate'] = all_draws

                df_xG_probs['percent_sg_mandante'] = all_homes_sg
                df_xG_probs['percent_sg_visitante'] = all_aways_sg
                df_xG_probs['percent_sg_empate'] = all_draws_sg

                df_xG_probs['sg_mandante_total'] = df_xG_probs['percent_sg_mandante'] + df_xG_probs['percent_sg_empate']
                df_xG_probs['sg_visitante_total'] = df_xG_probs['percent_sg_visitante'] + df_xG_probs['percent_sg_empate']

                df_xG_probs['sg_total'] = df_xG_probs['sg_mandante_total'] + df_xG_probs['sg_visitante_total']
                
                df_xG_probs = df_xG_probs.assign(val=df_xG_probs[['sg_mandante_total', 'sg_visitante_total']].max(1)).sort_values('val', ascending=False)\
                                                                      .drop('val', 1)\
                                                                      .reset_index(drop=True).copy()
                
                return df_xG_probs.dropna()

         #df_xG_clubes_probs =xG_clubes_probs(df_xg_diff,2022,7)


        def xG_clubes_defensivo(self, df_xg_diff):
                #O xGA de um time indica a habilidade de um time para evitar com que o adversário finalize com alta probabilidade de gol.
                df_defense = df_xg_diff.sort_values(by='xGA_geral',ascending=True).reset_index(drop=True)
                df_defense = df_defense[['clube', 'xGA_geral']]
                print(df_defense)
                
                return df_defense.dropna()


        #df_defense = xG_defensivo(df_xg_diff)


        def G_clubes_mando(self, df):
                # obter dataframes desejados para G e GA, por mando
                G_mandante, GA_mandante, G_visitante, GA_visitante = self.goals_expected_and_allowed(df, metric='Goal')
                # aglutinar dataframes
                frames = [G_mandante, G_visitante, GA_mandante, GA_visitante]
                # concatená-lo num único dataframe
                df_goals_mando = pd.concat(frames)
                # agrupar por equipe
                df_goals_mando = df_goals_mando.groupby('clube_id')[['Goal_casa','GoalA_casa','Goal_fora', 'GoalA_fora']].first().reset_index()
                # trazer nomes equipes
                df_goals_mando['clube'] = df_goals_mando['clube_id'].map(self.times_dict_r)
                # ordem des ejada das coluas
                df_goals_mando = df_goals_mando[['clube', 'Goal_casa', 'GoalA_casa', 'Goal_fora', 'GoalA_fora']]
                print(df_goals_mando)
                return df_goals_mando.dropna()


        #df_goals_mando = G_clubes_mando(df_brasileiro)


        def G_clubes_mando_diff(self, df_goals_mando):
                #A diferença entre G e GA de um time (G - GA) indica como um time está de facto performando
                df_gols_diff= df_goals_mando.copy()

                df_gols_diff['Goal_geral'] = (df_gols_diff['Goal_casa'] + df_gols_diff['Goal_fora'])/2
                df_gols_diff['GoalA_geral'] = (df_gols_diff['GoalA_casa'] + df_gols_diff['GoalA_fora'])/2
                df_gols_diff['Goal_diff_casa'] = df_gols_diff['Goal_casa'] - df_gols_diff['GoalA_casa']
                df_gols_diff['Goal_diff_fora'] = df_gols_diff['Goal_fora'] - df_gols_diff['GoalA_fora']
                df_gols_diff['Goal_diff_geral'] = df_gols_diff['Goal_geral'] - df_gols_diff['GoalA_geral']

                df_gols_diff = df_gols_diff.sort_values(by='Goal_diff_geral',ascending=False).reset_index(drop=True)

                print(df_gols_diff)
                return df_gols_diff.dropna()


        #df_gols_diff = G_clubes_mando_diff(df_goals_mando)


        def xG_clubes_potencial(self, df_xg_diff,df_gols_diff):
                # Uma G_diff *negativa* e xG_diff *positivo* pode indicar que o time experimenta azar ou performance de finalização abaixo da média
                # juntar diferenças entre xG e Gols
                df_diffs = df_xg_diff.merge(df_gols_diff, on='clube')
                df_diffs = df_diffs[['clube', 'Goal_diff_geral', 'xG_diff_geral']]
                # criar mascara para filtragem
                mask = (df_diffs['Goal_diff_geral']<0) & (df_diffs['xG_diff_geral']>0)
                df_diffs = df_diffs.loc[mask]
                print(df_diffs)
                return df_diffs.dropna()


        def xG_vs_G_composit(self, df_xg_diff, df_gols_diff):
                df_xG_vs_Goals = df_xg_diff.merge(df_gols_diff, on='clube')
                
                df_composite = df_xG_vs_Goals.copy()

                df_composite['xG_G_comp'] = df_composite['xG_geral']-df_composite['Goal_geral']
                df_composite = df_composite.sort_values(by='xG_G_comp',ascending=False).reset_index(drop=True)

                print(df_composite)

                # # get values for plotting
                # xg = list(df_composite['xG_geral'].values)
                # g = list(df_composite['Goal_geral'].values)
                # index = list(df_composite['clube'].values)
                # # dataframe for plotting
                # df = pd.DataFrame({'xG': xg,'G': g}, index=index)
                # # plot
                # ax = df.plot.bar(rot=90, color={"xG": "o", "G": "g"},figsize=(25,15))
                # # save plot ## TODO: put folder in .gitignore
                # ax.figure.savefig(f'gmodels/img/xg_g_composite_R{rodada}.png')

                return df_composite
                 

        #df_diffs = xG_clubes_potencial(df_gols_diff)

        ########################################### ATLETAS #############################################


        def xG_atletas(self, df):
                df_atletas = df.groupby(['atleta_id', 'apelido','clube_id'])['xG'].sum().reset_index()
                df_atletas = df_atletas.sort_values(by='xG',ascending=False).reset_index(drop=True)
                df_atletas = df_atletas[df_atletas['clube_id'].isin(self.clubes_ids)].reset_index(drop=True)
                df_atletas['clube'] = df_atletas['clube_id'].map(self.times_dict_r)
                print(df_atletas.head(20))
                return df_atletas.dropna()

        #df_atletas = xG_atletas(df_brasileiro)


        def xG_atletas_detalhado(self, df):
                # adicionar metricas
                df_atletas2 = df.groupby(['atleta_id', 'apelido', 'clube_id']).agg(xG_total=('xG', 'sum'),
                                                                             shots_total=('Lance', 'count'),
                                                                             jogos_num=('rodada_id', 'nunique'),
                                                                             goals_total=('Goal', 'sum'),                                                                              
                                                                             ).reset_index()
                df_atletas2['xG_per_match']=df_atletas2['xG_total']/df_atletas2['jogos_num']
                df_atletas2['G_per_match']=df_atletas2['goals_total']/df_atletas2['jogos_num']
                df_atletas2['xG_per_shot'] = df_atletas2['xG_total']/df_atletas2['shots_total']
                df_atletas2['shots_per_goal'] = df_atletas2['shots_total']/df_atletas2['goals_total']
                df_atletas2['clube'] = df_atletas2['clube_id'].map(self.times_dict_r)

                xg_total = df_atletas2.sort_values(by='xG_total',ascending=False).reset_index(drop=True)
                print(xg_total.head(10))

                return xg_total.dropna()

        #df_atletas2 = df_atletas2[df_atletas2['clube_id'].isin(self.clubes_ids)].sort_values(by='xG_total',ascending=False).reset_index(drop=True)

        #xg_total = xG_atletas_detalhado(df_brasileiro)


        def xG_atletas_per_match(self, df, xg_total):
                # xG per match
                rodada = self.df_brasileiro['rodada_id'].max()
                PORCENTAGEM_MINIMA_JOGOS = 2.5  # 40%
                min_jogos = rodada // PORCENTAGEM_MINIMA_JOGOS
                print(min_jogos)

                mask = (xg_total['jogos_num']>=int(min_jogos))
                xg_per_match = xg_total.loc[mask]
                xg_per_match = xg_per_match.sort_values(by='xG_per_match',ascending=False).reset_index(drop=True)
                print(xg_per_match.head(20))
                return xg_per_match.dropna()


        #xg_per_match = xG_atletas_per_match(df_brasileiro)


        def xG_atletas_eficiencia(self, xg_total):
                # eficiencia/potencial
                xg_total['eficiência'] = xg_total['goals_total']-xg_total['xG_total']
                xg_total = xg_total.loc[xg_total['goals_total']>=1].sort_values(by='eficiência',ascending=False).reset_index(drop=True).head(50)
                print(xg_total)
                return xg_total.dropna()


        #df_atletas_eficiencia = xG_atletas_eficiencia(xg_total)


        def xG_atletas_potencial(self, xg_total):
                # note: ascending = True
                df_potencial = xg_total.loc[xg_total['goals_total']>=1].head(20).sort_values(by='eficiência',ascending=True).reset_index(drop=True)
                print(df_potencial)
                return df_potencial.dropna()


        #df_atletas_potencial = xG_atletas_potencial(xg_total)



if __name__ == "__main__":

        xg = xG(2023)
        #xg.gerar_metricas_xG()
        clubes_xg, clubes_xg_mando, clubes_xg_diff, clubes_xG_probs, clubes_xg_defense, \
        clubes_goals_mando, clubes_gols_diff, clubes_xg_potencial, \
        atletas_xg, atletas_xg_detalhado, atletas_xg_per_match, \
        atletas_xg_eficiencia, atletas_xg_potencial = xg.gerar_metricas_xG()













