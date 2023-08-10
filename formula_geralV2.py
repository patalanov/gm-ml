# %%

import pandas as pd
import requests
import json
pd.set_option("display.max_columns", None)

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

dict_pos = {'Lateral':'LAT',
            'Atacante':'ATA',
            'Meia':'MEI',
            'Goleiro':'GOL',
            'Zagueiro':'ZAG',
            'Técnico':'TEC'}


database = 'database/2023/Cartola_2023'
dataCartola = pd.read_csv(database, compression='gzip')
dataCartola['clube'] = dataCartola['clube'].replace('Athletico-PR', 'Athlético-PR')

rodada_max = dataCartola.rodada_id.max()
rodada_max

# filtros
pontuou = (dataCartola['pontos_num'] != 0) | (dataCartola['variacao_num'] != 0)
jogou = (dataCartola['jogos_num'] > 0)

# colunas
columns = ['atleta_id', 'apelido','posicao', 'clube', 'jogos_num', 
           'preco_num', 'variacao_num', 'pontos_num', 'media_num', 'rodada_id']

atletasJogaram = dataCartola[jogou]
atletasNuncaJogaram = dataCartola[~jogou][~dataCartola['atleta_id'].isin(atletasJogaram['atleta_id'].unique())]

print('Atletas que jogaram: ', atletasJogaram.shape[0])
print('Atletas que nunca jogaram: ', atletasNuncaJogaram.shape[0])

dfAtletas = pd.DataFrame()
for atleta in atletasJogaram.atleta_id.unique():
    dfAtletas = dfAtletas.append(atletasJogaram[(atletasJogaram['atleta_id']==atleta) & pontuou].\
            sort_values('jogos_num', ascending=False).head(1)[columns], ignore_index = True)
    
for atleta in atletasNuncaJogaram.atleta_id.unique():
    dfAtletas = dfAtletas.append(atletasNuncaJogaram[(atletasNuncaJogaram['atleta_id']==atleta)].\
            sort_values('jogos_num', ascending=False).head(1)[columns], ignore_index = True)
    
    
dfAtletas.rename(columns={
    'preco_num': 'ultimo_preco',
    'variacao_num': 'ultima_variacao',
    'pontos_num': 'ultima_pontuacao',
    'media_num': 'ultima_media_pontos',
    'rodada_id': 'ultima_rodada_id'
}, inplace=True)
dfAtletas['pos_sigla'] = dfAtletas['posicao'].map(dict_pos)

# %%

def medias_atleta(atleta_id, column):
    atleta = dataCartola[(dataCartola['atleta_id']==atleta_id)]
    ultima_rodada = atleta['rodada_id'].max()
    atleta = atleta[atleta['rodada_id']<ultima_rodada]
    medias = atleta[['preco_open', 'pontos_num', 'variacao_num']].mean()
    return round(medias[column], 2)

def medias_atleta_completa(atleta_id, column, rodada_atual):
    atleta = dataCartola[(dataCartola['atleta_id']==atleta_id)]
    ultima_rodada = atleta['rodada_id'].max()
    atleta = atleta[atleta['rodada_id']<ultima_rodada]
    soma = atleta[['preco_open', 'pontos_num', 'variacao_num']].sum()
    return round((soma[column]/(rodada_atual-1)), 2)


def medias_rodada(rodada_id, column):
    rodada = dataCartola[(dataCartola['rodada_id']==rodada_id)]
    rodada = rodada[(rodada['pontos_num'] != 0) & (rodada['variacao_num'] != 0)]
    medias = rodada[['preco_open', 'pontos_num', 'variacao_num']].mean()
    return round(medias[column], 2)

dfAtletas['preco_medio_at'] = dfAtletas['atleta_id'].apply(lambda x: medias_atleta(x, 'preco_open'))
dfAtletas['pontuacao_media_at'] = dfAtletas['atleta_id'].apply(lambda x: medias_atleta(x, 'pontos_num'))
dfAtletas['variacao_media_at'] = dfAtletas['atleta_id'].apply(lambda x: medias_atleta(x, 'variacao_num'))

dfAtletas['preco_medio_completo_at'] = dfAtletas['atleta_id'].apply(lambda x: medias_atleta_completa(x, 'preco_open', 13))
dfAtletas['pontuacao_media_completa_at'] = dfAtletas['atleta_id'].apply(lambda x: medias_atleta_completa(x, 'pontos_num', 13))
dfAtletas['variacao_media_completa_at'] = dfAtletas['atleta_id'].apply(lambda x: medias_atleta_completa(x, 'variacao_num', 13))

dfAtletas['preco_medio_rodada'] = dfAtletas['ultima_rodada_id'].apply(lambda x: medias_rodada(x, 'preco_open'))
dfAtletas['pontuacao_media_rodada'] = dfAtletas['ultima_rodada_id'].apply(lambda x: medias_rodada(x, 'pontos_num'))
dfAtletas['variacao_media_rodada'] = dfAtletas['ultima_rodada_id'].apply(lambda x: medias_rodada(x, 'variacao_num'))

dfAtletas['diff_preco_medio'] = dfAtletas['ultimo_preco'] - dfAtletas['preco_medio_at']
dfAtletas['diff_pontuacao_media'] = dfAtletas['ultima_pontuacao'] - dfAtletas['pontuacao_media_at']
dfAtletas['diff_variacao_media'] = dfAtletas['ultima_variacao'] - dfAtletas['variacao_media_at']

dfAtletas['diff_preco_medio_completo'] = dfAtletas['ultimo_preco'] - dfAtletas['preco_medio_completo_at']
dfAtletas['diff_pontuacao_media_completa'] = dfAtletas['ultima_pontuacao'] - dfAtletas['pontuacao_media_completa_at']
dfAtletas['diff_variacao_media_completa'] = dfAtletas['ultima_variacao'] - dfAtletas['variacao_media_completa_at']

dfAtletas['diff_preco_medio_geral'] = dfAtletas['ultimo_preco'] - dfAtletas['preco_medio_rodada']
dfAtletas['diff_pontuacao_media_geral'] = dfAtletas['ultima_pontuacao'] - dfAtletas['pontuacao_media_rodada']
dfAtletas['diff_variacao_media_geral'] = dfAtletas['ultima_variacao'] - dfAtletas['variacao_media_rodada']
dfAtletas.fillna(0, inplace=True)

dfAtletas_Pontuaram = dfAtletas[~((dfAtletas['ultima_pontuacao'] == 0) & (dfAtletas['jogos_num'] > 0))]
dfAtletas_nao_Pontuaram = dfAtletas[((dfAtletas['ultima_pontuacao'] == 0) & (dfAtletas['jogos_num'] > 0))]
dfAtletas_nunca_jogaram = dfAtletas[((dfAtletas['jogos_num'] == 0))]

rodada_max
rodada = rodada_max
rodada

# %%
def linearRegression(X, y):
    print ('Dataset shape: ',X.shape)
    
    ####### Treinando o regressor:
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    print(reg_model)
    print(X.columns)
    print('coef_',reg_model.coef_)
    print('intercept_',reg_model.intercept_)

    # Extraindo a métrica R2 para treino:
    y_pred_test_model = reg_model.predict(X)
    r2_train = r2_score(y, y_pred_test_model)
    print(f'R2 treino: {r2_train : .6f}')
    
    print('R2:', r2_score(y, y_pred_test_model))
    print('MAE: ', mean_absolute_error(y, y_pred_test_model))
    print('MSE:', mean_squared_error(y, y_pred_test_model))
    print('-------------------------\n\n')

    plt.figure(figsize=(10,10))
    plt.scatter(y, y_pred_test_model, c='crimson')
    p1 = max(max(y_pred_test_model), max(y))
    p2 = min(min(y_pred_test_model), min(y))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    
    plt.figure(figsize=(10,10))
    sns.residplot(y, y_pred_test_model)
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show() 
    
    yhatFormula = y_pred_test_model
    
    return yhatFormula

# %%
def min_valorizar_final(X, rodada):
      val = 0
      if X['ultima_rodada_id'] == rodada:
            val = (  0.59026935   *  X['ultima_pontuacao'])    +\
                  (  -0.01730351  *  X['diff_pontuacao_media'])    +\
                  0.09689845205501157
      else:  
            if X['jogos_num'] > 1: 
                  
                  if X['ultima_pontuacao'] == 0:
                        val = (  1.4421092   *  X['diff_preco_medio_completo'])    +\
                              ( -1.40595357  *  X['ultima_variacao'])    +\
                              (  1.95448003  *  X['diff_variacao_media'])    +\
                              (  0.34281961  *  X['diff_pontuacao_media_geral'])    +\
                              1.3132283131074551
                  else:
                        val = (  -0.56443463  *  X['diff_preco_medio_completo'])    +\
                              (  -0.0856294   *  X['ultima_media_pontos'])    +\
                              (  0.53938066   *  X['ultima_pontuacao'])    +\
                              (  -0.2955632   *  X['ultima_variacao'])    +\
                              (  0.47467247   *  X['ultimo_preco'])    +\
                              -0.3027862786796067
            else:
                  val = ( -0.15417763  *  X['ultima_pontuacao'])    +\
                        ( 1.42  *  X['ultimo_preco'])    +\
                        -0.3027862
      return round(val,2)

# %%
dfAtletas['min_valorizar'] = np.nan
for index, x in dfAtletas.iterrows():
    val = min_valorizar_final(x, rodada)
    dfAtletas['min_valorizar'].iloc[index] = val

# %%
dfAtletasaval = dfAtletas[['atleta_id', 'apelido', 'clube', 'jogos_num', 'ultima_pontuacao',  'ultima_media_pontos', 'min_valorizar']]

# %%
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==90031] = 11.8
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==68952] = 2.0
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==109591] = 8.0
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==118774] = 8.6
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==117870] = 1.4
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==106829] = 4.6
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==89273] = 1.2
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==72595] = 4.9
dfAtletas['min_valorizar'][dfAtletas['atleta_id']==99412] = 4.7

# %%
ids_problema = [90031, 68952, 109591, 118774, 117870, 106829, 89273, 72595, 99412]
df_teste = dfAtletas[dfAtletas['atleta_id'].isin(ids_problema)]

# %%

def carrega_externo(rodada, make_request=False):
    if make_request:
        HOST = 'https://cartolaanalitico.com/api/scouts'
        USER_AGENT = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36'
        PAYLOAD = {
        "pos": "",
        "weight_cartoletas": 0
        }
        headers = {'User-Agent': USER_AGENT}
        response = requests.post(url=HOST, data=PAYLOAD, headers=headers)
        response.json()
        with open(f"saved_2023_r{rodada}.json", 'w') as f:
            json.dump(response.json(), f)
    
    
    
    df = pd.read_json(f"saved_2023_r{rodada}.json")
    df = pd.DataFrame.from_dict(df['players'].to_dict(), orient='index')
    df = df.replace('-','0')
    df[['ca', 'dd', 'ds', 'fc', 'fn', 'fs', 'ga', 'ip', 'minval', 'pi',
        'pontos', 'preco', 'sg', 'valor']] = df[['ca', 'dd', 'ds', 'fc', 'fn', 'fs', 'ga', 'ip', 'minval', 'pi',
        'pontos', 'preco', 'sg', 'valor']].astype(float)
    return df

# %%
df_analitico = carrega_externo(rodada+1, make_request=True)
df_analitico.head()

# %%
dfCompleto = df_analitico.merge(dfAtletas, how='left', left_on=['nome', 'pos', 'time'], right_on=['apelido', 'pos_sigla', 'clube'])
dfCompleto.reset_index(inplace=True)
dfCompleto['diff'] = dfCompleto['min_valorizar'] - dfCompleto['minval'] 
df_aval = dfCompleto.sort_values('diff')


# %% [markdown]
# # Aqui só se quiser alterar

# %%
###CASO 1: 

dfSelecionado = dfCompleto
dfSelecionado = dfSelecionado[dfSelecionado['ultima_rodada_id'] == rodada_max]
variables     = ['ultima_pontuacao',
                 'diff_pontuacao_media']

# %%
###CASO 2: 

dfSelecionado = dfCompleto
dfSelecionado = dfSelecionado[dfSelecionado['ultima_rodada_id'] < rodada_max]
dfSelecionado = dfSelecionado[dfSelecionado['jogos_num'] < 2]
variables     = ['ultima_pontuacao',
                 'ultimo_preco']

# %%


# %%
yHat = linearRegression(dfSelecionado[variables], dfSelecionado['minval'])

# %%
dfSelecionado = df_teste

# %%
dfSelecionado = dfSelecionado[dfSelecionado['jogos_num']>1]

# %%
dfSelecionado.corr()['min_valorizar'].sort_values(ascending=False)

# %%


# %%
variables = [
            'diff_preco_medio_completo',
            #'diff_preco_medio',
            #'ultima_media_pontos',
            #'ultima_pontuacao',
            #'diff_pontuacao_media',
            'ultima_variacao',
            'diff_variacao_media',
            #'ultimo_preco',
            #'diff_preco_medio',
            #'diff_preco_medio_geral',
            'diff_pontuacao_media_geral',
            #'diff_preco_medio_geral',
            #'ultimo_preco',
            #'diff_pontuacao_media_completa',
            #'diff_pontuacao_media',
            #'diff_preco_medio',
            #'ultima_variacao',
            #'diff_variacao_media',
            #'preco_medio_rodada'    
            #'ultimo_preco',
            #'diff_preco_medio',
            #'diff_variacao_media',
            #'diff_preco_medio',
            #'ultima_media_pontos',
            #'ultimo_preco',
            #'variacao_media_at'
            #'diff_pontuacao_media'
            #'preco_anterior',
            #'media_num_anterior',
            #'pontuacao_anterior',
            #'variacao_anterior',
            #'ultimo_preco',
            #'diff_pontuacao_media',
            #'pontuacao_media_at',
            ##'ultima_media_pontos',
            ##'ultima_pontuacao',
            #'ultima_variacao',
            #'valor', 
            #'variacao_num',
            #'ultimo_preco', 
            ]

yHat = linearRegression(dfSelecionado[variables], dfSelecionado['min_valorizar'])

# %%
dfSelecionado['min_valorizar'] = yHat

# %%
dfSelecionado['diff'] = dfSelecionado['min_valorizar'] - dfSelecionado['minval']

# %%
dfSelecionado.sort_values('diff')

# %% [markdown]
# # Aqui preparando para entregar

# %%
df_min_valorizar = dfAtletas.copy()
df_avaliacao = df_min_valorizar[['jogos_num', 'atleta_id', 'apelido', 'clube', 'min_valorizar']]

dict_json = {"assinante": True,
             "rodada": rodada+1,
             "atletas": df_min_valorizar[['atleta_id', 'min_valorizar']].set_index('atleta_id').to_dict()['min_valorizar']
             }

df = pd.DataFrame.from_dict(dict_json, orient='index')
df[0].to_json(path_or_buf=f'2023/minimor{rodada+1}.json')


# %%



