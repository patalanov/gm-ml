import sys 
#sys.path.append("../")
import pandas as pd
import numpy as np
from random import randint

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from maths.trig import calculate_distance_angles, calculate_angles
from sources.apis.utils import sct as sct


# In[Dicionário de coordenadas]


dictCoordenadas36 = {
    1:(20,50),
    2:(113,50),
    3:(206,50),
    4:(299,50),
    5:(392,50),
    6:(485,50),
    7:(20,183),
    8:(113,183),
    9:(206,183),
    10:(299,183),
    11:(392,183),
    12:(485,183),
    13:(20,316),
    14:(113,316),
    15:(206,316),
    16:(299,316),
    17:(392,316),
    18:(485,316),
    19:(20,449),
    20:(113,449),
    21:(206,449),
    22:(299,449),
    23:(392,449),
    24:(485,449),
    25:(20,582),
    26:(113,582),
    27:(206,582),
    28:(299,582),
    29:(392,582),
    30:(485,582),
    31:(20,715),
    32:(113,715),
    33:(206,715),
    34:(299,715),
    35:(392,715),
    36:(485,715),
}



# In[Functions]

def draw_pitch(x_min=0, x_max=105,
               y_min=0, y_max=68,
               pitch_color="w",
               line_color="grey",
               line_thickness=1.5,
               point_size=20,
               orientation="horizontal",
               aspect="full",
               ax=None
               ):

    if not ax:
        raise TypeError("This function is intended to be used with an existing fig and ax in order to allow flexibility in plotting of various sizes and in subplots.")


    if orientation.lower().startswith("h"):
        first = 0
        second = 1
        arc_angle = 0

        if aspect == "half":
            ax.set_xlim(x_max / 2, x_max + 5)

    elif orientation.lower().startswith("v"):
        first = 1
        second = 0
        arc_angle = 90

        if aspect == "half":
            ax.set_ylim(x_max / 2, x_max + 5)

    
    else:
        raise NameError("You must choose one of horizontal or vertical")

    
    ax.axis("off")

    rect = plt.Rectangle((x_min, y_min),
                         x_max, y_max,
                         facecolor=pitch_color,
                         edgecolor="none",
                         zorder=-2)

    ax.add_artist(rect)

    x_conversion = x_max / 100
    y_conversion = y_max / 100

    pitch_x = [0,5.8,11.5,17,50,83,88.5,94.2,100] # pitch x markings
    pitch_x = [x * x_conversion for x in pitch_x]

    pitch_y = [0, 21.1, 36.6, 50, 63.2, 78.9, 100] # pitch y markings
    pitch_y = [x * y_conversion for x in pitch_y]

    goal_y = [45.2, 54.8] # goal posts
    goal_y = [x * y_conversion for x in goal_y]

    # side and goal lines
    lx1 = [x_min, x_max, x_max, x_min, x_min]
    ly1 = [y_min, y_min, y_max, y_max, y_min]

    # outer boxed
    lx2 = [x_max, pitch_x[5], pitch_x[5], x_max]
    ly2 = [pitch_y[1], pitch_y[1], pitch_y[5], pitch_y[5]]

    lx3 = [0, pitch_x[3], pitch_x[3], 0]
    ly3 = [pitch_y[1], pitch_y[1], pitch_y[5], pitch_y[5]]

    # goals
    lx4 = [x_max, x_max+2, x_max+2, x_max]
    ly4 = [goal_y[0], goal_y[0], goal_y[1], goal_y[1]]

    lx5 = [0, -2, -2, 0]
    ly5 = [goal_y[0], goal_y[0], goal_y[1], goal_y[1]]

    # 6 yard boxes
    lx6 = [x_max, pitch_x[7], pitch_x[7], x_max]
    ly6 = [pitch_y[2],pitch_y[2], pitch_y[4], pitch_y[4]]

    lx7 = [0, pitch_x[1], pitch_x[1], 0]
    ly7 = [pitch_y[2],pitch_y[2], pitch_y[4], pitch_y[4]]


    # Halfway line, penalty spots, and kickoff spot
    lx8 = [pitch_x[4], pitch_x[4]]
    ly8 = [0, y_max]

    lines = [
        [lx1, ly1],
        [lx2, ly2],
        [lx3, ly3],
        [lx4, ly4],
        [lx5, ly5],
        [lx6, ly6],
        [lx7, ly7],
        [lx8, ly8],
        ]

    points = [
        [pitch_x[6], pitch_y[3]],
        [pitch_x[2], pitch_y[3]],
        [pitch_x[4], pitch_y[3]]
        ]

    circle_points = [pitch_x[4], pitch_y[3]]
    arc_points1 = [pitch_x[6], pitch_y[3]]
    arc_points2 = [pitch_x[2], pitch_y[3]]


    for line in lines:
        ax.plot(line[first], line[second],
                color=line_color,
                lw=line_thickness,
                zorder=-1)

    for point in points:
        ax.scatter(point[first], point[second],
                   color=line_color,
                   s=point_size,
                   zorder=-1)

    circle = plt.Circle((circle_points[first], circle_points[second]),
                        x_max * 0.088,
                        lw=line_thickness,
                        color=line_color,
                        fill=False,
                        zorder=-1)

    ax.add_artist(circle)

    arc1 = Arc((arc_points1[first], arc_points1[second]),
               height=x_max * 0.088 * 2,
               width=x_max * 0.088 * 2,
               angle=arc_angle,
               theta1=128.75,
               theta2=231.25,
               color=line_color,
               lw=line_thickness,
               zorder=-1)

    ax.add_artist(arc1)

    arc2 = Arc((arc_points2[first], arc_points2[second]),
               height=x_max * 0.088 * 2,
               width=x_max * 0.088 * 2,
               angle=arc_angle,
               theta1=308.75,
               theta2=51.25,
               color=line_color,
               lw=line_thickness,
               zorder=-1)

    ax.add_artist(arc2)

    ax.set_aspect("equal")

    return ax




def set_shots_coordinates(df,metric):
    list_goals = [11,23,6,17,55,60,38]
    df['Goal'] = df['Codigo'].apply (lambda x: 1 if x in list_goals else 0)
    # Inserindo informações de finalização
    list_finalizacoes = [3,4,5,6,7,8,9,17,18,19,77]
    df['header'] = df['Codigo'].apply (lambda x: 1 if x in list_finalizacoes else 0)


    df = sct.filter_db(df,scout_ids=[1,10,11,12,13,20,21,22,23,24,45,59,90,
                                     3,4,5,6,7,8,9,17,18,19,77,
                                     55,56,57,58,59,
                                     60,61,62,63])

    df['coordenadas'] = df['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))
    #print(df['coordenadas'].isnull().sum())
    df = df.dropna(subset=['coordenadas'])

    x = []
    y = []
    for i, row in df.iterrows():
        if row['PosicaoLance']==-1:
            x.append(-1)
            y.append(-1)
        elif row['PosicaoLance']>=36:
            x.append(-1)
            y.append(-1)
        else:
            x.append(round(row['coordenadas'][0]+randint(10, 100),2))
            y.append(round(row['coordenadas'][1]+randint(0, 130),2))
            # x.append(row['coordenadas'][0]) #+randint(20, 70)
            # y.append(row['coordenadas'][1]) #+randint(20, 100)
    df['x'] = x
    df['y'] = y

    # # Check if was penalty
    # list_penalty = [60,61,62,63]
    # lances['x'][lances['Codigo'].isin(list_penalty)] = 300
    # lances['y'][lances['Codigo'].isin(list_penalty)] = 777


    # Normalizando tamanho campo em x e y
    df['x'] = ((df['x']/550)*100) -6
    df['y'] = ((df['y']/800)*100) -7

    # Colocando na situação do campo
    df['X'] = df['x']*0.68
    df['Y'] = df['y']*1.05


    df['x'] = df['x']
    df['y'] = 100-df['y']

    df['Center_dist'] = abs(df['x']-50)

    df = df.apply(lambda x: calculate_distance_angles(x,metric), axis=1)
    return df



def set_shots_on_target_coordinates(df,metric):

    ## Finalização
    # 10: 'Finalização,Dentro Área,Defendido'
    # 11: 'Finalização,Dentro Área,Gol'
    # 13: 'Finalização,Fora da Área,Defendido'
    # 23: 'Finalização,Fora da Área,Gol'
    # 89: 'Finalização,Finalização,Certa'

    # ## Finalização de Cabeça
    # 3:  'Finalização Cabeça,Dentro Peq.Área,Defendido'
    # 6:  'Finalização Cabeça,Dentro Peq.Área,Gol'
    # 8:  'Finalização Cabeça,Grande Área,Defendido'
    # 17: 'Finalização Cabeça,Grande Área,Gol'

    # 57: 'Falta,Defendida'

    list_goals = [11,23,6,17,55,60,38]
    df['Goal'] = df['Codigo'].apply (lambda x: 1 if x in list_goals else 0)
    # Inserindo informações de finalização
    list_finalizacoes_no_gol = [3,6,8,17]
    df['header_ot'] = df['Codigo'].apply (lambda x: 1 if x in list_finalizacoes_no_gol else 0)

    df = sct.filter_db(df,scout_ids=[10,11,13,23,89,
                                     3,6,8,17,
                                     57])

    df['coordenadas_ot'] = df['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))
    #print(df['coordenadas'].isnull().sum())
    df = df.dropna(subset=['coordenadas_ot'])

    x = []
    y = []
    for i, row in df.iterrows():
        if row['PosicaoLance']==-1:
            x.append(-1)
            y.append(-1)
        elif row['PosicaoLance']>=36:
            x.append(-1)
            y.append(-1)
        else:
            x.append(round(row['coordenadas_ot'][0]+randint(10, 100),2))
            y.append(round(row['coordenadas_ot'][1]+randint(0, 130),2))
            # x.append(row['coordenadas_ot'][0]) #+randint(20, 70)
            # y.append(row['coordenadas_ot'][1]) #+randint(20, 100)
    df['x_ot'] = x
    df['y_ot'] = y

    # # Check if was penalty
    # list_penalty = [60,61,62,63]
    # lances['x'][lances['Codigo'].isin(list_penalty)] = 300
    # lances['y'][lances['Codigo'].isin(list_penalty)] = 777

    # Normalizando tamanho campo em x e y
    df['x_ot'] = ((df['x_ot']/550)*100) -6
    df['y_ot'] = ((df['y_ot']/800)*100) -7

    # Colocando na situação do campo
    df['X_ot'] = df['x_ot']*0.68
    df['Y_ot'] = df['y_ot']*1.05


    df['x_ot'] = df['x_ot']
    df['y_ot'] = 100-df['y_ot']

    df['Center_dist Ot'] = abs(df['x_ot']-50)

    df = df.apply(lambda x: calculate_distance_angles(x,metric), axis=1)
    return df



def set_assists_coordinates(df,metric):
    ## Passe
    # 14: 'Passe,Decisivo'
    # 25: 'Passe,Incompleto'
    # 74: 'Passe,Completo'

    df = sct.filter_db(df,scout_ids=[14])

    df['coordenadas_assist'] = df['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))
    #print(df['coordenadas'].isnull().sum())
    df = df.dropna(subset=['coordenadas_assist'])

    x = []
    y = []
    for i, row in df.iterrows():
        if row['PosicaoLance']==-1:
            x.append(-1)
            y.append(-1)
        elif row['PosicaoLance']>=36:
            x.append(-1)
            y.append(-1)
        else:
            x.append(round(row['coordenadas_assist'][0]+randint(10, 100),2))
            y.append(round(row['coordenadas_assist'][1]+randint(0, 130),2))
            # x.append(row['coordenadas'][0]) #+randint(20, 70)
            # y.append(row['coordenadas'][1]) #+randint(20, 100)
    
    df['x_assist'] = x
    df['y_assist'] = y

    # # Check if was penalty
    # list_penalty = [60,61,62,63]
    # lances['x'][lances['Codigo'].isin(list_penalty)] = 300
    # lances['y'][lances['Codigo'].isin(list_penalty)] = 777

    # Normalizando tamanho campo em x e y
    df['x_assist'] = ((df['x_assist']/550)*100) -6
    df['y_assist'] = ((df['y_assist']/800)*100) -7

    # Colocando na situação do campo
    df['X_assist'] = df['x_assist']*0.68
    df['Y_assist'] = df['y_assist']*1.05


    df['x_assist'] = df['x_assist']
    df['y_assist'] = 100-df['y_assist']

    df['Center_dist Assist'] = abs(df['x_assist']-50)

    df = df.apply(lambda x: calculate_distance_angles(x,metric), axis=1)
    return df


def set_preassists_coordinates(df,metric):
    ## Passe
    # 14: 'Passe,Decisivo'
    # 25: 'Passe,Incompleto'
    # 74: 'Passe,Completo'

    # list_assistencias = [14]
    # df['Assist'] = df['Codigo'].apply (lambda x: 1 if x in list_assistencias else 0)
    # # Inserindo informações de finalização
    # list_pre_assistencias = [3,4,5,6,7,8,9,17,18,19,77]
    # df['header'] = df['Codigo'].apply (lambda x: 1 if x in list_pre_assistencias else 0)

    df = sct.filter_db(df,scout_ids=[74])

    df['coordenadas_preassist'] = df['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))
    #print(df['coordenadas'].isnull().sum())
    df = df.dropna(subset=['coordenadas_preassist'])

    x = []
    y = []
    for i, row in df.iterrows():
        if row['PosicaoLance']==-1:
            x.append(-1)
            y.append(-1)
        elif row['PosicaoLance']>=36:
            x.append(-1)
            y.append(-1)
        else:
            x.append(round(row['coordenadas_preassist'][0]+randint(10, 100),2))
            y.append(round(row['coordenadas_preassist'][1]+randint(0, 130),2))
            # x.append(row['coordenadas'][0]) #+randint(20, 70)
            # y.append(row['coordenadas'][1]) #+randint(20, 100)
    
    df['x_preassist'] = x
    df['y_preassist'] = y

    # # Check if was penalty
    # list_penalty = [60,61,62,63]
    # lances['x'][lances['Codigo'].isin(list_penalty)] = 300
    # lances['y'][lances['Codigo'].isin(list_penalty)] = 777

    # Normalizando tamanho campo em x e y
    df['x_preassist'] = ((df['x_preassist']/550)*100) -6
    df['y_preassist'] = ((df['y_preassist']/800)*100) -7

    # Colocando na situação do campo
    df['X_preassist'] = df['x_preassist']*0.68
    df['Y_preassist'] = df['y_preassist']*1.05


    df['x_preassist'] = df['x_preassist']
    df['y_preassist'] = 100-df['y_preassist']

    df['Center_dist PreAssist'] = abs(df['x_preassist']-50)

    df = df.apply(lambda x: calculate_distance_angles(x,metric), axis=1)
    return df


############################################################################### NOVOS CALCULOS POS NOVOS CAMPOS DE DADOS SCT ###############################################################################


# Dimensões oficiais
goal_width_m = 7.32  # Largura do gol em metros
goal_height_m = 2.44  # Altura do gol em metros
field_width_m = 65  # Largura do campo em metros, conforme especificado
field_height_m = 50  # Metade da altura do campo em metros

# Proporções do 'Goal Field'
goal_field_width_px = 804
goal_field_height_px = 306
proporcao_px_m_goal_width = goal_width_m / (goal_field_width_px / 3)  # Largura do gol é um terço do 'Goal Field'
proporcao_px_m_goal_height = (goal_height_m * 3) / goal_field_height_px  # Altura do gol triplicada

# Proporções do 'Field'
field_width_px = 804
field_height_px = 409
proporcao_px_m_field_width = field_width_m / field_width_px
proporcao_px_m_field_height = field_height_m / field_height_px



# User-provided conversion rates
proporcao_px_m_goal_width = 0.0273  # 1 pixel corresponds to approximately 0.0273 meters
proporcao_px_m_goal_height = 0.0080  # 1 pixel corresponds to approximately 0.0080 meters
proporcao_px_m_field_width = 0.0808  # 1 pixel corresponds to approximately 0.0808 meters
proporcao_px_m_field_height = 0.1222  # 1 pixel corresponds to approximately 0.1222 meters







def calculate_shots_coordinates(df, metric):
    """
    Converts shot position coordinates from pixels to meters, both for goal and field.

    Parameters:
    df (DataFrame): Data containing shot positions in pixels and various other metrics.
    metric (str): The metric name used to store the angle results.

    Returns:
    DataFrame: The input dataframe with additional columns for shot coordinates in meters.

    The conversion rates are based on the real-world dimensions of a football goal and field:
    - Goal width: 7.32 meters
    - Goal height: 2.44 meters
    - Field width: 65 meters
    - Field height: 50 meters (full height considered for conversion)

    The image's full width (804 pixels) represents the full width of the field (65 meters),
    and the image's full height (306 pixels) represents the height of the goal area (2.44 meters x 3).
    """

    # Real-world dimensions of a football field and goal
    goal_width_m   = 7.32  # Width of the goal in meters
    goal_height_m  = 2.44  # Height of the goal in meters
    field_width_m  = 65  # Width of the field in meters
    field_height_m = 50  # Full height of the field in meters (not half)

    # Conversion rates from pixels to meters
    proporcao_px_m_goal_width   = goal_width_m / (804 / 3)  # Convert goal width from pixels to meters
    proporcao_px_m_goal_height  = goal_height_m / (306 / 3)  # Convert goal height from pixels to meters
    proporcao_px_m_field_width  = field_width_m / 804  # Convert field width from pixels to meters
    proporcao_px_m_field_height = field_height_m / 306  # Convert field height from pixels to meters

    list_goals = [11,23,6,17,55,60,38] # given by scout service api
    list_finalizacoes = [3,4,5,6,7,8,9,17,18,19,77] # given by scout service api    
    
    df['Goal']   = df['Codigo'].apply(lambda x: 1 if x in list_goals else 0)
    df['header'] = df['Codigo'].apply(lambda x: 1 if x in list_finalizacoes else 0)

    df = sct.filter_db(df, scout_ids=[1,10,11,12,13,20,21,22,23,24,45,59,90,
                                      3,4,5,6,7,8,9,17,18,19,77,
                                      55,56,57,58,59,
                                      60,61,62,63])

    # Cópia das colunas de posição em pixels
    df['goal_x_px']  = df['TravePosicaoX']
    df['goal_y_px']  = df['TravePosicaoY']
    df['field_x_px'] = df['CampoPosicaoX']
    df['field_y_px'] = df['CampoPosicaoY']

    # Adjusted conversion in the method
    df['goal_x_metros'] = df['goal_x_px'] * proporcao_px_m_goal_width
    df['goal_y_metros'] = df['goal_y_px'] * proporcao_px_m_goal_height

    # Conversion of pixel coordinates to meters for the field
    df['field_x_metros'] = df['field_x_px'] * proporcao_px_m_field_width
    df['field_y_metros'] = df['field_y_px'] * proporcao_px_m_field_height

    df['Center_dist']  = df['Metros']  # Assumindo que 'Metros' já representa a distância do centro
    df['ContraAtaque'] = df['ContraAtaque'].fillna(0).astype(int)  # categorical as int

        # calcular angulos
    df = df.apply(lambda x: calculate_angles(x,metric), axis=1)
    df = df.apply(lambda x: calculate_angles(x,metric), axis=1)

    return df



def calculate_shots_on_target_coordinates(df, metric):
    """
    Converts shot position coordinates from pixels to meters, both for goal and field.

    Parameters:
    df (DataFrame): Data containing shot positions in pixels and various other metrics.
    metric (str): The metric name used to store the angle results.

    Returns:
    DataFrame: The input dataframe with additional columns for shot coordinates in meters.

    The conversion rates are based on the real-world dimensions of a football goal and field:
    - Goal width: 7.32 meters
    - Goal height: 2.44 meters
    - Field width: 65 meters
    - Field height: 50 meters (full height considered for conversion)

    The image's full width (804 pixels) represents the full width of the field (65 meters),
    and the image's full height (306 pixels) represents the height of the goal area (2.44 meters x 3).
    """

    # Real-world dimensions of a football field and goal
    goal_width_m   = 7.32  # Width of the goal in meters
    goal_height_m  = 2.44  # Height of the goal in meters
    field_width_m  = 65  # Width of the field in meters
    field_height_m = 50  # Full height of the field in meters (not half)

    # Conversion rates from pixels to meters
    proporcao_px_m_goal_width   = goal_width_m / (804 / 3)  # Convert goal width from pixels to meters
    proporcao_px_m_goal_height  = goal_height_m / (306 / 3)  # Convert goal height from pixels to meters
    proporcao_px_m_field_width  = field_width_m / 804  # Convert field width from pixels to meters
    proporcao_px_m_field_height = field_height_m / 306  # Convert field height from pixels to meters

    # valores (int) do scout service para lances desejados
    list_goals = [11,23,6,17,55,60,38]
    list_finalizacoes_no_gol = [3,6,8,17]   
    
    df['Goal'] = df['Codigo'].apply(lambda x: 1 if x in list_goals else 0)
    # header on target
    df['header_ot'] = df['Codigo'].apply(lambda x: 1 if x in list_finalizacoes_no_gol else 0)

    df = sct.filter_db(df, scout_ids=[10, 11, 13, 23, 89, 3, 6, 8, 17, 57])

    # Cópia das colunas de posição em pixels
    df['goal_x_px_on_target'] = df['TravePosicaoX']
    df['goal_y_px_on_target'] = df['TravePosicaoY']
    df['field_x_px_on_target'] = df['CampoPosicaoX']
    df['field_y_px_on_target'] = df['CampoPosicaoY']

    # Conversão da coordenada de pixels para metros para 'Goal Field'
    df['goal_x_metros_on_target'] = df['goal_x_px_on_target'] * proporcao_px_m_goal_width
    df['goal_y_metros_on_target'] = df['goal_y_px_on_target'] * proporcao_px_m_goal_height

    # Conversion of pixel coordinates to meters for the field
    df['field_x_metros_on_target'] = df['field_x_px_on_target'] * proporcao_px_m_field_width
    df['field_y_metros_on_target'] = df['field_y_px_on_target'] * proporcao_px_m_field_height

    df['Center_dist_on_target'] = df['Metros']  # Assumindo que 'Metros' já representa a distância do centro
    df['ContraAtaque'] = df['ContraAtaque'].fillna(0).astype(int)  # categorical as int

    # calcular angulos
    df = df.apply(lambda row: calculate_angles(row, metric), axis=1)
    df = df.apply(lambda row: calculate_angles(row, metric), axis=1)

    return df











