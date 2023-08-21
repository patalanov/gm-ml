import sys 
#sys.path.append("../")
import pandas as pd
import numpy as np
from random import randint

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from maths.trig import calculate_distance_angles
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




def set_shots_coordinates(df):
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

    df = df.apply(lambda x: calculate_distance_angles(x), axis=1)
    return df


def set_assists_coordinates(df):
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

    df = df.apply(lambda x: calculate_distance_angles(x), axis=1)
    return df