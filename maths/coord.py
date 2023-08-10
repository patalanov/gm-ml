import sys 
sys.path.append("../../")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


# Randomizar os valores dos 36 quadrantes para retirar a coordenada do centro
# e gerar maior variância para os modelos de XG
dictCoordenadas36 = {
        1:(20+np.random.randint(20, 70),50+np.random.randint(20, 100)),
        2:(113+np.random.randint(20, 70),50+np.random.randint(20, 100)),
        3:(206+np.random.randint(20, 70),50+np.random.randint(20, 100)),
        4:(299+np.random.randint(20, 70),50+np.random.randint(20, 100)),
        5:(392+np.random.randint(20, 70),50+np.random.randint(20, 100)),
        6:(485+np.random.randint(20, 70),50+np.random.randint(20, 100)),
        7:(20+np.random.randint(20, 70),183+np.random.randint(20, 100)),
        8:(113+np.random.randint(20, 70),183+np.random.randint(20, 100)),
        9:(206+np.random.randint(20, 70),183+np.random.randint(20, 100)),
        10:(299+np.random.randint(20, 70),183+np.random.randint(20, 100)),
        11:(392+np.random.randint(20, 70),183+np.random.randint(20, 100)),
        12:(485+np.random.randint(20, 70),183+np.random.randint(20, 100)),
        13:(20+np.random.randint(20, 70),316+np.random.randint(20, 100)),
        14:(113+np.random.randint(20, 70),316+np.random.randint(20, 100)),
        15:(206+np.random.randint(20, 70),316+np.random.randint(20, 100)),
        16:(299+np.random.randint(20, 70),316+np.random.randint(20, 100)),
        17:(392+np.random.randint(20, 70),316+np.random.randint(20, 100)),
        18:(485+np.random.randint(20, 70),316+np.random.randint(20, 100)),
        19:(20+np.random.randint(20, 70),449+np.random.randint(20, 100)),
        20:(113+np.random.randint(20, 70),449+np.random.randint(20, 100)),
        21:(206+np.random.randint(20, 70),449+np.random.randint(20, 100)),
        22:(299+np.random.randint(20, 70),449+np.random.randint(20, 100)),
        23:(392+np.random.randint(20, 70),449+np.random.randint(20, 100)),
        24:(485+np.random.randint(20, 70),449+np.random.randint(20, 100)),
        25:(20+np.random.randint(20, 70),582+np.random.randint(20, 100)),
        26:(113+np.random.randint(20, 70),582+np.random.randint(20, 100)),
        27:(206+np.random.randint(20, 70),582+np.random.randint(20, 100)),
        28:(299+np.random.randint(20, 70),582+np.random.randint(20, 100)),
        29:(392+np.random.randint(20, 70),582+np.random.randint(20, 100)),
        30:(485+np.random.randint(20, 70),582+np.random.randint(20, 100)),
        31:(20+np.random.randint(20, 70),715+np.random.randint(20, 100)),
        32:(113+np.random.randint(20, 70),715+np.random.randint(20, 100)),
        33:(206+np.random.randint(20, 70),715+np.random.randint(20, 100)),
        34:(299+np.random.randint(20, 70),715+np.random.randint(20, 100)),
        35:(392+np.random.randint(20, 70),715+np.random.randint(20, 100)),
        36:(485+np.random.randint(20, 70),715+np.random.randint(20, 100)),
        }

dictCoordenadas36_to_100 = {
        1:(0.0+np.random.uniform(0.0,16.6),0.0+np.random.uniform(0.0,16.6)),
        2:(0.0+np.random.uniform(0.0,16.6),16.6+np.random.uniform(16.6, 33.2)),
        3:(0.0+np.random.uniform(0.0,16.6),33.2+np.random.uniform(33.2, 49.8)),
        4:(0.0+np.random.uniform(0.0,16.6),49.8+np.random.uniform(49.8, 66.4)),
        5:(0.0+np.random.uniform(0.0,16.6),66.4+np.random.uniform(66.4, 83.0)),
        6:(0.0+np.random.uniform(0.0,16.6),83.0+np.random.uniform(83.0, 100)),
        7:(16.6+np.random.uniform(16.6,33.2),0.0+np.random.uniform(0.0,16.6)),
        8:(16.6+np.random.uniform(16.6,33.2),16.6+np.random.uniform(16.6, 33.2)),
        9:(16.6+np.random.uniform(16.6,33.2),33.2+np.random.uniform(33.2, 49.8)),
        10:(16.6+np.random.uniform(16.6,33.2),49.8+np.random.uniform(49.8, 66.4)),
        11:(16.6+np.random.uniform(16.6,33.2),66.4+np.random.uniform(66.4, 83.0)),
        12:(16.6+np.random.uniform(16.6,33.2),83.0+np.random.uniform(83.0, 100)),
        13:(33.2+np.random.uniform(33.2, 49.8),0.0+np.random.uniform(0.0,16.6)),
        14:(33.2+np.random.uniform(33.2, 49.8),16.6+np.random.uniform(16.6, 33.2)),
        15:(33.2+np.random.uniform(33.2, 49.8),33.2+np.random.uniform(33.2, 49.8)),
        16:(33.2+np.random.uniform(33.2, 49.8),49.8+np.random.uniform(49.8, 66.4)),
        17:(33.2+np.random.uniform(33.2, 49.8),66.4+np.random.uniform(66.4, 83.0)),
        18:(33.2+np.random.uniform(33.2, 49.8),83.0+np.random.uniform(83.0, 100)),
        19:(49.8+np.random.uniform(49.8, 66.4),0.0+np.random.uniform(0.0,16.6)),
        20:(49.8+np.random.uniform(49.8, 66.4),16.6+np.random.uniform(16.6, 33.2)),
        21:(49.8+np.random.uniform(49.8, 66.4),33.2+np.random.uniform(33.2, 49.8)),
        22:(49.8+np.random.uniform(49.8, 66.4),49.8+np.random.uniform(49.8, 66.4)),
        23:(49.8+np.random.uniform(49.8, 66.4),66.4+np.random.uniform(66.4, 83.0)),
        24:(49.8+np.random.uniform(49.8, 66.4),83.0+np.random.uniform(83.0, 100)),
        25:(66.4+np.random.uniform(66.4, 83.0),0.0+np.random.uniform(0.0,16.6)),
        26:(66.4+np.random.uniform(66.4, 83.0),16.6+np.random.uniform(16.6, 33.2)),
        27:(66.4+np.random.uniform(66.4, 83.0),33.2+np.random.uniform(33.2, 49.8)),
        28:(66.4+np.random.uniform(66.4, 83.0),49.8+np.random.uniform(49.8, 66.4)),
        29:(66.4+np.random.uniform(66.4, 83.0),66.4+np.random.uniform(66.4, 83.0)),
        30:(66.4+np.random.uniform(66.4, 83.0),83.0+np.random.uniform(83.0, 100)),
        31:(83.0+np.random.uniform(83.0, 100),0.0+np.random.uniform(0.0,16.6)),
        32:(83.0+np.random.uniform(83.0, 100),16.6+np.random.uniform(16.6, 33.2)),
        33:(83.0+np.random.uniform(83.0, 100),33.2+np.random.uniform(33.2, 49.8)),
        34:(83.0+np.random.uniform(83.0, 100),49.8+np.random.uniform(49.8, 66.4)),
        35:(83.0+np.random.uniform(83.0, 100),66.4+np.random.uniform(66.4, 83.0)),
        36:(83.0+np.random.uniform(83.0, 100),83.0+np.random.uniform(83.0, 100)),
        }

def transform(df=None, tipo='', quadrantes=[6,6], xg=False):
    '''

    Parâmetros Obrigatórios:
        df: dataframe
        quadrantes: 36 ao todo
        xg: modelo que requer transformação de (x,y) para percentual, no range[0,100]

    Resposta: Retorna o dataframe de eventos com coordenadas
    '''
    
     # Configurações do campo
    width = 800
    height  = 550
    
    # transform coordinates to percentage
    def xy_to_dict(x,y):
        return {'y': (y/width)*100, 'x': (x/height)*100}

    # Data
    df['coordenadas'] = df['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))
    
    # replace Nan with tuple with zeros
    for row in df.loc[df.coordenadas.isnull(), 'coordenadas'].index:
        df.at[row, 'coordenadas'] = (0,0)

    if xg:
        df['coordenadas'] = df.apply(lambda row: xy_to_dict(row['coordenadas'][0], row['coordenadas'][1]), axis=1)
       
    return df





def campinho(df=None, tipo='', quadrantes=[6,6]):

    # Data
    df['coordenadas'] = df['PosicaoLance'].apply(lambda x: dictCoordenadas36.get(x))

    # replace Nan with tuple with zeros
    for row in df.loc[df.coordenadas.isnull(), 'coordenadas'].index:
        df.at[row, 'coordenadas'] = (0,0)
    # Configurações do campo
    height = 550
    width  = 800
    
    xbins = height/quadrantes[0]
    ybins = width/quadrantes[1]
    img = plt.imread("/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre/utils/static/campinho_cartola_v.jpg")

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.ylim([0,900])
    plt.xlim([0,600])
    x = []
    y = []
    fotos = []

    for i, row in df.iterrows():
        x.append(row['coordenadas'][0]+np.random.randint(20, 70))
        y.append(row['coordenadas'][1]+np.random.randint(20, 100))
        #fotos.append(row['foto'])
        ax.add_artist(patches.Rectangle(xy=row['coordenadas'],
                      color='firebrick',
                      width=xbins, alpha=0.05, height=ybins))
        
    plt.plot(x,y, "ro", alpha=0.5,  markersize=10)
    plt.show()

