import json
import pandas as pd


def minuto_ocorrencia(lance):
    if 'S' not in lance:
        lance = lance+'00'
    if 'M' in lance:
        if len(lance.split('M')[0].split('T')[1])==1:
            tempo = '0'+lance.split('M')[0].split('T')[1]+':'+lance.split('M')[1].replace('S','')
        else:
            tempo = lance.split('M')[0].split('T')[1]+':'+lance.split('M')[1].replace('S','')
    else:
        tempo = lance.replace('PT','00:').replace('S','')

    if len(tempo.split(':')[1])==1:
        tempo = tempo.split(':')[0]+':0'+tempo.split(':')[1]
    return tempo



def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}



def pivot_columns(df, colunas, append=False):
    for coluna in colunas:
        df[coluna] = df[coluna].apply(lambda x: flatten_dict(x))

    for coluna in colunas:
        try:
            df[coluna] = df[coluna].apply(lambda x: str(x).replace('\'','"').replace('None', '0').replace('True', '1').replace('False', '0'))
        except:
            df[coluna] = pd.DataFrame()
        try:
            df_temp = df[coluna].apply(json.loads).apply(pd.Series)
        except:
            df_temp = pd.DataFrame()
        if append:
            df_temp = df_temp.add_prefix(f"{coluna}_")
        df = pd.concat([df, df_temp],axis=1)
        df = df.drop(coluna, axis=1)
    return df


def get_escudos():
    icon = {'Vasco':'![Vasco](https://s.glbimg.com/es/sde/f/organizacoes/2016/07/29/Vasco-65.png)',
            'Internacional':'![Internacional](https://s.glbimg.com/es/sde/f/organizacoes/2016/05/03/inter65.png)',
            'Santos':'![Santos](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/14/santos_60x60.png)',
            'Ceará':'![Ceará](https://s.glbimg.com/es/sde/f/organizacoes/2019/10/10/ceara-65x65.png)',
            'Bragantino':'![Bragantino](https://s.glbimg.com/es/sde/f/organizacoes/2020/01/01/65.png)',
            'Flamengo':'![Flamengo](https://s.glbimg.com/es/sde/f/organizacoes/2018/04/09/Flamengo-65.png)',
            'Fortaleza':'![Fortaleza](https://s.glbimg.com/es/sde/f/organizacoes/2018/06/10/fortaleza-ec-65px.png)',
            'Fluminense':'![Fluminense](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/14/fluminense_60x60.png)',
            'Palmeiras':'![Palmeiras](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/14/palmeiras_60x60.png)',
            'Bahia':'![Bahia](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/14/bahia_60x60.png)',
            'Athletico-PR':'![Athletico-PRR](https://s.glbimg.com/es/sde/f/organizacoes/2019/09/09/Athletico-PR-65x65.png)',
            'Atlético-MG':'![Atlético-MG](https://s.glbimg.com/es/sde/f/organizacoes/2017/11/23/Atletico-Mineiro-escudo65px.png)',
            'Corinthians':'![Corinthians](https://s.glbimg.com/es/sde/f/organizacoes/2019/09/30/Corinthians_65.png)',
            'Coritiba':'![Coritiba](https://s.glbimg.com/es/sde/f/organizacoes/2017/03/29/coritiba65.png)',
            'Sport':'![Sport](https://s.glbimg.com/es/sde/f/organizacoes/2015/07/21/sport65.png)',
            'Grêmio':'![Grêmio](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/14/gremio_60x60.png)',
            'Goiás':'![Goiás](https://s.glbimg.com/es/sde/f/organizacoes/2019/05/01/Goias_65px.png)',
            'Atlético-GO':'![Atletico-GO](https://s.glbimg.com/es/sde/f/organizacoes/2020/07/02/atletico-go-2020-65.png)',
            'São Paulo':'![São Paulo](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/14/sao_paulo_60x60.png)',
            'Botafogo':'![Botafogo](https://s.glbimg.com/es/sde/f/equipes/2013/12/16/botafogo_30x30.png)',
            'América-MG':' ![América-MG](https://s.glbimg.com/es/sde/f/organizacoes/2019/02/28/escudo65_1.png)',
            'Chapecoense':' ![Chapecoense](https://s.glbimg.com/es/sde/f/organizacoes/2015/08/03/Escudo-Chape-165.png)',
            'Juventude':' ![Juventude](https://s.glbimg.com/es/sde/f/organizacoes/2016/05/08/juventude65.png)',
            'Cuiabá':' ![Cuiabá](https://s.glbimg.com/es/sde/f/organizacoes/2014/04/16/cuiaba65.png)'}    
    return icon