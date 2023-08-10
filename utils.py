import json
from decouple import config
import swiftclient


def swift_upload(file_name, data_json):
    """
    Realiza o upload de um arquivo .json para o Vault da Globo

    Args:
        file_name (str): Apenas nome do arquivo com extensão
        data_json: Dados no formato JSON

    Examples:
        swift_upload('IEO-GM_Clubes_Atacantes_Neutro_Geral_R37.json', dataframe.to_json(...))
    """

    # Abre uma conexão no Vault
    container = 'static'
    swift_conn = swiftclient.Connection(authurl=config('SWIFT_AUTH_URL'),
                                        user=config('SWIFT_USER'),
                                        key=config('SWIFT_PASSWORD'),
                                        tenant_name=config('SWIFT_TENANT'),
                                        os_options={"endpoint_type": "admin"},
                                        auth_version='2.0')

    # Realiza o upload
    swift_conn.put_object(container, file_name, contents=data_json, content_type='application/json')


def exportar_dataframe(dataframe, conteudo, nome_arquivo, formato=None, vault=False):
    """
    Nomeia e exporta o arquivo para o formato necessário

    Args:
        dataframe: Dataframe usado pelo índice
        conteudo: tipo de conteudo a ser exportado (a classe)
        nome_arquivo(str): Apenas o nome do arquivo, sem a pasta ou formato
        formato(str): 'excel', 'csv' ou 'json'
        vault(bool): Se deve fazer upload para o vault (somente quando formato='json')

    Examples:
        self.exportar(df, 'IEO-GM_Clubes_Atacante_Neutro_Geral_R37', 'json', True):

    """

    path_arquivo = 'static/%s/{}/%s.{}' % (conteudo, nome_arquivo)

    if formato == 'excel':
        return dataframe.to_excel(path_arquivo.format('excel', 'xlsx'))

    elif formato == 'csv':
        return dataframe.to_csv(path_arquivo.format('csv', 'csv'))

    elif formato == 'json':
        with open(path_arquivo.format('json', 'json'), 'w', encoding='utf-8') as f:
            data_json = dataframe.to_json(f, force_ascii=False, indent=4)

            if vault:
                swift_upload(file_name=nome_arquivo + '.json', data_json=data_json)

            return data_json

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
