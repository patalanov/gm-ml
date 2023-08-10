

def filter_db(database=None, jogo_id_externo=None, jogo_id_interno=None, clube_id=None, 
                                 atleta_id=None, scout_slug=None, scout_ids=None, rodada=None, tempo=None):
    '''
    Função para filtrar a base de dados do ScoutService
    Parâmetros Obrigatórios:
        database: base de dados do scout service (dataframe)
    Parâmetros Opcionais:
        jogo_id: lista de partidas (ID SDEL) (list))
        clube_id: Lista de clubes (ID SDE) (list)
        atleta_id: Lista de atletas que você deseja analisar (ID SDE) (list)
        scout_slug: Lista de slugs que você deseja analisar (Ex: [Finalização]) (list)
        scout_ids: Lista dos scouts que você deseja analisar (Olhar dicionario scout service) (list)
        rodada: Lista de rodadas específicas que você deseja analisar (list)
        tempo: Tempo  (int)
    Resposta: Retorna informações dos técnicos informados.
    '''
    df = database
    try:
        del df['Unnamed: 0']
    except:
        None
    
    # Código exererno
    if jogo_id_externo:
        df = df[df['Partida_CodigoExterno']==jogo_id_externo]
    
    if jogo_id_interno: 
        df = df[df['Partida_CodigoInterno']==jogo_id_interno]
        
    # Clube
    if clube_id:
        df = df[df['clube_id'].isin(clube_id)]

    if atleta_id:
        df = df[df['atleta_id'].isin(atleta_id)]
        
    # Scout Slug
    if scout_slug:
        df = df[df['Lance'].isin(scout_slug)]
    
    # Scout ID
    if scout_ids:
        df = df[df['Codigo'].isin(scout_ids)]
    
    # Rodada
    if rodada:
        df = df[df['Rodada'].isin(rodada)]
    
    # Tempo
    if tempo:
        df = df[df['TempoPartida']==tempo]
    
    return df
