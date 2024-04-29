from google.cloud import storage
from collect import stats_atletas
from decouple import config

BUCKET_NAME = config('BUCKET_NAME')

def run_stats_atletas(event, context):
    """Background Cloud Function to be triggered by Pub/Sub.
    Args:
         event (dict):  The dictionary with data specific to this type of event.
         context (google.cloud.functions.Context): The event metadata.
    """
    torneios = ['Libertadores', 'Brasileiro', 'BrasileiroB', 'CopaBrasil', 'CopaNordeste', 'CopaSPJunior', 'Carioca', 'Gaucho', 'Mineiro', 'Baiano', 'Paulista', 'Pernambucano', 'PreLibertadores', 'RecopaSulAmericana', 'SulAmericana', 'SuperCopaBrasil']
    edicao = '2024'
    stats_atletas(torneios, edicao, BUCKET_NAME)
    print("Stats processed successfully")
