
from sklearn import datasets

from gflow.workflow import Step

class ShotsDatasetStep(Step):
    def run(self, context):

        import pandas as pd
        # Lets parametrize the name of the dataset
        dataset_name = context.get_input('dataset_name')
        
        df_shots = pd.read_csv('Finalizacoes.csv')
        del df_shots['Unnamed: 0']
        df_shots.index.name = 'index'
        df_shots['label'] = df_shots['Goal']

        df_shots = df_shots[['label','Distance','Angle Radians','header']].copy()
                
        # Lets log the number of rows of this dataset
        context.log_metric('shots_dataset_size', df_shots.shape[0])

        context.save_dataset(dataset_name, df_shots)
