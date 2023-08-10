
from sklearn import datasets
from gflow.workflow import Step

class ShotsDatasetStep(Step):
    def run(self, context):
        # Lets parametrize the name of the dataset
        dataset_name = context.get_input('dataset_name')
        
        df_shots = context.load_dataset('shots.csv')
        df_shots.index.name = 'index'
        df_shots['label'] = df_shots['Goal']
        
        # Lets log the number of rows of this dataset
        context.log_metric('shots_dataset_size', df_shots.shape[0])

        context.save_dataset(dataset_name, df_shots)
