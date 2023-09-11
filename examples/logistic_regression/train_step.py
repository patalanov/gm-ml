
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gflow.workflow import Step

class TrainStep(Step):
    def run(self, context):
        # Lets parametrize the name of the dataset and the max_iter parameter
        dataset_name = context.get_input('dataset_name')
        max_iter = context.get_input('max_iter') or 150
        
        # Load the dataset and the sklearn pipeline
        df = context.load_dataset(dataset_name).set_index('index')

        X = df.drop(columns=['label']).values.tolist()
        y = df['label']
        
        pipeline = Pipeline([('clf', LogisticRegression(random_state=0, max_iter=max_iter))])

        # Train!
        pipeline.fit(X, y)
        context.save_model('model', pipeline)
        
        #Version and tag the model
        model_version = context.get_input('model_version')
        model_tag = context.get_input('model_tag')

        if model_version:
            context.version_model('model', model_version)

        if model_version and model_tag:
            context.tag_model('model', model_version, model_tag)
