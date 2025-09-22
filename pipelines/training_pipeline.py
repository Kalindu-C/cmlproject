import os
import sys
import joblib
import logging
import pandas as pd
from data_pipeline import data_pipeline
from typing import Dict,Any, Tuple, Optional

from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.model_building import XgBoostModelBuilder
from utils.config import get_model_config, get_data_paths
from utils.mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(
        data_path: str = 'data/raw/ChurnModelling.csv',
        model_params: Optional[Dict[str,Any]] = None,
        test_size: float = 0.2,
        random_state:int = 42,
        model_path:str = 'artifacts/models/churn_analysis.joblib'
):
    if(not os.path.exists(get_data_paths()['X_train']) or 
       (not os.path.exists(get_data_paths()['X_test'])) or 
       (not os.path.exists(get_data_paths()['Y_train'])) or 
       (not os.path.exists(get_data_paths()['Y_test']))):
        
        data_pipeline()
    else:
        print("loading data artifacts from data pipeline")

    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags(
        'training_pipeline', {
            'model_type': 'XGBoost',
            'training_strategy': 'simple',
            'other_models': 'random_forest'
        }
    )
    run = mlflow_tracker.start_run(
        run_name='training_pipeline',
        tags=run_tags
    )
    
    X_train = pd.read_csv(get_data_paths()['X_train'])
    X_test = pd.read_csv(get_data_paths()['X_test'])
    Y_train = pd.read_csv(get_data_paths()['Y_train'])
    Y_test = pd.read_csv(get_data_paths()['Y_test'])
    
    model_builder = XgBoostModelBuilder(**model_params)
    model = model_builder.build_model()

    trainer = ModelTrainer()
    model, _ = trainer.train(
        model=model,
        X_train=X_train,
        Y_train=Y_train
    )

    trainer.save_model(model, model_path)
    # print("model_saved")

    evaluator = ModelEvaluator(
        model=model,
        model_name="XGBoost"
    )

    evaluation_result = evaluator.evaluate(X_test,Y_test)
    # print("Evaluation result", evaluation_result)
    evaluation_result_cp = evaluation_result.copy()
    del evaluation_result_cp['cm']

    model_params = get_model_config()['model_params']
    mlflow_tracker.log_training_metrics(model, evaluation_result_cp, model_params)

    mlflow_tracker.end_run()


if __name__ == '__main__':
    model_config = get_model_config()
    model_params = model_config.get('model_params')
    # print(model_params)
    training_pipeline(model_params=model_params)

         
    

