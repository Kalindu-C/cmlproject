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




if __name__ == '__main__':
    model_config = get_model_config()
    model_params = model_config.get('model_params')
    # print(model_params)
    training_pipeline(model_params=model_params)
    # 1.06.00

         
    

