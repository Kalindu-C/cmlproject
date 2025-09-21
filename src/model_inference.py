import json
import logging
import os
import sys
import joblib
from typing import Any, Dict, List, Optional, Tuple, Union
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.feature_bining import CustomBiningStrategy
from src.feture_encoding import OrdinalENcodingStrategy
from utils.config import get_binning_config, get_encoding_config



logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference():
    def __init__(
            self,
            model_path
    ):
        self.model_path = model_path
        self.encoders = {}
        self.model = self.load_model()
        self.binning_config = get_binning_config()
        self.encoding_config = get_encoding_config()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError("can't load file not found")
        
        return joblib.load(self.model_path)

    def load_encoders(self, encoder_dir):
        for file in os.listdir(encoder_dir):
            feature_name = file.split('_encoder.json')[0]
            with open(os.path.join(encoder_dir, file), 'r') as f:
                self.encoders[feature_name] = json.load(f)

    def preprocess_input(self, data):
        data = pd.DataFrame([data])

        for col, encoder in self.encoders.items():
            data[col] = data[col].map(encoder)

        binning = CustomBiningStrategy(self.binning_config['credit_score_bins'])
        data = binning.bin_feature(data, 'CreditScore')

        ordinal_strategy = OrdinalENcodingStrategy(self.encoding_config['ordinal_mappings'])
        data = ordinal_strategy.encode(data)

        data.drop(columns=['RowNumber', 'CustomerId', 'Firstname', 'Lastname'], inplace=True)
        # print(data)

        return data
    
    def predict(self, data):
        pp_data = self.preprocess_input(data)
        Y_pred = self.model.predict(pp_data)
        Y_proba = float(self.model.predict_proba(pp_data)[:,1])

        Y_pred = 'Churn' if Y_pred == 1 else 'Retain'
        Y_proba = round(Y_proba*100,2)

        return{
            "Status": Y_pred,
            "Confidence": f"{Y_proba} %"
        }
    



