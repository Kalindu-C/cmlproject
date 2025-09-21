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
from src.model_inference import ModelInference

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

inference = ModelInference('artifacts/models/churn_analysis.joblib')

def streaming_inference(inference, data):
    inference.load_encoders('artifacts/encode')
    pred = inference.predict(data)
    return pred


if __name__ == '__main__':
    data = {
            "RowNumber": 1,
            "CustomerId": 15634602,
            "Firstname": "Grace",
            "Lastname": "Williams",
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
            }
    pred = streaming_inference(inference, data)
    print(pred)