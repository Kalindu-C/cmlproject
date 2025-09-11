from genericpath import exists
import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class FeatureEncodingStratergy(ABC):
    @abstractmethod
    def encode(self, df:pd.DataFrame)->pd.DataFrame:
        pass


class VariableType(str, Enum):
    NORMINAL = 'norminal'
    ORDINAL = 'ordinal'


# for norminal encoding 
class NorminalEncodingStrategy(FeatureEncodingStratergy):
    def __init__(self, norminal_columns):
        self.norminal_columns = norminal_columns
        self.encoder_dicts = {}
        os.mkdir('artifacts/encode', exist_ok=True)

    def encode(self, df):
        for column in self.norminal_columns:
            unique_values = df[column].unique()
            encoder_dict = {value:i for i, value in enumerate(unique_values)}
            self.encoder_dicts[column] = encoder_dict

            encoder_path = os.path.join('artifacts/encode', f"{column}_encoder.json")
            with open(encoder_path, "w") as f:
                json.dump(encoder_dict)

            df[column] = df[column].map(encoder_dict)
        return df
    
    def get_encoder_dicts(self):
        return self.encoder_dicts
    

# for ordinal encoding
class OrdinalENcodingStrategy(FeatureEncodingStratergy):
    def __init__(self, ordinal_mapping):
        self.ordinal_mapping = ordinal_mapping

    def encode(self,df):
        for column, mapping in self.ordinal_mapping.items():
            df[column] = df[column].map(mapping)
            logging.info(f"Encoded ordinal variable '{column} with {len(mapping)} categories")
