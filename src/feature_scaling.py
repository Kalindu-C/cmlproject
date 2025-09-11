import logging
from time import sleep
import pandas as pd
from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df:pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'


class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False

    def scale(self, df, columns_to_scale):
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        self.fitted = True
        logging.info(f'Applied MinMax scaling: {columns_to_scale}')
        return df
    
    def get_scale(self):
        return self.scaler