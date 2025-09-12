import logging
import pandas as pd
from abc import ABC, abstractmethod


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df:pd.DataFrame, columns:list) -> pd.DataFrame:
        pass


class IQROutliersDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            df[col] = df[col].astype(float)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3-Q1
            outliers[col] = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        logging.info("Outliers detected using IQR method")
        return outliers
    

class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy

    def detect_outliers(self, df, selected_columns):
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='remove'):
        outliers = self.detect_outliers(df, selected_columns)
        outliers_count = outliers.sum(axis=1)
        rows_to_remove = outliers_count >= 2
        return df[~rows_to_remove]