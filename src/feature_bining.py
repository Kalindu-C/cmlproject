import logging
from multiprocessing import Value
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureBiningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df:pd.DataFrame, column:str) -> pd.DataFrame:
        pass


class CustomBiningStrategy(FeatureBiningStrategy):
    def __init__(self, bin_definition) -> None:
        self.bin_definition = bin_definition

    def bin_feature(self, df, column):
        def assing_bin(value):
            if value == 850:
                return "Excellent"
            
            for bin_label, bin_range in self.bin_definition.items():
                if(bin_range) == 2:
                    if bin_range[0] <= value <= bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return bin_label
                    
            if value > 850:
                return "Invalid"
            
            return "Invalid"
        
        df[f'{column}Bins'] = df[column].apply(assing_bin)
        del df[column]

        return df