import os
import pandas as pd
from abc import ABC, abstractmethod

class DataIngestor(ABC):  # build abstract class that has method called ingest (Abstraction)
    @abstractmethod
    def ingest(self, file_path_or_link:str):
        pass

# we inherit DataIngestor to the DataIngestorCSV (Inheritence)
# we have to override all the methods inside super class
class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path_or_link: str):
        return pd.read_csv(file_path_or_link)

# both classes use same method(ingest) seperately. so here method overriding.(polymorphism) there

class DataIngestorExcel(DataIngestor):
    def ingest(self, file_path_or_link):
        return pd.read_excel(file_path_or_link)