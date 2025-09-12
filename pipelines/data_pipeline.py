import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestorCSV
from src.handle_missing_values import DropMissingValueStrategy, FillMissingValuesStrategy, GenderImputer
from src.outliers_detection import OutlierDetector, IQROutliersDetection
from src.feature_bining import CustomBiningStrategy
from src.feture_encoding import OrdinalENcodingStrategy, NorminalEncodingStrategy
from src.feature_scaling import MinMaxScalingStrategy
from src.data_splitter import SimpleTrainTestSplitStrategy

from utils.config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config


def data_pipeline(
    data_path: str = 'data/raw/ChurnModelling.csv',
    target_column: str = 'Exited',
    test_size: float = 0.2,
    force_rebuild: bool = False
) -> Dict[str, np.ndarray]:
    
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    print("step 1 : Data ingestion")
    artifacts_dir = os.path.join(os.path.dirname(__file__),'..', data_paths['data_artifacts_dir'])
    x_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

    if os.path.exists(x_train_path) and \
        os.path.exists(x_test_path) and \
        os.path.exists(y_train_path) and \
        os.path.exists(y_test_path):
        
        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        Y_train = pd.read_csv(y_train_path)
        Y_test = pd.read_csv(y_test_path)

    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
    if not os.path.exists('temp_imputed.csv'):

        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        print(f"loaded data shape : {df.shape}")

    
        print("\nStep2: Handle Missing Values")

        drop_handler = DropMissingValueStrategy(critical_columns=columns['critical_columns'])

        age_handler = FillMissingValuesStrategy(
            method='mean',
            relevant_column='Age'
        )

        gender_handler = FillMissingValuesStrategy(
            relevant_column='Gender',
            is_custom_imputer=True,
            custom_imputer=GenderImputer()
        )

        df = drop_handler.handle(df)
        df = age_handler.handle(df)
        df = gender_handler.handle(df)

        df.to_csv('temp_imputed.csv', index=False)
    
    df = pd.read_csv('temp_imputed.csv')
    print(f'data shape after imputation: {df.shape}')

    print("\nStep 03: Handle Outliers")

    outlier_detector = OutlierDetector(strategy=IQROutliersDetection())
    df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
    print(f"data shape outlier removal: {df.shape}")

    print("\nStep 04: Feature Bining")

    binning = CustomBiningStrategy(binning_config['credit_score_bins'])
    df = binning.bin_feature(df, 'CreditScore')
    print(f"data shape after binning: \n{df.head()}")

    print("\nStep 05: Feature Encoding")
    
    norminal_strategy = NorminalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_strategy = OrdinalENcodingStrategy(encoding_config['ordinal_mappings'])

    df = norminal_strategy.encode(df)
    df = ordinal_strategy.encode(df)

    print(f"data shape after encoding: \n{df.head()}")

    print("\nStep 06: Feature Scaling")
    min_max_strategy = MinMaxScalingStrategy()
    df = min_max_strategy.scale(df, scaling_config['columns_to_scale'])
    print(f"data shape after scaling: \n{df.head()}")

    print("\nStep 07: Post Processing")
    df.drop(columns=['Unnamed: 0', 'RowNumber', 'CustomerId', 'Firstname', 'Lastname'], inplace=True)
    print(df)

    print("\nStep 08: Data Splitting")
    splitting_strategy = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'])
    X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, 'Exited')

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")


data_pipeline()