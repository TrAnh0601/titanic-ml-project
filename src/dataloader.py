import pandas as pd
from src import config

def load_data():
    """
    Load train and test datasets from paths defined in config.
    """
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    return train_df, test_df
