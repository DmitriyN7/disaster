"""Data loading and preprocessing entry points for disaster tweets."""

import numpy as np
import pandas as pd

from src.config import TEST_PATH, TRAIN_PATH
from src.preprocessing import preprocess_dataframe


def load_train_data(path=TRAIN_PATH) -> pd.DataFrame:
    """Load and preprocess the labeled Kaggle training data."""
    df = pd.read_csv(path, dtype={"id": np.int32, "target": np.int8})
    return preprocess_dataframe(df, correct_labels=True)


def load_test_data(path=TEST_PATH) -> pd.DataFrame:
    """Load and preprocess the unlabeled Kaggle test data."""
    df = pd.read_csv(path, dtype={"id": np.int32})
    return preprocess_dataframe(df, correct_labels=False)


df_train = load_train_data()
df_test = load_test_data()
