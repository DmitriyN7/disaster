"""Shared data splits for training pipelines."""

from sklearn.model_selection import train_test_split

from src.config import SEED
from src.eda import df_train

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["text"],
    df_train["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_train["target"],
)
