from typing import List

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _one_hot_encode(df: pd.DataFrame, categorical_columns: List[str] = []) -> pd.DataFrame:
    if len(categorical_columns) == 0:
        return df
    
    return pd.get_dummies(df, columns=categorical_columns)


def _scale_numerical_features(df: pd.DataFrame, numerical_columns: List[str] = [], scaler: StandardScaler = None) -> pd.DataFrame:
    if (len(numerical_columns) == 0) or (not scaler):
        return df
    
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    
    return df


def preprocess_data(df: pd.DataFrame, categorical_columns: List[str] = [], numerical_columns: List[str] = []) -> pd.DataFrame:
    df = _one_hot_encode(df, categorical_columns)
    df = _scale_numerical_features(df, numerical_columns)

    return df
