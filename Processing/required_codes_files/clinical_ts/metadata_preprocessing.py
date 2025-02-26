# metadata_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_metadata(df, metadata_cols):
    metadata_df = df[metadata_cols].fillna(df[metadata_cols].mean())
    metadata_scaler = StandardScaler()
    normalized_metadata = metadata_scaler.fit_transform(metadata_df)
    return normalized_metadata
