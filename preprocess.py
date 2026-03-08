import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype


def preprocess_data(df):

    df = df.copy()

    for col in df.columns:
        if df[col].nunique() == len(df):
            df.drop(columns=[col], inplace=True)

    for col in df.columns:

        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() > 0.5 * len(df):

            df[col] = converted
            df[col] = df[col].fillna(df[col].mean())

        else:
            
            df[col] = df[col].astype(str)
            df[col] = df[col].fillna("unknown")


    df = df.loc[:, df.nunique() > 1]

    encoders = {}

    for col in df.columns[:-1]:
        if not is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    target = df.columns[-1]

    le_y = LabelEncoder()
    df[target] = le_y.fit_transform(df[target])

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    x = np.nan_to_num(x)

    # Feature Scaling
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x = np.clip(x, -10, 10)

    x = x.astype("float32")

    x = torch.tensor(x)
    y = torch.tensor(y, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # DataLoader
    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    num_features = x_train.shape[1]

    return train_loader, x_test, y_test, num_features