# prepreparing.py
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y