from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# wine_df = pd.read_csv('../newdataset/new_winequality.csv')
def load_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

# print(load_and_scale_data(wine_df, 'quality'))