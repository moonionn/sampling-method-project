from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# wine_df = pd.read_csv('../newdataset/new_winequality.csv')
def load_and_scale_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled, y

# print(load_and_scale_data(wine_df, 'quality'))