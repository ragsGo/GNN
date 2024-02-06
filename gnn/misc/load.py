import pandas as pd


def load(filename, y_column=0):
    df = pd.read_csv(filename, header=None)
    y = df[y_column].to_numpy()
    x = df[list(set(df.columns)-{y_column})].to_numpy()

    return x, y
