import pandas as pd

def load_pima_data(path='D:\portfolio\Generalized-Linear-Models\data\pima_diabetes.csv'):
    return pd.read_csv(path)